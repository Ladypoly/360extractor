"""Live grading in the UI.

The point of these endpoints is speed. Seeking an 8K source costs around 600 ms, which
is unusable under a slider, so the server keeps the decoded frame and re-grades *that*.
The browser asks for a small proxy while a slider is moving and the full frame when it
is released.
"""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from threesixty.web.server import Handler, Session

pytestmark = pytest.mark.ffmpeg


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def ui(ffmpeg):
    port = free_port()
    session = Session(ffmpeg)
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), type("Bound", (Handler,), {"session": session}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", session
    server.shutdown()
    server.server_close()


def post(base, path, payload):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def fetch(base, url):
    with urllib.request.urlopen(base + url, timeout=60) as response:
        return response.read()


class TestPreviewCache:
    def test_loading_a_preview_caches_the_ungraded_frame(self, ui, equirect_clip):
        base, session = ui
        assert session.preview_source is None
        status, _ = post(base, "/api/preview", {"path": str(equirect_clip), "time": 0})
        assert status == 200
        assert session.preview_source is not None
        assert session.preview_source.exists()

    def test_regrading_needs_a_preview_first(self, ui):
        base, _ = ui
        status, body = post(base, "/api/preview/grade", {"grade": {"exposure": 1.0}})
        assert status == 400
        assert "no preview loaded" in body["error"]

    def test_regrade_does_not_touch_the_video(self, ui, equirect_clip):
        """The cached frame is what gets graded, so the source can go away."""
        base, session = ui
        post(base, "/api/preview", {"path": str(equirect_clip), "time": 0})
        cached = session.preview_source

        status, body = post(base, "/api/preview/grade",
                            {"grade": {"exposure": 1.0}, "width": 320})
        assert status == 200
        # Same cached source, a new output file.
        assert session.preview_source == cached
        assert body["url"].endswith(".jpg")

    def test_grading_changes_the_picture(self, ui, equirect_clip):
        base, _ = ui
        post(base, "/api/preview", {"path": str(equirect_clip), "time": 0})

        _, plain = post(base, "/api/preview/grade", {"grade": {}, "width": 320})
        _, bright = post(base, "/api/preview/grade",
                         {"grade": {"exposure": 1.5}, "width": 320})
        assert fetch(base, plain["url"]) != fetch(base, bright["url"])

    def test_width_is_honoured_and_capped(self, ui, equirect_clip):
        base, _ = ui
        post(base, "/api/preview", {"path": str(equirect_clip), "time": 0})

        _, small = post(base, "/api/preview/grade", {"grade": {}, "width": 240})
        assert small["width"] == 240
        _, huge = post(base, "/api/preview/grade", {"grade": {}, "width": 99999})
        assert huge["width"] <= 1600, "an unbounded width would be a denial of service"

    def test_the_proxy_is_smaller_and_quicker(self, ui, equirect_clip):
        """What makes the slider usable: the proxy must actually be cheaper."""
        base, _ = ui
        post(base, "/api/preview", {"path": str(equirect_clip), "time": 0})

        started = time.monotonic()
        _, proxy = post(base, "/api/preview/grade",
                        {"grade": {"exposure": 0.5}, "width": 320})
        proxy_time = time.monotonic() - started

        _, full = post(base, "/api/preview/grade",
                       {"grade": {"exposure": 0.5}, "width": 1600})
        assert len(fetch(base, proxy["url"])) < len(fetch(base, full["url"]))
        assert proxy_time < 5.0, "a regrade has to be fast enough to sit under a slider"

    def test_a_bad_grade_value_is_a_clean_400(self, ui, equirect_clip):
        base, _ = ui
        post(base, "/api/preview", {"path": str(equirect_clip), "time": 0})
        status, body = post(base, "/api/preview/grade",
                            {"grade": {"exposure": 99.0}, "width": 320})
        assert status == 400
        assert "grade.exposure" in body["error"]


class TestAutoEndpoint:
    def test_needs_a_loaded_source(self, ui):
        base, _ = ui
        status, body = post(base, "/api/grade/auto", {})
        assert status == 400
        assert "load a source first" in body["error"]

    def test_returns_a_grade_and_an_explanation(self, ui, equirect_clip):
        base, _ = ui
        post(base, "/api/preview", {"path": str(equirect_clip), "time": 0})
        status, body = post(base, "/api/grade/auto", {})
        assert status == 200
        assert set(body["grade"]) >= {"exposure", "contrast", "saturation"}
        assert any("median" in line for line in body["notes"])

    def test_the_proposed_grade_is_applicable(self, ui, equirect_clip):
        """Whatever auto proposes must be something the regrade endpoint accepts."""
        base, _ = ui
        post(base, "/api/preview", {"path": str(equirect_clip), "time": 0})
        _, auto = post(base, "/api/grade/auto", {})
        status, _ = post(base, "/api/preview/grade",
                         {"grade": auto["grade"], "width": 320})
        assert status == 200
