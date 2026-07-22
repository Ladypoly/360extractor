"""UI endpoints for painting occluders and measuring what they cost."""

import base64
import json
import socket
import subprocess
import threading
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
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), type("Bound", (Handler,), {"session": Session(ffmpeg)}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
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


def data_url(ffmpeg, tmp_path, filter_chain, size="512x256"):
    """A PNG data: URL, built the way the browser would send one."""
    path = tmp_path / "painted.png"
    subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=white:size={size}",
         *(["-vf", filter_chain] if filter_chain else []),
         "-frames:v", "1", str(path)], check=True, capture_output=True)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


class TestPaint:
    def test_stores_a_painted_mask(self, ui, ffmpeg, tmp_path):
        painted = data_url(ffmpeg, tmp_path,
                           "drawbox=x=0:y=170:w=512:h=86:color=black:t=fill")
        status, body = post(ui, "/api/mask/paint", {"image": painted})
        assert status == 200
        assert body["path"] and body["path"].endswith(".png")

    def test_blank_paint_stores_nothing(self, ui, ffmpeg, tmp_path):
        """Clearing the brush must not leave a no-op mask behind.

        An all-white mask changes nothing but still forces every camera into masked
        alpha handling and costs one file per extracted frame.
        """
        status, body = post(ui, "/api/mask/paint",
                            {"image": data_url(ffmpeg, tmp_path, "")})
        assert status == 200
        assert body["path"] is None

    def test_missing_image_is_a_clean_400(self, ui):
        status, body = post(ui, "/api/mask/paint", {"image": ""})
        assert status == 400
        assert "no image data" in body["error"]


class TestCoverage:
    def _rig(self, occluders, pitch=0.0):
        return {
            "cameras": [{"name": "level", "yaw": 0, "pitch": pitch,
                         "h_fov": 90, "v_fov": 67.5},
                        {"name": "down", "yaw": 90, "pitch": -70,
                         "h_fov": 60, "v_fov": 60}],
            "output": {"auto": True},
            "occluders": occluders,
        }

    def test_no_occluders_means_no_coverage(self, ui):
        status, body = post(ui, "/api/mask/coverage", {"rig": self._rig([])})
        assert status == 200
        assert body["coverage"] == {}

    def test_nadir_cone_hits_the_downward_camera_hardest(self, ui):
        status, body = post(ui, "/api/mask/coverage", {
            "rig": self._rig([{"type": "nadir_cone", "angle": 30}]),
            "source_width": 2048, "source_height": 1024,
        })
        assert status == 200
        coverage = body["coverage"]
        assert coverage["down"] > 0.95, "a camera aimed into the cone is nearly all occluder"
        assert coverage["level"] < 0.2, "a level camera should barely be touched"

    def test_painted_occluder_is_measured(self, ui, ffmpeg, tmp_path):
        """The whole point of measuring server-side: an arbitrary painted shape."""
        painted = data_url(ffmpeg, tmp_path,
                           "drawbox=x=0:y=128:w=512:h=128:color=black:t=fill")
        status, body = post(ui, "/api/mask/paint", {"image": painted})
        assert status == 200 and body["path"]

        status, body = post(ui, "/api/mask/coverage", {
            "rig": self._rig([{"type": "equirect_mask", "path": body["path"]}]),
            "source_width": 2048, "source_height": 1024,
        })
        assert status == 200
        # Everything below the horizon was painted out.
        assert body["coverage"]["down"] > 0.95
        assert 0.3 < body["coverage"]["level"] < 0.7

    def test_coverage_rejects_a_broken_rig(self, ui):
        status, body = post(ui, "/api/mask/coverage", {
            "rig": {"cameras": [{"name": "a", "h_fov": 0}],
                    "occluders": [{"type": "nadir_cone", "angle": 20}]},
        })
        assert status == 400
        assert "h_fov" in body["error"]
