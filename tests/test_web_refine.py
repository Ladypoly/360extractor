"""The Refine tab's endpoints.

Detection itself is covered in test_ml.py; what matters here is that the tab can
describe an extracted dataset and composite a mask over a frame, since that preview is
what the user checks before committing to a long reconstruction.
"""

import json
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from threesixty.extract import run_extraction
from threesixty.ffmpeg import probe_media
from threesixty.plan import FrameSelection, plan_extraction
from threesixty.project import Project
from threesixty.rig import Camera, Output, Rig
from threesixty.web.server import Handler, Session

pytestmark = pytest.mark.ffmpeg


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def dataset(ffmpeg, equirect_clip, tmp_path):
    """A small real extraction with a nadir cone, so masks exist on disk."""
    rig = Rig(
        cameras=[Camera(name="a", yaw=0, pitch=-20, h_fov=90, v_fov=90),
                 Camera(name="b", yaw=90, pitch=-20, h_fov=90, v_fov=90)],
        output=Output(width=160, height=160, format="jpg", auto=False),
        occluders=[{"type": "nadir_cone", "angle": 20}],
    )
    project = Project.create(tmp_path / "p", sources=[str(equirect_clip)], rig=rig)
    media = probe_media(equirect_clip, ffmpeg)
    plan = plan_extraction(media, rig, FrameSelection("fps", 1), project.root,
                           ffmpeg=ffmpeg, mask_mode="sidecar")
    run_extraction(plan, ffmpeg)
    return project


@pytest.fixture
def ui(ffmpeg, dataset):
    port = free_port()
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        type("Bound", (Handler,), {"session": Session(ffmpeg, dataset)}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
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


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as response:
        return response.status, response.read(), response.headers.get("Content-Type", "")


class TestStatus:
    def test_reports_whether_the_ml_extra_is_installed(self, ui):
        status, body, _ = get(ui, "/api/detect/status")
        assert status == 200
        assert isinstance(json.loads(body)["available"], bool)


class TestFrames:
    def test_lists_cameras_and_their_frames(self, ui):
        status, body = post(ui, "/api/detect/frames", {})
        assert status == 200
        names = {c["name"] for c in body["cameras"]}
        assert names == {"a", "b"}
        assert all(c["frames"] for c in body["cameras"])

    def test_counts_masks_already_on_disk(self, ui):
        """The cone is at -20 and the cameras look down, so masks were written."""
        status, body = post(ui, "/api/detect/frames", {})
        assert status == 200
        assert body["masked"] > 0

    def test_needs_an_open_project(self, ffmpeg):
        port = free_port()
        server = ThreadingHTTPServer(
            ("127.0.0.1", port),
            type("Bound", (Handler,), {"session": Session(ffmpeg, None)}))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            status, body = post(f"http://127.0.0.1:{port}", "/api/detect/frames", {})
            assert status == 400
            assert "no project is open" in body["error"]
        finally:
            server.shutdown()
            server.server_close()


class TestPreview:
    def _first_frame(self, ui):
        _, body = post(ui, "/api/detect/frames", {})
        camera = body["cameras"][0]
        return camera["name"], camera["frames"][0]

    def test_composites_a_fetchable_image(self, ui):
        name, frame = self._first_frame(ui)
        status, body = post(ui, "/api/detect/preview",
                            {"camera": name, "frame": frame, "opacity": 0.5})
        assert status == 200
        assert body["has_mask"] is True

        status, data, content_type = get(ui, body["url"])
        assert status == 200
        assert content_type == "image/jpeg"
        assert data[:2] == b"\xff\xd8"

    def test_overlay_opacity_changes_the_result(self, ui):
        """Proves the mask is actually composited rather than the frame passed through."""
        name, frame = self._first_frame(ui)
        images = []
        for opacity in (0.0, 0.9):
            _, body = post(ui, "/api/detect/preview",
                           {"camera": name, "frame": frame, "opacity": opacity})
            images.append(get(ui, body["url"])[1])
        assert images[0] != images[1]

    def test_unknown_camera_is_a_clean_400(self, ui):
        status, body = post(ui, "/api/detect/preview", {"camera": "nope", "frame": 1})
        assert status == 400
        assert "no camera named" in body["error"]

    def test_unknown_frame_is_a_clean_400(self, ui):
        name, _ = self._first_frame(ui)
        status, body = post(ui, "/api/detect/preview", {"camera": name, "frame": 99999})
        assert status == 400
        assert "no frame" in body["error"]


class TestRun:
    def test_settings_are_stored_on_the_project(self, ui, dataset):
        """Even when detection cannot start, the choices belong to the project."""
        status, body = post(ui, "/api/detect/run", {
            "backend": "yolo", "classes": ["person"], "confidence": 0.4,
            "dilate": 9, "fuse": False,
        })
        from threesixty.mask import ml
        if not ml.available():
            assert status == 400 and "ML extra" in body["error"]
            return

        assert status == 200
        reopened = Project.load(dataset.root)
        assert reopened.detect.backend == "yolo"
        assert reopened.detect.classes == ["person"]
        assert reopened.detect.confidence == 0.4
        assert reopened.detect.dilate == 9
        assert reopened.detect.fuse is False
