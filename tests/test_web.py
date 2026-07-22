"""Tests for the local UI server.

The server is started for real on a free port and driven over HTTP, because the
things worth testing here -- routing, JSON shapes, the preview pipeline -- all live
at that boundary.
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
        ("127.0.0.1", port), type("Bound", (Handler,), {"session": session})
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return response.status, response.read(), response.headers.get("Content-Type", "")


def post(base, path, payload):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


class TestStatic:
    def test_serves_the_page(self, ui):
        status, body, content_type = get(ui, "/")
        assert status == 200
        assert "text/html" in content_type
        assert b"<canvas" in body

    def test_unknown_endpoint_is_404_json(self, ui):
        with pytest.raises(urllib.error.HTTPError) as info:
            get(ui, "/api/nope")
        assert info.value.code == 404


class TestPresets:
    def test_returns_every_preset_as_a_valid_rig(self, ui):
        status, body = post(ui, "/api/rig/validate", {"rig": {"cameras": []}})
        assert status == 400  # sanity: validation is actually wired up

        with urllib.request.urlopen(ui + "/api/presets", timeout=10) as response:
            presets = json.loads(response.read())["presets"]
        assert {"ring", "cube", "dome", "car-forward", "handheld"} <= set(presets)
        for name, rig in presets.items():
            status, body = post(ui, "/api/rig/validate", {"rig": rig})
            assert status == 200, f"preset {name} did not validate: {body}"
            assert body["enabled"] > 0


class TestProbeAndPreview:
    def test_probe_reports_media(self, ui, equirect_clip):
        status, body = post(ui, "/api/probe", {"path": str(equirect_clip)})
        assert status == 200
        assert body["media"]["width"] == 1024
        assert body["media"]["looks_equirectangular"] is True

    def test_probe_missing_file_is_a_clean_400(self, ui, tmp_path):
        status, body = post(ui, "/api/probe", {"path": str(tmp_path / "ghost.mp4")})
        assert status == 400
        assert "no such file" in body["error"]

    def test_preview_returns_a_fetchable_jpeg(self, ui, equirect_clip):
        status, body = post(ui, "/api/preview", {"path": str(equirect_clip), "time": 0})
        assert status == 200
        status, image, content_type = get(ui, body["url"])
        assert status == 200
        assert content_type == "image/jpeg"
        assert image[:2] == b"\xff\xd8"  # JPEG magic

    def test_camera_preview_uses_the_named_camera(self, ui, equirect_clip):
        rig = {
            "cameras": [
                {"name": "fwd", "yaw": 0, "pitch": 0, "h_fov": 90, "v_fov": 90},
                {"name": "back", "yaw": 180, "pitch": 0, "h_fov": 90, "v_fov": 90},
            ],
            "output": {"width": 200, "height": 200, "format": "png"},
        }
        images = []
        for name in ("fwd", "back"):
            status, body = post(ui, "/api/camera-preview", {
                "path": str(equirect_clip), "rig": rig, "camera": name, "time": 0, "width": 200,
            })
            assert status == 200, body
            _, data, _ = get(ui, body["url"])
            images.append(data)
        assert images[0] != images[1], "opposite cameras returned the same preview"

    def test_camera_preview_rejects_unknown_camera(self, ui, equirect_clip):
        status, body = post(ui, "/api/camera-preview", {
            "path": str(equirect_clip),
            "rig": {"cameras": [{"name": "fwd"}]},
            "camera": "nope", "time": 0,
        })
        assert status == 400
        assert "no enabled camera" in body["error"]


class TestRigFiles:
    def test_save_then_load_roundtrips(self, ui, tmp_path):
        rig = {
            "name": "test",
            "cameras": [{"name": "a", "yaw": 45, "pitch": -10, "h_fov": 90, "v_fov": 67.5}],
            "output": {"width": 640, "height": 480, "format": "png", "quality": 2},
            "occluders": [{"type": "nadir_cone", "angle": 40}],
        }
        target = tmp_path / "rigs" / "test.json"
        status, body = post(ui, "/api/rig/save", {"rig": rig, "path": str(target)})
        assert status == 200 and target.exists()

        status, body = post(ui, "/api/rig/load", {"path": str(target)})
        assert status == 200
        assert body["rig"]["cameras"][0]["yaw"] == 45
        assert body["rig"]["occluders"] == [{"type": "nadir_cone", "angle": 40}]

    def test_invalid_rig_is_rejected_with_a_reason(self, ui):
        status, body = post(ui, "/api/rig/validate", {
            "rig": {"cameras": [{"name": "a", "h_fov": 0}]},
        })
        assert status == 400
        assert "h_fov" in body["error"]

    def test_validate_surfaces_warnings(self, ui):
        status, body = post(ui, "/api/rig/validate", {
            "rig": {
                "cameras": [{"name": "a", "h_fov": 90, "v_fov": 90}],
                "output": {"width": 1920, "height": 1440},
            },
        })
        assert status == 200
        assert any("stretched" in w for w in body["warnings"])


class TestExtraction:
    def test_runs_and_reports_completion(self, ui, equirect_clip, tmp_path):
        rig = {
            "cameras": [{"name": f"c{i}", "yaw": i * 90 - 180, "pitch": 0,
                         "h_fov": 90, "v_fov": 90} for i in range(2)],
            "output": {"width": 160, "height": 160, "format": "png"},
        }
        status, body = post(ui, "/api/extract", {
            "sources": [str(equirect_clip)], "rig": rig,
            "mode": "fps", "value": 1, "output_dir": str(tmp_path / "out"),
        })
        assert status == 200 and body["started"]

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            _, data, _ = get(ui, "/api/progress")
            progress = json.loads(data)
            if progress["state"] not in ("running", "idle"):
                break
            time.sleep(0.2)

        assert progress["state"] == "done", progress
        assert progress["images"] == 4
        assert progress["fraction"] == 1.0

    def test_rejects_extraction_with_no_sources(self, ui):
        status, body = post(ui, "/api/extract", {
            "sources": [], "rig": {"cameras": [{"name": "a"}]},
        })
        assert status == 400
        assert "no source" in body["error"]

    def test_rejects_invalid_frame_selection(self, ui, equirect_clip):
        status, body = post(ui, "/api/extract", {
            "sources": [str(equirect_clip)],
            "rig": {"cameras": [{"name": "a"}]},
            "mode": "fps", "value": 0,
        })
        assert status == 400
        assert "positive" in body["error"]
