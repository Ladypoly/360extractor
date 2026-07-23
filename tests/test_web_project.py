"""Project endpoints, and the painted occluder living inside the project.

The occluder used to be written into a temp directory, which meant the rig referenced a
file that vanished on reboot. These tests pin it to the project.
"""

import base64
import json
import shutil
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from threesixty.project import Project
from threesixty.web.server import Handler, Session

pytestmark = pytest.mark.ffmpeg

# A due-east track at the equator, ~100 m/s, 4 s -> ~300 m total.
GPX = """<?xml version="1.0"?>
<gpx version="1.1"><trk><trkseg>
<trkpt lat="0.0" lon="0.000000"><time>2020-01-01T00:00:00Z</time></trkpt>
<trkpt lat="0.0" lon="0.000898"><time>2020-01-01T00:00:01Z</time></trkpt>
<trkpt lat="0.0" lon="0.001796"><time>2020-01-01T00:00:02Z</time></trkpt>
<trkpt lat="0.0" lon="0.002694"><time>2020-01-01T00:00:03Z</time></trkpt>
</trkseg></trk></gpx>
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def make_ui(ffmpeg):
    servers = []

    def build(project=None):
        port = free_port()
        session = Session(ffmpeg, project)
        server = ThreadingHTTPServer(
            ("127.0.0.1", port), type("Bound", (Handler,), {"session": session}))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{port}", session

    yield build
    for server in servers:
        server.shutdown()
        server.server_close()


def post(base, path, payload):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return json.loads(response.read())


class TestProjectEndpoints:
    def test_reports_no_project_when_none_is_open(self, make_ui):
        base, _ = make_ui()
        assert get(base, "/api/project")["project"] is None

    def test_reports_the_project_the_server_started_with(self, make_ui, tmp_path):
        project = Project.create(tmp_path / "p", name="startup")
        base, _ = make_ui(project)
        payload = get(base, "/api/project")["project"]
        assert payload["name"] == "startup"
        assert payload["stages"] == {"extract": "pending", "mask": "pending",
                                     "export": "pending"}

    def test_new_then_open_round_trips(self, make_ui, tmp_path):
        base, _ = make_ui()
        status, body = post(base, "/api/project/new",
                            {"root": str(tmp_path / "fresh"), "sources": []})
        assert status == 200

        status, body = post(base, "/api/project/open", {"path": str(tmp_path / "fresh")})
        assert status == 200
        assert body["project"]["name"] == "fresh"

    def test_new_refuses_to_clobber(self, make_ui, tmp_path):
        Project.create(tmp_path / "p")
        base, _ = make_ui()
        status, body = post(base, "/api/project/new", {"root": str(tmp_path / "p")})
        assert status == 400
        assert "already exists" in body["error"]

    def test_opening_something_that_is_not_a_project(self, make_ui, tmp_path):
        base, _ = make_ui()
        status, body = post(base, "/api/project/open", {"path": str(tmp_path)})
        assert status == 400
        assert "project new" in body["error"]

    def test_save_persists_the_ui_state(self, make_ui, tmp_path):
        project = Project.create(tmp_path / "p")
        base, _ = make_ui(project)

        status, body = post(base, "/api/project/save", {
            "rig": {"cameras": [{"name": "solo", "yaw": 33, "h_fov": 90, "v_fov": 67.5}],
                    "output": {"auto": True}},
            "frames": {"mode": "every", "value": 7},
            "output": {"mask_mode": "burn"},
        })
        assert status == 200

        reopened = Project.load(tmp_path / "p")
        assert [c.name for c in reopened.rig.cameras] == ["solo"]
        assert reopened.frames.mode == "every"
        assert reopened.frames.value == 7
        assert reopened.output.mask_mode == "burn"

    def test_save_can_take_a_snapshot_at_the_same_time(self, make_ui, tmp_path):
        project = Project.create(tmp_path / "p")
        base, _ = make_ui(project)
        status, _ = post(base, "/api/project/save", {"snapshot": "before-change"})
        assert status == 200
        assert Project.load(tmp_path / "p").snapshots() == ["before-change"]

    def test_save_without_an_open_project_needs_a_root(self, make_ui):
        base, _ = make_ui()
        status, body = post(base, "/api/project/save", {"frames": {"value": 3}})
        assert status == 400
        assert "choose a folder" in body["error"]

    def test_partial_settings_do_not_wipe_the_rest(self, make_ui, tmp_path):
        """The UI sends only what it holds; unmentioned settings must survive."""
        project = Project.create(tmp_path / "p")
        project.detect.confidence = 0.6
        project.save()

        base, _ = make_ui(Project.load(tmp_path / "p"))
        post(base, "/api/project/save", {"frames": {"value": 4}})

        reopened = Project.load(tmp_path / "p")
        assert reopened.detect.confidence == 0.6
        assert reopened.frames.value == 4


class TestOpenSourceCreatesProject:
    """Opening a video is opening a project: one is created in a folder beside it."""

    def test_creates_a_project_in_a_subfolder_named_after_the_clip(self, make_ui, tmp_path):
        source = tmp_path / "Q360_0001.mp4"
        source.write_bytes(b"not really a video")
        base, session = make_ui()

        status, body = post(base, "/api/project/for-source", {"path": str(source)})
        assert status == 200
        assert body["project"]["root"] == str(tmp_path / "Q360_0001")
        assert (tmp_path / "Q360_0001" / "project.json").exists()
        assert session.project is not None
        # The source is registered, so export and extraction have something to run on.
        assert body["project"]["sources"] == [str(source)]

    def test_seeds_a_new_project_from_the_ui_settings(self, make_ui, tmp_path):
        source = tmp_path / "clip.mp4"
        source.write_bytes(b"x")
        base, _ = make_ui()

        post(base, "/api/project/for-source", {
            "path": str(source), "frames": {"mode": "every", "value": 9}})
        assert Project.load(tmp_path / "clip").frames.value == 9

    def test_reopening_the_same_clip_resumes_its_project(self, make_ui, tmp_path):
        source = tmp_path / "clip.mp4"
        source.write_bytes(b"x")
        existing = Project.create(tmp_path / "clip", name="clip")
        existing.frames.value = 5
        existing.save()
        base, _ = make_ui()

        # Disk wins over whatever the UI is currently showing.
        _, body = post(base, "/api/project/for-source", {
            "path": str(source), "frames": {"mode": "fps", "value": 99}})
        assert body["project"]["frames"]["value"] == 5

    def test_missing_source_is_rejected(self, make_ui, tmp_path):
        base, _ = make_ui()
        status, body = post(base, "/api/project/for-source",
                            {"path": str(tmp_path / "nope.mp4")})
        assert status == 400
        assert "does not exist" in body["error"]


class TestRecentProjects:
    def test_opened_projects_appear_newest_first(self, make_ui, tmp_path):
        Project.create(tmp_path / "a", name="alpha")
        Project.create(tmp_path / "b", name="beta")
        base, _ = make_ui()

        post(base, "/api/project/open", {"path": str(tmp_path / "a")})
        post(base, "/api/project/open", {"path": str(tmp_path / "b")})

        recent = get(base, "/api/recent")["recent"]
        assert [e["name"] for e in recent] == ["beta", "alpha"]
        assert all(e["exists"] for e in recent)

    def test_reopening_moves_to_front_without_duplicating(self, make_ui, tmp_path):
        Project.create(tmp_path / "a", name="alpha")
        Project.create(tmp_path / "b", name="beta")
        base, _ = make_ui()
        for name in ("a", "b", "a"):
            post(base, "/api/project/open", {"path": str(tmp_path / name)})

        recent = get(base, "/api/recent")["recent"]
        assert [e["name"] for e in recent] == ["alpha", "beta"]

    def test_remove_drops_an_entry(self, make_ui, tmp_path):
        Project.create(tmp_path / "a", name="alpha")
        base, _ = make_ui()
        post(base, "/api/project/open", {"path": str(tmp_path / "a")})

        status, body = post(base, "/api/recent/remove",
                            {"root": str((tmp_path / "a").resolve())})
        assert status == 200
        assert body["recent"] == []


class TestTwoStageCapture:
    def _wait(self, base, timeout=120):
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, snap = post(base, "/api/job/status", {"stage": "capture"})
            if snap.get("state") not in ("running", "pending"):
                return snap
            time.sleep(0.2)
        raise AssertionError("capture job did not finish in time")

    def test_extract_frames_then_generate_cameras(self, make_ui, tmp_path, equirect_clip):
        source = tmp_path / "drive.mp4"
        shutil.copy(equirect_clip, source)
        project = Project.create(tmp_path / "proj", sources=[str(source)])
        base, _ = make_ui(project)

        status, body = post(base, "/api/frames/extract", {"mode": "fps", "value": 5})
        assert status == 200 and body["started"]
        snap = self._wait(base)
        assert snap["state"] == "done", snap
        assert snap["result"]["frames"] >= 1
        assert (tmp_path / "proj" / "frames" / "drive").is_dir()

        rig = {"cameras": [{"name": f"c{i}", "yaw": i * 180 - 90, "pitch": 0,
                            "h_fov": 90, "v_fov": 90} for i in range(2)],
               "output": {"width": 160, "height": 160, "format": "jpg"}}
        status, body = post(base, "/api/cameras/generate", {"rig": rig})
        assert status == 200 and body["started"]
        snap = self._wait(base)
        assert snap["state"] == "done", snap
        assert snap["result"]["images"] > 0

        reopened = Project.load(tmp_path / "proj")
        assert reopened.status("extract") == "done"
        for i in range(2):
            assert (tmp_path / "proj" / "images" / "drive" / f"c{i}").is_dir()

    def test_frames_list_and_serving(self, make_ui, tmp_path, equirect_clip):
        source = tmp_path / "drive.mp4"
        shutil.copy(equirect_clip, source)
        project = Project.create(tmp_path / "proj", sources=[str(source)])
        base, _ = make_ui(project)

        assert get(base, "/api/frames/list")["frames"] == []      # none yet

        post(base, "/api/frames/extract", {"mode": "fps", "value": 5})
        self._wait(base)
        listing = get(base, "/api/frames/list")
        assert listing["clip"] == "drive" and len(listing["frames"]) >= 1

        with urllib.request.urlopen(
                f"{base}/frames/drive/{listing['frames'][0]}", timeout=30) as response:
            assert response.status == 200
            assert response.read(3) == b"\xff\xd8\xff"            # JPEG magic

    def test_generate_before_extract_is_a_clear_error(self, make_ui, tmp_path,
                                                      equirect_clip):
        source = tmp_path / "drive.mp4"
        shutil.copy(equirect_clip, source)
        project = Project.create(tmp_path / "proj", sources=[str(source)])
        base, _ = make_ui(project)
        status, body = post(base, "/api/cameras/generate", {})
        assert status == 400
        assert "extract frames" in body["error"]


class TestMaskPreview:
    def test_returns_a_tinted_preview_for_the_sky_cone(self, make_ui, tmp_path,
                                                       equirect_clip):
        source = tmp_path / "drive.mp4"
        shutil.copy(equirect_clip, source)
        project = Project.create(tmp_path / "proj", sources=[str(source)])
        base, _ = make_ui(project)

        status, body = post(base, "/api/mask/preview",
                            {"path": str(source), "sky_cone_angle": 30})
        assert status == 200 and body["url"].startswith("/preview/")
        with urllib.request.urlopen(base + body["url"], timeout=30) as response:
            assert response.status == 200
            assert response.read(3) == b"\xff\xd8\xff"        # JPEG

    def test_no_occluders_returns_the_plain_frame(self, make_ui, tmp_path, equirect_clip):
        source = tmp_path / "drive.mp4"
        shutil.copy(equirect_clip, source)
        base, _ = make_ui(Project.create(tmp_path / "proj", sources=[str(source)]))
        status, body = post(base, "/api/mask/preview", {"path": str(source)})
        assert status == 200 and body.get("empty") is True


class TestSegmentEndpoint:
    def _drive(self, tmp_path, equirect_clip):
        source = tmp_path / "drive.mp4"
        shutil.copy(equirect_clip, source)
        return source

    def test_duration_creates_a_project_per_segment(self, make_ui, tmp_path, equirect_clip):
        source = self._drive(tmp_path, equirect_clip)          # 2 s clip
        base, _ = make_ui()
        status, body = post(base, "/api/segment",
                            {"path": str(source), "mode": "duration", "seconds": 1.0})
        assert status == 200
        segs = body["segments"]
        assert len(segs) == 2
        assert segs[0]["name"] == "drive_seg01"
        for seg in segs:
            root = Path(seg["root"])
            assert (root / "project.json").exists()
            loaded = Project.load(root)
            assert loaded.frames.start == pytest.approx(seg["start"], abs=0.01)
            assert loaded.frames.end == pytest.approx(seg["end"], abs=0.01)
            assert loaded.sources == [str(source)]

    def test_gpx_mode_cuts_by_distance(self, make_ui, tmp_path, equirect_clip):
        source = self._drive(tmp_path, equirect_clip)
        (tmp_path / "drive.gpx").write_text(GPX, encoding="utf-8")
        base, _ = make_ui()
        status, body = post(base, "/api/segment",
                            {"path": str(source), "mode": "gpx", "meters": 100.0})
        assert status == 200
        assert len(body["segments"]) >= 2       # ~300 m / 100 m
        assert body["segments"][0]["distance"] == pytest.approx(100.0, rel=0.02)

    def test_gpx_mode_without_a_sidecar_is_a_clear_error(self, make_ui, tmp_path,
                                                         equirect_clip):
        source = self._drive(tmp_path, equirect_clip)
        base, _ = make_ui()
        status, body = post(base, "/api/segment",
                            {"path": str(source), "mode": "gpx", "meters": 100.0})
        assert status == 400
        assert "GPX" in body["error"]

    def test_unknown_mode_errors(self, make_ui, tmp_path, equirect_clip):
        source = self._drive(tmp_path, equirect_clip)
        base, _ = make_ui()
        status, _ = post(base, "/api/segment", {"path": str(source), "mode": "nope"})
        assert status == 400


class TestRigPresets:
    RIG = {"cameras": [{"name": "solo", "yaw": 0, "h_fov": 90, "v_fov": 67.5}],
           "output": {"auto": True}}

    def test_builtins_are_listed(self, make_ui):
        base, _ = make_ui()
        body = get(base, "/api/presets")
        assert {"ring", "cube"} <= set(body["presets"])
        assert body["user"] == []

    def test_saved_preset_joins_the_list_and_survives_reload(self, make_ui, tmp_path):
        base, _ = make_ui()
        status, body = post(base, "/api/preset/save", {"name": "my rig", "rig": self.RIG})
        assert status == 200
        assert "my rig" in body["presets"]
        assert body["user"] == ["my rig"]

        # A fresh server (same state dir) still has it -- presets are global, not per-run.
        base2, _ = make_ui()
        assert "my rig" in get(base2, "/api/presets")["presets"]

    def test_save_refuses_a_builtin_name(self, make_ui):
        base, _ = make_ui()
        status, body = post(base, "/api/preset/save", {"name": "ring", "rig": self.RIG})
        assert status == 400
        assert "built-in" in body["error"]

    def test_save_rejects_an_empty_name(self, make_ui):
        base, _ = make_ui()
        status, body = post(base, "/api/preset/save", {"name": "  ", "rig": self.RIG})
        assert status == 400

    def test_project_specific_occluders_are_stripped(self, make_ui):
        base, _ = make_ui()
        rig = {**self.RIG, "occluders": [
            {"type": "nadir_cone", "angle": 20},
            {"type": "equirect_mask", "path": "C:/proj/assets/painted.png"}]}
        _, body = post(base, "/api/preset/save", {"name": "coned", "rig": rig})
        kept = body["presets"]["coned"]["occluders"]
        assert kept == [{"type": "nadir_cone", "angle": 20}]

    def test_delete_removes_a_saved_preset(self, make_ui):
        base, _ = make_ui()
        post(base, "/api/preset/save", {"name": "temp", "rig": self.RIG})
        status, body = post(base, "/api/preset/delete", {"name": "temp"})
        assert status == 200
        assert "temp" not in body["presets"]

    def test_delete_refuses_a_builtin(self, make_ui):
        base, _ = make_ui()
        status, body = post(base, "/api/preset/delete", {"name": "cube"})
        assert status == 400
        assert "cube" in get(base, "/api/presets")["presets"]


class TestPaintedOccluderLocation:
    def _painted(self, ffmpeg, tmp_path):
        path = tmp_path / "painted.png"
        subprocess.run(
            [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=white:size=256x128",
             "-vf", "drawbox=x=0:y=90:w=256:h=38:color=black:t=fill",
             "-frames:v", "1", str(path)], check=True, capture_output=True)
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

    def test_painted_mask_goes_into_the_project(self, make_ui, ffmpeg, tmp_path):
        """Regression: it used to land in a temp folder wiped on reboot, leaving the
        rig pointing at an occluder that no longer existed."""
        project = Project.create(tmp_path / "p")
        base, _ = make_ui(project)

        status, body = post(base, "/api/mask/paint",
                            {"image": self._painted(ffmpeg, tmp_path)})
        assert status == 200 and body["path"]

        stored = tmp_path / "p" / "assets" / "painted_occluder.png"
        assert stored.exists()
        assert body["path"] == str(stored)

    def test_without_a_project_it_still_works(self, make_ui, ffmpeg, tmp_path):
        base, session = make_ui()
        status, body = post(base, "/api/mask/paint",
                            {"image": self._painted(ffmpeg, tmp_path)})
        assert status == 200 and body["path"]
        assert str(session.cache) in body["path"]
