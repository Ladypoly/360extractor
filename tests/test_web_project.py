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

    def test_a_deleted_project_stops_being_offered(self, make_ui, tmp_path):
        """Listing a folder that is gone is offering the user an error."""
        Project.create(tmp_path / "a", name="alpha")
        Project.create(tmp_path / "b", name="beta")
        base, _ = make_ui()
        post(base, "/api/project/open", {"path": str(tmp_path / "a")})
        post(base, "/api/project/open", {"path": str(tmp_path / "b")})

        shutil.rmtree(tmp_path / "a")

        recent = get(base, "/api/recent")["recent"]
        assert [e["name"] for e in recent] == ["beta"]

    def test_forgetting_is_written_back(self, make_ui, tmp_path):
        """The pruning heals the file, not just the one response."""
        from threesixty import recent as recent_store

        Project.create(tmp_path / "a", name="alpha")
        base, _ = make_ui()
        post(base, "/api/project/open", {"path": str(tmp_path / "a")})
        shutil.rmtree(tmp_path / "a")

        get(base, "/api/recent")
        assert recent_store._read_raw() == []

    def test_an_unmounted_drive_is_kept_greyed(self, make_ui, tmp_path, monkeypatch):
        """An unplugged drive is not a deleted project: it comes back."""
        from threesixty import recent as recent_store

        Project.create(tmp_path / "a", name="alpha")
        base, _ = make_ui()
        post(base, "/api/project/open", {"path": str(tmp_path / "a")})
        shutil.rmtree(tmp_path / "a")
        monkeypatch.setattr(recent_store, "_volume_available", lambda path: False)

        recent = get(base, "/api/recent")["recent"]
        assert [e["name"] for e in recent] == ["alpha"]
        assert recent[0]["exists"] is False


class TestTwoStageCapture:
    def _wait(self, base, stage="capture", timeout=120):
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, snap = post(base, "/api/job/status", {"stage": stage})
            if snap.get("state") not in ("running", "pending"):
                return snap
            time.sleep(0.2)
        raise AssertionError(f"{stage} job did not finish in time")

    def test_extract_frames_runs_on_start_not_capture(self, make_ui, tmp_path,
                                                      equirect_clip):
        """The reported complaint: Capture spun, then ticked, during an import."""
        source = tmp_path / "drive.mp4"
        shutil.copy(equirect_clip, source)
        project = Project.create(tmp_path / "proj", sources=[str(source)])
        base, _ = make_ui(project)

        status, body = post(base, "/api/frames/extract", {"mode": "fps", "value": 5})
        assert status == 200 and body["stage"] == "start"
        assert self._wait(base, "start")["state"] == "done"

        _, capture = post(base, "/api/job/status", {"stage": "capture"})
        assert capture["state"] == "pending"
        # ...and the project does not claim its images exist yet either.
        assert Project.load(tmp_path / "proj").status("extract") == "pending"

    def test_extract_frames_then_generate_cameras(self, make_ui, tmp_path, equirect_clip):
        source = tmp_path / "drive.mp4"
        shutil.copy(equirect_clip, source)
        project = Project.create(tmp_path / "proj", sources=[str(source)])
        base, _ = make_ui(project)

        status, body = post(base, "/api/frames/extract", {"mode": "fps", "value": 5})
        assert status == 200 and body["started"]
        snap = self._wait(base, "start")
        assert snap["state"] == "done", snap
        assert snap["result"]["frames"] >= 1
        assert any((tmp_path / "proj" / "frames").glob("*.jpg"))

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
            assert (tmp_path / "proj" / "images" / f"c{i}").is_dir()

    def test_frames_list_and_serving(self, make_ui, tmp_path, equirect_clip):
        source = tmp_path / "drive.mp4"
        shutil.copy(equirect_clip, source)
        project = Project.create(tmp_path / "proj", sources=[str(source)])
        base, _ = make_ui(project)

        assert get(base, "/api/frames/list")["frames"] == []      # none yet

        post(base, "/api/frames/extract", {"mode": "fps", "value": 5})
        self._wait(base, "start")
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


class TestReconstructPoints:
    def _write_points(self, model_dir):
        import struct as st
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "points3D.bin", "wb") as handle:
            pts = [(1.0, 2.0, 3.0, 255, 0, 0), (4.0, 5.0, 6.0, 0, 255, 0)]
            handle.write(st.pack("<Q", len(pts)))
            for i, (x, y, z, r, g, b) in enumerate(pts):
                handle.write(st.pack("<Q", i + 1))
                handle.write(st.pack("<ddd", x, y, z))
                handle.write(st.pack("<BBB", r, g, b))
                handle.write(st.pack("<d", 0.1))
                handle.write(st.pack("<Q", 1))
                handle.write(st.pack("<ii", 1, 0))

    def test_serves_the_sparse_cloud_as_binary(self, make_ui, tmp_path):
        import struct as st
        project = Project.create(tmp_path / "proj")
        self._write_points(project.root / "sparse" / "0")
        base, _ = make_ui(project)

        with urllib.request.urlopen(base + "/api/reconstruct/points", timeout=30) as r:
            blob = r.read()
        mtime, count = st.unpack_from("<dI", blob, 0)
        assert count == 2 and mtime > 0
        assert st.unpack_from("<3f", blob, 12) == (1.0, 2.0, 3.0)      # first xyz

    def test_204_when_the_model_has_not_changed(self, make_ui, tmp_path):
        import struct as st
        project = Project.create(tmp_path / "proj")
        self._write_points(project.root / "sparse" / "0")
        base, _ = make_ui(project)
        with urllib.request.urlopen(base + "/api/reconstruct/points", timeout=30) as r:
            mtime = st.unpack_from("<dI", r.read(), 0)[0]
        with urllib.request.urlopen(
                f"{base}/api/reconstruct/points?since={mtime}", timeout=30) as r:
            assert r.status == 204

    def test_no_model_is_204(self, make_ui, tmp_path):
        project = Project.create(tmp_path / "proj")
        base, _ = make_ui(project)
        with urllib.request.urlopen(base + "/api/reconstruct/points", timeout=30) as r:
            assert r.status == 204


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


class TestFrameRemoval:
    """The "nothing was detected on these frames" warning, and acting on it."""

    def _dataset(self, tmp_path):
        from test_dataset import build_working_set
        from threesixty import dataset

        project = Project.create(tmp_path / "p")
        project.sources = ["clip.mp4"]
        project.save()
        build_working_set(project.root, "clip", ["c00", "c01"], ["00001", "00002"])
        dataset.export_dataset(project.root, "clip", ["c00", "c01"])
        return project

    def test_removing_a_frame_prunes_every_tree(self, make_ui, tmp_path):
        project = self._dataset(tmp_path)
        base, _ = make_ui(project)

        status, body = post(base, "/api/frames/remove", {"frames": ["00002"]})
        assert status == 200
        assert body["frames"] == 1 and body["images"] == 2 and body["remaining"] == 1
        assert not (project.root / "images" / "clip" / "c00" / "00002.jpg").exists()
        assert not (project.root / "RC_Dataset" / "view_00" / ".geometry"
                    / "frame_000002_v00.jpg").exists()

    def test_an_empty_list_is_refused(self, make_ui, tmp_path):
        base, _ = make_ui(self._dataset(tmp_path))
        status, body = post(base, "/api/frames/remove", {"frames": []})
        assert status == 400 and "no frames" in body["error"]

    def test_the_warning_survives_a_reload(self, make_ui, tmp_path):
        """A page refresh must not lose the list of frames nothing was detected on."""
        project = self._dataset(tmp_path)
        project.mark_done("extract", images=4)
        project.mark_done("mask", masks=4, undetected=["00002"])
        project.save()
        base, _ = make_ui(project)

        assert get(base, "/api/project")["project"]["undetected"] == ["00002"]

    def test_removing_a_frame_clears_it_from_the_warning(self, make_ui, tmp_path):
        project = self._dataset(tmp_path)
        project.mark_done("extract", images=4)
        project.mark_done("mask", masks=4, undetected=["00001", "00002"])
        project.save()
        base, _ = make_ui(project)

        post(base, "/api/frames/remove", {"frames": ["00002"]})
        assert get(base, "/api/project")["project"]["undetected"] == ["00001"]


class TestMaskFrameOverlay:
    """The Capture overlay: a mask per frame, answered from disk wherever possible."""

    def _project_with_frames(self, tmp_path, ffmpeg, equirect_clip):
        from threesixty.frames import extract_frames, frames_dir
        from threesixty.ffmpeg import probe_media as probe
        from threesixty.plan import FrameSelection

        project = Project.create(tmp_path / "p", sources=[str(equirect_clip)])
        media = probe(equirect_clip, ffmpeg)
        extract_frames(ffmpeg, media, FrameSelection(mode="fps", value=5.0), project.root)
        clip = equirect_clip.stem
        names = sorted(p.name for p in frames_dir(project.root, clip).glob("*.jpg"))
        return project, names

    def test_a_generated_mask_is_served_without_re_detecting(
            self, make_ui, ffmpeg, tmp_path, equirect_clip):
        """Once camera generation has masked a frame, the overlay is a file read --
        no model, no inference, which is what makes scrubbing bearable."""
        import subprocess as sp

        project, names = self._project_with_frames(tmp_path, ffmpeg, equirect_clip)
        equirect = project.root / ".threesixty" / "masks" / "equirect_masks"
        equirect.mkdir(parents=True)
        sp.run([str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=white:size=256x128",
                "-vf", "drawbox=x=0:y=0:w=256:h=30:color=black:t=fill",
                "-frames:v", "1", str(equirect / f"{Path(names[0]).stem}.png")],
               check=True, capture_output=True)
        project.mark_done("extract", images=1)
        project.mark_done("mask", masks=1)
        project.save()

        base, _ = make_ui(project)
        status, body = post(base, "/api/mask/frame", {"frame": names[0], "width": 256})
        assert status == 200 and body["source"] == "generated"

        with urllib.request.urlopen(base + body["url"], timeout=30) as response:
            assert response.read()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_a_stale_mask_is_not_reused(self, make_ui, ffmpeg, tmp_path, equirect_clip):
        """Masks written under settings that have since changed must not be served as
        if they still described what would happen."""
        project, names = self._project_with_frames(tmp_path, ffmpeg, equirect_clip)
        equirect = project.root / ".threesixty" / "masks" / "equirect_masks"
        equirect.mkdir(parents=True)
        (equirect / f"{Path(names[0]).stem}.png").write_bytes(b"not really a png")
        project.mark_done("extract", images=1)
        project.mark_done("mask", masks=1)
        project.detect.confidence = 0.42       # invalidates the mask fingerprint
        project.save()
        assert project.status("mask") == "stale"

        base, _ = make_ui(project)
        status, body = post(base, "/api/mask/frame", {"frame": names[0], "width": 256})
        # No ML installed in CI -> a clear error rather than the stale file.
        assert body.get("source") != "generated"

    def test_an_unknown_frame_is_refused(self, make_ui, ffmpeg, tmp_path, equirect_clip):
        project, _ = self._project_with_frames(tmp_path, ffmpeg, equirect_clip)
        base, _ = make_ui(project)
        status, body = post(base, "/api/mask/frame", {"frame": "99999.jpg"})
        assert status == 400 and "no such frame" in body["error"]

    def test_grading_works_from_an_extracted_frame(
            self, make_ui, ffmpeg, tmp_path, equirect_clip):
        """Capture has no decoded video preview in the two-stage flow, so Auto and the
        sliders have to grade the frame on the canvas instead of erroring."""
        project, names = self._project_with_frames(tmp_path, ffmpeg, equirect_clip)
        base, _ = make_ui(project)

        status, body = post(base, "/api/grade/auto", {"frame": names[0]})
        assert status == 200 and "exposure" in body["grade"]

        status, body = post(base, "/api/preview/grade",
                            {"frame": names[0], "width": 320,
                             "grade": {**body["grade"], "brightness": 0.2}})
        assert status == 200 and body["url"]
        with urllib.request.urlopen(base + body["url"], timeout=30) as response:
            assert response.read()[:2] == b"\xff\xd8"      # a real jpeg came back

    def test_grading_without_a_frame_or_preview_says_so(self, make_ui, tmp_path):
        base, _ = make_ui(Project.create(tmp_path / "p"))
        status, body = post(base, "/api/grade/auto", {})
        assert status == 400 and "load a source first" in body["error"]

    def test_a_graded_frame_answers_to_a_stable_name(
            self, make_ui, ffmpeg, tmp_path, equirect_clip):
        """Same frame, same grade -> same URL, so the browser keeps it cached."""
        project, names = self._project_with_frames(tmp_path, ffmpeg, equirect_clip)
        base, _ = make_ui(project)
        request = {"frame": names[0], "width": 320, "grade": {"brightness": 0.1}}

        _, first = post(base, "/api/preview/grade", request)
        _, again = post(base, "/api/preview/grade", request)
        assert first["url"] == again["url"]

    def test_frames_are_served_at_the_asked_width(
            self, make_ui, ffmpeg, tmp_path, equirect_clip):
        """The canvas asks for canvas-sized pixels, not the 8K original."""
        project, names = self._project_with_frames(tmp_path, ffmpeg, equirect_clip)
        base, _ = make_ui(project)
        clip = equirect_clip.stem

        with urllib.request.urlopen(
                f"{base}/frames/{clip}/{names[0]}?w=256", timeout=30) as response:
            small = response.read()
        with urllib.request.urlopen(
                f"{base}/frames/{clip}/{names[0]}", timeout=30) as response:
            full = response.read()
        assert len(small) < len(full)
