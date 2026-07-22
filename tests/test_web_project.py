"""Project endpoints, and the painted occluder living inside the project.

The occluder used to be written into a temp directory, which meant the rig referenced a
file that vanished on reboot. These tests pin it to the project.
"""

import base64
import json
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from threesixty.project import Project
from threesixty.web.server import Handler, Session

pytestmark = pytest.mark.ffmpeg


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
