"""UI endpoints for painting occluders and measuring what they cost."""

import base64
import json
import tempfile
import socket
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

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


class TestRawSourceViews:
    """A raw two-lens file has two honest pictures, and the UI can ask for either."""

    FORMAT = {"projection": "dfisheye", "lens_fov": 190,
              "layout": "streams", "rotate": [90, -90]}

    def _size(self, ffmpeg, base, url):
        from threesixty.ffmpeg import probe_media
        import urllib.request
        name = url.rsplit("/", 1)[-1]
        with urllib.request.urlopen(base + f"/preview/{name}", timeout=60) as response:
            blob = response.read()
        path = Path(tempfile.mkdtemp()) / name
        path.write_bytes(blob)
        info = probe_media(path, ffmpeg)
        return info.width, info.height

    def test_the_panorama_is_the_default_even_for_a_raw_source(self, ui, two_stream_clip):
        """Silence means the panorama, always.

        Every tab but Start draws over this picture in equirect coordinates -- footprints,
        masks, the grade preview -- so a caller that does not ask for a view must never
        be handed two fisheye circles. Start asks for the lenses by name.
        """
        status, body = post(ui, "/api/preview",
                            {"path": str(two_stream_clip), "source_format": self.FORMAT})
        assert status == 200, body
        assert body["view"] == "panorama"

    def test_the_lens_view_is_there_for_the_asking(self, ui, two_stream_clip):
        status, body = post(ui, "/api/preview",
                            {"path": str(two_stream_clip), "source_format": self.FORMAT,
                             "view": "lenses"})
        # Two square lenses side by side come out 2:1, the same shape as a panorama --
        # so the canvas does not have to know which of the two it is showing.
        assert status == 200 and body["view"] == "lenses"

    def test_the_panorama_view_can_be_asked_for(self, ui, two_stream_clip):
        status, body = post(ui, "/api/preview",
                            {"path": str(two_stream_clip), "source_format": self.FORMAT,
                             "view": "panorama"})
        assert status == 200 and body["view"] == "panorama"

    def test_an_equirect_source_stays_on_the_panorama(self, ui, equirect_clip):
        status, body = post(ui, "/api/preview", {"path": str(equirect_clip)})
        assert status == 200 and body["view"] == "panorama"

    def test_the_file_suggests_what_it_is(self, ui, two_stream_clip):
        status, body = post(ui, "/api/preview", {"path": str(two_stream_clip)})
        assert status == 200
        suggested = body["media"]["suggested_source"]
        assert suggested["projection"] == "dfisheye"
        assert suggested["layout"] == "streams"
        assert suggested["rotate"] == [90.0, -90.0]

    def test_a_stitched_file_suggests_nothing(self, ui, equirect_clip):
        status, body = post(ui, "/api/preview", {"path": str(equirect_clip)})
        assert status == 200 and body["media"]["suggested_source"] is None

    def test_trimming_shows_up_on_the_lenses(self, ui, two_stream_clip):
        """The point of the trim: a ring cut off the rim of each circle, seen as such."""
        status, body = post(ui, "/api/mask/preview",
                            {"path": str(two_stream_clip), "view": "lenses",
                             "source_format": {**self.FORMAT, "trim": 10}})
        assert status == 200, body
        assert not body.get("empty"), "a trim of 10 degrees must mask something"

    def test_no_trim_masks_nothing(self, ui, two_stream_clip):
        status, body = post(ui, "/api/mask/preview",
                            {"path": str(two_stream_clip), "view": "lenses",
                             "source_format": self.FORMAT})
        assert status == 200 and body.get("empty")


class TestFitLensFov:
    def test_it_finds_the_field_of_view_the_lenses_were_built_with(self, ui,
                                                                  two_stream_clip):
        """The fixture's lenses are 190 degrees; the fit has to land on that.

        This is the number a spec sheet rounds, and being a few degrees out puts a
        visible step across the stitch line -- so it is measured, not guessed.
        """
        status, body = post(ui, "/api/source/fit",
                            {"path": str(two_stream_clip), "fit_rotation": False,
                             "source_format": {"projection": "dfisheye",
                                               "lens_fov": 180, "layout": "streams",
                                               "rotate": [90, -90]}})
        assert status == 200, body
        assert abs(body["lens_fov"] - 190) <= 2, body["scores"][:5]

    def test_it_works_out_how_the_lenses_are_mounted(self, ui, two_stream_clip):
        """Told nothing about the mounting, it has to find one that stitches.

        Only the *relative* rotation is measurable: turning both lenses over stitches
        exactly as well and puts the world upside down, so the pair is checked for the
        180-degree offset this camera has rather than for two specific numbers.
        """
        status, body = post(ui, "/api/source/fit",
                            {"path": str(two_stream_clip),
                             "source_format": {"projection": "dfisheye",
                                               "layout": "streams"}})
        assert status == 200, body
        first, second = body["rotate"]
        assert (first - second) % 360 == 180, body["rotations"]

    def test_it_refuses_a_source_with_only_one_lens(self, ui, equirect_clip):
        status, body = post(ui, "/api/source/fit", {"path": str(equirect_clip)})
        assert status == 400 and "dual-fisheye" in body["error"]


class TestWhatDetectionSees:
    """Masking is decided on the panorama, whichever view is on screen.

    The bug this pins: with the lenses showing, detection ran on *them* -- two circles in
    a black frame, which a detector trained on photographs reads as nothing much -- and
    the result was then reprojected as if it had been measured on a panorama. The preview
    came back with no mask at all while the panorama view masked the sky and the car
    correctly.
    """

    FORMAT = {"projection": "dfisheye", "lens_fov": 190,
              "layout": "streams", "rotate": [90, -90]}

    def test_the_detector_is_handed_a_panorama(self, ui, ffmpeg, two_stream_clip,
                                               monkeypatch):
        from threesixty.web import server as server_module

        seen = []

        class Fake:
            """Masks the left half of whatever it is given, and keeps the frame."""

            def detect(self, frames):
                import numpy as np

                seen.append(Path(frames[0]))
                from threesixty.ffmpeg import probe_media
                info = probe_media(frames[0], ffmpeg)
                mask = np.full((info.height, info.width), 255, dtype="uint8")
                mask[:, : info.width // 2] = 0
                return [type("Result", (), {"mask": mask})()]

        monkeypatch.setattr(server_module.Handler, "_mask_backend",
                            lambda self, detect: Fake())

        status, body = post(ui, "/api/mask/preview",
                            {"path": str(two_stream_clip), "view": "lenses",
                             "objects": True, "source_format": self.FORMAT,
                             "width": 512})
        assert status == 200, body
        assert not body.get("empty")
        assert seen, "detection never ran"

        # What it measured has to be the panorama, not the lens pair. Both are 2:1 at
        # this width, so the shape cannot tell them apart -- the pixels can.
        from threesixty.ffmpeg import probe_media
        measured = probe_media(seen[0], ffmpeg)
        assert (measured.width, measured.height) == (512, 256)

        reference = Path(tempfile.mkdtemp()) / "panorama.jpg"
        subprocess.run(
            [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(two_stream_clip), "-filter_complex",
             "[0:v:0]transpose=1[a];[0:v:1]transpose=2[b];[a][b]hstack=inputs=2[s];"
             "[s]v360=dfisheye:e:ih_fov=190:iv_fov=190:w=512:h=256[o]",
             "-map", "[o]", "-frames:v", "1", str(reference)],
            check=True, capture_output=True)
        assert psnr(ffmpeg, seen[0], reference) > 25, "detection saw the lenses, not the panorama"


def psnr(ffmpeg, first, second):
    """How alike two images are, in dB. Above ~25 is "the same picture"."""
    import re

    result = subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-i", str(first), "-i", str(second),
         "-lavfi", "psnr", "-f", "null", "-"], capture_output=True, text=True)
    match = re.search(r"average:([0-9.]+|inf)", result.stderr)
    assert match, result.stderr
    return float("inf") if match.group(1) == "inf" else float(match.group(1))
