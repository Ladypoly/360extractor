"""Does real COLMAP accept what we write?

Everything else about the export is checked against our own understanding of the format.
This checks it against COLMAP itself, which is the only authority that matters. It skips
when COLMAP is absent rather than silently passing.

The specific claims under test are the two that would fail *quietly*:

* `rig_config.json` produces **one rig containing every camera**, not a separate rig per
  camera -- which is what happens when the file is ignored or misread.
* Each frame gathers **all** the cameras, which only works if their filenames match
  across folders.
"""

import sqlite3
import struct
import subprocess

import pytest

from threesixty.colmap import export as colmap_export
from threesixty.colmap import locate
from threesixty.extract import run_extraction
from threesixty.ffmpeg import probe_media
from threesixty.plan import FrameSelection, plan_extraction
from threesixty.rig import Camera, Output, Rig

pytestmark = [pytest.mark.ffmpeg, pytest.mark.colmap]

CAMERAS = 3
FRAMES = 4


@pytest.fixture(scope="module")
def colmap():
    found = locate.resolve()
    if found is None:
        pytest.skip("no COLMAP with rig support found")
    return found


@pytest.fixture
def dataset(ffmpeg, equirect_clip, tmp_path):
    """A real extraction, small but shaped exactly like a full one."""
    rig = Rig(
        cameras=[Camera(name=f"c{i:02d}", yaw=i * 120 - 120, h_fov=90, v_fov=67.5)
                 for i in range(CAMERAS)],
        output=Output(width=320, height=240, format="jpg", auto=False),
    )
    media = probe_media(equirect_clip, ffmpeg)
    plan = plan_extraction(media, rig, FrameSelection("fps", 2), tmp_path,
                           ffmpeg=ffmpeg, mask_mode="none")
    run_extraction(plan, ffmpeg)

    colmap_export.export(tmp_path, rig, "clip", media.width, has_masks=False)
    return tmp_path, rig


def run(colmap, *args):
    result = subprocess.run([str(colmap.path), *args], capture_output=True, text=True,
                            errors="replace", timeout=900)
    assert result.returncode == 0, (
        f"colmap {args[0]} failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return result.stdout + result.stderr


@pytest.fixture
def configured(colmap, dataset):
    """A database with our rig applied, which is the thing worth checking."""
    root, rig = dataset
    run(colmap, "feature_extractor",
        "--image_path", str(root / "images"),
        "--database_path", str(root / "database.db"),
        "--ImageReader.single_camera_per_folder", "1")
    output = run(colmap, "rig_configurator",
                 "--database_path", str(root / "database.db"),
                 "--rig_config_path", str(root / "rig_config.json"))
    return root, rig, output


class TestRigConfigurator:
    def test_our_rig_config_is_accepted(self, configured):
        _, _, output = configured
        assert "Configured: Rig(" in output

    def test_one_rig_holds_every_camera(self, configured):
        """A separate rig per camera means the file was ignored, and the whole point
        of the export is lost -- silently."""
        _, rig, output = configured
        rig_lines = [line for line in output.splitlines() if "Configured: Rig(" in line]
        assert len(rig_lines) == 1, f"expected exactly one rig, got:\n{rig_lines}"
        # One reference sensor plus the rest listed as sensors.
        assert rig_lines[0].count("(CAMERA,") == len(rig.cameras)

    def test_every_frame_gathers_all_the_cameras(self, configured):
        """Only possible because each camera's frame N shares a filename."""
        _, rig, output = configured
        frames = [line for line in output.splitlines() if "Configured: Frame(" in line]
        assert frames, "no frames were configured"
        for line in frames:
            assert line.count("(CAMERA,") == len(rig.cameras), (
                f"a frame did not collect every camera:\n{line}")

    def test_frame_count_matches_the_extraction(self, configured):
        _, _, output = configured
        frames = [line for line in output.splitlines() if "Configured: Frame(" in line]
        assert len(frames) == FRAMES


class TestIntrinsics:
    def test_our_exact_intrinsics_reach_the_database(self, configured):
        """`rig_configurator` should adopt the camera_params we supplied, rather than
        leaving COLMAP's guess in place."""
        root, rig, _ = configured
        database = sqlite3.connect(root / "database.db")
        rows = list(database.execute(
            "SELECT model, width, height, params, prior_focal_length FROM cameras"))
        database.close()

        assert len(rows) == len(rig.cameras)
        expected = colmap_export.focal_from_fov(320, 90.0)
        for model, width, height, blob, prior in rows:
            params = struct.unpack("<" + "d" * (len(blob) // 8), blob)
            assert model == 1, "expected PINHOLE (model id 1)"
            assert (width, height) == (320, 240)
            assert params[0] == pytest.approx(expected, rel=1e-6)
            assert params[2] == pytest.approx(160.0)
            assert prior == 1, "the focal length should be marked as known, not guessed"


class TestSparseRoundTrip:
    def test_we_can_read_a_model_colmap_wrote(self, colmap, tmp_path):
        """Our reader against COLMAP's own writer, via model_converter.

        Guards the binary parsing against COLMAP changing its layout.
        """
        from threesixty.colmap.model import (ColmapCamera, ColmapImage, read_model,
                                             write_cameras_text, write_images_text)

        source = tmp_path / "txt"
        cameras = {1: ColmapCamera(1, "PINHOLE", 640, 480, [500.0, 500.0, 320.0, 240.0])}
        images = {
            1: ColmapImage(1, (1, 0, 0, 0), (0.0, 0.0, 0.0), 1, "clip/c00/00001.jpg"),
            2: ColmapImage(2, (0.9238795, 0, 0.3826834, 0), (1.0, 2.0, 3.0), 1,
                           "clip/c01/00001.jpg"),
        }
        write_cameras_text(cameras, source / "cameras.txt")
        write_images_text(images, source / "images.txt")
        (source / "points3D.txt").write_text("", encoding="utf-8")

        binary = tmp_path / "bin"
        binary.mkdir()
        run(colmap, "model_converter", "--input_path", str(source),
            "--output_path", str(binary), "--output_type", "BIN")

        ours = read_model(binary)
        assert set(ours.images) == {1, 2}
        for key, original in images.items():
            import numpy as np
            assert np.allclose(ours.images[key].center, original.center, atol=1e-6)
            assert ours.images[key].name == original.name
