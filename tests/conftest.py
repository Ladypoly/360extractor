import subprocess

import pytest

from threesixty.ffmpeg import FFmpegError, resolve_ffmpeg


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    """Keep the recent-projects list out of the developer's real home directory."""
    monkeypatch.setenv("THREESIXTY_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture(scope="session")
def ffmpeg():
    """The ffmpeg 360extract would use, or skip the test."""
    try:
        return resolve_ffmpeg()
    except FFmpegError as exc:
        pytest.skip(f"no usable ffmpeg: {exc}")


@pytest.fixture(scope="session")
def equirect_clip(ffmpeg, tmp_path_factory):
    """A synthetic 2:1 clip.

    testsrc2 is deliberate: it is spatially distinctive, so a yaw error or an
    axis flip is visible rather than plausible.
    """
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=1024x512:rate=10:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def dfisheye_clip(ffmpeg, tmp_path_factory):
    """The same content as `equirect_clip`, but as a camera's raw dual-fisheye file.

    Made by projecting the equirect testsrc2 back out through two 190-degree lenses, so
    a tile cut from this one can be compared against the tile cut from the panorama --
    which is the only way to tell a correct projection from a plausible one.
    """
    path = tmp_path_factory.mktemp("media") / "raw.mp4"
    subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=1024x512:rate=10:duration=2",
         "-vf", "v360=e:dfisheye:h_fov=190:v_fov=190:w=1024:h=512",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def two_stream_clip(ffmpeg, tmp_path_factory):
    """The same dual fisheye, written the way a QooCam 8K writes one.

    One video stream per lens, each rotated a quarter turn the *opposite* way -- which
    is what its two sensors do, and what makes the back half of the panorama arrive
    upside down if nothing puts it right. Built by cutting a known-good dual-fisheye
    frame in half, so a test can compare the reassembled panorama against the
    side-by-side file of the same content and see only resampling between them.
    """
    path = tmp_path_factory.mktemp("media") / "twolens.mp4"
    subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=1024x512:rate=10:duration=2",
         "-filter_complex",
         "[0:v]v360=e:dfisheye:h_fov=190:v_fov=190:w=1024:h=512,split=2[a][b];"
         "[a]crop=512:512:0:0,transpose=2[l0];"
         "[b]crop=512:512:512:0,transpose=1[l1]",
         "-map", "[l0]", "-map", "[l1]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path
