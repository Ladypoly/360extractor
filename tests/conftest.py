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
