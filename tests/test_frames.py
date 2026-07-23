"""Stage-A equirect frame extraction.

Real ffmpeg over the synthetic clip: this is the working set Capture later rigs and masks,
so it must land the right count of panorama frames in frames/<clip>/.
"""

import pytest

from threesixty.ffmpeg import probe_media
from threesixty.frames import extract_frames, frames_dir
from threesixty.plan import FrameSelection

pytestmark = pytest.mark.ffmpeg


def test_fps_mode_writes_equirect_frames(ffmpeg, equirect_clip, tmp_path):
    media = probe_media(equirect_clip, ffmpeg)      # 2 s @ 10 fps
    seen = []
    result = extract_frames(ffmpeg, media, FrameSelection(mode="fps", value=5.0),
                            tmp_path, on_progress=lambda frac, n, t: seen.append(frac))
    # ~10 frames (2 s x 5 fps); allow ffmpeg's boundary rounding.
    assert 8 <= result.count <= 12
    assert result.directory == frames_dir(tmp_path, "clip")
    assert result.count == len(list(result.directory.glob("*.jpg")))
    assert seen and seen[-1] <= 1.0


def test_a_start_end_window_limits_the_frames(ffmpeg, equirect_clip, tmp_path):
    media = probe_media(equirect_clip, ffmpeg)
    full = extract_frames(ffmpeg, media, FrameSelection(mode="fps", value=10.0), tmp_path)
    windowed = extract_frames(
        ffmpeg, media, FrameSelection(mode="fps", value=10.0, start=0.0, end=1.0),
        tmp_path / "half")
    assert windowed.count < full.count


def test_sharp_mode_selects_frames(ffmpeg, equirect_clip, tmp_path):
    media = probe_media(equirect_clip, ffmpeg)
    notes = []
    result = extract_frames(ffmpeg, media, FrameSelection(mode="sharp", value=2.0),
                            tmp_path, on_analysis=notes.append)
    assert result.count >= 1
    assert notes            # sharp analysis reported a summary
