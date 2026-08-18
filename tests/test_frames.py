"""Stage-A equirect frame extraction.

Real ffmpeg over the synthetic clip: this is the working set Capture later rigs and masks,
so it must land the right count of panorama frames in frames/<clip>/.
"""

import pytest

from threesixty.ffmpeg import probe_media
from threesixty.frames import extract_frames, frames_dir
from threesixty.plan import FrameSelection
from threesixty.source import SourceFormat

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


def test_a_dual_fisheye_source_lands_as_equirect_frames(ffmpeg, dfisheye_clip, tmp_path):
    """The working set is equirect whatever the camera wrote.

    Everything after this -- the rig, the masks, the canvas -- reads a panorama, so the
    projection has to be spent here and nowhere else.
    """
    media = probe_media(dfisheye_clip, ffmpeg)
    result = extract_frames(ffmpeg, media, FrameSelection(mode="fps", value=2.0),
                            tmp_path, source_format=SourceFormat("dfisheye", 190))
    assert result.count >= 1

    frame = probe_media(sorted(result.directory.glob("*.jpg"))[0], ffmpeg)
    # 1024 across two 190-degree lenses is 512 per lens, which is ~970 across a full
    # turn (972 once rounded to an encodable size) -- and 2:1, which the raw file is not.
    assert (frame.width, frame.height) == (972, 486)
    assert frame.looks_equirectangular


def test_an_equirect_source_is_not_resampled(ffmpeg, equirect_clip, tmp_path):
    media = probe_media(equirect_clip, ffmpeg)
    result = extract_frames(ffmpeg, media, FrameSelection(mode="fps", value=2.0),
                            tmp_path, source_format=SourceFormat())
    frame = probe_media(sorted(result.directory.glob("*.jpg"))[0], ffmpeg)
    assert (frame.width, frame.height) == (1024, 512)


def test_a_lens_per_stream_is_stacked_back_into_one_panorama(
        ffmpeg, two_stream_clip, dfisheye_clip, tmp_path):
    """A QooCam-shaped file has to come out as the same panorama a side-by-side one does.

    Both fixtures carry identical content; the only difference is that one file split its
    lenses across two video streams and rotated each a quarter turn. If the stacking or
    either rotation is wrong the pictures diverge -- most visibly with the back half
    upside down, which still produces frames and still looks like a panorama.
    """
    streams = extract_frames(
        ffmpeg, probe_media(two_stream_clip, ffmpeg), FrameSelection("fps", 1.0),
        tmp_path / "streams",
        source_format=SourceFormat("dfisheye", 190, layout="streams", rotate=(90, -90)))
    sbs = extract_frames(
        ffmpeg, probe_media(dfisheye_clip, ffmpeg), FrameSelection("fps", 1.0),
        tmp_path / "sbs", source_format=SourceFormat("dfisheye", 190))

    assert streams.count == sbs.count
    from test_extract_integration import psnr
    first = sorted(streams.directory.glob("*.jpg"))[0]
    second = sorted(sbs.directory.glob("*.jpg"))[0]
    assert probe_media(first, ffmpeg).width == probe_media(second, ffmpeg).width
    assert psnr(ffmpeg, first, second) > 30


def test_the_stream_layout_sizes_from_both_lenses(ffmpeg, two_stream_clip, tmp_path):
    """Each stream is one lens, so the pair is twice as wide as the file claims."""
    media = probe_media(two_stream_clip, ffmpeg)
    assert media.video_streams == 2
    result = extract_frames(
        ffmpeg, media, FrameSelection("fps", 1.0), tmp_path,
        source_format=SourceFormat("dfisheye", 190, layout="streams", rotate=(90, -90)))
    frame = probe_media(sorted(result.directory.glob("*.jpg"))[0], ffmpeg)
    # 512 per lens is a 1024-wide pair: the same panorama the side-by-side file gives.
    assert (frame.width, frame.height) == (972, 486)
