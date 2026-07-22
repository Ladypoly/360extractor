"""End-to-end tests that actually run ffmpeg."""

import hashlib

import pytest

from threesixty.extract import run_extraction
from threesixty.ffmpeg import probe_media
from threesixty.plan import FrameSelection, plan_extraction
from threesixty.rig import Camera, Output, Rig, ring

pytestmark = pytest.mark.ffmpeg

SMALL = Output(width=320, height=240, format="png")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def images_in(directory):
    return sorted(p for p in directory.iterdir() if p.suffix in {".png", ".jpg"})


def test_probe_reads_the_synthetic_clip(ffmpeg, equirect_clip):
    media = probe_media(equirect_clip, ffmpeg)
    assert (media.width, media.height) == (1024, 512)
    assert media.looks_equirectangular
    assert media.is_video
    assert media.duration == pytest.approx(2.0, abs=0.2)


def test_extracts_expected_image_count(ffmpeg, equirect_clip, tmp_path):
    media = probe_media(equirect_clip, ffmpeg)
    plan = plan_extraction(media, ring(4, output=SMALL), FrameSelection("fps", 2), tmp_path)
    result = run_extraction(plan, ffmpeg)

    assert result.cameras_done == 4
    # 2 seconds at 2 fps is 4 frames per camera.
    assert result.images_written == 16
    for job in plan.passes[0].jobs:
        assert len(images_in(job.directory)) == 4


def test_single_pass_matches_separate_runs_byte_for_byte(ffmpeg, equirect_clip, tmp_path):
    """The correctness anchor for the whole planner.

    If fanning 4 cameras out of one decode ever diverges from running them
    individually, the optimisation is silently changing the output.
    """
    media = probe_media(equirect_clip, ffmpeg)
    rig = ring(4, output=SMALL)
    selection = FrameSelection("fps", 2)

    together = tmp_path / "together"
    plan = plan_extraction(media, rig, selection, together, max_streams=4)
    assert len(plan.passes) == 1
    run_extraction(plan, ffmpeg)

    apart = tmp_path / "apart"
    solo = plan_extraction(media, rig, selection, apart, max_streams=1)
    assert len(solo.passes) == 4
    run_extraction(solo, ffmpeg)

    for camera in rig.cameras:
        left = images_in(together / "clip" / camera.name)
        right = images_in(apart / "clip" / camera.name)
        assert [p.name for p in left] == [p.name for p in right]
        assert [digest(p) for p in left] == [digest(p) for p in right], \
            f"camera {camera.name} differs between batched and solo extraction"


def test_cameras_agree_on_frame_numbering(ffmpeg, equirect_clip, tmp_path):
    """Same sequence number must mean the same instant across every camera.

    Downstream photogrammetry pairs images by index; if the split ever handed
    cameras different frame sets, poses from different moments would be fused.
    """
    media = probe_media(equirect_clip, ffmpeg)
    rig = ring(3, output=SMALL)
    plan = plan_extraction(media, rig, FrameSelection("fps", 2), tmp_path)
    run_extraction(plan, ffmpeg)

    counts = {job.camera.name: len(images_in(job.directory)) for job in plan.passes[0].jobs}
    assert len(set(counts.values())) == 1, f"cameras produced different frame counts: {counts}"


def test_distinct_yaws_produce_distinct_images(ffmpeg, equirect_clip, tmp_path):
    """Guards against every camera silently rendering the same direction."""
    media = probe_media(equirect_clip, ffmpeg)
    plan = plan_extraction(media, ring(4, output=SMALL), FrameSelection("fps", 1), tmp_path)
    run_extraction(plan, ffmpeg)

    first_frames = [images_in(job.directory)[0] for job in plan.passes[0].jobs]
    digests = {digest(p) for p in first_frames}
    assert len(digests) == 4, "cameras at different yaws produced identical images"


def test_unwrapped_yaw_would_be_rejected_but_rig_wraps_it(ffmpeg, equirect_clip, tmp_path):
    """ring(3) yields 0/120/240; 240 is outside v360's range and must be wrapped."""
    media = probe_media(equirect_clip, ffmpeg)
    rig = ring(3, output=SMALL)
    assert any(c.yaw == pytest.approx(-120.0) for c in rig.normalized_cameras())

    plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path)
    result = run_extraction(plan, ffmpeg)
    assert result.cameras_done == 3


def test_pitched_camera_differs_from_level_one(ffmpeg, equirect_clip, tmp_path):
    media = probe_media(equirect_clip, ffmpeg)
    rig = Rig(
        cameras=[Camera(name="level", pitch=0, h_fov=90, v_fov=67.5),
                 Camera(name="down", pitch=-40, h_fov=90, v_fov=67.5)],
        output=SMALL,
    )
    plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path)
    run_extraction(plan, ffmpeg)

    level = images_in(tmp_path / "clip" / "level")[0]
    down = images_in(tmp_path / "clip" / "down")[0]
    assert digest(level) != digest(down)


def test_time_range_limits_output(ffmpeg, equirect_clip, tmp_path):
    media = probe_media(equirect_clip, ffmpeg)
    plan = plan_extraction(media, ring(1, output=SMALL),
                           FrameSelection("fps", 4, start=0.5, end=1.5), tmp_path)
    run_extraction(plan, ffmpeg)
    produced = len(images_in(plan.passes[0].jobs[0].directory))
    assert 3 <= produced <= 5, f"expected about 4 frames from a 1s window, got {produced}"


def test_resume_skips_completed_cameras(ffmpeg, equirect_clip, tmp_path):
    media = probe_media(equirect_clip, ffmpeg)
    rig = ring(2, output=SMALL)
    selection = FrameSelection("fps", 1)

    first = plan_extraction(media, rig, selection, tmp_path, resume=True)
    run_extraction(first, ffmpeg)
    assert all(job.marker.exists() for job in first.passes[0].jobs)

    second = plan_extraction(media, rig, selection, tmp_path, resume=True)
    assert second.passes == []
    assert len(second.skipped) == 2


def test_failed_pass_leaves_no_completion_marker(ffmpeg, equirect_clip, tmp_path):
    """A camera must never look done unless its pass actually succeeded."""
    from threesixty.ffmpeg import FFmpegError

    media = probe_media(equirect_clip, ffmpeg)
    rig = ring(1, output=SMALL)
    plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path)
    job = plan.passes[0].jobs[0]
    job.directory.mkdir(parents=True, exist_ok=True)
    job.marker.write_text("stale marker from an earlier run\n", encoding="utf-8")

    # Point the plan at a nonexistent source so the pass fails.
    plan.media = media.__class__(**{**media.__dict__, "path": tmp_path / "missing.mp4"})
    with pytest.raises(FFmpegError):
        run_extraction(plan, ffmpeg)
    assert not job.marker.exists(), "a failed pass left a stale marker that would skip a redo"


def test_dry_run_writes_nothing(ffmpeg, equirect_clip, tmp_path, capsys):
    media = probe_media(equirect_clip, ffmpeg)
    plan = plan_extraction(media, ring(2, output=SMALL), FrameSelection("fps", 1), tmp_path)
    result = run_extraction(plan, ffmpeg, dry_run=True)

    assert result.images_written == 0
    assert "v360=e:rectilinear" in capsys.readouterr().out
    assert not (tmp_path / "clip").exists()


def test_still_image_source_yields_one_image_per_camera(ffmpeg, equirect_clip, tmp_path):
    import subprocess

    still = tmp_path / "shot.png"
    subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(equirect_clip), "-frames:v", "1", str(still)],
        check=True, capture_output=True,
    )

    media = probe_media(still, ffmpeg)
    assert not media.is_video

    plan = plan_extraction(media, ring(3, output=SMALL), FrameSelection("fps", 2), tmp_path / "out")
    result = run_extraction(plan, ffmpeg)
    assert result.images_written == 3
