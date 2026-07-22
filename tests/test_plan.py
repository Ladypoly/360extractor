from pathlib import Path

import pytest

from threesixty.ffmpeg import MediaInfo
from threesixty.plan import (
    FrameSelection,
    build_filter_graph,
    build_pass_argv,
    plan_extraction,
    safe_stem,
)
from threesixty.rig import Camera, Output, Rig, ring


@pytest.fixture
def video():
    return MediaInfo(path=Path("clip.mp4"), width=3840, height=1920, fps=30.0,
                     duration=10.0, frame_count=300, codec="h264", is_video=True)


@pytest.fixture
def still():
    return MediaInfo(path=Path("shot.jpg"), width=5760, height=2880, fps=0.0,
                     duration=0.0, frame_count=1, codec="mjpeg", is_video=False)


class TestFrameSelection:
    def test_fps_mode_builds_fps_filter(self, video):
        assert FrameSelection("fps", 2).filter_prefix(video) == "fps=2"

    def test_every_mode_escapes_its_comma(self, video):
        # An unescaped comma would be read as a filter separator and break the graph.
        prefix = FrameSelection("every", 5).filter_prefix(video)
        assert prefix == r"select='not(mod(n\,5))'"
        assert r"\," in prefix

    def test_all_mode_adds_no_filter(self, video):
        assert FrameSelection("all").filter_prefix(video) == ""

    def test_stills_never_get_a_frame_filter(self, still):
        assert FrameSelection("fps", 2).filter_prefix(still) == ""

    def test_estimates_frames_from_duration(self, video):
        assert FrameSelection("fps", 2).estimate_frames(video) == 20

    def test_estimate_respects_time_range(self, video):
        assert FrameSelection("fps", 2, start=2.0, end=7.0).estimate_frames(video) == 10

    def test_estimate_for_still_is_one(self, still):
        assert FrameSelection("fps", 30).estimate_frames(still) == 1

    @pytest.mark.parametrize("selection,message", [
        (FrameSelection("nonsense"), "fps|every|all"),
        (FrameSelection("fps", 0), "must be positive"),
        (FrameSelection("fps", -1), "must be positive"),
        (FrameSelection("every", 0), "whole number"),
        (FrameSelection("every", 2.5), "whole number"),
        (FrameSelection("fps", 1, start=-5), "must not be negative"),
        (FrameSelection("fps", 1, start=5, end=3), "greater than"),
    ])
    def test_rejects_nonsense(self, selection, message):
        with pytest.raises(ValueError, match=message):
            selection.validate()


class TestFilterGraph:
    def test_single_camera_skips_the_split(self):
        graph, labels = build_filter_graph([Camera(name="a")], ring(1), "fps=2")
        assert "split" not in graph
        assert graph.startswith("[0:v]fps=2,v360=e:rectilinear:")
        assert labels == ["o0"]

    def test_multi_camera_decodes_once_and_splits(self):
        rig = ring(3)
        graph, labels = build_filter_graph(rig.normalized_cameras(), rig, "fps=2")
        # One decode, one frame-thinning filter, three chains.
        assert graph.count("[0:v]") == 1
        assert graph.count("fps=2") == 1
        assert "split=3[s0][s1][s2]" in graph
        assert graph.count("v360=") == 3
        assert labels == ["o0", "o1", "o2"]

    def test_no_prefix_produces_valid_graph(self):
        rig = ring(2)
        graph, _ = build_filter_graph(rig.normalized_cameras(), rig, "")
        assert graph.startswith("[0:v]split=2")
        assert ",," not in graph  # an empty prefix must not leave a dangling comma

    def test_every_yaw_is_within_v360_range(self):
        # ring(3) yields 0/120/240; 240 must never reach ffmpeg.
        rig = ring(3)
        graph, _ = build_filter_graph(rig.normalized_cameras(), rig, "")
        yaws = [float(part.removeprefix("yaw=")) for part in graph.split(":") if part.startswith("yaw=")]
        assert len(yaws) == 3
        assert all(-180 <= y <= 180 for y in yaws)

    def test_output_size_and_interp_come_from_rig(self):
        rig = ring(1, output=Output(width=800, height=600, interp="lanczos"))
        graph, _ = build_filter_graph(rig.normalized_cameras(), rig, "")
        assert "w=800:h=600" in graph
        assert "interp=lanczos" in graph

    def test_rejects_empty_camera_list(self):
        with pytest.raises(ValueError, match="no cameras"):
            build_filter_graph([], ring(1), "")


class TestPlanning:
    def test_chunks_cameras_into_passes(self, video):
        plan = plan_extraction(video, ring(20), FrameSelection(), "out", max_streams=8)
        assert [len(p.jobs) for p in plan.passes] == [8, 8, 4]
        assert plan.total_cameras == 20

    def test_single_pass_when_rig_fits(self, video):
        plan = plan_extraction(video, ring(6), FrameSelection(), "out", max_streams=8)
        assert len(plan.passes) == 1

    def test_max_streams_of_one_gives_a_pass_each(self, video):
        plan = plan_extraction(video, ring(4), FrameSelection(), "out", max_streams=1)
        assert len(plan.passes) == 4

    def test_rejects_zero_max_streams(self, video):
        with pytest.raises(ValueError, match="at least 1"):
            plan_extraction(video, ring(4), FrameSelection(), "out", max_streams=0)

    def test_brush_layout_puts_cameras_under_images(self, video, tmp_path):
        """Brush and COLMAP both read an images/ root, and Brush requires masks/ to
        mirror its nested subpaths exactly."""
        plan = plan_extraction(video, ring(2), FrameSelection(), tmp_path, layout="brush")
        directories = {j.directory for p in plan.passes for j in p.jobs}
        assert directories == {tmp_path / "images" / "clip" / "c00",
                               tmp_path / "images" / "clip" / "c01"}

    def test_brush_layout_names_frames_identically_across_cameras(self, video, tmp_path):
        """COLMAP groups images into frames by matching filenames across camera
        folders, so every camera's frame N must be called the same thing.

        Embedding the camera name in the filename -- which is what the flat layout
        does -- silently prevents the rig from ever being formed.
        """
        plan = plan_extraction(video, ring(3), FrameSelection(), tmp_path, layout="brush")
        patterns = {j.pattern for p in plan.passes for j in p.jobs}
        assert len(patterns) == 1, f"cameras must share a filename pattern, got {patterns}"
        assert patterns.pop().startswith("%0")

    def test_flat_layout_keeps_names_distinct(self, video, tmp_path):
        """One shared folder needs unique names, so the camera goes back in."""
        plan = plan_extraction(video, ring(3), FrameSelection(), tmp_path, layout="flat")
        patterns = {j.pattern for p in plan.passes for j in p.jobs}
        assert len(patterns) == 3

    def test_mask_directories_mirror_image_directories(self, video, tmp_path):
        plan = plan_extraction(video, ring(2), FrameSelection(), tmp_path, layout="brush")
        for job in plan.passes[0].jobs:
            assert job.directory.relative_to(tmp_path / "images") == \
                   job.mask_directory.relative_to(tmp_path / "masks")

    def test_flat_layout_keeps_the_older_shape(self, video, tmp_path):
        plan = plan_extraction(video, ring(2), FrameSelection(), tmp_path, layout="flat")
        directories = {j.directory for p in plan.passes for j in p.jobs}
        assert directories == {tmp_path / "clip" / "c00", tmp_path / "clip" / "c01"}

    def test_rejects_unknown_layout(self, video, tmp_path):
        with pytest.raises(ValueError, match="--layout"):
            plan_extraction(video, ring(2), FrameSelection(), tmp_path, layout="sideways")

    def test_estimated_image_count(self, video):
        plan = plan_extraction(video, ring(8), FrameSelection("fps", 2), "out")
        assert plan.estimated_frames == 20
        assert plan.estimated_images == 160

    def test_resume_skips_cameras_with_markers(self, video, tmp_path):
        plan = plan_extraction(video, ring(4), FrameSelection(), tmp_path, resume=True)
        first = plan.passes[0].jobs[0]
        first.marker.parent.mkdir(parents=True, exist_ok=True)
        first.marker.write_text("images=20\n", encoding="utf-8")

        resumed = plan_extraction(video, ring(4), FrameSelection(), tmp_path, resume=True)
        assert len(resumed.skipped) == 1
        assert resumed.total_cameras == 3
        assert first.camera.name not in {j.camera.name for p in resumed.passes for j in p.jobs}

    def test_markers_ignored_without_resume(self, video, tmp_path):
        plan = plan_extraction(video, ring(4), FrameSelection(), tmp_path, resume=True)
        job = plan.passes[0].jobs[0]
        job.marker.parent.mkdir(parents=True, exist_ok=True)
        job.marker.write_text("images=20\n", encoding="utf-8")

        fresh = plan_extraction(video, ring(4), FrameSelection(), tmp_path, resume=False)
        assert fresh.total_cameras == 4


class TestPassArgv:
    def test_never_uses_a_shell_and_maps_every_camera(self, video, tmp_path):
        plan = plan_extraction(video, ring(3), FrameSelection("fps", 2), tmp_path)
        argv = build_pass_argv(Path("ffmpeg"), plan, plan.passes[0])
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
        assert argv.count("-map") == 3
        assert "-filter_complex" in argv
        assert "-progress" in argv

    def test_time_range_seeks_before_the_input(self, video, tmp_path):
        plan = plan_extraction(video, ring(1), FrameSelection("fps", 2, start=3, end=8), tmp_path)
        argv = build_pass_argv(Path("ffmpeg"), plan, plan.passes[0])
        # -ss after -i decodes and discards everything before the start point.
        assert argv.index("-ss") < argv.index("-i")
        assert argv.index("-to") < argv.index("-i")

    def test_jpeg_quality_applied_per_output(self, video, tmp_path):
        rig = ring(2, output=Output(format="jpg", quality=4))
        plan = plan_extraction(video, rig, FrameSelection(), tmp_path)
        argv = build_pass_argv(Path("ffmpeg"), plan, plan.passes[0])
        assert argv.count("-q:v") == 2

    def test_png_uses_compression_not_quality(self, video, tmp_path):
        rig = ring(1, output=Output(format="png"))
        plan = plan_extraction(video, rig, FrameSelection(), tmp_path)
        argv = build_pass_argv(Path("ffmpeg"), plan, plan.passes[0])
        assert "-q:v" not in argv
        assert "-compression_level" in argv


class TestSafeStem:
    @pytest.mark.parametrize("raw,expected", [
        ("clip", "clip"),
        ("my clip 01", "my_clip_01"),
        ("VID_20260101_120000", "VID_20260101_120000"),
        ("../etc/passwd", "etc_passwd"),
        ("...", "unnamed"),
        ("", "unnamed"),
    ])
    def test_produces_safe_path_components(self, raw, expected):
        assert safe_stem(raw) == expected
