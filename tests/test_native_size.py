"""Automatic output sizing and the rig-save path bug."""

import pytest

from threesixty.ffmpeg import MediaInfo
from threesixty.plan import FrameSelection, camera_size, plan_extraction
from threesixty.rig import Camera, Output, Rig, RigError, native_size, ring
from pathlib import Path


def media(width=5760):
    return MediaInfo(path=Path("clip.mp4"), width=width, height=width // 2, fps=30.0,
                     duration=10.0, frame_count=300, codec="h264", is_video=True)


class TestNativeSize:
    def test_90_degrees_of_a_5760_source_is_a_quarter_of_it(self):
        # 5760 px spans 360 degrees, so 90 degrees is exactly 1440 px.
        assert native_size(5760, 90, 90) == (1440, 1440)

    def test_vertical_density_matches_horizontal(self):
        # Equirect is 2:1, so degrees-per-pixel is the same on both axes.
        width, height = native_size(4096, 90, 45)
        assert width == 1024 and height == 512

    def test_narrow_camera_gets_fewer_pixels(self):
        wide = native_size(4096, 90, 67.5)
        narrow = native_size(4096, 45, 33.75)
        assert narrow[0] == wide[0] // 2

    @pytest.mark.parametrize("source,h_fov", [(3840, 90), (5760, 77), (7680, 120), (1024, 33)])
    def test_sizes_are_even(self, source, h_fov):
        # Odd dimensions upset several encoders.
        width, height = native_size(source, h_fov, h_fov * 0.75)
        assert width % 2 == 0 and height % 2 == 0

    def test_never_returns_zero(self):
        assert native_size(64, 0.1, 0.1) == (2, 2)


class TestCameraSize:
    def test_auto_uses_the_source_resolution(self):
        rig = ring(4, output=Output(auto=True))
        camera = rig.cameras[0]
        assert camera_size(camera, rig, media(5760)) == (1440, 1080)

    def test_manual_uses_the_rig_values(self):
        rig = ring(4, output=Output(width=800, height=600, auto=False))
        assert camera_size(rig.cameras[0], rig, media(5760)) == (800, 600)

    def test_auto_falls_back_when_source_width_unknown(self):
        rig = ring(4, output=Output(width=800, height=600, auto=True))
        assert camera_size(rig.cameras[0], rig, media(0)) == (800, 600)

    def test_mixed_rig_gets_per_camera_sizes(self):
        """A narrow camera should not be inflated to match a wide one."""
        rig = Rig(
            cameras=[Camera(name="wide", h_fov=90, v_fov=67.5),
                     Camera(name="narrow", yaw=90, h_fov=45, v_fov=33.75)],
            output=Output(auto=True),
        )
        plan = plan_extraction(media(4096), rig, FrameSelection(), "out")
        by_name = {job.camera.name: (job.width, job.height)
                   for p in plan.passes for job in p.jobs}
        assert by_name["wide"] == (1024, 768)
        assert by_name["narrow"] == (512, 384)

    def test_sizes_reach_the_filter_graph(self):
        rig = Rig(
            cameras=[Camera(name="wide", h_fov=90, v_fov=67.5),
                     Camera(name="narrow", yaw=90, h_fov=45, v_fov=33.75)],
            output=Output(auto=True),
        )
        from threesixty.plan import build_pass_argv
        plan = plan_extraction(media(4096), rig, FrameSelection(), "out")
        argv = build_pass_argv(Path("ffmpeg"), plan, plan.passes[0])
        graph = argv[argv.index("-filter_complex") + 1]
        assert "w=1024:h=768" in graph
        assert "w=512:h=384" in graph

    def test_auto_suppresses_the_aspect_warning(self):
        # With automatic sizing the output aspect always matches the fov by
        # construction, so the stretch warning would be noise.
        rig = Rig(cameras=[Camera(name="c", h_fov=90, v_fov=90)],
                  output=Output(width=1920, height=1440, auto=True))
        assert rig.warnings() == []
        rig.output.auto = False
        assert any("stretched" in w for w in rig.warnings())


class TestSavePath:
    """Regression: saving with an empty filename wrote to a directory.

    It surfaced as `PermissionError: [Errno 13] Permission denied: '.'` from deep
    inside pathlib, which said nothing about what the caller did wrong.
    """

    def test_empty_path_is_rejected_clearly(self):
        with pytest.raises(RigError, match="no filename given"):
            ring(2).save("")

    def test_whitespace_path_is_rejected(self):
        with pytest.raises(RigError, match="no filename given"):
            ring(2).save("   ")

    def test_directory_path_is_rejected_clearly(self, tmp_path):
        with pytest.raises(RigError, match="is a directory"):
            ring(2).save(tmp_path)

    def test_dot_is_rejected(self):
        with pytest.raises(RigError, match="is a directory"):
            ring(2).save(".")

    def test_missing_suffix_gets_json(self, tmp_path):
        path = ring(2).save(tmp_path / "myrig")
        assert path.name == "myrig.json" and path.exists()

    def test_normal_save_still_works(self, tmp_path):
        path = ring(2).save(tmp_path / "nested" / "rig.json")
        assert path.exists()
