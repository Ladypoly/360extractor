"""Static occluder masking.

The thing that has to be true: the mask covers exactly the part of the picture the
occluder covers. Not approximately, and not "the fraction looks about right" -- if the
mask is off, the user believes the car is excluded when it is still being trained on.

Polarity throughout: white keeps, black is ignored. Brush copies mask luma into image
alpha and treats alpha 0 as "do not train here"; COLMAP and nerfstudio agree.
"""

import subprocess

import pytest

from threesixty.extract import run_extraction
from threesixty.ffmpeg import probe_media
from threesixty.mask import apply as mask_apply
from threesixty.mask import geometric
from threesixty.plan import FrameSelection, plan_extraction
from threesixty.rig import Camera, Output, Rig, ring

pytestmark = pytest.mark.ffmpeg

SMALL = Output(width=320, height=240, format="jpg", auto=False)


def luma_grid(ffmpeg, path, size=32):
    """The image as a size x size grid of luma values, area-averaged."""
    raw = subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"scale={size}:{size}:flags=area,format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True).stdout
    assert len(raw) >= size * size, f"short read: {len(raw)} of {size*size}"
    return [list(raw[row * size:(row + 1) * size]) for row in range(size)]


def cone(angle):
    return [{"type": "nadir_cone", "angle": angle}]


class TestEquirectMask:
    def test_nadir_cone_is_black_below_and_white_above(self, ffmpeg, tmp_path):
        occluders = [geometric.Occluder("nadir_cone", angle=30)]
        mask = geometric.build_equirect_mask(ffmpeg, occluders, 512, 256,
                                             tmp_path / "eq.png")
        grid = luma_grid(ffmpeg, mask)

        # Equirect is linear in elevation: -30 degrees sits at (90+30)/180 = 2/3 down.
        boundary = int(32 * 2 / 3)
        assert all(v > 250 for v in grid[boundary - 3]), "above the cone should be kept"
        assert all(v < 5 for v in grid[boundary + 3]), "below the cone should be ignored"

    def test_zenith_cone_masks_the_top(self, ffmpeg, tmp_path):
        occluders = [geometric.Occluder("zenith_cone", angle=30)]
        mask = geometric.build_equirect_mask(ffmpeg, occluders, 512, 256,
                                             tmp_path / "eq.png")
        grid = luma_grid(ffmpeg, mask)
        assert all(v < 5 for v in grid[2]), "sky should be ignored"
        assert all(v > 250 for v in grid[16]), "horizon should be kept"

    def test_no_occluders_leaves_everything_white(self, ffmpeg, tmp_path):
        mask = geometric.build_equirect_mask(ffmpeg, [], 256, 128, tmp_path / "eq.png")
        grid = luma_grid(ffmpeg, mask)
        assert all(v > 250 for row in grid for v in row)

    def test_painted_mask_is_combined(self, ffmpeg, tmp_path):
        """A hand-painted occluder darkens on top of the cones."""
        painted = tmp_path / "painted.png"
        subprocess.run(
            [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=white:size=512x256",
             "-vf", "drawbox=x=0:y=0:w=256:h=256:color=black:t=fill",
             "-frames:v", "1", str(painted)], check=True, capture_output=True)

        occluders = [geometric.Occluder("equirect_mask", path=str(painted))]
        mask = geometric.build_equirect_mask(ffmpeg, occluders, 512, 256, tmp_path / "eq.png")
        grid = luma_grid(ffmpeg, mask)
        assert grid[16][4] < 5, "painted left half should be ignored"
        assert grid[16][28] > 250, "unpainted right half should be kept"

    def test_combining_is_order_independent(self, ffmpeg, tmp_path):
        """Darken-only combination means occluders can never restore coverage."""
        painted = tmp_path / "painted.png"
        subprocess.run(
            [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=black:size=512x256",
             "-frames:v", "1", str(painted)], check=True, capture_output=True)

        both = [geometric.Occluder("nadir_cone", angle=30),
                geometric.Occluder("equirect_mask", path=str(painted))]
        mask = geometric.build_equirect_mask(ffmpeg, both, 512, 256, tmp_path / "eq.png")
        grid = luma_grid(ffmpeg, mask)
        assert all(v < 5 for row in grid for v in row), "an all-black paint must win"

    def test_unknown_occluder_type_is_rejected(self):
        with pytest.raises(geometric.MaskError, match="unknown occluder"):
            geometric.Occluder.from_dict({"type": "wormhole"})

    def test_missing_painted_file_is_reported(self, ffmpeg, tmp_path):
        occluders = [geometric.Occluder("equirect_mask", path=str(tmp_path / "gone.png"))]
        with pytest.raises(geometric.MaskError, match="not found"):
            geometric.build_equirect_mask(ffmpeg, occluders, 256, 128, tmp_path / "eq.png")

    def test_ml_occluders_are_left_to_the_dynamic_stage(self):
        rig = ring(2)
        rig.occluders = [{"type": "nadir_cone", "angle": 20},
                         {"type": "ml", "backend": "sam2.1", "prompts": ["person"]}]
        kinds = [o.kind for o in geometric.occluders_of(rig)]
        assert kinds == ["nadir_cone"]


class TestCameraProjection:
    def test_downward_camera_is_fully_masked(self, ffmpeg, tmp_path):
        mask = geometric.build_equirect_mask(
            ffmpeg, [geometric.Occluder("nadir_cone", angle=30)], 1024, 512,
            tmp_path / "eq.png")
        camera = Camera(name="down", pitch=-80, h_fov=40, v_fov=40)
        rendered = geometric.render_camera_mask(ffmpeg, mask, camera, 128, 128,
                                                tmp_path / "down.png")
        assert geometric.ignored_fraction(ffmpeg, rendered) > 0.99

    def test_upward_camera_is_untouched(self, ffmpeg, tmp_path):
        mask = geometric.build_equirect_mask(
            ffmpeg, [geometric.Occluder("nadir_cone", angle=30)], 1024, 512,
            tmp_path / "eq.png")
        camera = Camera(name="up", pitch=60, h_fov=40, v_fov=40)
        rendered = geometric.render_camera_mask(ffmpeg, mask, camera, 128, 128,
                                                tmp_path / "up.png")
        assert geometric.ignored_fraction(ffmpeg, rendered) < 0.01

    def test_mask_covers_the_bottom_of_a_horizon_camera(self, ffmpeg, tmp_path):
        """The load-bearing case: which *part* of the frame is masked.

        A level camera looking at a nadir cone must have the cone at the bottom of the
        frame and clear sky at the top. A vertical flip would leave the fraction
        identical and mask precisely the wrong half.
        """
        mask = geometric.build_equirect_mask(
            ffmpeg, [geometric.Occluder("nadir_cone", angle=20)], 2048, 1024,
            tmp_path / "eq.png")
        camera = Camera(name="level", pitch=0, h_fov=90, v_fov=67.5)
        rendered = geometric.render_camera_mask(ffmpeg, mask, camera, 320, 240,
                                                tmp_path / "level.png")
        grid = luma_grid(ffmpeg, rendered)

        assert grid[1][16] > 250, "top of the frame is above the cone and must be kept"
        assert grid[30][16] < 5, "bottom of the frame is inside the cone and must be ignored"

    def test_coverage_grows_with_cone_angle(self, ffmpeg, tmp_path):
        camera = Camera(name="level", pitch=0, h_fov=90, v_fov=67.5)
        shares = []
        for angle in (10, 20, 30):
            mask = geometric.build_equirect_mask(
                ffmpeg, [geometric.Occluder("nadir_cone", angle=angle)], 1024, 512,
                tmp_path / f"eq{angle}.png")
            rendered = geometric.render_camera_mask(ffmpeg, mask, camera, 160, 120,
                                                    tmp_path / f"c{angle}.png")
            shares.append(geometric.ignored_fraction(ffmpeg, rendered))
        assert shares[0] > shares[1] > shares[2], f"tighter cones should mask more: {shares}"

    def test_mask_matches_the_image_size_exactly(self, ffmpeg, tmp_path):
        mask = geometric.build_equirect_mask(
            ffmpeg, [geometric.Occluder("nadir_cone", angle=30)], 1024, 512,
            tmp_path / "eq.png")
        rendered = geometric.render_camera_mask(
            ffmpeg, mask, Camera(name="c"), 640, 480, tmp_path / "c.png")
        out = subprocess.run(
            [str(ffmpeg.path.parent / "ffprobe.exe"), "-v", "error", "-select_streams",
             "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(rendered)], check=True, capture_output=True, text=True).stdout.strip()
        assert out.startswith("640,480")


class TestApplyModes:
    def _rig(self, angle=20):
        rig = ring(4, output=SMALL)
        rig.occluders = cone(angle)
        return rig

    def test_sidecar_writes_one_mask_per_image(self, ffmpeg, equirect_clip, tmp_path):
        media = probe_media(equirect_clip, ffmpeg)
        plan = plan_extraction(media, self._rig(), FrameSelection("fps", 2), tmp_path,
                               ffmpeg=ffmpeg, mask_mode="sidecar")
        result = run_extraction(plan, ffmpeg)

        assert result.masks_written == result.images_written
        for job in plan.passes[0].jobs:
            images = [p for p in job.directory.iterdir() if p.suffix == ".jpg"]
            masks = [p for p in job.mask_directory.iterdir() if p.suffix == ".png"]
            assert len(images) == len(masks)
            # Brush pairs by matching stem under a mirrored subpath.
            assert {p.stem for p in images} == {p.stem for p in masks}

    def test_mask_paths_mirror_image_paths(self, ffmpeg, equirect_clip, tmp_path):
        """Brush requires nested mask directories to match nested image directories."""
        media = probe_media(equirect_clip, ffmpeg)
        plan = plan_extraction(media, self._rig(), FrameSelection("fps", 1), tmp_path,
                               ffmpeg=ffmpeg, mask_mode="sidecar")
        run_extraction(plan, ffmpeg)
        for job in plan.passes[0].jobs:
            relative_image = job.directory.relative_to(tmp_path / "images")
            relative_mask = job.mask_directory.relative_to(tmp_path / "masks")
            assert relative_image == relative_mask

    def test_untouched_cameras_get_no_masks(self, ffmpeg, equirect_clip, tmp_path):
        """An all-white mask is a no-op that still costs a file per frame."""
        rig = ring(4, output=SMALL)
        rig.occluders = cone(80)  # far below anything a horizon ring can see
        media = probe_media(equirect_clip, ffmpeg)
        plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path,
                               ffmpeg=ffmpeg, mask_mode="sidecar")
        result = run_extraction(plan, ffmpeg)
        assert result.images_written > 0
        assert result.masks_written == 0

    def test_skip_drops_heavily_occluded_cameras(self, ffmpeg, equirect_clip, tmp_path):
        rig = Rig(
            cameras=[Camera(name="level", pitch=0, h_fov=90, v_fov=90),
                     Camera(name="down", yaw=90, pitch=-80, h_fov=60, v_fov=60)],
            output=SMALL,
            occluders=cone(30),
        )
        media = probe_media(equirect_clip, ffmpeg)
        plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path,
                               ffmpeg=ffmpeg, mask_mode="skip")
        names = [job.camera.name for p in plan.passes for job in p.jobs]
        assert "down" not in names, "a camera buried in the occluder should be dropped"
        assert "level" in names

    def test_none_mode_ignores_occluders(self, ffmpeg, equirect_clip, tmp_path):
        media = probe_media(equirect_clip, ffmpeg)
        plan = plan_extraction(media, self._rig(), FrameSelection("fps", 1), tmp_path,
                               ffmpeg=ffmpeg, mask_mode="none")
        result = run_extraction(plan, ffmpeg)
        assert result.masks_written == 0
        assert plan.burn_mask is None

    def test_burn_blacks_out_the_image_itself(self, ffmpeg, equirect_clip, tmp_path):
        """Burn must darken the bottom of the frame and leave the top alone."""
        media = probe_media(equirect_clip, ffmpeg)
        rig = Rig(cameras=[Camera(name="level", pitch=0, h_fov=90, v_fov=67.5)],
                  output=Output(width=320, height=240, format="png", auto=False),
                  occluders=cone(15))
        plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path,
                               ffmpeg=ffmpeg, mask_mode="burn")
        assert plan.burn_mask is not None
        run_extraction(plan, ffmpeg)

        job = plan.passes[0].jobs[0]
        image = sorted(p for p in job.directory.iterdir() if p.suffix == ".png")[0]
        grid = luma_grid(ffmpeg, image)
        assert grid[31][16] < 8, "the occluded bottom row should be burned to black"
        assert grid[1][16] > 8, "the top of the frame should be untouched"

    def test_burn_does_not_tint_the_kept_area(self, ffmpeg, equirect_clip, tmp_path):
        """Multiplying in YUV would drag chroma toward grey; this must be RGB."""
        media = probe_media(equirect_clip, ffmpeg)
        rig = Rig(cameras=[Camera(name="level", pitch=0, h_fov=90, v_fov=67.5)],
                  output=Output(width=320, height=240, format="png", auto=False))
        plain = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path / "plain",
                                ffmpeg=ffmpeg, mask_mode="none")
        run_extraction(plain, ffmpeg)

        rig.occluders = cone(15)
        burned = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path / "burn",
                                 ffmpeg=ffmpeg, mask_mode="burn")
        run_extraction(burned, ffmpeg)

        first = lambda plan: sorted(
            p for p in plan.passes[0].jobs[0].directory.iterdir() if p.suffix == ".png")[0]
        untouched = luma_grid(ffmpeg, first(plain))
        after = luma_grid(ffmpeg, first(burned))
        # Compare a row well above the occluder: it must be pixel-identical.
        assert untouched[2] == after[2], "burning altered pixels outside the occluder"

    def test_rejects_unknown_mode(self, ffmpeg):
        with pytest.raises(ValueError, match="mask mode"):
            mask_apply.prepare(ffmpeg, ring(2), {}, "x", mode="obliterate")
