"""Sphere fusion: making overlapping cameras agree.

The failure this prevents is specific and nasty. A pedestrian caught by camera A and
missed by overlapping camera B leaves inconsistent supervision, and a splat trainer
bakes the ghost in from B. Every camera that could see a direction has to agree on it.

The reverse projection is written by hand rather than handed to ffmpeg, because v360
clamps a tile's border pixels outward across the whole sphere and its `alpha_mask`
option does not mark visibility (measured: a 60x60 camera has true coverage 0.052 while
alpha reports 0.725, and at yaw 90 it reports nothing at all).

So `TestRoundTrip` is the load-bearing test in this file: it pins the hand-written
inverse directly against v360's forward projection, which everything else is verified
against.
"""

import numpy as np
import pytest

from threesixty.mask import fuse, geometric
from threesixty.mask.geometric import MaskError
from threesixty.rig import Camera

pytestmark = pytest.mark.ffmpeg


def write(ffmpeg, path, array):
    return fuse.write_gray(ffmpeg, array.astype(np.uint8), path)


def solid(value, size=(120, 160)):
    return np.full(size, value, dtype=np.uint8)


CAMERAS = [
    Camera(name="fwd", yaw=0, pitch=0, h_fov=90, v_fov=90),
    Camera(name="east", yaw=90, pitch=0, h_fov=60, v_fov=60),
    Camera(name="tilted", yaw=-45, pitch=-20, h_fov=90, v_fov=67.5),
    Camera(name="seam", yaw=175, pitch=0, h_fov=90, v_fov=67.5),
    Camera(name="steep", yaw=0, pitch=-70, h_fov=60, v_fov=60),
]


class TestRoundTrip:
    """Does the hand-written inverse actually invert v360's forward projection?"""

    @pytest.mark.parametrize("camera", CAMERAS, ids=[c.name for c in CAMERAS])
    def test_recovered_mask_matches_the_original(self, ffmpeg, tmp_path, camera):
        source = np.full((256, 512), 255, dtype=np.uint8)
        source[96:128, :] = 0
        source[:, 200:240] = 0
        equirect = write(ffmpeg, tmp_path / "eq.png", source)

        tile_path = geometric.render_camera_mask(
            ffmpeg, equirect, camera, 240, 180, tmp_path / "tile.png")
        tile = fuse.read_gray(ffmpeg, tile_path, 240, 180)
        recovered, visible = fuse.project_to_sphere(tile, camera, 512, 256)

        # Ignore a margin around the frustum edge and around the pattern's own
        # boundaries: both round-trips resample, so single-pixel disagreement at a hard
        # edge is expected and uninteresting.
        interior = _shrink(visible, 3)
        stable = _stable(source == 0, 3)
        compare = interior & stable
        assert compare.sum() > 200, "not enough interior area to compare"

        agreement = ((recovered < 128) == (source < 128))[compare].mean()
        assert agreement > 0.98, (
            f"the inverse projection disagrees with v360's forward projection for "
            f"{camera.name}: {agreement:.3f} agreement"
        )

    @pytest.mark.parametrize("camera", CAMERAS, ids=[c.name for c in CAMERAS])
    def test_visibility_matches_the_measured_coverage(self, ffmpeg, tmp_path, camera):
        """`visible` must agree with how much of the sphere the camera really covers."""
        white = write(ffmpeg, tmp_path / "white.png", solid(255, (256, 512)))
        tile_path = geometric.render_camera_mask(
            ffmpeg, white, camera, 160, 120, tmp_path / "t.png")
        tile = fuse.read_gray(ffmpeg, tile_path, 160, 120)
        _, visible = fuse.project_to_sphere(tile, camera, 512, 256)

        # Solid-angle share of a rectilinear frustum, weighted for equirect row area.
        rows = np.cos(np.radians(90.0 - (np.arange(256) + 0.5) / 256 * 180.0))
        weights = np.repeat(rows[:, None], 512, axis=1)
        solid_share = (visible * weights).sum() / weights.sum()

        expected = (camera.h_fov / 360.0) * (camera.v_fov / 180.0)
        assert 0.4 * expected < solid_share < 2.5 * expected, (
            f"{camera.name}: visible share {solid_share:.4f} is nowhere near the "
            f"{expected:.4f} its field of view implies"
        )


def _shrink(mask, amount):
    """Erode a boolean mask by `amount` pixels, without scipy."""
    out = mask.copy()
    for _ in range(amount):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        out = shrunk
    return out


def _stable(mask, amount):
    """Where a boolean pattern is not within `amount` pixels of its own boundary."""
    return _shrink(mask, amount) | _shrink(~mask, amount)


class TestProjection:
    def test_a_camera_only_votes_inside_its_field_of_view(self, ffmpeg, tmp_path):
        """An all-black tile must darken only what that camera could see.

        This is precisely what v360's edge clamping would get wrong.
        """
        camera = Camera(name="only", yaw=0, h_fov=60, v_fov=60)
        mask, visible = fuse.project_to_sphere(solid(0), camera, 512, 256)

        assert (mask < 128).mean() < 0.12, "the black tile leaked outside the frustum"
        assert mask[128, 256] < 128, "straight ahead should be masked"
        assert mask[128, 0] == 255, "behind the camera must be untouched"

    def test_yaw_places_coverage_where_expected(self):
        camera = Camera(name="east", yaw=90, pitch=0, h_fov=60, v_fov=60)
        _, visible = fuse.project_to_sphere(solid(255), camera, 360, 180)
        # x = (yaw + 180) / 360 * width, so yaw 90 sits three quarters across.
        assert visible[90, 270]
        assert not visible[90, 90]

    def test_pitch_places_coverage_where_expected(self):
        camera = Camera(name="down", yaw=0, pitch=-60, h_fov=60, v_fov=60)
        _, visible = fuse.project_to_sphere(solid(255), camera, 360, 180)
        # y = (90 - pitch) / 180 * height, so pitch -60 is 150 of 180.
        assert visible[150, 180]
        assert not visible[30, 180]

    def test_rejects_a_non_2d_tile(self):
        with pytest.raises(MaskError, match="2-D"):
            fuse.project_to_sphere(np.zeros((4, 4, 3), np.uint8),
                                   Camera(name="c"), 64, 32)


class TestFuse:
    def _pair(self):
        # Overlapping: 90 degree fields of view 60 degrees apart.
        return (Camera(name="a", yaw=0, h_fov=90, v_fov=90),
                Camera(name="b", yaw=60, h_fov=90, v_fov=90))

    def test_a_detection_in_one_camera_reaches_the_sphere(self, ffmpeg, tmp_path):
        a, b = self._pair()
        flagged = solid(255).copy()
        flagged[40:80, 60:110] = 0
        tiles = [(a, write(ffmpeg, tmp_path / "a.png", flagged)),
                 (b, write(ffmpeg, tmp_path / "b.png", solid(255)))]

        sphere = fuse.fuse(ffmpeg, tiles, 512, 256, tmp_path / "s.png")
        array = fuse.read_gray(ffmpeg, sphere, 512, 256)
        assert (array < 128).any(), "camera A's detection vanished during fusion"
        assert (array < 128).mean() < 0.2

    def test_clean_cameras_leave_a_clean_sphere(self, ffmpeg, tmp_path):
        a, b = self._pair()
        tiles = [(a, write(ffmpeg, tmp_path / "a.png", solid(255))),
                 (b, write(ffmpeg, tmp_path / "b.png", solid(255)))]
        sphere = fuse.fuse(ffmpeg, tiles, 512, 256, tmp_path / "s.png")
        assert (fuse.read_gray(ffmpeg, sphere, 512, 256) < 128).sum() == 0

    def test_union_is_order_independent(self, ffmpeg, tmp_path):
        a, b = self._pair()
        first = solid(255).copy();  first[20:60, 20:60] = 0
        second = solid(255).copy(); second[60:100, 80:140] = 0
        pa = write(ffmpeg, tmp_path / "a.png", first)
        pb = write(ffmpeg, tmp_path / "b.png", second)

        one = fuse.read_gray(ffmpeg, fuse.fuse(ffmpeg, [(a, pa), (b, pb)], 256, 128,
                                               tmp_path / "s1.png"), 256, 128)
        two = fuse.read_gray(ffmpeg, fuse.fuse(ffmpeg, [(b, pb), (a, pa)], 256, 128,
                                               tmp_path / "s2.png"), 256, 128)
        assert np.array_equal(one, two)

    def test_overlap_transfers_between_cameras(self, ffmpeg, tmp_path):
        """The actual point: B ends up masking what only A detected.

        Fused sphere re-projected into B must carry A's detection, in the region the
        two cameras share.
        """
        a = Camera(name="a", yaw=0, h_fov=90, v_fov=90)
        b = Camera(name="b", yaw=45, h_fov=90, v_fov=90)

        flagged = solid(255, (200, 200)).copy()
        flagged[:, 150:200] = 0            # right-hand side of A, inside the overlap
        tiles = [(a, write(ffmpeg, tmp_path / "a.png", flagged)),
                 (b, write(ffmpeg, tmp_path / "b.png", solid(255, (200, 200))))]

        sphere = fuse.fuse(ffmpeg, tiles, 1024, 512, tmp_path / "s.png")
        back_in_b = geometric.render_camera_mask(ffmpeg, sphere, b, 200, 200,
                                                 tmp_path / "b_final.png")
        recovered = fuse.read_gray(ffmpeg, back_in_b, 200, 200)
        assert (recovered < 128).any(), (
            "camera B came back clean; A's detection did not cross the overlap"
        )

    def test_rejects_an_empty_camera_list(self, ffmpeg, tmp_path):
        with pytest.raises(MaskError, match="at least one"):
            fuse.fuse(ffmpeg, [], 256, 128, tmp_path / "s.png")
