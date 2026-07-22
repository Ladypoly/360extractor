"""Splat PLY handling and the cleanup that removes floaters at the rig positions.

Two properties matter most:

* **Nothing is lost.** A splat file carries properties this tool does not understand
  (spherical harmonics of whatever degree, and anything a trainer decided to add).
  Filtering must drop rows and nothing else, so an unfiltered round trip has to come
  back byte-identical.
* **Only the right gaussians go.** `cleaned + removed == original`, with exactly the
  ones inside the radius on the removed side.
"""

import numpy as np
import pytest

from threesixty.colmap.model import ColmapImage, SparseModel, matrix_to_quaternion
from threesixty.splat import clean as splat_clean
from threesixty.splat import ply

# A realistic INRIA layout: positions, normals, DC colour, 9 SH rest terms, the rest.
PROPERTIES = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(9)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)


def make_ply(path, positions, extra_comment=True):
    """Write a splat PLY the way a trainer would."""
    positions = np.asarray(positions, dtype=np.float32)
    count = positions.shape[0]

    header = ["ply", "format binary_little_endian 1.0"]
    if extra_comment:
        header.append("comment Exported from Brush")
    header.append(f"element vertex {count}")
    header += [f"property float {name}" for name in PROPERTIES]
    header.append("end_header")

    dtype = np.dtype([(name, "<f4") for name in PROPERTIES])
    data = np.zeros(count, dtype=dtype)
    data["x"], data["y"], data["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    # Give the other fields distinct values so a round trip can be checked properly.
    for index, name in enumerate(PROPERTIES[3:], start=3):
        data[name] = np.arange(count, dtype=np.float32) * 0.5 + index

    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + data.tobytes())
    return path


class TestPlyIO:
    def test_reads_positions_and_count(self, tmp_path):
        positions = [[0, 0, 0], [1, 2, 3], [-4, 5, -6]]
        splats = ply.read(make_ply(tmp_path / "a.ply", positions))
        assert len(splats) == 3
        assert np.allclose(splats.positions, positions)

    def test_unfiltered_round_trip_is_byte_identical(self, tmp_path):
        """Everything this tool does not understand has to survive untouched."""
        source = make_ply(tmp_path / "a.ply", np.random.RandomState(0).randn(50, 3))
        splats = ply.read(source)
        ply.write(splats, tmp_path / "b.ply")
        assert (tmp_path / "b.ply").read_bytes() == source.read_bytes()

    def test_filtering_keeps_every_other_property(self, tmp_path):
        source = make_ply(tmp_path / "a.ply", np.arange(30).reshape(10, 3))
        splats = ply.read(source)

        keep = np.zeros(10, dtype=bool)
        keep[[1, 4, 7]] = True
        subset = splats.select(keep)

        assert len(subset) == 3
        for name in PROPERTIES:
            assert np.allclose(subset.data[name], splats.data[name][keep]), name

    def test_written_header_reports_the_new_count(self, tmp_path):
        splats = ply.read(make_ply(tmp_path / "a.ply", np.zeros((10, 3))))
        keep = np.zeros(10, dtype=bool)
        keep[:4] = True
        out = ply.write(splats.select(keep), tmp_path / "b.ply")

        header = out.read_bytes().split(b"end_header")[0].decode()
        assert "element vertex 4" in header
        assert len(ply.read(out)) == 4

    def test_comments_survive(self, tmp_path):
        splats = ply.read(make_ply(tmp_path / "a.ply", np.zeros((5, 3))))
        out = ply.write(splats, tmp_path / "b.ply")
        assert b"comment Exported from Brush" in out.read_bytes()

    def test_compressed_splat_is_refused_by_name(self, tmp_path):
        """Packed positions are quantised, so filtering by location is impossible."""
        path = tmp_path / "packed.ply"
        header = ["ply", "format binary_little_endian 1.0", "element vertex 1",
                  "property uint packed_position", "property uint packed_scale",
                  "property uint packed_rotation", "property uint packed_color",
                  "end_header"]
        path.write_bytes(("\n".join(header) + "\n").encode() + b"\x00" * 16)
        with pytest.raises(ply.PlyError, match="compressed splat file"):
            ply.read(path)

    def test_ascii_ply_is_refused(self, tmp_path):
        path = tmp_path / "a.ply"
        path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\n"
                         b"property float x\nend_header\n0\n")
        with pytest.raises(ply.PlyError, match="ascii"):
            ply.read(path)

    def test_truncated_file_is_caught(self, tmp_path):
        source = make_ply(tmp_path / "a.ply", np.zeros((20, 3)))
        data = source.read_bytes()
        source.write_bytes(data[:len(data) - 200])
        with pytest.raises(ply.PlyError, match="truncated"):
            ply.read(source)

    def test_not_a_ply_is_caught(self, tmp_path):
        path = tmp_path / "a.ply"
        path.write_bytes(b"this is not a ply\n")
        with pytest.raises(ply.PlyError, match="not a PLY"):
            ply.read(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ply.PlyError, match="no such file"):
            ply.read(tmp_path / "nope.ply")


def straight_trajectory(count=10, spacing=1.0, height=0.0):
    positions = np.zeros((count, 3))
    positions[:, 2] = np.arange(count) * spacing   # travelling along +Z
    positions[:, 1] = height
    return splat_clean.Trajectory(positions=positions, frames=list(range(1, count + 1)))


class TestTrajectory:
    def test_length_and_spacing(self):
        trajectory = straight_trajectory(11, spacing=2.0)
        assert trajectory.length == pytest.approx(20.0)
        assert trajectory.median_spacing == pytest.approx(2.0)

    def test_from_model_averages_the_cameras_of_a_frame(self):
        """Every camera in a frame shares an optical centre, so their mean is the rig."""
        images = {}
        for index, (frame, offset) in enumerate(
                [(1, [0, 0, 0]), (1, [0.01, 0, 0]), (2, [0, 0, 5]), (2, [0.01, 0, 5])],
                start=1):
            centre = np.array(offset, dtype=float)
            images[index] = ColmapImage(index, (1, 0, 0, 0), tuple(-centre), 1,
                                        f"clip/c{index % 2:02d}/{frame:05d}.jpg")
        trajectory = splat_clean.trajectory_from_model(SparseModel(images=images))

        assert len(trajectory) == 2
        assert np.allclose(trajectory.positions[0], [0.005, 0, 0])
        assert np.allclose(trajectory.positions[1], [0.005, 0, 5])

    def test_frame_spread_is_reported(self):
        """A large spread means COLMAP did not honour the rig; worth surfacing."""
        images = {
            1: ColmapImage(1, (1, 0, 0, 0), (0, 0, 0), 1, "clip/c00/00001.jpg"),
            2: ColmapImage(2, (1, 0, 0, 0), (-2.0, 0, 0), 1, "clip/c01/00001.jpg"),
        }
        trajectory = splat_clean.trajectory_from_model(SparseModel(images=images))
        assert trajectory.spread == pytest.approx(1.0)

    def test_unrecognisable_names_are_reported(self):
        images = {1: ColmapImage(1, (1, 0, 0, 0), (0, 0, 0), 1, "no_digits_here.jpg")}
        with pytest.raises(ValueError, match="grouped into frames"):
            splat_clean.trajectory_from_model(SparseModel(images=images))

    def test_up_axis_is_the_direction_of_least_travel(self):
        # Travelling in the X/Z plane, so up should come out close to +/-Y.
        positions = np.zeros((20, 3))
        positions[:, 0] = np.arange(20)
        positions[:, 2] = np.sin(np.arange(20))
        direction, _ = splat_clean.Trajectory(positions=positions).estimate_up()
        assert direction is not None
        assert abs(direction[1]) > 0.99

    def test_a_straight_trajectory_cannot_reveal_up(self):
        """Driving down a street is the main use case, and a line is symmetric about
        its own axis -- so every perpendicular direction is equally 'least'. Guessing
        one would silently put the floor in the wrong plane."""
        direction, reason = straight_trajectory(20, spacing=3.0).estimate_up()
        assert direction is None
        assert "straight" in reason

    def test_a_stationary_trajectory_cannot_reveal_up(self):
        direction, reason = splat_clean.Trajectory(positions=np.zeros((10, 3))).estimate_up()
        assert direction is None
        assert "does not move" in reason


class TestCleanupGeometry:
    def test_removes_exactly_what_is_inside(self):
        trajectory = straight_trajectory(5, spacing=10.0)   # points at z = 0,10,...,40
        positions = np.array([
            [0, 0, 0],       # on a camera -> out
            [0.5, 0, 10],    # near a camera -> out
            [0, 0, 5],       # midway, 5 from both -> out at radius 6
            [0, 0, 100],     # far away -> stays
            [50, 0, 20],     # far to the side -> stays
        ])
        inside, _ = splat_clean.cleanup_mask(positions, trajectory, radius=6.0)
        assert list(inside) == [True, True, True, False, False]

    def test_radius_is_a_hard_boundary(self):
        trajectory = splat_clean.Trajectory(positions=np.zeros((1, 3)))
        positions = np.array([[0.999, 0, 0], [1.001, 0, 0]])
        inside, _ = splat_clean.cleanup_mask(positions, trajectory, radius=1.0)
        assert list(inside) == [True, False]

    def test_zero_radius_is_rejected(self):
        with pytest.raises(ValueError, match="radius must be positive"):
            splat_clean.cleanup_mask(np.zeros((1, 3)),
                                     splat_clean.Trajectory(np.zeros((1, 3))), 0)

    def test_floor_spares_what_is_below_the_rig(self):
        """The road under a vehicle is real data, seen from other frames."""
        positions = np.zeros((10, 3))
        positions[:, 0] = np.arange(10) * 5.0     # travelling along X
        positions[:, 1] = 2.0                     # rig 2 units up
        trajectory = splat_clean.Trajectory(positions=positions)
        up = np.array([0.0, 1.0, 0.0])

        points = np.array([
            [0, 2.0, 0],    # level with the rig
            [0, 0.0, 0],    # road, 2 below
        ])
        without = splat_clean.cleanup_mask(points, trajectory, radius=3.0, up=up)[0]
        with_floor = splat_clean.cleanup_mask(points, trajectory, radius=3.0,
                                              floor=1.0, up=up)[0]

        assert list(without) == [True, True]
        assert list(with_floor) == [True, False], "the floor must spare the road"

    def test_floor_on_a_straight_path_demands_an_up_direction(self):
        """Rather than silently choosing a plane, it asks."""
        trajectory = straight_trajectory(20, spacing=3.0)
        with pytest.raises(ValueError, match="--up"):
            splat_clean.cleanup_mask(np.zeros((3, 3)), trajectory, radius=2.0, floor=1.0)

    def test_sphere_alone_works_without_knowing_up(self):
        trajectory = straight_trajectory(20, spacing=3.0)
        points = np.array([[0, 0, 0], [0, 0, 1000.0]])
        inside, counts = splat_clean.cleanup_mask(points, trajectory, radius=2.0)
        assert list(inside) == [True, False]
        assert counts["up_known"] is False

    def test_reports_how_much_was_below_the_rig(self):
        """Without a floor, this is the number that says whether the road got holes."""
        positions = np.zeros((5, 3))
        positions[:, 0] = np.arange(5) * 4.0
        positions[:, 1] = 2.0
        trajectory = splat_clean.Trajectory(positions=positions)

        points = np.array([[0, 0.0, 0], [4, 0.0, 0], [8, 2.0, 0]])
        _, counts = splat_clean.cleanup_mask(points, trajectory, radius=3.0,
                                             up=np.array([0.0, 1.0, 0.0]))
        assert counts["removed_below_rig"] == 2

    def test_chunking_matches_a_single_pass(self, monkeypatch):
        """Blocked distance computation must not change the answer."""
        rng = np.random.RandomState(3)
        positions = rng.randn(5000, 3) * 5
        trajectory = straight_trajectory(20, spacing=2.0)

        full, _ = splat_clean.cleanup_mask(positions, trajectory, radius=3.0)
        monkeypatch.setattr(splat_clean, "BLOCK", 97)
        chunked, _ = splat_clean.cleanup_mask(positions, trajectory, radius=3.0)
        assert np.array_equal(full, chunked)


class TestStreetScenario:
    """A synthetic drive down a street, with the three things that matter present.

    Floaters hug the trajectory, buildings stand either side, and road lies underneath.
    A good cleanup takes the first and leaves the other two. Measured rather than
    asserted loosely, because the interesting failure -- eating the road -- is exactly
    what a plain sphere does.
    """

    @pytest.fixture
    def scene(self):
        rng = np.random.RandomState(7)
        frames = 30
        rig = np.zeros((frames, 3))
        rig[:, 0] = np.linspace(0, 60, frames)
        rig[:, 1] = 2.0                                  # 2 units above the road

        floaters = rig[rng.randint(0, frames, 2000)] + rng.randn(2000, 3) * 0.6
        buildings = np.column_stack([
            rng.uniform(-5, 65, 3000), rng.uniform(0, 12, 3000),
            rng.choice([-9.0, 9.0], 3000) + rng.randn(3000) * 0.4])
        road = np.column_stack([
            rng.uniform(-5, 65, 3000), rng.randn(3000) * 0.05,
            rng.uniform(-6, 6, 3000)])

        positions = np.vstack([floaters, buildings, road])
        labels = np.array(["floater"] * len(floaters) + ["building"] * len(buildings)
                          + ["road"] * len(road))
        return positions, labels, splat_clean.Trajectory(positions=rig)

    UP = np.array([0.0, 1.0, 0.0])

    def _shares(self, scene, radius, floor):
        positions, labels, trajectory = scene
        inside, _ = splat_clean.cleanup_mask(positions, trajectory, radius, floor,
                                             up=self.UP)
        return {label: inside[labels == label].mean() for label in np.unique(labels)}

    def test_removes_the_floaters(self, scene):
        assert self._shares(scene, 2.5, None)["floater"] > 0.98

    def test_never_touches_the_buildings(self, scene):
        assert self._shares(scene, 2.5, None)["building"] == 0.0

    def test_a_plain_sphere_eats_the_road(self, scene):
        """Measured, and the reason --floor exists: a sphere centred on a rig 2 units
        up necessarily reaches the surface below it."""
        assert self._shares(scene, 2.5, None)["road"] > 0.1

    def test_the_floor_saves_the_road_almost_for_free(self, scene):
        with_floor = self._shares(scene, 2.5, 1.5)
        assert with_floor["road"] == 0.0
        assert with_floor["floater"] > 0.98, (
            "sparing the road must not cost meaningful floater removal")

    def test_a_larger_radius_makes_the_trade_worse(self, scene):
        assert self._shares(scene, 4.0, None)["road"] > \
               self._shares(scene, 2.5, None)["road"]
        assert self._shares(scene, 4.0, 1.5)["road"] == 0.0


class TestClean:
    def test_split_is_exhaustive_and_disjoint(self, tmp_path):
        rng = np.random.RandomState(1)
        positions = rng.randn(400, 3) * 10
        splats = ply.read(make_ply(tmp_path / "a.ply", positions))
        trajectory = straight_trajectory(5, spacing=4.0)

        kept, removed, report = splat_clean.clean(splats, trajectory, radius=5.0)

        assert len(kept) + len(removed) == len(splats) == report.total
        assert report.removed == len(removed)
        # Every original position appears in exactly one of the two halves.
        combined = np.vstack([kept.positions, removed.positions])
        assert sorted(map(tuple, np.round(combined, 5))) == \
               sorted(map(tuple, np.round(splats.positions, 5)))

    def test_removed_are_the_close_ones(self, tmp_path):
        positions = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 100.0]])
        splats = ply.read(make_ply(tmp_path / "a.ply", positions))
        trajectory = splat_clean.Trajectory(positions=np.zeros((1, 3)))

        kept, removed, _ = splat_clean.clean(splats, trajectory, radius=2.0)
        assert len(removed) == 2
        assert np.allclose(kept.positions, [[0, 0, 100]])

    def test_report_mentions_the_floor_when_a_lot_is_underneath(self, tmp_path):
        positions = np.array([[0, -1.0, 0], [0, -1.0, 1], [0, -1.0, 2]])
        splats = ply.read(make_ply(tmp_path / "a.ply", positions))
        rig = np.zeros((3, 3))
        rig[:, 2] = [0, 1, 2]
        trajectory = splat_clean.Trajectory(positions=rig)

        _, _, report = splat_clean.clean(splats, trajectory, radius=5.0,
                                         up=np.array([0.0, 1.0, 0.0]))
        assert any("--floor" in line for line in report.lines())

    def test_report_says_when_up_is_unknown(self, tmp_path):
        splats = ply.read(make_ply(tmp_path / "a.ply", np.zeros((3, 3))))
        trajectory = straight_trajectory(20, spacing=3.0)
        _, _, report = splat_clean.clean(splats, trajectory, radius=5.0)
        assert any("--up" in line for line in report.lines())

    def test_written_files_reload(self, tmp_path):
        positions = np.random.RandomState(2).randn(100, 3) * 3
        splats = ply.read(make_ply(tmp_path / "a.ply", positions))
        trajectory = straight_trajectory(3, spacing=2.0)

        kept, removed, _ = splat_clean.clean(splats, trajectory, radius=2.0)
        ply.write(kept, tmp_path / "kept.ply")
        ply.write(removed, tmp_path / "removed.ply")

        assert len(ply.read(tmp_path / "kept.ply")) == len(kept)
        assert len(ply.read(tmp_path / "removed.ply")) == len(removed)
