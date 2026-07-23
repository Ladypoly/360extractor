"""Reading the sparse point cloud for the live reconstruction view."""

import struct

import numpy as np

from threesixty.colmap.model import read_points


def _write_points_binary(path, points):
    """points: list of (x, y, z, r, g, b), with a couple of track entries each."""
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for i, (x, y, z, r, g, b) in enumerate(points):
            handle.write(struct.pack("<Q", i + 1))            # point3D id
            handle.write(struct.pack("<ddd", x, y, z))
            handle.write(struct.pack("<BBB", r, g, b))
            handle.write(struct.pack("<d", 0.5))              # error
            handle.write(struct.pack("<Q", 2))                # track length
            handle.write(struct.pack("<iiii", 1, 0, 2, 0))    # two (image, pt2d) pairs


def test_reads_positions_and_colours_from_binary(tmp_path):
    pts = [(1.0, 2.0, 3.0, 255, 0, 0), (-4.0, 5.0, -6.0, 10, 20, 30)]
    _write_points_binary(tmp_path / "points3D.bin", pts)

    positions, colors = read_points(tmp_path)
    assert positions.shape == (2, 3) and colors.shape == (2, 3)
    assert positions.dtype == np.float32 and colors.dtype == np.uint8
    np.testing.assert_allclose(positions[0], [1, 2, 3])
    np.testing.assert_array_equal(colors[1], [10, 20, 30])


def test_reads_from_text(tmp_path):
    (tmp_path / "points3D.txt").write_text(
        "# comment\n1 1.0 2.0 3.0 200 100 50 0.4 1 0 2 0\n", encoding="utf-8")
    positions, colors = read_points(tmp_path)
    assert positions.shape == (1, 3)
    np.testing.assert_array_equal(colors[0], [200, 100, 50])


def test_missing_points_is_empty_not_an_error(tmp_path):
    positions, colors = read_points(tmp_path)
    assert positions.shape == (0, 3) and colors.shape == (0, 3)


def test_limit_subsamples(tmp_path):
    pts = [(float(i), 0.0, 0.0, 0, 0, 0) for i in range(100)]
    _write_points_binary(tmp_path / "points3D.bin", pts)
    positions, _ = read_points(tmp_path, limit=10)
    assert len(positions) <= 10
