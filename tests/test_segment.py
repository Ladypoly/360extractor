"""Segmenting a capture by duration, GPX distance, or video motion.

Pure maths, no ffmpeg: the real optical-flow estimator (`motion.forward_motion`) is
validated separately on real footage. Here the motion *series* is synthetic so the cut
logic is checked deterministically.
"""

import pytest

from threesixty import gps, segment
from threesixty.segment import Segment, SegmentError, segment_by_duration, \
    segment_by_gpx, segment_by_motion


# -- distance maths ---------------------------------------------------------

class TestHaversine:
    def test_a_degree_of_longitude_at_the_equator(self):
        # Spherical mean radius: a degree is ~111.195 km (not the WGS84 111.319 km).
        metres = gps.haversine(gps.Fix(0, 0), gps.Fix(0, 1))
        assert metres == pytest.approx(111_195, rel=0.001)

    def test_a_degree_of_latitude(self):
        metres = gps.haversine(gps.Fix(0, 0), gps.Fix(1, 0))
        assert metres == pytest.approx(111_195, rel=0.001)

    def test_zero_distance(self):
        assert gps.haversine(gps.Fix(51.5, -0.1), gps.Fix(51.5, -0.1)) == 0.0

    def test_cumulative_is_monotonic_and_starts_at_zero(self):
        fixes = [gps.Fix(0, i * 0.001) for i in range(5)]
        totals = gps.cumulative_distance(fixes)
        assert totals[0] == 0.0
        assert all(b > a for a, b in zip(totals, totals[1:]))
        assert len(totals) == len(fixes)


# -- duration ---------------------------------------------------------------

class TestByDuration:
    def test_even_split(self):
        segs = segment_by_duration(10.0, 4.0)
        assert [(s.start, s.end) for s in segs] == [(0, 4), (4, 8), (8, 10)]

    def test_short_tail_is_folded_into_the_previous_segment(self):
        segs = segment_by_duration(9.5, 3.0)          # tail of 0.5 s is a runt
        assert [(s.start, s.end) for s in segs] == [(0, 3), (3, 6), (6, 9.5)]

    def test_segment_longer_than_clip_is_one_segment(self):
        segs = segment_by_duration(10.0, 30.0)
        assert len(segs) == 1 and segs[0].end == 10.0

    def test_rejects_nonsense(self):
        with pytest.raises(SegmentError):
            segment_by_duration(0.0, 5.0)
        with pytest.raises(SegmentError):
            segment_by_duration(10.0, 0.0)


# -- GPX distance -----------------------------------------------------------

class TestByGpx:
    def _straight_track(self, hops, metres_per_hop=100.0):
        """A due-east track at the equator, one fix per second."""
        deg = metres_per_hop / 111_319.49
        return [gps.Fix(0.0, i * deg, time=1000.0 + i) for i in range(hops + 1)]

    def test_cuts_every_target_distance(self):
        fixes = self._straight_track(10)              # 1000 m over 10 s
        segs = segment_by_gpx(fixes, 500.0)
        assert len(segs) == 2
        assert segs[0].end == pytest.approx(5.0, abs=0.05)
        assert segs[0].distance == pytest.approx(500.0, rel=0.01)

    def test_target_bigger_than_track_is_one_segment(self):
        segs = segment_by_gpx(self._straight_track(5), 5000.0)
        assert len(segs) == 1

    def test_needs_timed_points(self):
        with pytest.raises(SegmentError):
            segment_by_gpx([gps.Fix(0, 0), gps.Fix(0, 1)], 100.0)  # no timestamps


# -- motion -----------------------------------------------------------------

class TestByMotion:
    def _steady(self, n, step=1.0, mag=1.0):
        return [(i * step, 0.0 if i == 0 else mag) for i in range(n)]

    def test_count_splits_into_equal_travel(self):
        segs = segment_by_motion(self._steady(11), count=2)
        assert len(segs) == 2
        assert segs[0].end == pytest.approx(5.0, abs=1e-6)
        assert all(s.approximate for s in segs)

    def test_a_stop_does_not_get_its_own_segment(self):
        # move, stop, move: the boundary should land in the moving stretch, not the stop.
        samples = [(float(i), m) for i, m in enumerate([0, 1, 1, 0, 0, 1, 1])]
        segs = segment_by_motion(samples, count=2)
        assert len(segs) == 2
        assert segs[0].end == pytest.approx(2.0, abs=1e-6)

    def test_metric_mode_with_speed(self):
        # 10 m/s for 10 s = 100 m; 30 m segments -> cuts at 3, 6, 9 s.
        segs = segment_by_motion(self._steady(11), meters=30.0, speed_kph=36.0)
        assert len(segs) == 4
        assert segs[0].end == pytest.approx(3.0, abs=1e-6)
        assert segs[0].distance == pytest.approx(30.0, rel=0.01)
        assert not segs[0].approximate

    def test_metric_mode_skips_stationary_time(self):
        # 5 s moving, 5 s stopped, 5 s moving, at 10 m/s -> 100 m of actual travel.
        mags = [0] + [1] * 5 + [0] * 5 + [1] * 4
        samples = [(float(i), float(m)) for i, m in enumerate(mags)]
        segs = segment_by_motion(samples, meters=40.0, speed_kph=36.0)
        # 100 m / 40 = cuts at 40 m and 80 m of *moving* distance, never during the stop.
        assert len(segs) >= 2
        for s in segs:
            assert s.end > s.start

    def test_needs_count_or_speed(self):
        with pytest.raises(SegmentError):
            segment_by_motion(self._steady(5))
        with pytest.raises(SegmentError):
            segment_by_motion([(0.0, 0.0)], count=2)   # too few samples


def test_segment_is_frozen_with_duration():
    s = Segment(index=0, start=2.0, end=7.0)
    assert s.duration == 5.0
    with pytest.raises(Exception):
        s.start = 3.0
