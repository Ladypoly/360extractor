"""Splitting one long capture into independent segments.

A long drive overwhelms COLMAP and Brush -- they drift over a long sequence -- so a
kilometres-long clip is better reconstructed as a handful of shorter datasets. A segment
is just a `(start, end)` time window over the source; the extraction pipeline already
honours a windowed `FrameSelection` via ffmpeg `-ss/-to`, and each window becomes its
own project (see the web layer).

Three ways to decide the cuts, in order of how much they need:

* **duration** -- every N seconds. Needs nothing but the clip length.
* **motion** -- every N metres of *forward travel*, estimated from the video itself
  (`motion.forward_motion`). Skips stationary stretches, so a red light does not waste a
  segment. Monocular flow has no absolute scale, so exact metres need an average speed;
  without one, split into a requested number of equal-travel segments.
* **gpx** -- every N metres along a GPX track (`gps.cumulative_distance`), when a sidecar
  is present. The only source of true metres without a speed guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import gps

#: A trailing window shorter than this fraction of a full segment is folded into the one
#: before it rather than left as a runt that is too short to reconstruct on its own.
MIN_TAIL_FRACTION = 0.25


class SegmentError(ValueError):
    """A segmentation request could not be satisfied."""


@dataclass(frozen=True)
class Segment:
    """One time window of the source, and how far it is believed to cover."""

    index: int
    start: float                    # seconds from the clip start
    end: float                      # seconds
    distance: float | None = None   # metres, if known
    approximate: bool = False       # True when distance is a scale-free motion proxy

    @property
    def duration(self) -> float:
        return self.end - self.start


def _finish(bounds: list[float], distances: list[float | None],
            approximate: bool) -> list[Segment]:
    """Turn a sorted list of cut times into Segments, folding a runt tail in."""
    # bounds is [0, t1, t2, ..., total]; N segments from N+1 bounds.
    spans = list(zip(bounds, bounds[1:]))
    distances = list(distances)
    if len(spans) >= 2:
        last_start, last_end = spans[-1]
        typical = (bounds[-1] - bounds[0]) / len(spans)
        if (last_end - last_start) < MIN_TAIL_FRACTION * typical:
            # Absorb the runt into the segment before it, and its distance too.
            spans[-2] = (spans[-2][0], last_end)
            spans.pop()
            if len(distances) >= 2 and distances[-1] is not None \
                    and distances[-2] is not None:
                distances[-2] += distances[-1]
            distances = distances[:-1]
    return [Segment(index=i, start=s, end=e,
                    distance=distances[i] if i < len(distances) else None,
                    approximate=approximate)
            for i, (s, e) in enumerate(spans)]


def segment_by_duration(total_seconds: float, seconds: float) -> list[Segment]:
    """Cut every `seconds`. `total_seconds` is the clip length."""
    if total_seconds <= 0:
        raise SegmentError("clip has no duration to segment")
    if seconds <= 0:
        raise SegmentError("segment length must be positive")
    if seconds >= total_seconds:
        return [Segment(index=0, start=0.0, end=total_seconds)]

    bounds = [0.0]
    while bounds[-1] + seconds < total_seconds:
        bounds.append(bounds[-1] + seconds)
    bounds.append(total_seconds)
    return _finish(bounds, [None] * (len(bounds) - 1), approximate=False)


def segment_by_gpx(fixes: list[gps.Fix], meters: float) -> list[Segment]:
    """Cut every `meters` of travel along a timed GPX track."""
    timed = [f for f in fixes if f.time is not None]
    if len(timed) < 2:
        raise SegmentError("GPX has too few timed points to measure distance")
    if meters <= 0:
        raise SegmentError("segment length must be positive")

    base = timed[0].time
    times = [f.time - base for f in timed]
    cumulative = gps.cumulative_distance(timed)
    total = cumulative[-1]
    if meters >= total:
        return [Segment(index=0, start=0.0, end=times[-1],
                        distance=total, approximate=False)]

    bounds = [0.0]
    distances: list[float | None] = []
    target = meters
    for i in range(1, len(cumulative)):
        # A single hop can cross several targets on a fast stretch; emit each.
        while target < cumulative[i]:
            span = cumulative[i] - cumulative[i - 1] or 1e-9
            frac = (target - cumulative[i - 1]) / span
            bounds.append(times[i - 1] + frac * (times[i] - times[i - 1]))
            distances.append(meters)
            target += meters
    bounds.append(times[-1])
    distances.append(total - meters * (len(bounds) - 2))
    return _finish(bounds, distances, approximate=False)


def segment_by_motion(samples: list[tuple[float, float]], *,
                      meters: float | None = None, count: int | None = None,
                      speed_kph: float | None = None) -> list[Segment]:
    """Cut by accumulated forward motion.

    `samples` is `(time_seconds, magnitude)` from `motion.forward_motion`, where
    magnitude is the forward travel in the interval ending at that time (~0 while
    stationary). Two ways to set the cuts:

    * `speed_kph` + `meters` -- metric: travel accrues as speed x moving-time (only while
      magnitude clears the noise floor), cut every `meters`. Distances are real-ish.
    * `count` -- scale-free: split the *total* accumulated magnitude into `count` equal
      parts. Distances are unknown (``approximate=True``).
    """
    if len(samples) < 2:
        raise SegmentError("not enough motion samples to segment")
    times = [t for t, _ in samples]

    if speed_kph is not None and meters is not None:
        if speed_kph <= 0 or meters <= 0:
            raise SegmentError("speed and segment length must be positive")
        speed_mps = speed_kph / 3.6
        floor = _motion_floor(samples)
        bounds, distances = [times[0]], []
        travelled, target = 0.0, meters
        for (t_prev, _), (t, mag) in zip(samples, samples[1:]):
            if mag > floor:
                travelled += speed_mps * (t - t_prev)
            while travelled >= target:
                bounds.append(t)
                distances.append(meters)
                target += meters
        bounds.append(times[-1])
        distances.append(max(travelled - meters * (len(bounds) - 2), 0.0))
        return _finish(bounds, distances, approximate=False)

    if count is not None:
        if count < 1:
            raise SegmentError("segment count must be at least 1")
        total = sum(mag for _, mag in samples)
        if count == 1 or total <= 0:
            return [Segment(index=0, start=times[0], end=times[-1], approximate=True)]
        step = total / count
        bounds, accumulated, target = [times[0]], 0.0, step
        for (_, _), (t, mag) in zip(samples, samples[1:]):
            accumulated += mag
            while len(bounds) < count and accumulated >= target:
                bounds.append(t)
                target += step
        bounds.append(times[-1])
        return _finish(bounds, [None] * (len(bounds) - 1), approximate=True)

    raise SegmentError("motion segmentation needs either a segment count, "
                       "or an average speed together with a segment length")


def _motion_floor(samples: list[tuple[float, float]]) -> float:
    """A noise floor below which a sample counts as stationary.

    A fraction of the median non-zero magnitude: robust to the odd bright frame and to
    scenes that never truly stop.
    """
    magnitudes = sorted(mag for _, mag in samples if mag > 0)
    if not magnitudes:
        return 0.0
    median = magnitudes[len(magnitudes) // 2]
    return 0.15 * median
