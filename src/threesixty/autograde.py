"""Choosing a grade automatically from what the picture actually looks like.

This grades **for a splat trainer, not for a viewer**, and those want different things.
3DGS fits what is in the images: anything the grade destroys on the way in is destroyed
for good, and no amount of training recovers it. A clipped highlight has no gradient to
descend, so gaussians land on it flat and white. A crushed shadow takes the features
COLMAP was going to match with. Boosted chroma amplifies 4:2:0 subsampling and sensor
noise into per-gaussian colour speckle. So the goal is *unclipped and honest*, not
*attractive* -- and where those conflict, unclipped wins.

Three ideas do most of the work.

**Measure the part that gets trained.** A third of an equirectangular frame is sky, and
the sky is masked out of the splat. Statistics taken over the whole panorama are
therefore statistics of pixels that will be thrown away: a bright overcast sky drags the
median up and darkens the street to compensate, or a dark street drags it down and
blows the sky. Only the band between `SKY_ELEVATION` and `NADIR_ELEVATION` is measured
-- above it is sky, below it is whatever the camera was bolted to.

**Simulate before committing.** Every correction is applied to the measured pixels and
the result is checked for clipping; a correction that would clip is backed off until it
does not. This is the difference between "target a median of 0.46" and "target a median
of 0.46 *if that can be done without destroying the top of the histogram*".

**Never add saturation.** The previous version targeted a mean chroma and would push
saturation to 1.4 on ordinary overcast footage. There is no reading of "photometrically
faithful input" under which inventing colour helps, and plenty of ways it hurts.
Saturation is a ceiling now: it comes down when a camera profile is lurid, and otherwise
stays exactly 1.0.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg import FFmpegError, FFmpegInfo
from .rig import Grade

#: The band that gets trained on, in degrees of elevation. Above `SKY_ELEVATION` is sky
#: on any outdoor capture; below `NADIR_ELEVATION` is the car roof, the tripod, or the
#: hand holding the stick. Both are masked out of the splat, so neither should have a
#: vote in how the street is exposed.
SKY_ELEVATION = 40.0
NADIR_ELEVATION = -60.0

#: Where a well-exposed midtone should sit, measured on that band.
TARGET_MEDIAN = 0.46
#: Target for the 10th-90th percentile band.
TARGET_SPAN = 0.55

#: Nothing may be pushed past these.
HIGHLIGHT_CEILING = 0.97
SHADOW_FLOOR = 0.012
#: ...where "nothing" means all but this share of the trained band, at each end.
#: Specular highlights on wet road exist and are not worth darkening a street for.
#:
#: Measured on real drives, the budget buys very little and costs a lot: going from 1%
#: to 4% gained around 0.2 stops of lift while taking clipped pixels from ~0.5% to 3%.
#: The dark median on that footage is real scene contrast -- black tarmac under a bright
#: overcast -- not underexposure, and no global lift fixes it without destroying the
#: bright end.
CLIP_BUDGET = 1.0

#: Corrections are clamped to these, so auto can never produce something wild.
EXPOSURE_LIMIT = 1.5
#: Contrast is only ever *raised*, and barely. A wide span means the scene really is
#: high-dynamic-range, not that it needs flattening -- and flattening cannot recover
#: anything, it only discards separation the capture already had. The ceiling is low
#: because contrast pushes *both* ends outward, which is the fastest way to clip.
CONTRAST_RANGE = (1.0, 1.18)
#: Saturation is a ceiling, not a target: never above 1.0.
SATURATION_RANGE = (0.8, 1.0)
#: Mean chroma above which a camera profile is considered oversaturated for this
#: purpose. Overcast material genuinely sits near 0.10; anything past this is the
#: camera's look, not the scene.
CHROMA_CEILING = 0.30

#: Black is only pulled when the picture is visibly lifted (haze, flare through a 360
#: lens), and never by more than this -- crushing shadows costs COLMAP its features.
BLACK_LIFT_MIN = 0.05
BLACK_LIMIT = 0.08
#: ...and only when the picture is also flatter than this, which together with a lifted
#: bottom is what haze looks like. A picture sitting exactly on the span target is not
#: hazy, so the margin below TARGET_SPAN matters.
FLAT_SPAN = TARGET_SPAN * 0.9

#: Corrections smaller than these leave the control exactly neutral.
#:
#: Sized to be visually imperceptible, and deliberately generous: a grade that is not
#: quite the identity is not free. It changes the rig, which changes the extraction
#: fingerprint, which marks an already-extracted dataset stale and invites re-running
#: the whole thing for a tenth of a stop nobody can see. Auto has to be safe to press
#: on footage that is already fine.
DEADBAND = {
    "exposure": 0.12,     # stops
    "contrast": 0.06,
    "saturation": 0.06,
    "black": 0.02,
}


@dataclass
class Analysis:
    """What the picture looks like now, all in 0..1, measured on the trained band."""

    median: float
    low: float           # 1st percentile
    high: float          # 99th percentile
    inner_low: float     # 10th percentile
    inner_high: float    # 90th percentile
    chroma: float
    clipped_high: float  # share of pixels already at the top
    clipped_low: float
    #: True when the measurement excluded sky and nadir, i.e. the sample knew its shape.
    banded: bool = False

    @property
    def span(self) -> float:
        """The 10-90 band: the part of the picture that carries the scene."""
        return max(self.inner_high - self.inner_low, 1e-4)

    @property
    def full_span(self) -> float:
        return max(self.high - self.low, 1e-4)


def sample(ffmpeg: FFmpegInfo, image: Path, width: int = 256) -> np.ndarray:
    """Read an image as an (H, W, 3) float array in 0..1.

    Shaped rather than flat, because which *row* a pixel is on decides whether it is sky.
    """
    height = max(width // 2, 1)
    proc = subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-i", str(image),
         "-vf", f"scale={width}:{height}:flags=area,format=rgb24",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    expected = width * height * 3
    if len(proc.stdout) < expected:
        raise FFmpegError(
            f"could not read {image} for analysis: "
            f"{proc.stderr.decode(errors='replace').strip()}")
    raw = np.frombuffer(proc.stdout[:expected], dtype=np.uint8)
    return raw.reshape(height, width, 3).astype(np.float32) / 255.0


def trained_band(pixels: np.ndarray) -> np.ndarray:
    """The rows that reach the splat, flattened to (N, 3).

    A flat array is passed through: a caller that hands over pixels with no geometry has
    already decided which ones to measure.
    """
    if pixels.ndim != 3:
        return pixels.reshape(-1, 3)
    height = pixels.shape[0]
    top = int(height * (90.0 - SKY_ELEVATION) / 180.0)
    bottom = int(math.ceil(height * (90.0 - NADIR_ELEVATION) / 180.0))
    band = pixels[top:bottom]
    # A frame too short to have a band at all still has to be measurable.
    return (band if band.size else pixels).reshape(-1, 3)


def luma(pixels: np.ndarray) -> np.ndarray:
    """Rec.709 luma: green carries most of the perceived brightness."""
    return pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def analyse(pixels: np.ndarray) -> Analysis:
    """Measure a picture. `pixels` is (H, W, 3) or an already-chosen (N, 3)."""
    banded = pixels.ndim == 3
    flat = trained_band(pixels)
    values = luma(flat)
    low, inner_low, median, inner_high, high = (
        float(v) for v in np.percentile(values, [1, 10, 50, 90, 99]))
    chroma = float((flat.max(axis=1) - flat.min(axis=1)).mean())
    return Analysis(
        median=median, low=low, high=high,
        inner_low=inner_low, inner_high=inner_high, chroma=chroma,
        clipped_high=float((values > 0.995).mean()),
        clipped_low=float((values < 0.005).mean()),
        banded=banded,
    )


# -- the model ---------------------------------------------------------------
#
# Close enough to ffmpeg's `exposure` and `eq` to predict what will clip, which is all it
# is asked to do. `exposure` is vf_exposure's own arithmetic; `eq` is its contrast about
# the midpoint. The grade is chosen against these numbers rather than hoped about.


def apply_exposure(values: np.ndarray, exposure: float, black: float) -> np.ndarray:
    if exposure == 0.0 and black == 0.0:
        return values
    return np.clip((values - black) / (2.0 ** -exposure - black), 0.0, 1.0)


def apply_contrast(values: np.ndarray, contrast: float) -> np.ndarray:
    if contrast == 1.0:
        return values
    return np.clip((values - 0.5) * contrast + 0.5, 0.0, 1.0)


def _deadband(value: float, neutral: float, control: str) -> float:
    """Snap near-neutral corrections to exactly neutral."""
    return neutral if abs(value - neutral) < DEADBAND[control] else value


def _percentiles(values: np.ndarray) -> tuple[float, float]:
    """The ends of the histogram the guard rails apply to."""
    low, high = np.percentile(values, [CLIP_BUDGET, 100.0 - CLIP_BUDGET])
    return float(low), float(high)


def grade_for(analysis: Analysis, values: np.ndarray | None = None) -> Grade:
    """Work out a conservative grade from a measurement.

    `values` is the trained band's luma, when the caller has it. With it, each correction
    is simulated and backed off until it stops clipping; without it the same corrections
    are chosen from the percentiles alone, which is very nearly as good and is what makes
    this testable on synthetic data.
    """
    grade = Grade()
    if values is None:
        values = np.array([analysis.low, analysis.inner_low, analysis.median,
                           analysis.inner_high, analysis.high], dtype=np.float32)

    # 1. Black. Only for a picture that is *both* lifted and flat, which is what haze and
    #    lens flare look like; a full-range picture that simply has no deep shadows is not
    #    hazy and must be left alone. Never past the limit either: pulling black crushes
    #    everything under it, and the shadows hold the features COLMAP matches on.
    if analysis.low > BLACK_LIFT_MIN and analysis.span < FLAT_SPAN:
        black = min(BLACK_LIMIT, (analysis.low - 0.01) * 0.9)
        grade.black = round(_deadband(max(0.0, black), 0.0, "black"), 3)
    working = apply_exposure(values, 0.0, grade.black)

    # 2. Exposure, from the median, held to whatever headroom is left. A dark street
    #    under a blown sky is a real situation and the answer is to expose the street --
    #    the sky is masked out anyway, which is why it is not in `values`.
    median = float(np.percentile(working, 50))
    if median > 1e-4:
        stops = math.log2(TARGET_MEDIAN / median)
        _, top = _percentiles(working)
        if stops > 0:
            headroom = math.log2(HIGHLIGHT_CEILING / max(top, 1e-4))
            stops = min(stops, max(headroom, 0.0))
        else:
            bottom, _ = _percentiles(working)
            floor = math.log2(max(SHADOW_FLOOR, 1e-4) / max(bottom, 1e-4))
            stops = max(stops, min(floor, 0.0))
        grade.exposure = round(_deadband(
            max(-EXPOSURE_LIMIT, min(EXPOSURE_LIMIT, stops)), 0.0, "exposure"), 3)
    working = apply_exposure(values, grade.exposure, grade.black)

    # 3. Contrast, only for footage that is genuinely flat, and only as far as the
    #    histogram has room for. Contrast pushes both ends outward at once, so this is
    #    the correction most likely to be refused outright.
    span = max(float(np.percentile(working, 90) - np.percentile(working, 10)), 1e-4)
    if span < TARGET_SPAN:
        wanted = min(TARGET_SPAN / span, CONTRAST_RANGE[1])
        bottom, top = _percentiles(working)
        # (v - 0.5) * c + 0.5 stays inside the guard rails for c up to:
        limits = []
        if top > 0.5:
            limits.append((HIGHLIGHT_CEILING - 0.5) / (top - 0.5))
        if bottom < 0.5:
            limits.append((SHADOW_FLOOR - 0.5) / (bottom - 0.5))
        safe = min(limits) if limits else CONTRAST_RANGE[1]
        contrast = max(CONTRAST_RANGE[0], min(wanted, safe))
        grade.contrast = round(_deadband(contrast, 1.0, "contrast"), 3)

    # 4. Saturation: a ceiling, never a target. Nothing here ever adds colour.
    if analysis.chroma > CHROMA_CEILING:
        saturation = max(SATURATION_RANGE[0], CHROMA_CEILING / analysis.chroma)
        grade.saturation = round(_deadband(min(saturation, 1.0), 1.0, "saturation"), 3)

    return grade


def auto_grade(ffmpeg: FFmpegInfo, image: Path) -> tuple[Grade, Analysis]:
    """Measure an image and return the grade it wants, plus the measurement."""
    pixels = sample(ffmpeg, image)
    analysis = analyse(pixels)
    return grade_for(analysis, luma(trained_band(pixels))), analysis


def predict(analysis: Analysis, grade: Grade, values: np.ndarray) -> tuple[float, float]:
    """Share of the trained band that would end up clipped, high and low."""
    after = apply_contrast(
        apply_exposure(values, grade.exposure, grade.black), grade.contrast)
    return float((after > 0.995).mean()), float((after < 0.005).mean())


def describe(analysis: Analysis, grade: Grade) -> list[str]:
    """Lines explaining what was measured and why, for the CLI and the UI."""
    where = "street level" if analysis.banded else "frame"
    lines = [
        f"{where}: median {analysis.median:.2f} (target {TARGET_MEDIAN:.2f}), "
        f"mid-range {analysis.inner_low:.2f}-{analysis.inner_high:.2f}, "
        f"chroma {analysis.chroma:.2f}",
    ]
    if analysis.banded:
        lines.append(
            f"measured between {NADIR_ELEVATION:+.0f}° and {SKY_ELEVATION:+.0f}°, "
            "since sky and nadir are masked out of the splat")
    if analysis.clipped_high > 0.02:
        lines.append(
            f"{analysis.clipped_high * 100:.0f}% of it is already at full brightness, "
            f"so the exposure lift is held back")

    # The most useful thing auto can say is when it *declined* to do the obvious. A dark
    # street under a bright overcast wants more exposure than its highlights can survive,
    # and the user deserves to know that rather than assume auto did nothing.
    if analysis.median > 1e-4:
        wanted = math.log2(TARGET_MEDIAN / analysis.median)
        if wanted - grade.exposure > DEADBAND["exposure"]:
            lines.append(
                f"{wanted:+.2f} stops would hit the target, but only {grade.exposure:+.2f} "
                f"fits under the highlights — the rest would clip, which a splat cannot "
                f"recover. Raise exposure by hand if you want it brighter anyway.")
    if grade.is_identity:
        lines.append("already well exposed; nothing to change")
    else:
        parts = [f"exposure {grade.exposure:+.2f} stops"]
        if grade.black:
            parts.append(f"black {grade.black:.3f}")
        if grade.contrast != 1.0:
            parts.append(f"contrast {grade.contrast:.2f}")
        if grade.saturation != 1.0:
            parts.append(f"saturation {grade.saturation:.2f} (never raised)")
        lines.append(", ".join(parts))
    return lines
