"""Choosing a grade automatically from what the picture actually looks like.

Deliberately conservative. The job here is to rescue flat or dark footage, not to
restyle it: a photogrammetry dataset wants an honest, well-spread image, and anything
clipped is detail thrown away that no amount of later work recovers.

Three measurements, three corrections:

* the **median** luma sets exposure -- robust where the mean is not, because a big
  overcast sky drags a mean upwards and would darken the whole street to compensate;
* the **1st and 99th percentiles** set contrast, since that is the range actually
  occupied rather than the range two stray pixels reach;
* the mean **chroma spread** sets saturation.

Every correction is clamped, and the whole thing is a no-op on footage that is already
well exposed.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg import FFmpegError, FFmpegInfo
from .rig import Grade

#: These are calibrated on real 360 footage, not on ordinary photographs, and the
#: difference matters: **30-40% of an equirectangular frame is sky**. Measured across
#: frames of an overcast village drive, the 1st-99th percentile span sits at 0.90-0.97
#: and the 10th-90th at 0.75-0.88, purely because sky and shadow occupy both ends. A
#: target tuned for normal photos reads that as "too contrasty" and flattens the scene.

#: Where a well-exposed midtone should sit. Slightly below half, because a bright sky
#: pulls the top of the histogram up regardless of how dark the street is.
TARGET_MEDIAN = 0.46
#: Target for the 10th-90th percentile band. The 1-99 band is almost pure sky-and-shadow
#: on a panorama and says nothing useful about the part of the picture that matters.
TARGET_SPAN = 0.60
#: Mean chroma of footage that looks properly coloured. Overcast material genuinely sits
#: near 0.10, so targeting the 0.2+ of a sunny photograph only ever produces something
#: lurid.
TARGET_CHROMA = 0.14

#: Corrections are clamped to these, so auto can never produce something wild.
EXPOSURE_LIMIT = 1.5
#: Contrast is only ever *raised*. A wide span means the scene really is
#: high-dynamic-range, not that it needs flattening -- and flattening cannot recover
#: anything, it only discards separation the capture already had.
CONTRAST_RANGE = (1.0, 1.6)
SATURATION_RANGE = (0.85, 1.4)

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
}


@dataclass
class Analysis:
    """What the picture looks like now, all in 0..1."""

    median: float
    low: float           # 1st percentile
    high: float          # 99th percentile
    inner_low: float     # 10th percentile
    inner_high: float    # 90th percentile
    chroma: float
    clipped_high: float  # share of pixels already at the top
    clipped_low: float

    @property
    def span(self) -> float:
        """The 10-90 band: the part of the picture that carries the scene."""
        return max(self.inner_high - self.inner_low, 1e-4)

    @property
    def full_span(self) -> float:
        return max(self.high - self.low, 1e-4)


def sample(ffmpeg: FFmpegInfo, image: Path, width: int = 256) -> np.ndarray:
    """Read an image as an (N, 3) float array in 0..1."""
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
    return raw.reshape(-1, 3).astype(np.float32) / 255.0


def analyse(pixels: np.ndarray) -> Analysis:
    """Measure a picture. `pixels` is (N, 3) in 0..1."""
    # Rec.709 luma: green carries most of the perceived brightness.
    luma = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    low, inner_low, median, inner_high, high = (
        float(v) for v in np.percentile(luma, [1, 10, 50, 90, 99]))
    chroma = float((pixels.max(axis=1) - pixels.min(axis=1)).mean())
    return Analysis(
        median=median, low=low, high=high,
        inner_low=inner_low, inner_high=inner_high, chroma=chroma,
        clipped_high=float((luma > 0.995).mean()),
        clipped_low=float((luma < 0.005).mean()),
    )


def _deadband(value: float, neutral: float, control: str) -> float:
    """Snap near-neutral corrections to exactly neutral."""
    return neutral if abs(value - neutral) < DEADBAND[control] else value


def grade_for(analysis: Analysis) -> Grade:
    """Work out a conservative grade from a measurement."""
    grade = Grade()

    # Exposure, from the median. Held back where the picture is already clipping,
    # since brightening those pixels only destroys more of them.
    if analysis.median > 1e-4:
        stops = math.log2(TARGET_MEDIAN / analysis.median)
        if stops > 0 and analysis.clipped_high > 0.02:
            stops *= max(0.0, 1.0 - analysis.clipped_high * 10.0)
        grade.exposure = round(_deadband(
            max(-EXPOSURE_LIMIT, min(EXPOSURE_LIMIT, stops)), 0.0, "exposure"), 3)

    # Contrast, from how much of the range the picture actually occupies. Measured on
    # the original span: exposure shifts the picture but barely changes its spread.
    contrast = TARGET_SPAN / analysis.span
    grade.contrast = round(_deadband(
        max(CONTRAST_RANGE[0], min(CONTRAST_RANGE[1], contrast)), 1.0, "contrast"), 3)

    # Saturation, from mean chroma. Skipped on nearly monochrome footage, where the
    # ratio explodes and the result would be lurid.
    if analysis.chroma > 0.02:
        saturation = TARGET_CHROMA / analysis.chroma
        grade.saturation = round(_deadband(
            max(SATURATION_RANGE[0], min(SATURATION_RANGE[1], saturation)), 1.0,
            "saturation"), 3)

    return grade


def auto_grade(ffmpeg: FFmpegInfo, image: Path) -> tuple[Grade, Analysis]:
    """Measure an image and return the grade it wants, plus the measurement."""
    analysis = analyse(sample(ffmpeg, image))
    return grade_for(analysis), analysis


def describe(analysis: Analysis, grade: Grade) -> list[str]:
    """Lines explaining what was measured and why, for the CLI and the UI."""
    lines = [
        f"median {analysis.median:.2f} (target {TARGET_MEDIAN:.2f}), "
        f"mid-range {analysis.inner_low:.2f}-{analysis.inner_high:.2f}, "
        f"chroma {analysis.chroma:.2f}",
    ]
    if analysis.clipped_high > 0.02:
        lines.append(
            f"{analysis.clipped_high * 100:.0f}% of pixels are already at full "
            f"brightness, so the exposure lift is held back")
    if grade.is_identity:
        lines.append("already well exposed; nothing to change")
    else:
        lines.append(f"exposure {grade.exposure:+.2f} stops, contrast {grade.contrast:.2f}, "
                     f"saturation {grade.saturation:.2f}")
    return lines
