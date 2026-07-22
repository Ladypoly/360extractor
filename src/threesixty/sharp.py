"""Sharp frame selection.

Uniform time sampling is blind to motion blur: it takes whatever frame lands on the
tick, and on a walking or driving capture a good share of those are smeared. Blurred
frames are worse than useless for photogrammetry -- they contribute no matchable
features and drag down the reconstruction.

So instead of "every 0.5 seconds", this picks "the sharpest frame in each 0.5 second
window". Same frame count, materially better dataset. The idea is Florian Bruggisser's
sharp-frame-extractor (github.com/cansik/sharp-frame-extractor).

Sharpness comes from ffmpeg's own `blurdetect` filter, so there is no extra
dependency and no second decoder to keep in step. It reports *blurriness*, so lower
is sharper -- verified monotonic against known Gaussian blur:
sigma 0 -> 4.47, sigma 1 -> 6.08, sigma 3 -> 11.10, sigma 8 -> 20.06.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg import FFmpegError, FFmpegInfo, MediaInfo

#: Analysis runs on a downscaled copy: blur ranking survives it and it is several
#: times faster than scoring full 4K or 8K frames.
ANALYSIS_WIDTH = 640

_BLUR = re.compile(r"lavfi\.blur=([\d.eE+-]+)")
_FRAME = re.compile(r"^frame:(\d+)", re.M)


@dataclass(frozen=True)
class Sharpness:
    """Per-frame blur scores for one source, indexed from the analysis start."""

    scores: list[float]

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def sharpest(self) -> int | None:
        return min(range(len(self.scores)), key=self.scores.__getitem__) if self.scores else None


def analyze(ffmpeg: FFmpegInfo, media: MediaInfo, start: float | None = None,
            end: float | None = None, width: int = ANALYSIS_WIDTH,
            block_pct: int = 80) -> Sharpness:
    """Score every frame's blurriness. Lower is sharper."""
    argv = [str(ffmpeg.path), "-hide_banner", "-loglevel", "info", "-nostdin"]
    # Must match the extraction pass exactly, or frame numbers refer to
    # different frames in the two runs.
    if start is not None:
        argv += ["-ss", f"{start:g}"]
    if end is not None:
        argv += ["-to", f"{end:g}"]
    argv += [
        "-i", str(media.path),
        "-vf", f"scale={width}:-2,blurdetect=block_pct={block_pct},metadata=print:file=-",
        "-an", "-f", "null", "-",
    ]

    proc = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(f"sharpness analysis failed:\n{proc.stderr.strip()}")

    scores = [float(value) for value in _BLUR.findall(proc.stdout + proc.stderr)]
    if not scores:
        raise FFmpegError(
            "sharpness analysis produced no scores -- this ffmpeg's blurdetect filter "
            "did not report metadata. Use --fps instead of --sharp."
        )
    return Sharpness(scores=scores)


def choose(sharpness: Sharpness, source_fps: float, target_fps: float) -> list[int]:
    """Pick the sharpest frame within each 1/target_fps window.

    Returns frame indices counted from the start of the analysed range, which is the
    same origin ffmpeg's `select` filter uses when given the same seek options.
    """
    if not sharpness.scores:
        return []
    if target_fps <= 0:
        raise ValueError(f"target fps must be positive, got {target_fps}")

    block = max(int(round(source_fps / target_fps)), 1) if source_fps else 1
    chosen = []
    for begin in range(0, len(sharpness.scores), block):
        window = range(begin, min(begin + block, len(sharpness.scores)))
        chosen.append(min(window, key=sharpness.scores.__getitem__))
    return chosen


def select_expression(frames: list[int]) -> str:
    """An ffmpeg `select` expression matching exactly these frame numbers.

    Commas are escaped because the filtergraph parser would otherwise read them as
    filter separators. Long expressions are fine: the caller writes the graph to a
    script file rather than the command line.
    """
    if not frames:
        return "select=0"
    terms = "+".join(rf"eq(n\,{frame})" for frame in frames)
    return f"select='{terms}'"


def summarize(sharpness: Sharpness, chosen: list[int]) -> str:
    """One line for the log: was this actually worth doing?"""
    if not chosen or not sharpness.scores:
        return "no frames analysed"
    picked = [sharpness.scores[i] for i in chosen]
    everything = sharpness.scores
    return (f"picked {len(chosen)} of {len(everything)} frames, "
            f"mean blur {sum(picked)/len(picked):.2f} vs {sum(everything)/len(everything):.2f} "
            f"across all frames (lower is sharper)")
