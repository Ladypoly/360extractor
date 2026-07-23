"""Stage A of capture: pull equirect frames from the source into the working set.

The capture flow is two stages: choose *which frames*, then choose *which cameras*. This
module is the first half. It decodes the video once, thins it to the chosen frames (fps /
sharpest-per-second / every-Nth / all), and writes the panorama frames straight to
``frames/<clip>/`` -- no rig, no projection, no grade. Those belong to Stage B, applied
when the equirect frames become camera tiles, so the working set stays a neutral source of
truth the user can re-rig and re-mask without decoding the video again.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

from . import sharp
from .ffmpeg import FFmpegError, FFmpegInfo, MediaInfo
from .plan import FrameSelection, safe_stem


@dataclass
class FramesResult:
    directory: Path
    clip: str
    count: int
    cancelled: bool = False


def frames_dir(root: str | Path, clip: str) -> Path:
    """Where a clip's extracted equirect frames live inside a project."""
    return Path(root) / "frames" / clip


def extract_frames(ffmpeg: FFmpegInfo, media: MediaInfo, selection: FrameSelection,
                   output_root: str | Path, quality: int = 2,
                   on_progress=None, on_analysis=None, should_cancel=None,
                   overwrite: bool = True) -> FramesResult:
    """Decode `media`, thin to `selection`, write equirect JPEGs to frames/<clip>/."""
    selection.validate()

    # Sharp mode needs a decode pass to score frames before it can name the ones to keep.
    if selection.mode == "sharp" and media.is_video and not selection.frames:
        scores = sharp.analyze(ffmpeg, media, selection.start, selection.end)
        chosen = sharp.choose(scores, media.fps, selection.value)
        if on_analysis is not None:
            on_analysis(sharp.summarize(scores, chosen))
        selection = replace(selection, frames=tuple(chosen))

    clip = safe_stem(media.path.stem)
    out_dir = frames_dir(output_root, clip)
    out_dir.mkdir(parents=True, exist_ok=True)
    digits = max(5, len(str(selection.estimate_frames(media))) + 1)
    pattern = f"%0{digits}d.jpg"

    argv = [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-y" if overwrite else "-n"]
    # Input-side seek keeps the decoder off skipped frames (matches build_pass_argv).
    if selection.start is not None:
        argv += ["-ss", f"{selection.start:g}"]
    if selection.end is not None:
        argv += ["-to", f"{selection.end:g}"]
    argv += ["-i", str(media.path)]

    prefix = selection.filter_prefix(media)
    if prefix:
        argv += ["-vf", prefix]
    # vfr so a `select` expression drops rather than duplicates the frames it rejects.
    argv += ["-fps_mode", "vfr", "-q:v", str(quality), "-progress", "pipe:1",
             str(out_dir / pattern)]

    expected = selection.estimate_frames(media)
    started = time.monotonic()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    cancelled = False
    try:
        for line in process.stdout:
            if should_cancel and should_cancel():
                process.terminate()
                cancelled = True
                break
            key, _, value = line.strip().partition("=")
            if key == "frame" and on_progress is not None:
                try:
                    frame = int(value)
                except ValueError:
                    continue
                on_progress(min(frame / max(expected, 1), 1.0), frame,
                            time.monotonic() - started)
    finally:
        error = process.stderr.read() if process.stderr else ""
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
        code = process.wait()

    if not cancelled and code not in (0, None):
        raise FFmpegError(f"frame extraction failed: {error.strip()}")

    count = sum(1 for p in out_dir.iterdir() if p.suffix.lower() == ".jpg")
    return FramesResult(directory=out_dir, clip=clip, count=count, cancelled=cancelled)
