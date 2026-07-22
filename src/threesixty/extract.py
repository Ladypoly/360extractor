"""Running the planned passes and reporting progress."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .ffmpeg import FFmpegError, FFmpegInfo
from .plan import ExtractPlan, Pass, build_pass_argv

ProgressFn = Callable[["Progress"], None]


@dataclass
class Progress:
    """A snapshot handed to the progress callback."""

    pass_index: int
    pass_count: int
    frame: int
    expected_frames: int
    cameras_in_pass: int
    elapsed: float
    speed: str = ""

    @property
    def fraction(self) -> float:
        if self.pass_count == 0:
            return 1.0
        within = min(self.frame / self.expected_frames, 1.0) if self.expected_frames else 0.0
        return (self.pass_index + within) / self.pass_count


@dataclass
class ExtractResult:
    """What happened, in enough detail to report honestly."""

    images_written: int = 0
    masks_written: int = 0
    cameras_done: int = 0
    cameras_skipped: int = 0
    passes_run: int = 0
    elapsed: float = 0.0
    cancelled: bool = False
    directories: list[Path] = field(default_factory=list)


def _write_sidecars(plan: ExtractPlan, job) -> int:
    """Link this camera's mask beside each of its images, if masking is in sidecar mode."""
    mask_plan = getattr(plan, "mask_plan", None)
    if mask_plan is None or mask_plan.mode != "sidecar" or job.mask_directory is None:
        return 0
    mask = mask_plan.camera_masks.get(job.camera.name)
    if mask is None:
        return 0

    from .mask.apply import link_sidecars
    return link_sidecars(mask, job.directory, job.mask_directory)


def _count_outputs(directory: Path, pattern: str) -> int:
    """Count files actually produced for one camera's pattern."""
    prefix, _, rest = pattern.partition("%")
    suffix = rest.partition("d")[2]
    if not directory.exists():
        return 0
    return sum(
        1 for p in directory.iterdir()
        if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix)
    )


def _stream_progress(
    proc: subprocess.Popen[str],
    single_pass: Pass,
    plan: ExtractPlan,
    pass_count: int,
    on_progress: ProgressFn | None,
    started: float,
) -> None:
    """Consume ffmpeg's -progress stream so the user sees movement, not a frozen bar."""
    speed = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key == "speed":
            speed = value
        elif key == "frame" and on_progress is not None:
            try:
                frame = int(value)
            except ValueError:
                continue
            on_progress(Progress(
                pass_index=single_pass.index,
                pass_count=pass_count,
                frame=frame,
                expected_frames=plan.estimated_frames,
                cameras_in_pass=len(single_pass.jobs),
                elapsed=time.monotonic() - started,
                speed=speed,
            ))


def run_extraction(
    plan: ExtractPlan,
    ffmpeg: FFmpegInfo,
    on_progress: ProgressFn | None = None,
    dry_run: bool = False,
    overwrite: bool = True,
) -> ExtractResult:
    """Execute every pass in the plan.

    A pass either completes and marks its cameras done, or it fails and marks
    nothing -- so a resumed run never treats a half-written camera as finished.
    """
    result = ExtractResult(cameras_skipped=len(plan.skipped))
    started = time.monotonic()

    if dry_run:
        for single_pass in plan.passes:
            print(" ".join(build_pass_argv(ffmpeg.path, plan, single_pass, overwrite)))
        return result

    for single_pass in plan.passes:
        for job in single_pass.jobs:
            job.directory.mkdir(parents=True, exist_ok=True)
            # A stale marker from an earlier run would otherwise outlive its images.
            job.marker.unlink(missing_ok=True)

        graph_path = plan.output_root / ".threesixty" / f"pass{single_pass.index}.filter"
        argv = build_pass_argv(ffmpeg.path, plan, single_pass, overwrite, graph_path)
        # stderr goes to a file rather than a second pipe: we only drain stdout while
        # the process runs, and an unread stderr pipe filling its buffer would deadlock
        # ffmpeg mid-render.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errfile:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=errfile,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            try:
                _stream_progress(proc, single_pass, plan, len(plan.passes), on_progress, started)
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                result.cancelled = True
                result.elapsed = time.monotonic() - started
                return result

            errfile.seek(0)
            stderr = errfile.read().strip()

        if proc.returncode != 0:
            raise FFmpegError(
                f"pass {single_pass.index + 1} failed (exit {proc.returncode})\n"
                f"  cameras: {', '.join(j.camera.name for j in single_pass.jobs)}\n"
                f"{stderr}"
            )

        result.passes_run += 1
        for job in single_pass.jobs:
            written = _count_outputs(job.directory, job.pattern)
            result.images_written += written
            result.cameras_done += 1
            if job.directory not in result.directories:
                result.directories.append(job.directory)

            # Sidecars are written once the images exist, because there has to be one
            # mask file per image for Brush to pair them up.
            result.masks_written += _write_sidecars(plan, job)

            if written:
                job.marker.write_text(
                    f"images={written}\ncamera={job.camera.name}\n", encoding="utf-8"
                )

    result.elapsed = time.monotonic() - started
    return result


def clear_markers(root: Path) -> int:
    """Drop resume markers under `root` so the next run redoes everything."""
    removed = 0
    for marker in root.rglob(".*.done"):
        marker.unlink(missing_ok=True)
        removed += 1
    return removed


def terminal_progress(stream=sys.stderr) -> ProgressFn:
    """A single-line progress bar, quiet when not attached to a terminal."""
    width = 28
    last = [0.0]

    def render(progress: Progress) -> None:
        now = time.monotonic()
        if now - last[0] < 0.1 and progress.fraction < 1.0:
            return
        last[0] = now
        filled = int(progress.fraction * width)
        bar = "#" * filled + "-" * (width - filled)
        columns = shutil.get_terminal_size((100, 24)).columns
        text = (
            f"\r[{bar}] {progress.fraction * 100:5.1f}%  "
            f"pass {progress.pass_index + 1}/{progress.pass_count}  "
            f"frame {progress.frame}  x{progress.cameras_in_pass} cams  {progress.speed}"
        )
        stream.write(text[:columns - 1].ljust(min(columns - 1, len(text))))
        stream.flush()

    def noop(progress: Progress) -> None:
        return

    return render if stream.isatty() else noop


def finish_progress(stream=sys.stderr) -> None:
    if stream.isatty():
        stream.write("\n")
        stream.flush()
