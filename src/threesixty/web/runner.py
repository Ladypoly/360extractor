"""Running external programs and turning their chatter into progress.

COLMAP and Brush are long-lived binaries that say useful things on stdout. This is one
place that runs a command, tails its output into a job's log, extracts a fraction with a
supplied pattern, and stops when asked -- rather than three stages each inventing their
own subprocess handling.

Cancellation terminates the child rather than merely setting a flag: a mapper left
running would hold the database and quietly corrupt the next attempt.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .jobs import Cancelled, Job

#: How often a cancelled process is checked before it is killed outright.
GRACE_SECONDS = 8

#: Lines matching these are worth colouring in the log viewer. COLMAP prefixes its
#: severity, and most tools shout in a recognisable way.
_WARN = re.compile(r"^\s*(W\d{8}|WARN|WARNING)\b|warning:", re.I)
_ERROR = re.compile(r"^\s*(E\d{8}|F\d{8}|ERROR|FATAL)\b|error:|failed", re.I)


def classify(line: str) -> str:
    if _ERROR.search(line):
        return "error"
    if _WARN.search(line):
        return "warn"
    return "info"


@dataclass
class ProgressPattern:
    """How to read progress out of a program's output.

    `pattern` should capture a current value and, optionally, a total. COLMAP's
    ``Registering image #24 (40)`` and Brush's step counters both fit this shape.
    """

    pattern: re.Pattern
    total: int | None = None
    message: str = ""

    def read(self, line: str) -> tuple[float, str] | None:
        match = self.pattern.search(line)
        if not match:
            return None
        groups = match.groups()
        try:
            current = float(groups[0])
        except (TypeError, ValueError):
            return None

        total = self.total
        if len(groups) > 1 and groups[1]:
            try:
                total = float(groups[1])
            except ValueError:
                pass
        if not total:
            return None

        fraction = max(0.0, min(1.0, current / total))
        label = self.message or "working"
        return fraction, f"{label} {int(current)} / {int(total)}"


@dataclass
class RunResult:
    returncode: int
    lines: list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def tail(self, count: int = 25) -> str:
        return "\n".join(self.lines[-count:])


def run(job: Job, argv: Sequence[str], *, cwd: Path | None = None,
        progress: ProgressPattern | None = None,
        on_line: Callable[[str], None] | None = None,
        label: str = "", base: float = 0.0, span: float = 1.0) -> RunResult:
    """Run a command, streaming its output into `job`.

    `base` and `span` place this command's progress inside a larger sequence, so a
    four-step reconstruction can report one continuous 0-100% rather than resetting
    at every step.
    """
    if label:
        job.update(message=label)
    job.log(f"$ {' '.join(str(a) for a in argv)}", "info")

    process = subprocess.Popen(
        [str(a) for a in argv],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,     # one stream: interleaving is what a log wants
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines: list[str] = []
    watcher = threading.Thread(target=_watch_for_cancel, args=(job, process), daemon=True)
    watcher.start()

    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        lines.append(line)
        job.log(line, classify(line))
        if on_line:
            on_line(line)
        if progress:
            read = progress.read(line)
            if read:
                fraction, message = read
                job.update(fraction=base + fraction * span, message=message)

    process.wait()
    if job.cancel.is_set():
        raise Cancelled()
    return RunResult(process.returncode, lines)


def _watch_for_cancel(job: Job, process: subprocess.Popen) -> None:
    """Terminate the child when the job is cancelled, and kill it if it dawdles."""
    while process.poll() is None:
        if job.cancel.wait(0.25):
            process.terminate()
            deadline = time.time() + GRACE_SECONDS
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if process.poll() is None:
                process.kill()
            return


@dataclass
class Step:
    """One named step of a multi-step run, for the Reconstruct step list."""

    key: str
    label: str
    argv: Sequence[str]
    progress: ProgressPattern | None = None
    optional: bool = False

    def describe(self) -> dict:
        return {"key": self.key, "label": self.label, "optional": self.optional}


def run_steps(job: Job, steps: Sequence[Step], cwd: Path | None = None,
              on_step: Callable[[str, str, float], None] | None = None) -> list[dict]:
    """Run steps in order, giving each an equal slice of the overall progress.

    Returns a record per step so the UI can show status and duration. Stops at the
    first failure: a mapper run on a database whose features never extracted only
    produces a more confusing error.
    """
    records: list[dict] = []
    total = max(len(steps), 1)

    for index, step in enumerate(steps):
        job.raise_if_cancelled()
        started = time.time()
        if on_step:
            on_step(step.key, "running", 0.0)
        job.update(message=step.label, detail=f"step {index + 1} of {total}")

        result = run(job, step.argv, cwd=cwd, progress=step.progress,
                     label=step.label, base=index / total, span=1 / total)
        duration = time.time() - started

        record = {"key": step.key, "label": step.label, "seconds": round(duration, 1),
                  "state": "done" if result.ok else "error"}
        if not result.ok:
            record["error"] = result.tail()
            records.append(record)
            if on_step:
                on_step(step.key, "error", duration)
            raise RuntimeError(f"{step.label} failed:\n{result.tail(12)}")

        records.append(record)
        if on_step:
            on_step(step.key, "done", duration)
        job.update(fraction=(index + 1) / total)

    return records
