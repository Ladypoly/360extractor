"""Long-running work, one job per pipeline stage.

The old design had a single job for the whole session, which meant extraction and
detection could not be told apart, could not run independently, and a stuck one blocked
everything. Worse, the only signal a caller got was "something is already running" --
true, unhelpful, and the reason detection looked dead while it was working.

So: a job per stage, each carrying state, a *fraction*, a message, a log, and a cancel
flag. Two rules follow from what the UI needs.

**A running job must report a fraction.** A progress bar sitting at zero while work
happens is worse than no progress bar, because it reads as a hang.

**A job outlives the page.** Navigating away from a stage must not stop it, and coming
back has to restore the percentage, the message and the recent log. That is why the log
lives here rather than being streamed and forgotten.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

#: Pipeline stages, in order. The UI renders navigation straight from this.
STAGES = ("capture", "refine", "reconstruct", "train", "inspect")

#: How much log each job keeps. Enough to explain a failure, bounded so a chatty
#: process cannot grow without limit.
LOG_LIMIT = 2000

PENDING = "pending"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"


@dataclass
class LogLine:
    """One line of output, tagged so the viewer can style it without parsing."""

    text: str
    level: str = "info"   # info | warn | error
    at: float = field(default_factory=time.time)


class Job:
    """One stage's long-running work."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self._lock = threading.Lock()
        self._log: deque[LogLine] = deque(maxlen=LOG_LIMIT)

        self.state = PENDING
        self.message = ""
        self.fraction = 0.0
        self.detail = ""           # secondary line, e.g. "camera c03"
        self.error = ""
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.result: dict = {}
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        #: Bumped on every change, so a client can tell "nothing new" cheaply.
        self.revision = 0

    # -- state ---------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.state == RUNNING

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    def update(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)
            self.revision += 1

    def log(self, text: str, level: str = "info") -> None:
        for line in str(text).splitlines():
            if line.strip():
                with self._lock:
                    self._log.append(LogLine(line.rstrip(), level))
                    self.revision += 1

    def log_tail(self, limit: int = 200, since: int = 0) -> list[dict]:
        with self._lock:
            lines = list(self._log)
        selected = lines[max(len(lines) - limit, since):] if since else lines[-limit:]
        return [{"text": l.text, "level": l.level} for l in selected]

    def snapshot(self, log_limit: int = 200) -> dict:
        """Everything the UI needs to render this stage, including after a reload."""
        with self._lock:
            base = {
                "stage": self.stage,
                "state": self.state,
                "message": self.message,
                "detail": self.detail,
                "fraction": round(self.fraction, 4),
                "error": self.error,
                "elapsed": round(self.elapsed, 1),
                "revision": self.revision,
                "result": self.result,
                "cancellable": self.state == RUNNING,
            }
        base["log"] = self.log_tail(log_limit)
        return base

    # -- running -------------------------------------------------------------

    def start(self, work: Callable[["Job"], dict | None], name: str = "") -> None:
        """Run `work` on a thread, with state handled here rather than by every caller."""
        if self.running:
            raise AlreadyRunning(self.stage, self.message or name)

        self.cancel.clear()
        self.update(state=RUNNING, message=name or "starting", detail="", fraction=0.0,
                    error="", result={}, started_at=time.time(), finished_at=None)

        def wrapper() -> None:
            try:
                returned = work(self)
                # Tolerate a work function that returns nothing useful: the job's own
                # bookkeeping should not fail because a caller forgot to return a dict.
                result = returned if isinstance(returned, dict) else {}
                if self.cancel.is_set():
                    self.update(state=CANCELLED, message="cancelled",
                                finished_at=time.time())
                else:
                    self.update(state=DONE, fraction=1.0, result=result,
                                message=result.get("summary", "finished"),
                                finished_at=time.time())
            except Cancelled:
                self.update(state=CANCELLED, message="cancelled", finished_at=time.time())
                self.log("cancelled", "warn")
            except Exception as exc:                      # noqa: BLE001 - reported to UI
                traceback.print_exc()
                self.update(state=ERROR, error=str(exc), message="failed",
                            finished_at=time.time())
                self.log(str(exc), "error")

        self.thread = threading.Thread(target=wrapper, daemon=True, name=f"job-{self.stage}")
        self.thread.start()

    def raise_if_cancelled(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def progress(self, fraction: float, message: str = "", detail: str = "") -> None:
        """Report real progress. Callers should prefer this over `update`."""
        fields = {"fraction": max(0.0, min(1.0, float(fraction)))}
        if message:
            fields["message"] = message
        if detail:
            fields["detail"] = detail
        self.update(**fields)
        self.raise_if_cancelled()


class Cancelled(Exception):
    """Raised inside a job when the user asked it to stop."""


class AlreadyRunning(RuntimeError):
    """Something is running -- and, unlike before, we say exactly what and where."""

    def __init__(self, stage: str, what: str = "") -> None:
        self.stage = stage
        self.what = what
        label = {
            "capture": "Frame extraction",
            "refine": "Dynamic mask detection",
            "reconstruct": "Reconstruction",
            "train": "Splat training",
            "inspect": "Splat cleanup",
        }.get(stage, stage)
        super().__init__(f"{label} is currently running.")


class JobRegistry:
    """The session's jobs, one per stage."""

    def __init__(self) -> None:
        self._jobs = {stage: Job(stage) for stage in STAGES}

    def __getitem__(self, stage: str) -> Job:
        if stage not in self._jobs:
            raise KeyError(f"unknown stage {stage!r}; expected one of {STAGES}")
        return self._jobs[stage]

    def snapshot(self, log_limit: int = 0) -> dict[str, dict]:
        return {stage: job.snapshot(log_limit) for stage, job in self._jobs.items()}

    def running(self) -> list[Job]:
        return [job for job in self._jobs.values() if job.running]

    def any_running(self) -> Job | None:
        found = self.running()
        return found[0] if found else None

    def cancel_all(self) -> None:
        for job in self._jobs.values():
            job.cancel.set()


def steps_progress(job: Job, index: int, total: int, message: str,
                   inner: float = 0.0) -> None:
    """Fraction for work made of `total` equal steps, `index` of them complete.

    Saves every caller from doing the same arithmetic slightly differently.
    """
    total = max(total, 1)
    job.progress((index + max(0.0, min(1.0, inner))) / total, message)


def iterate(job: Job, items: Iterable, message: str = "") -> Iterable:
    """Iterate, reporting a fraction and honouring cancellation between items."""
    items = list(items)
    for index, item in enumerate(items):
        job.raise_if_cancelled()
        job.progress(index / max(len(items), 1),
                     message or job.message,
                     detail=f"{index + 1} / {len(items)}")
        yield item
    job.progress(1.0, message or job.message, detail=f"{len(items)} / {len(items)}")
