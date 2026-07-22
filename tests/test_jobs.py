"""Per-stage jobs.

Two properties come straight from the brief and are the reason this exists:

* a running job reports a **fraction**, because a bar sitting at zero reads as a hang --
  the exact complaint that started the redesign;
* a job **outlives the view**, so leaving a stage does not stop it and returning
  restores percentage, message and log.
"""

import threading
import time

import pytest

from threesixty.web.jobs import (
    AlreadyRunning,
    Cancelled,
    Job,
    JobRegistry,
    STAGES,
    iterate,
    steps_progress,
)


def wait_for(job, state, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.state == state:
            return True
        time.sleep(0.02)
    return False


class TestLifecycle:
    def test_starts_pending(self):
        assert Job("capture").state == "pending"

    def test_runs_and_finishes(self):
        job = Job("capture")
        job.start(lambda j: {"summary": "all done"})
        assert wait_for(job, "done")
        assert job.fraction == 1.0
        assert job.message == "all done"

    def test_failure_is_recorded_not_raised(self):
        job = Job("capture")

        def boom(j):
            raise RuntimeError("the mapper fell over")

        job.start(boom)
        assert wait_for(job, "error")
        assert "mapper fell over" in job.error
        assert any("mapper fell over" in line["text"] for line in job.log_tail())

    def test_result_is_kept(self):
        job = Job("train")
        job.start(lambda j: {"splat": "a.ply", "summary": "trained"})
        assert wait_for(job, "done")
        assert job.result["splat"] == "a.ply"

    def test_elapsed_is_measured(self):
        job = Job("capture")
        job.start(lambda j: (time.sleep(0.05), {})[1])
        assert wait_for(job, "done")
        assert job.elapsed > 0


class TestProgress:
    def test_progress_moves_the_fraction(self):
        job = Job("refine")
        job.progress(0.42, "halfway", detail="camera c03")
        assert job.fraction == pytest.approx(0.42)
        assert job.message == "halfway"
        assert job.detail == "camera c03"

    def test_fraction_is_clamped(self):
        job = Job("refine")
        job.progress(5.0)
        assert job.fraction == 1.0
        job.progress(-2.0)
        assert job.fraction == 0.0

    def test_a_running_job_reports_a_real_fraction(self):
        """The bug that started the redesign: a bar stuck at zero reads as a hang."""
        job = Job("refine")
        seen = []
        release = threading.Event()

        def work(j):
            for index in range(4):
                j.progress((index + 1) / 4, f"step {index + 1}")
                seen.append(j.fraction)
            release.set()
            return {}

        job.start(work)
        assert release.wait(5)
        assert wait_for(job, "done")
        assert seen == [0.25, 0.5, 0.75, 1.0]

    def test_steps_progress_spreads_evenly(self):
        job = Job("reconstruct")
        steps_progress(job, 1, 4, "matching")
        assert job.fraction == pytest.approx(0.25)
        steps_progress(job, 1, 4, "matching", inner=0.5)
        assert job.fraction == pytest.approx(0.375)

    def test_iterate_reports_and_finishes_at_one(self):
        job = Job("refine")
        assert list(iterate(job, [1, 2, 3, 4])) == [1, 2, 3, 4]
        assert job.fraction == 1.0


class TestCancel:
    def test_cancel_stops_the_work(self):
        job = Job("refine")
        started = threading.Event()

        def work(j):
            started.set()
            for _ in range(1000):
                j.raise_if_cancelled()
                time.sleep(0.01)
            return {}

        job.start(work)
        assert started.wait(5)
        job.cancel.set()
        assert wait_for(job, "cancelled")

    def test_progress_raises_once_cancelled(self):
        job = Job("refine")
        job.cancel.set()
        with pytest.raises(Cancelled):
            job.progress(0.5)

    def test_only_a_running_job_is_cancellable(self):
        job = Job("capture")
        assert job.snapshot()["cancellable"] is False


class TestAlreadyRunning:
    def test_second_start_is_refused(self):
        job = Job("refine")
        release = threading.Event()
        job.start(lambda j: (release.wait(5), {})[1])
        with pytest.raises(AlreadyRunning):
            job.start(lambda j: {})
        release.set()

    def test_the_message_names_the_stage(self):
        """"Something is already running" was the unhelpful version of this."""
        with pytest.raises(AlreadyRunning, match="Dynamic mask detection"):
            raise AlreadyRunning("refine")
        with pytest.raises(AlreadyRunning, match="Splat training"):
            raise AlreadyRunning("train")

    def test_it_carries_the_stage_so_the_ui_can_link_there(self):
        error = AlreadyRunning("reconstruct")
        assert error.stage == "reconstruct"


class TestLog:
    def test_lines_are_kept_and_classified(self):
        job = Job("reconstruct")
        job.log("all good")
        job.log("something failed", "error")
        levels = [line["level"] for line in job.log_tail()]
        assert levels == ["info", "error"]

    def test_multiline_input_is_split(self):
        job = Job("reconstruct")
        job.log("one\ntwo\nthree")
        assert len(job.log_tail()) == 3

    def test_blank_lines_are_dropped(self):
        job = Job("reconstruct")
        job.log("\n\n   \n")
        assert job.log_tail() == []

    def test_log_is_bounded(self):
        job = Job("reconstruct")
        for index in range(5000):
            job.log(f"line {index}")
        assert len(job.log_tail(limit=10_000)) <= 2000

    def test_snapshot_carries_the_log_for_a_returning_view(self):
        """Coming back to a stage must restore what happened while you were away."""
        job = Job("reconstruct")
        job.log("registering image 1")
        job.progress(0.5, "mapping")
        snapshot = job.snapshot()
        assert snapshot["fraction"] == 0.5
        assert snapshot["message"] == "mapping"
        assert snapshot["log"][-1]["text"] == "registering image 1"


class TestRegistry:
    def test_has_a_job_for_every_stage(self):
        registry = JobRegistry()
        assert set(registry.snapshot()) == set(STAGES)

    def test_unknown_stage_is_rejected(self):
        with pytest.raises(KeyError, match="unknown stage"):
            JobRegistry()["reticulate"]

    def test_stages_run_independently(self):
        """One shared job meant detection blocked extraction for no reason."""
        registry = JobRegistry()
        release = threading.Event()
        registry["refine"].start(lambda j: (release.wait(5), {})[1])
        registry["capture"].start(lambda j: {"summary": "extracted"})

        assert wait_for(registry["capture"], "done")
        assert registry["refine"].running
        release.set()

    def test_any_running_reports_which(self):
        registry = JobRegistry()
        release = threading.Event()
        registry["train"].start(lambda j: (release.wait(5), {})[1])
        running = registry.any_running()
        assert running is not None and running.stage == "train"
        release.set()

    def test_cancel_all_stops_everything(self):
        registry = JobRegistry()
        for stage in ("capture", "refine"):
            def spin(j):
                for _ in range(500):
                    j.raise_if_cancelled()
                    time.sleep(0.01)
                return {}

            registry[stage].start(spin)
        registry.cancel_all()
        assert wait_for(registry["capture"], "cancelled")
        assert wait_for(registry["refine"], "cancelled")
