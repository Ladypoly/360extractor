"""Browser tests for the application shell.

These exist because the structural problems the redesign fixed are invisible to unit
tests: an Extract button on the wrong stage, a stage that cannot say why it is
disabled, a job that vanishes when you navigate away. All of those are only observable
in a rendered page.

Skipped when Playwright or its browser is unavailable, rather than silently passing.
"""

import socket
import threading
from http.server import ThreadingHTTPServer

import pytest

from threesixty.project import Project
from threesixty.web.server import Handler, Session

pytestmark = [pytest.mark.ffmpeg, pytest.mark.ui]

sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="playwright is not installed").sync_playwright

STAGES = ["capture", "refine", "reconstruct", "train", "inspect"]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as play:
        try:
            instance = play.chromium.launch()
        except Exception as exc:                       # noqa: BLE001
            pytest.skip(f"no chromium available: {exc}")
        yield instance
        instance.close()


@pytest.fixture
def app(ffmpeg, tmp_path, equirect_clip):
    """A server with an empty project: nothing extracted, so gating is visible."""
    project = Project.create(tmp_path / "job", sources=[str(equirect_clip)])
    port = free_port()
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        type("Bound", (Handler,), {"session": Session(ffmpeg, project)}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", project
    server.shutdown()
    server.server_close()


@pytest.fixture
def app_ready(ffmpeg, tmp_path, equirect_clip):
    """A project that already has extracted images, so later stages are unlocked.

    The images are placeholders: readiness is a question about what exists on disk, and
    running a real extraction here would cost minutes for nothing.
    """
    project = Project.create(tmp_path / "ready", sources=[str(equirect_clip)])
    for camera in ("c00", "c01"):
        folder = project.root / "images" / "clip" / camera
        folder.mkdir(parents=True)
        for frame in range(1, 4):
            (folder / f"{frame:05d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    port = free_port()
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        type("Bound", (Handler,), {"session": Session(ffmpeg, project)}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", project
    server.shutdown()
    server.server_close()


@pytest.fixture
def empty_app(ffmpeg):
    """A server with no project open: the front-door / landing case."""
    port = free_port()
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        type("Bound", (Handler,), {"session": Session(ffmpeg, None)}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def open_page(browser, url):
    page = browser.new_page(viewport={"width": 1600, "height": 900})
    problems = []
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.on("console", lambda message:
            problems.append(message.text) if message.type == "error" else None)
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector(".pipeline .stage")
    page.wait_for_timeout(400)
    page.problems = problems
    return page


@pytest.fixture
def ready_page(browser, app_ready):
    """A page whose project has images, so Refine and Reconstruct are reachable."""
    page = open_page(browser, app_ready[0])
    yield page
    page.close()


@pytest.fixture
def page(browser, app):
    url, project = app
    page = browser.new_page(viewport={"width": 1600, "height": 900})
    problems = []
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.on("console", lambda message:
            problems.append(message.text) if message.type == "error" else None)
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector(".pipeline .stage")
    page.wait_for_timeout(400)
    page.problems = problems
    page.project = project
    yield page
    page.close()


class TestShell:
    def test_it_loads_without_console_errors(self, page):
        assert page.title() == "360extract"
        assert page.problems == []

    def test_the_pipeline_has_five_stages_in_order(self, page):
        labels = [page.locator(f"#stage-tab-{key} .stage__label").inner_text()
                  for key in STAGES]
        for label, key in zip(labels, STAGES):
            assert key.split("_")[0][:4].lower() in label.lower()

    def test_the_top_bar_shows_the_project(self, page):
        assert "job" in page.locator(".brand__project").inner_text()

    def test_system_status_lists_the_tools(self, page):
        page.click("text=System")
        page.wait_for_selector("#system-dialog .tool-row")
        names = page.locator("#system-dialog .tool-row__name").all_inner_texts()
        assert {"FFmpeg", "COLMAP", "Brush", "SuperSplat"} <= set(names)


class TestLanding:
    def test_with_no_project_it_lands_on_capture_showing_the_front_door(
            self, browser, empty_app):
        page = open_page(browser, empty_app)
        try:
            # Capture is the visible panel, not whatever stage was last used.
            assert page.locator("#stage-panel-capture").is_visible()
            assert page.locator("#stage-tab-capture").get_attribute("aria-selected") == "true"
            # The landing (drop zone + Browse + Open project) is shown; the editor is not.
            assert page.locator(".stage-panel--empty .landing__drop").is_visible()
            assert page.locator("#stage-panel-capture .inspector").is_hidden()
            assert page.problems == []
        finally:
            page.close()

    def test_a_stale_last_stage_does_not_strand_the_user(self, browser, empty_app):
        """Reported bug: reopening landed on Reconstruct with no way back in."""
        page = open_page(browser, empty_app)
        try:
            page.evaluate("localStorage.setItem('stage', 'reconstruct')")
            page.reload(wait_until="networkidle")
            page.wait_for_selector(".pipeline .stage")
            page.wait_for_timeout(300)
            assert page.locator("#stage-panel-capture").is_visible()
            assert page.locator("#stage-panel-reconstruct").is_hidden()
        finally:
            page.close()


class TestStageOwnership:
    def test_extract_belongs_only_to_capture(self, page):
        """The reported complaint: an Extract button while working in Refine."""
        capture = page.locator("#stage-panel-capture .actionbar")
        assert "Extract frames" in capture.inner_text()

        for key in ["refine", "reconstruct", "train", "inspect"]:
            text = page.locator(f"#stage-panel-{key} .actionbar").inner_text()
            assert "Extract" not in text, f"{key} offers an Extract action"

    @pytest.mark.parametrize("key,label", [
        ("capture", "Extract frames"), ("refine", "Run Detection"),
        ("reconstruct", "Run All"), ("train", "Start Training"),
        ("inspect", "Apply Cleanup"),
    ])
    def test_each_stage_has_its_own_primary_action(self, page, key, label):
        bar = page.locator(f"#stage-panel-{key} .actionbar__actions")
        assert label in bar.inner_text()

    def test_there_is_no_global_footer(self, page):
        """Every action bar belongs to a panel, so exactly one is ever visible."""
        visible = page.locator(".stage-panel:not([hidden]) .actionbar")
        assert visible.count() == 1


class TestGating:
    def test_later_stages_are_disabled_before_extraction(self, page):
        for key in ["train", "inspect"]:
            assert page.locator(f"#stage-tab-{key}").is_disabled(), \
                f"{key} should not be available yet"

    def test_a_disabled_stage_explains_why(self, page):
        """Hiding the reason is what makes a disabled control infuriating."""
        title = page.locator("#stage-tab-train").get_attribute("title")
        assert "reconstruction" in title.lower()

        title = page.locator("#stage-tab-inspect").get_attribute("title")
        assert ".ply" in title.lower() or "trained" in title.lower()

    def test_capture_is_available(self, page):
        assert not page.locator("#stage-tab-capture").is_disabled()


class TestNavigation:
    def test_selecting_a_stage_shows_only_that_panel(self, ready_page):
        ready_page.click("#stage-tab-reconstruct")
        ready_page.wait_for_timeout(200)
        assert ready_page.locator("#stage-panel-reconstruct").is_visible()
        assert ready_page.locator("#stage-panel-capture").is_hidden()
        assert ready_page.locator("#stage-tab-reconstruct")            .get_attribute("aria-selected") == "true"

    def test_the_stage_survives_a_reload(self, ready_page):
        ready_page.click("#stage-tab-reconstruct")
        ready_page.wait_for_timeout(300)
        ready_page.reload(wait_until="networkidle")
        ready_page.wait_for_selector(".pipeline .stage")
        ready_page.wait_for_timeout(500)
        assert ready_page.locator("#stage-panel-reconstruct").is_visible()

    def test_the_pipeline_is_a_tablist(self, page):
        assert page.locator(".pipeline").get_attribute("role") == "tablist"
        assert page.locator("#stage-tab-capture").get_attribute("role") == "tab"
        panel = page.locator("#stage-panel-capture")
        assert panel.get_attribute("role") == "tabpanel"
        assert panel.get_attribute("aria-labelledby") == "stage-tab-capture"


class TestReconstructWorkspace:
    def test_the_colmap_steps_are_listed_with_state(self, ready_page):
        ready_page.click("#stage-tab-reconstruct")
        ready_page.wait_for_timeout(200)
        steps = ready_page.locator("#stage-panel-reconstruct .step__label").all_inner_texts()
        assert "Feature extraction" in steps
        assert "Rig configuration" in steps
        assert "Mapping" in steps

    def test_generate_script_is_a_secondary_action(self, ready_page):
        """It used to be the whole stage; now it is an escape hatch."""
        ready_page.click("#stage-tab-reconstruct")
        ready_page.wait_for_timeout(200)
        primary = ready_page.locator("#stage-panel-reconstruct .actionbar .btn--primary")
        assert "Run All" in primary.inner_text()
        # It lives in the inspector, not the action bar.
        assert ready_page.locator("#stage-panel-reconstruct .inspector")            .inner_text().count("Generate script") == 1


class TestJobsAcrossStages:
    def test_a_running_job_shows_in_the_pipeline_from_another_stage(self, page):
        """Leaving a stage must not hide, or stop, its work."""
        page.click("#stage-tab-capture")
        page.wait_for_timeout(300)
        # The primary opens the extract-frames dialog; confirm it to start the job.
        page.click("#stage-panel-capture .actionbar .btn--primary")
        page.wait_for_selector("#frames-dialog[open]")
        page.click("#frames-dialog .btn--primary")

        # Move away immediately; the pipeline must still report it. Reconstruct is
        # legitimately disabled here (frame extraction hasn't produced camera images
        # yet, that is the second capture step), so navigate through the app's handler.
        page.evaluate(
            "{ const b = document.querySelector('#stage-tab-reconstruct');"
            "  b.disabled = false; b.click(); }")
        page.wait_for_function(
            """() => {
                 const tab = document.querySelector('#stage-tab-capture');
                 return tab && (tab.className.includes('running')
                             || tab.className.includes('done'));
               }""",
            timeout=45000)
        assert page.locator("#stage-panel-reconstruct").is_visible()


@pytest.mark.parametrize("key", STAGES)
def test_screenshot(page, key, tmp_path_factory):
    """Screenshots for review, at a standard desktop size."""
    # Navigate straight through the app's own handler rather than clicking the tab: a
    # disabled (not-yet-ready) stage cannot be clicked, and the job poll re-disables it
    # between enabling and clicking. This is deterministic regardless of poll timing.
    page.evaluate(
        f"{{ const b = document.querySelector('#stage-tab-{key}');"
        f"   b.disabled = false; b.click(); }}")
    page.wait_for_timeout(500)
    output = tmp_path_factory.mktemp("shots") / f"{key}.png"
    page.screenshot(path=str(output))
    assert output.exists() and output.stat().st_size > 5000
