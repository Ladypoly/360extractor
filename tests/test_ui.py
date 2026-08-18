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

STAGES = ["start", "capture", "reconstruct", "train"]


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
def raw_app(ffmpeg, tmp_path, two_stream_clip):
    """A project on a raw two-lens source: 1:1 on disk, a panorama on screen."""
    from threesixty.source import SourceFormat

    project = Project.create(tmp_path / "raw", sources=[str(two_stream_clip)])
    project.source_format = SourceFormat("dfisheye", 190, layout="streams",
                                         rotate=(90.0, -90.0))
    project.save()
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

    def test_the_pipeline_has_the_stages_in_order(self, page):
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
    def test_with_no_project_it_lands_on_start(self, browser, empty_app):
        page = open_page(browser, empty_app)
        try:
            # Start is the entry point, not whatever stage was last used.
            assert page.locator("#stage-panel-start").is_visible()
            assert page.locator("#stage-tab-start").get_attribute("aria-selected") == "true"
            # Its drop zone / project hub is shown.
            assert page.locator("#stage-panel-start .landing__drop").is_visible()
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
            assert page.locator("#stage-panel-start").is_visible()
            assert page.locator("#stage-panel-reconstruct").is_hidden()
        finally:
            page.close()


class TestAutosaveOwnership:
    def test_autosave_keeps_the_source(self, page):
        """Regression: Capture sent sources:[] on autosave and wiped the project source."""
        page.click("#stage-tab-start")
        page.wait_for_timeout(200)
        # Change a Start control to trigger an autosave.
        page.evaluate("""() => {
            const s = document.querySelector('#start-frame-mode');
            s.value = 'fps'; s.dispatchEvent(new Event('change'));
        }""")
        page.wait_for_timeout(1600)   # autosave debounce + round trip
        project = page.evaluate(
            "async () => (await (await fetch('/api/project')).json()).project")
        assert project and project["sources"], "autosave wiped the source"


class TestSourceProjection:
    """A raw dual-fisheye file has to be declared, because it looks like a panorama."""

    def test_start_offers_the_projection(self, page):
        page.click("#stage-tab-start")
        options = page.eval_on_selector(
            "#start-projection", "s => [...s.options].map(o => o.value)")
        assert options == ["equirect", "dfisheye", "fisheye"]
        assert page.input_value("#start-projection") == "equirect"

    def test_the_lens_field_appears_only_for_a_fisheye_source(self, page):
        page.click("#stage-tab-start")
        assert page.locator("#start-lens-fov").is_hidden()

        page.select_option("#start-projection", "dfisheye")
        assert page.locator("#start-lens-fov").is_visible()

    def test_the_lens_controls_appear_only_for_a_raw_source(self, page):
        page.click("#stage-tab-start")
        for control in ("#start-lens-layout", "#start-lens-rotate", "#start-lens-trim",
                        "#start-view"):
            assert page.locator(control).is_hidden(), control

        page.select_option("#start-projection", "dfisheye")
        for control in ("#start-lens-layout", "#start-lens-rotate", "#start-lens-trim",
                        "#start-view"):
            assert page.locator(control).is_visible(), control

    def test_a_single_fisheye_has_no_layout_to_choose(self, page):
        # One lens cannot be side by side with anything.
        page.click("#stage-tab-start")
        page.select_option("#start-projection", "fisheye")
        assert page.locator("#start-lens-layout").is_hidden()
        assert page.locator("#start-lens-rotate").is_visible()

    def test_the_qoocam_rotation_is_offered_by_name(self, page):
        page.click("#stage-tab-start")
        labels = page.eval_on_selector(
            "#start-lens-rotate", "s => [...s.options].map(o => o.textContent)")
        assert any("QooCam" in label for label in labels)

    def test_trimming_says_what_it_costs(self, page):
        """The trim takes the poles with it, which is worth saying before it happens."""
        page.click("#stage-tab-start")
        page.select_option("#start-projection", "dfisheye")
        page.fill("#start-lens-trim", "8")
        page.dispatch_event("#start-lens-trim", "change")
        hint = page.locator("#stage-panel-start .section", has=page.locator(
            "#start-lens-trim")).inner_text()
        assert "8°" in hint, hint
        # The seam runs through the poles, so trimming takes the sky and the ground
        # straight below with it. Surprising enough to be said before it happens.
        assert "sky" in hint.lower(), hint

    def test_the_choice_is_saved_to_the_project(self, page):
        page.click("#stage-tab-start")
        page.select_option("#start-projection", "dfisheye")
        page.wait_for_timeout(1600)   # autosave debounce + round trip
        project = page.evaluate(
            "async () => (await (await fetch('/api/project')).json()).project")
        assert project["source_format"]["projection"] == "dfisheye"


class TestCaptureOnRawFootage:
    """Capture works in panorama coordinates, whatever shape the file on disk is."""

    def test_the_canvas_is_a_panorama_not_the_file_s_own_shape(self, browser, raw_app):
        """The reported break: a 1:1 raw file squashed the whole 360 into a square.

        The canvas is sized from the *picture* now, not from the source's dimensions --
        every footprint is drawn in equirect coordinates over it, so a square canvas put
        all of them in the wrong place.
        """
        base, _ = raw_app
        page = open_page(browser, base)
        page.click("#stage-tab-capture")
        page.wait_for_timeout(4000)

        size = page.eval_on_selector(
            "#stage-panel-capture canvas",
            "c => ({ width: c.width, height: c.height })")
        assert size["width"] == 2 * size["height"], size
        assert not page.problems, page.problems

    def test_it_shows_the_source_before_anything_is_extracted(self, browser, raw_app):
        """Arriving from Start used to leave this tab with an empty canvas."""
        base, _ = raw_app
        page = open_page(browser, base)
        page.click("#stage-tab-capture")
        page.wait_for_timeout(4000)
        # The empty-state class is what covers the editor with the front door.
        empty = page.eval_on_selector(
            "#stage-panel-capture", "p => p.classList.contains('stage-panel--empty')")
        assert not empty
        assert page.eval_on_selector(
            "#stage-panel-capture canvas",
            "c => c.getContext('2d').getImageData(c.width / 2, c.height / 2, 1, 1)"
            ".data[3] > 0"), "nothing was drawn on the canvas"
        assert not page.problems, page.problems


class TestTheViewFollowsTheFootage:
    """One choice of picture for the whole app.

    Picking the lens view in Start and then being handed a panorama in Capture is the
    same capture described two different ways, and the rig placed on one does not read
    against the other.
    """

    def test_capture_offers_the_view_for_a_raw_source(self, browser, raw_app):
        base, _ = raw_app
        page = open_page(browser, base)
        page.click("#stage-tab-capture")
        page.wait_for_timeout(1500)
        assert page.locator("#cap-view").is_visible()
        assert not page.problems, page.problems

    def test_a_stitched_source_has_no_view_to_choose(self, page):
        page.click("#stage-tab-capture")
        page.wait_for_timeout(500)
        assert page.locator("#cap-view").is_hidden()

    def test_choosing_the_lenses_in_start_carries_into_capture(self, browser, raw_app):
        base, _ = raw_app
        page = open_page(browser, base)
        page.click("#stage-tab-start")
        page.select_option("#start-view", "lenses")
        page.wait_for_timeout(600)
        page.click("#stage-tab-capture")
        page.wait_for_timeout(2500)
        assert page.input_value("#cap-view") == "lenses"

    def test_switching_back_in_capture_moves_start_too(self, browser, raw_app):
        base, _ = raw_app
        page = open_page(browser, base)
        page.click("#stage-tab-capture")
        page.wait_for_timeout(1500)
        page.select_option("#cap-view", "panorama")
        page.wait_for_timeout(1500)
        page.click("#stage-tab-start")
        assert page.input_value("#start-view") == "panorama"


class TestAimingOnTheLenses:
    """Dragging a camera on the lens view has to mean what it looks like it means."""

    def test_dragging_aims_the_camera_where_it_was_dropped(self, browser, raw_app):
        """Drop a camera half way out along the front lens and it should be aimed there.

        Half the radius of a 190-degree lens is 47.5 degrees off its axis, and the axis
        is dead ahead -- so this checks the inverse projection through the real UI,
        not just the arithmetic in isolation.
        """
        base, _ = raw_app
        page = open_page(browser, base)
        page.click("#stage-tab-capture")
        page.wait_for_timeout(2500)
        page.select_option("#cap-view", "lenses")
        page.wait_for_timeout(2500)

        canvas = page.locator("#stage-panel-capture canvas")
        box = canvas.bounding_box()
        size = page.eval_on_selector("#stage-panel-capture canvas",
                                     "c => ({ w: c.width, h: c.height })")

        def client(x, y):
            return (box["x"] + x / size["w"] * box["width"],
                    box["y"] + y / size["h"] * box["height"])

        # The centre of the right-hand circle is the front lens's axis: bearing zero.
        start = client(3 * size["w"] / 4, size["h"] / 2)
        end = client(3 * size["w"] / 4 - size["w"] / 8, size["h"] / 2)
        page.mouse.move(*start)
        page.mouse.down()
        page.mouse.move(*end, steps=8)
        page.mouse.up()
        page.wait_for_timeout(600)

        aimed = page.evaluate("""() => {
            const rows = [...document.querySelectorAll('#stage-panel-capture .inspector input')]
              .filter((i) => i.type === 'text' && i.value === 'c00');
            return rows.length ? rows[0].closest('div').textContent : null;
        }""")
        assert aimed is not None, "the first camera vanished from the list"
        degrees = float(aimed.split("°")[0].split()[-1])
        assert -60 < degrees < -35, aimed
        assert not page.problems, page.problems


class TestStageOwnership:
    def test_extract_belongs_only_to_capture(self, page):
        """The reported complaint: an Extract button on the wrong stage."""
        capture = page.locator("#stage-panel-capture .actionbar")
        assert "Extract frames" in capture.inner_text()

        for key in ["reconstruct", "train"]:
            text = page.locator(f"#stage-panel-{key} .actionbar").inner_text()
            assert "Extract" not in text, f"{key} offers an Extract action"

    @pytest.mark.parametrize("key,label", [
        ("capture", "Extract frames"),
        ("reconstruct", "Run All"), ("train", "Start Training"),
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
        for key in ["train"]:
            assert page.locator(f"#stage-tab-{key}").is_disabled(), \
                f"{key} should not be available yet"

    def test_a_disabled_stage_explains_why(self, page):
        """Hiding the reason is what makes a disabled control infuriating."""
        title = page.locator("#stage-tab-train").get_attribute("title")
        assert "reconstruction" in title.lower()

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
        # Start a frame extraction -- a Start-tab action, and so a Start-tab job.
        page.evaluate("""async () => {
            await fetch('/api/frames/extract', {method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: 'fps', value: 1})});
        }""")

        # Move away immediately; the pipeline must still report it. Reconstruct is
        # legitimately disabled here (no camera images yet), so navigate via the handler.
        page.evaluate(
            "{ const b = document.querySelector('#stage-tab-reconstruct');"
            "  b.disabled = false; b.click(); }")
        page.wait_for_function(
            """() => {
                 const tab = document.querySelector('#stage-tab-start');
                 return tab && (tab.className.includes('running')
                             || tab.className.includes('done'));
               }""",
            timeout=45000)
        # ...and Capture, which has not run, must not be claiming it did.
        assert "running" not in page.locator("#stage-tab-capture").get_attribute("class")
        assert "done" not in page.locator("#stage-tab-capture").get_attribute("class")
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


class TestCleanupMovedIntoTrain:
    """Inspect is gone -- SuperSplat covers the viewing -- but the splat cleanup was
    never SuperSplat's to do, and had to come along rather than disappear."""

    def test_there_is_no_inspect_tab(self, page):
        assert page.locator("#stage-tab-inspect").count() == 0
        assert page.locator("#stage-panel-inspect").count() == 0

    # Read the markup rather than clicking: a stage panel that is not the active one is
    # hidden, so nothing inside it can be interacted with from here.
    def test_train_offers_the_cleanup(self, page):
        markup = page.locator("#stage-panel-train .inspector").inner_html()
        assert "Splat cleanup" in markup

    def test_the_cleanup_controls_came_with_it(self, page):
        markup = page.locator("#stage-panel-train .inspector").inner_html()
        for label in ("radius", "floor", "Preview", "Apply"):
            assert label in markup, f"cleanup lost its {label} control"
