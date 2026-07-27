// Train: drive Brush, and hand the result straight to Inspect.

import {
  InspectorSection, LogViewer, MetricStrip, StageActionBar, el, formatCount, formatClock,
} from "../components.js";
import { icon } from "../icons.js";
export function TrainStage(ctx) {
  const metrics = MetricStrip();

  // The real splat, in the same viewer Inspect uses, loaded from the exports Brush
  // writes as it goes. Not decoration: a piped Brush prints nothing at all, so watching
  // the thing take shape is most of what tells you training is alive.
  const frame = el("iframe", { class: "viewer-frame", src: "about:blank",
                               title: "Splat being trained" });
  const viewerInfo = el("div", { class: "pointcloud__info" }, "No export yet");
  const viewerHost = el("div", { class: "pointcloud__host" }, frame, viewerInfo);

  // A cleanup preview is only worth anything if the points it would delete can be
  // looked at, so the viewer switches between the splat and what would go.
  const VIEWS = [["result", "Result"], ["removed", "Removed points"]];
  const viewButtons = el("div", { class: "segmented" },
    ...VIEWS.map(([value, label]) => el("button", {
      type: "button", "aria-pressed": String(value === "result"),
      onclick: () => setView(value),
    }, label)));
  const viewerBar = el("div", { class: "log__bar" },
    el("span", {}, "Viewer"), viewButtons, el("span", { style: "flex:1" }),
    el("button", {
      class: "btn btn--ghost btn--icon", type: "button", title: "Reload the viewer",
      html: icon("refresh", { size: 13 }), onclick: () => showCurrent(true),
    }));

  const log = LogViewer({ title: "Brush output" });
  log.root.classList.add("log--compact");
  const nextStep = el("div", { style: "padding:0 16px 16px", hidden: true });

  const workspace = el("div", { class: "workspace" },
                      metrics.root, viewerHost, viewerBar, nextStep, log.root);

  const inspector = el("aside", { class: "inspector" });

  const basic = InspectorSection("Training", { id: "train-basic" });
  const totalSteps = el("input", { type: "number", value: 30000, step: 1000, min: 100 });
  const maxResolution = el("input", { type: "number", value: 1920, step: 160, min: 256 });
  basic.body.append(field("steps", totalSteps), field("max res", maxResolution),
    el("p", { class: "hint" },
      "Fewer steps train faster and are fine for checking a capture; 30,000 is Brush's "
      + "default for a finished result."));

  const advanced = InspectorSection("Advanced", { id: "train-adv", open: false });
  const exportEvery = el("input", { type: "number", value: 5000, step: 1000, min: 100 });
  const evalSplit = el("input", { type: "number", value: 0, step: 1, min: 0 });
  const withViewer = el("input", { type: "checkbox" });
  advanced.body.append(
    field("export every", exportEvery),
    field("eval split", evalSplit),
    el("div", { class: "field" }, el("label", {}, "own viewer"), withViewer),
    el("p", { class: "hint" },
      "Eval split 0 keeps every image for training. Brush's own viewer opens a separate "
      + "window alongside this one."));

  // Splat cleanup: removing the gaussians that sit where the rig was. SuperSplat cannot
  // do this -- it needs the reconstruction's camera trajectory -- so it lives here now
  // that Inspect is gone, next to the splat it operates on.
  const cleanup = InspectorSection("Splat cleanup", { id: "train-clean", open: false });
  const splatSelect = el("select", {});
  const radius = el("input", { type: "range", min: 0.05, max: 8, step: 0.05, value: 2.5 });
  const radiusOut = el("output", {}, "2.50");
  const floorEnabled = el("input", { type: "checkbox", checked: true });
  const floor = el("input", { type: "number", value: 1.5, step: 0.1, min: 0 });
  const up = el("select", {},
    el("option", { value: "" }, "auto"),
    el("option", { value: "enu" }, "enu (+Z)"),
    el("option", { value: "y" }, "+Y"),
    el("option", { value: "z" }, "+Z"));
  const cleanNotes = el("p", { class: "hint" });
  radius.addEventListener("input", () => {
    radiusOut.textContent = Number(radius.value).toFixed(2);
  });
  cleanup.body.append(
    field("splat", splatSelect),
    el("div", { class: "field" }, el("label", {}, "radius"),
       el("div", { class: "slider" }, radius, radiusOut)),
    el("div", { class: "field" }, el("label", {}, "floor"), floorEnabled, floor),
    field("up", up),
    el("p", { class: "hint" },
      "A sphere on a roof-mounted rig also reaches the road below it. The floor spares "
      + "anything further down, which costs almost no floater removal."),
    el("div", { class: "field", style: "margin-bottom:0" },
      el("button", { class: "btn", type: "button", onclick: () => clean(false) },
         "Preview"),
      el("button", { class: "btn btn--primary", type: "button", onclick: () => clean(true) },
         "Apply")),
    cleanNotes);

  inspector.append(basic.section, advanced.section, cleanup.section);

  const actionBar = StageActionBar({
    primaryLabel: "Start Training",
    onPrimary: start,
    onCancel: () => ctx.api.jobs.cancel("train").then(ctx.pokeJobs),
  });

  const panel = el("div", { class: "stage-panel" }, workspace, inspector, actionBar.bar);

  function field(label, control) {
    return el("div", { class: "field" }, el("label", {}, label), control);
  }

  async function start() {
    try {
      await ctx.api.post("/api/train/run", {
        total_steps: parseInt(totalSteps.value, 10) || 30000,
        max_resolution: parseInt(maxResolution.value, 10) || 1920,
        export_every: parseInt(exportEvery.value, 10) || 5000,
        eval_split_every: parseInt(evalSplit.value, 10) || 0,
        with_viewer: withViewer.checked,
      });
      log.clear();
      nextStep.hidden = true;
      ctx.pokeJobs();
    } catch (error) { ctx.report(error); }
  }

  function render(job) {
    if (!job) return;
    const steps = parseInt(totalSteps.value, 10) || 30000;
    const current = Math.round((job.fraction || 0) * steps);

    const items = [
      { label: "Step", value: `${formatCount(current)} / ${formatCount(steps)}` },
      { label: "Progress", value: `${Math.round((job.fraction || 0) * 100)}%` },
      { label: "Elapsed", value: formatClock(job.elapsed || 0) },
    ];
    if (job.detail) items.push({ label: "Remaining", value: job.detail.split("·").pop().trim() });
    metrics.render(items);

    log.render(job.log || []);

    // The application just made this file; the user should not have to go and find it.
    if (job.state === "done" && job.result && job.result.splat) {
      nextStep.hidden = false;
      nextStep.replaceChildren(el("button", {
        class: "btn btn--primary", type: "button",
        html: `${icon("arrow-right", { size: 14 })}<span>Open in Inspect</span>`,
        onclick: () => ctx.goTo("inspect"),
      }));
    }
  }

  // ── following the exports ────────────────────────────────────────────
  const local = { view: "result", removed: "", cleaned: "" };
  let shownMtime = 0;
  let lastPoll = 0;

  function setView(view) {
    local.view = view;
    [...viewButtons.children].forEach((button, index) =>
      button.setAttribute("aria-pressed", String(VIEWS[index][0] === view)));
    showCurrent(true);
  }

  function relativeToProject(path) {
    const root = ctx.state.project && ctx.state.project.root;
    if (!root || !path.startsWith(root)) return "";
    return path.slice(root.length).replace(/^[\\/]+/, "").replace(/\\/g, "/");
  }

  /** Point the viewer at whichever file the toggle is asking for. */
  function showCurrent(force) {
    if (local.view === "removed" && local.removed) {
      const relative = relativeToProject(local.removed);
      if (relative) {
        show(`${location.origin}/splat/${relative}`, Date.now(), "removed points");
        return;
      }
    }
    shownMtime = force ? 0 : shownMtime;
    loadLatest();
  }

  function show(url, marker, label) {
    // The reload marker goes on the viewer's own URL, never on the file's: SuperSplat
    // reads the format from the extension, and `.ply?v=123` is not one it knows.
    // `v` comes *before* `load` so even a parser taking everything after `load=` gets a
    // clean path.
    frame.src = `/viewer/?v=${marker}&load=${encodeURIComponent(url)}`;
    if (label) viewerInfo.textContent = label;
  }

  async function loadLatest() {
    try {
      const latest = await ctx.api.get("/api/train/latest");
      populateSplats(latest.splats || []);
      if (!latest.splat) { viewerInfo.textContent = "No export yet"; return; }
      if (latest.mtime === shownMtime) return;    // already showing this one
      shownMtime = latest.mtime;
      const url = `${location.origin}/splat/${latest.splat.split(/[\\/]/).join("/")}`;
      show(url, latest.mtime,
           `${formatCount(latest.step)} steps · ${(latest.bytes / 1e6).toFixed(0)} MB`);
    } catch { /* keep whatever is loaded */ }
  }

  function populateSplats(splats) {
    const wanted = splatSelect.value;
    if (splatSelect.options.length === splats.length
        && [...splatSelect.options].every((o, i) => o.value === splats[i].path)) return;
    splatSelect.replaceChildren(...splats.map((entry) =>
      el("option", { value: entry.path }, entry.name)));
    if (wanted && splats.some((entry) => entry.path === wanted)) splatSelect.value = wanted;
  }

  async function clean(apply) {
    if (!splatSelect.value) {
      ctx.flash("Train a splat before cleaning one.", { level: "warn" }); return;
    }
    try {
      await ctx.api.post("/api/inspect/clean", {
        splat: splatSelect.value,
        radius: parseFloat(radius.value),
        floor: floorEnabled.checked ? parseFloat(floor.value) : null,
        up: up.value,
        apply,
      });
      ctx.pokeJobs();
    } catch (error) { ctx.report(error); }
  }

  function renderCleanup(job) {
    if (!job || !job.result) return;
    const result = job.result;
    if (result.before !== undefined) {
      cleanNotes.innerHTML =
        `${formatCount(result.would_remove)} of ${formatCount(result.before)} inside the `
        + `volume · ${formatCount(result.remaining)} remaining`
        + (result.notes ? `<br>${result.notes.join("<br>")}` : "");
    }
    if (result.removed_file && result.removed_file !== local.removed) {
      local.removed = result.removed_file;
      // Seeing what would go is the point of a preview.
      if (job.state === "done") setView("removed");
    }
    if (result.cleaned && result.cleaned !== local.cleaned) {
      local.cleaned = result.cleaned;
      ctx.flash("Cleaned splat written.", { level: "info" });
    }
  }

  function maybePoll() {
    // An export lands every few thousand steps; asking more often than this only costs
    // a directory listing, but reloading the viewer is not free, so it is gated on the
    // file actually changing.
    const now = Date.now();
    if (now - lastPoll < 5000) return;
    lastPoll = now;
    loadLatest();
  }

  return {
    panel,
    onJobs: (job, allJobs) => {
      actionBar.render(job);
      render(job);
      // Cleanup runs on its own job, so it reports independently of training.
      if (allJobs) renderCleanup(allJobs.inspect);
      if (!job) return;
      if (job.state === "running" || job.state === "done") maybePoll();
    },
    onEnter() {
      loadLatest();
      ctx.api.jobs.status("train").then(render).catch(() => {});
      ctx.api.jobs.status("inspect").then(renderCleanup).catch(() => {});
    },
  };
}
