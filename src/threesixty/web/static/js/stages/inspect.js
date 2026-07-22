// Inspect: the final destination — look at the splat, and clean the rig out of it.

import {
  EmptyState, InspectorSection, MetricStrip, StageActionBar, el, formatCount,
} from "../components.js";
import { icon } from "../icons.js";

const VIEWS = [["result", "Result"], ["removed", "Removed points"]];

export function InspectStage(ctx) {
  const local = { splat: "", removed: "", view: "result", ready: false };

  const frame = el("iframe", { class: "viewer-frame", src: "about:blank",
                               title: "Splat viewer" });
  const viewerHost = el("div", { style: "flex:1;min-height:0;display:flex" }, frame);
  const metrics = MetricStrip();
  metrics.root.style.display = "none";

  const viewButtons = el("div", { class: "segmented" },
    ...VIEWS.map(([value, label]) => el("button", {
      type: "button", "aria-pressed": String(value === "result"),
      onclick: () => setView(value),
    }, label)));

  const toolbar = el("div", { class: "log__bar" },
    el("span", {}, "Viewer"), viewButtons,
    el("span", { style: "flex:1" }),
    el("button", {
      class: "btn btn--ghost btn--icon", type: "button", title: "Reload the viewer",
      html: icon("refresh", { size: 13 }), onclick: () => loadViewer(local.view),
    }));

  const workspace = el("div", { class: "workspace" }, metrics.root, viewerHost, toolbar);

  const inspector = el("aside", { class: "inspector" });

  const cleanup = InspectorSection("Splat cleanup", { id: "insp-clean" });
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
    el("div", { class: "field", style: "margin-top:12px;margin-bottom:0" },
      el("button", { class: "btn", type: "button", onclick: () => clean(false) },
         "Preview cleanup")));

  const notes = el("p", { class: "hint" });
  cleanup.body.append(notes);

  inspector.append(cleanup.section);

  const actionBar = StageActionBar({
    primaryLabel: "Apply Cleanup",
    onPrimary: () => clean(true),
    onCancel: () => ctx.api.jobs.cancel("inspect").then(ctx.pokeJobs),
  });

  const panel = el("div", { class: "stage-panel" }, workspace, inspector, actionBar.bar);

  function field(label, control) {
    return el("div", { class: "field" }, el("label", {}, label), control);
  }

  function setView(view) {
    local.view = view;
    [...viewButtons.children].forEach((button, index) =>
      button.setAttribute("aria-pressed", String(VIEWS[index][0] === view)));
    loadViewer(view);
  }

  /** Point the embedded SuperSplat at a file this application serves. */
  function loadViewer(view) {
    const path = view === "removed" ? local.removed : local.splat;
    if (!path) { frame.src = "/viewer/"; return; }
    const relative = relativeToProject(path);
    frame.src = relative
      ? `/viewer/?load=${encodeURIComponent(`${location.origin}/splat/${relative}`)}`
      : "/viewer/";
  }

  function relativeToProject(path) {
    const root = ctx.state.project && ctx.state.project.root;
    if (!root || !path.startsWith(root)) return "";
    return path.slice(root.length).replace(/^[\\/]+/, "").replace(/\\/g, "/");
  }

  async function clean(apply) {
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

  function render(job) {
    if (!job || !job.result) return;
    const result = job.result;
    if (result.before !== undefined) {
      metrics.render([
        { label: "Gaussians before", value: formatCount(result.before) },
        { label: "Would remove", value: formatCount(result.would_remove) },
        { label: "Remaining", value: formatCount(result.remaining) },
      ]);
      metrics.root.style.display = "";
    }
    if (result.notes) notes.innerHTML = result.notes.join("<br>");
    if (result.removed_file) {
      local.removed = result.removed_file;
      // Seeing what would go is the point of a preview.
      if (job.state === "done") setView("removed");
    }
    if (result.cleaned) {
      local.splat = result.cleaned;
      ctx.flash("Cleaned splat written.", { level: "info" });
    }
  }

  async function discover() {
    if (!ctx.state.project) return;
    try {
      const { stages } = await ctx.api.jobs.all(0);
      local.ready = stages.inspect && stages.inspect.ready;
    } catch { /* readiness is advisory here */ }

    try {
      const job = await ctx.api.jobs.status("train", 0);
      const splats = (job.result && job.result.splats) || [];
      if (splats.length) {
        splatSelect.replaceChildren(...splats.map((path) =>
          el("option", { value: path }, path.split(/[\\/]/).pop())));
        local.splat = splats[0];
        loadViewer("result");
        return;
      }
    } catch { /* fall through to the empty state */ }

    if (!local.splat) {
      viewerHost.replaceChildren(EmptyState({
        iconName: "inspect", title: "No trained splat yet",
        body: "Train a splat and it will open here automatically.",
        actionLabel: "Go to Train", onAction: () => ctx.goTo("train"),
      }));
    }
  }

  return {
    panel,
    onJobs: (job) => { actionBar.render(job); render(job); },
    onEnter() {
      if (!viewerHost.contains(frame)) viewerHost.replaceChildren(frame);
      discover();
      ctx.api.jobs.status("inspect").then(render).catch(() => {});
    },
  };
}
