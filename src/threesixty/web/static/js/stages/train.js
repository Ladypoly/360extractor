// Train: drive Brush, and hand the result straight to Inspect.

import {
  InspectorSection, LogViewer, MetricStrip, StageActionBar, el, formatCount, formatClock,
} from "../components.js";
import { icon } from "../icons.js";

export function TrainStage(ctx) {
  const metrics = MetricStrip();
  const log = LogViewer({ title: "Brush output" });
  const nextStep = el("div", { style: "padding:0 16px 16px", hidden: true });

  const workspace = el("div", { class: "workspace" }, metrics.root, nextStep, log.root);

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

  inspector.append(basic.section, advanced.section);

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

  return {
    panel,
    onJobs: (job) => { actionBar.render(job); render(job); },
    onEnter() {
      ctx.api.jobs.status("train").then(render).catch(() => {});
    },
  };
}
