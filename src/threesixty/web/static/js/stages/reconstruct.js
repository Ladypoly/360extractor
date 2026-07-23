// Reconstruct: run COLMAP, rather than describe how to run it.

import {
  InspectorSection, LogViewer, MetricStrip, StageActionBar, el, formatCount,
} from "../components.js";
import { icon } from "../icons.js";
import { PointCloud } from "../pointcloud.js";

const STEPS = [
  ["features", "Feature extraction"],
  ["rig", "Rig configuration"],
  ["match", "Feature matching"],
  ["map", "Mapping"],
  ["geo", "Geo alignment"],
];

export function ReconstructStage(ctx) {
  const rows = new Map();

  const steps = el("div", { class: "steps" });
  for (const [key, label] of STEPS) {
    const iconSpan = el("span", { class: "step__icon", html: icon("pending", { size: 14 }) });
    const time = el("span", { class: "step__time" });
    const rerun = el("button", {
      class: "btn btn--ghost btn--icon", type: "button", title: `Run ${label} again`,
      html: icon("refresh", { size: 13 }),
      onclick: () => run({ only: key }),
    });
    const row = el("div", { class: "step" }, iconSpan,
                   el("span", { class: "step__label" }, label), time, rerun);
    if (key === "geo") row.style.display = "none";
    rows.set(key, { row, iconSpan, time });
    steps.append(row);
  }

  const metrics = MetricStrip();
  metrics.root.style.display = "none";

  // The sparse cloud, shown as COLMAP builds it (snapshots), with the log demoted to a
  // strip beneath rather than filling the page.
  const canvas = el("canvas", { class: "pointcloud" });
  const cloud = PointCloud(canvas);
  const cloudInfo = el("div", { class: "pointcloud__info" }, "No point cloud yet");
  const cloudHost = el("div", { class: "pointcloud__host" }, canvas, cloudInfo);

  const log = LogViewer({ title: "COLMAP output" });
  log.root.classList.add("log--compact");
  const workspace = el("div", { class: "workspace" }, metrics.root, cloudHost, log.root);
  if (cloud.ok) cloud.start();

  const inspector = el("aside", { class: "inspector" });
  const pipeline = InspectorSection("Pipeline", { id: "rec-steps" });
  pipeline.body.append(steps);

  const options = InspectorSection("Options", { id: "rec-options" });
  const matcher = el("select", {},
    el("option", { value: "sequential" }, "sequential (video)"),
    el("option", { value: "exhaustive" }, "exhaustive (slower, unordered)"));
  const geo = el("input", { type: "checkbox" });
  options.body.append(
    field("matching", matcher),
    el("div", { class: "field" }, el("label", {}, "geo-align"), geo),
    el("p", { class: "hint" },
      "Geo alignment needs a GPX track. It gives the model a real scale, which is what "
      + "makes a cleanup radius mean metres."),
    el("div", { class: "field", style: "margin-top:12px;margin-bottom:0" },
      el("button", { class: "btn btn--ghost", type: "button", onclick: writeScript },
         "Generate script")));

  inspector.append(pipeline.section, options.section);

  const actionBar = StageActionBar({
    primaryLabel: "Run All",
    onPrimary: () => run({}),
    onCancel: () => ctx.api.jobs.cancel("reconstruct").then(ctx.pokeJobs),
  });

  const panel = el("div", { class: "stage-panel" }, workspace, inspector, actionBar.bar);

  function field(label, control) {
    return el("div", { class: "field" }, el("label", {}, label), control);
  }

  geo.addEventListener("change", () => {
    rows.get("geo").row.style.display = geo.checked ? "" : "none";
  });

  async function run(extra) {
    try {
      await ctx.api.post("/api/reconstruct/run", {
        matcher: matcher.value, geo: geo.checked, ...extra,
      });
      log.clear();
      ctx.pokeJobs();
    } catch (error) { ctx.report(error); }
  }

  async function writeScript() {
    try {
      const data = await ctx.api.post("/api/export/colmap", {});
      ctx.flash(`Wrote ${data.written.join(", ")}`, { level: "info" });
    } catch (error) { ctx.report(error); }
  }

  function renderSteps(result = {}) {
    const states = result.steps || {};
    for (const [key] of STEPS) {
      const entry = rows.get(key);
      const state = states[key] ? states[key].state : "pending";
      entry.row.className = `step step--${state}`;
      entry.iconSpan.innerHTML = icon(
        { running: "running", done: "done", error: "error" }[state] || "pending",
        { size: 14 });
      entry.time.textContent = states[key] && states[key].seconds
        ? `${states[key].seconds}s` : "";
    }
  }

  function renderMetrics(result = {}) {
    const values = result.metrics;
    if (!values || !Object.keys(values).length) {
      metrics.root.style.display = "none";
      return;
    }
    const items = [
      { label: "Registered images", value: formatCount(values.registered_images || 0) },
      { label: "Frames", value: formatCount(values.frames || 0) },
      { label: "Cameras", value: formatCount(values.cameras || 0) },
    ];
    if (values.rig_spread !== undefined) {
      items.push({ label: "Rig spread", value: values.rig_spread.toFixed(6) });
    }
    if (values.path_length !== undefined) {
      items.push({ label: "Path length", value: values.path_length.toFixed(2) });
    }
    metrics.render(items);
    metrics.root.style.display = "";
  }

  // ── point cloud polling ──────────────────────────────────────────────
  let lastMtime = 0;
  let lastPoll = 0;
  async function loadPoints() {
    if (!cloud.ok) return;
    try {
      const response = await fetch(`/api/reconstruct/points?since=${lastMtime}`);
      if (response.status !== 200) return;   // 204: unchanged
      const buffer = await response.arrayBuffer();
      lastMtime = new DataView(buffer).getFloat64(0, true);
      cloud.load(buffer);
      cloudInfo.textContent = cloud.count
        ? `${formatCount(cloud.count)} points` : "No point cloud yet";
    } catch { /* keep what is drawn */ }
  }
  function maybePoll(running) {
    const now = Date.now();
    if (running && now - lastPoll < 1500) return;
    lastPoll = now;
    loadPoints();
  }

  return {
    panel,
    onJobs(job) {
      actionBar.render(job);
      if (!job) return;
      log.render(job.log || []);
      renderSteps(job.result);
      if (job.state === "done") renderMetrics(job.result);
      // Follow the cloud while mapping runs, and pick up the final model when it lands.
      if (job.state === "running") maybePoll(true);
      else if (job.state === "done") loadPoints();
    },
    onEnter() {
      loadPoints();
      ctx.api.jobs.status("reconstruct").then((job) => {
        log.render(job.log || []);
        renderSteps(job.result);
        if (job.state === "done") renderMetrics(job.result);
      }).catch(() => {});
    },
  };
}
