// Application shell: top bar, pipeline, stage panels, and the job poller they share.
//
// One poller for every stage, because a job must keep running -- and keep showing as
// running in the pipeline -- while the user is looking at a different stage.

import * as api from "./api.js";
import { EmptyState, el, StatusBadge } from "./components.js";
import { icon } from "./icons.js";
import { Pipeline, STAGES } from "./pipeline.js";

import { CaptureStage } from "./stages/capture.js";
import { RefineStage } from "./stages/refine.js";
import { ReconstructStage } from "./stages/reconstruct.js";
import { TrainStage } from "./stages/train.js";
import { InspectStage } from "./stages/inspect.js";

const POLL_IDLE = 1500;
const POLL_ACTIVE = 400;

export const state = {
  project: null,
  media: null,
  jobs: {},
  readiness: {},
  active: localStorage.getItem("stage") || "capture",
};

const stages = {};
let pipeline;
let pollTimer = null;

// ── shell ──────────────────────────────────────────────────────────────

function buildTopBar() {
  const projectName = el("span", { class: "brand__project" }, "no project");
  const sourceName = el("span", { class: "source-meta__name" });
  const sourceInfo = el("span", {});

  const button = (label, iconName, onclick, title) => el("button", {
    class: "btn btn--ghost", type: "button", onclick, title: title || label,
    html: `${icon(iconName, { size: 14 })}<span>${label}</span>`,
  });

  const bar = el("header", { class: "topbar" },
    el("div", { class: "brand" },
      el("span", { class: "brand__name" }, "360extract"),
      projectName),
    el("div", { class: "source-meta" }, sourceName, sourceInfo),
    el("div", { class: "topbar__spacer" }),
    el("div", { class: "topbar__actions" },
      button("Open", "folder", openProject, "Open a project folder"),
      button("Save", "save", saveProject, "Save the project"),
      button("System", "system", showSystem, "Detected tools")));

  return {
    bar,
    setProject: (project) => {
      projectName.textContent = project ? project.name : "no project";
    },
    setSource: (media) => {
      if (!media) { sourceName.textContent = ""; sourceInfo.textContent = ""; return; }
      sourceName.textContent = media.path.split(/[\\/]/).pop();
      const bits = [`${media.width}×${media.height}`];
      if (media.is_video) {
        bits.push(`${media.duration.toFixed(1)}s`, `${media.fps} fps`);
        if (media.frame_count) bits.push(`${media.frame_count} frames`);
      } else {
        bits.push("still");
      }
      if (!media.looks_equirectangular) bits.push("not 2:1 — geometry will be wrong");
      sourceInfo.textContent = bits.join("  ·  ");
    },
  };
}

let topbar;

// ── project ────────────────────────────────────────────────────────────

async function openProject() {
  try {
    const paths = await api.pick("directory", "Open a project folder");
    if (!paths.length) return;
    const { project } = await api.post("/api/project/open", { path: paths[0] });
    applyProject(project);
  } catch (error) { report(error); }
}

async function saveProject() {
  try {
    const payload = stages.capture ? stages.capture.projectPayload() : {};
    if (!state.project) {
      const paths = await api.pick("directory", "Choose a folder for the project");
      if (!paths.length) return;
      payload.root = paths[0];
    }
    const { project } = await api.post("/api/project/save", payload);
    applyProject(project, { keepMedia: true });
    flash("Project saved");
  } catch (error) { report(error); }
}

export function applyProject(project, { keepMedia = false } = {}) {
  state.project = project;
  topbar.setProject(project);
  if (stages.capture) stages.capture.applyProject(project, { keepMedia });
  refreshJobs();
}

// ── system status ──────────────────────────────────────────────────────

async function showSystem() {
  const dialog = document.getElementById("system-dialog");
  const body = dialog.querySelector(".dialog__body");
  body.replaceChildren(el("div", { class: "hint" }, "Checking…"));
  dialog.showModal();

  try {
    const { tools } = await api.get("/api/system");
    body.replaceChildren(...tools.map((tool) => el("div", { class: "tool-row" },
      el("span", { class: "tool-row__name" }, tool.name),
      StatusBadge(tool.found ? "done" : "error", tool.found ? "detected" : "missing"),
      el("span", { class: "tool-row__path" },
        tool.found ? `${tool.version || ""} ${tool.path}`.trim() : tool.detail))));
  } catch (error) {
    body.replaceChildren(el("div", { class: "hint hint--error" }, error.message));
  }
}

// ── job polling ────────────────────────────────────────────────────────

async function refreshJobs() {
  try {
    const { jobs, stages: readiness } = await api.jobs.all(0);
    state.jobs = jobs;
    state.readiness = readiness;
    pipeline.render({ jobs, readiness, project: state.project, active: state.active });

    const current = stages[state.active];
    if (current && current.onJobs) current.onJobs(jobs[state.active], jobs);
  } catch (error) {
    // A poll failure is not worth shouting about; the next one usually succeeds.
  } finally {
    const busy = Object.values(state.jobs).some((job) => job.state === "running");
    clearTimeout(pollTimer);
    pollTimer = setTimeout(refreshJobs, busy ? POLL_ACTIVE : POLL_IDLE);
  }
}

export function pokeJobs() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(refreshJobs, 60);
}

// ── navigation ─────────────────────────────────────────────────────────

export function goTo(key) {
  state.active = key;
  localStorage.setItem("stage", key);
  for (const stage of STAGES) {
    const panel = document.getElementById(`stage-panel-${stage.key}`);
    if (panel) panel.hidden = stage.key !== key;
  }
  pipeline.render({ jobs: state.jobs, readiness: state.readiness,
                    project: state.project, active: key });
  const entered = stages[key];
  if (entered && entered.onEnter) entered.onEnter();
}

// ── errors ─────────────────────────────────────────────────────────────

export function report(error) {
  if (error && error.isAlreadyRunning && error.runningStage) {
    // Say what is running and offer to go there, rather than "already running".
    flash(error.message, {
      actionLabel: "Show it",
      onAction: () => goTo(error.runningStage),
    });
    return;
  }
  flash(error && error.message ? error.message : String(error), { level: "error" });
}

let flashTimer = null;
export function flash(message, { level = "info", actionLabel, onAction } = {}) {
  const host = document.getElementById("flash");
  host.className = `flash flash--${level}`;
  host.replaceChildren(el("span", {}, message));
  if (actionLabel && onAction) {
    host.append(el("button", {
      class: "btn btn--ghost", type: "button",
      onclick: () => { onAction(); host.hidden = true; },
    }, actionLabel));
  }
  host.hidden = false;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { host.hidden = true; }, level === "error" ? 9000 : 4000);
}

// ── boot ───────────────────────────────────────────────────────────────

async function boot() {
  topbar = buildTopBar();
  document.getElementById("app").prepend(topbar.bar);

  pipeline = Pipeline({ onSelect: goTo });
  topbar.bar.after(pipeline.nav);

  const host = document.getElementById("panels");
  const context = {
    api, state,
    setSource: (media) => { state.media = media; topbar.setSource(media); },
    goTo, report, flash, pokeJobs,
    applyProject,
  };

  stages.capture = CaptureStage(context);
  stages.refine = RefineStage(context);
  stages.reconstruct = ReconstructStage(context);
  stages.train = TrainStage(context);
  stages.inspect = InspectStage(context);

  for (const stage of STAGES) {
    const panel = stages[stage.key].panel;
    panel.id = `stage-panel-${stage.key}`;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `stage-tab-${stage.key}`);
    panel.hidden = true;
    host.append(panel);
  }

  document.getElementById("system-close")
    .addEventListener("click", () => document.getElementById("system-dialog").close());

  try {
    const { project } = await api.get("/api/project");
    if (project) applyProject(project);
  } catch { /* nothing open is the normal case */ }

  goTo(state.active);
  refreshJobs();
}

boot();
