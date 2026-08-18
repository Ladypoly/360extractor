// Application shell: top bar, pipeline, stage panels, and the job poller they share.
//
// One poller for every stage, because a job must keep running -- and keep showing as
// running in the pipeline -- while the user is looking at a different stage.

import * as api from "./api.js";
import { EmptyState, el, StatusBadge } from "./components.js";
import { icon } from "./icons.js";
import { Pipeline, STAGES } from "./pipeline.js";

import { StartStage } from "./stages/start.js";
import { CaptureStage } from "./stages/capture.js";
import { ReconstructStage } from "./stages/reconstruct.js";
import { TrainStage } from "./stages/train.js";

const POLL_IDLE = 1500;
const POLL_ACTIVE = 400;

export const state = {
  project: null,
  media: null,
  jobs: {},
  readiness: {},
  active: localStorage.getItem("stage") || "start",
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

  const recentMenu = el("div", { class: "menu", role: "menu", hidden: true });
  const recentBtn = button("Recent", "layers",
    () => toggleRecent(recentMenu, recentBtn), "Recently opened projects");

  const bar = el("header", { class: "topbar" },
    el("div", { class: "brand" },
      el("span", { class: "brand__name" }, "360extract"),
      projectName),
    el("div", { class: "source-meta" }, sourceName, sourceInfo),
    el("div", { class: "topbar__spacer" }),
    el("div", { class: "topbar__actions" },
      el("div", { class: "menu-anchor" }, recentBtn, recentMenu),
      button("Open", "folder", openProject, "Open a project folder"),
      button("System", "system", showSystem, "Detected tools")));

  return {
    bar,
    recentMenu,
    setProject: (project) => {
      projectName.textContent = project ? project.name : "no project";
    },
    setSource: (media, format) => {
      if (!media) { sourceName.textContent = ""; sourceInfo.textContent = ""; return; }
      sourceName.textContent = media.path.split(/[\\/]/).pop();
      const bits = [`${media.width}×${media.height}`];
      if (media.is_video) {
        bits.push(`${media.duration.toFixed(1)}s`, `${media.fps} fps`);
        if (media.frame_count) bits.push(`${media.frame_count} frames`);
      } else {
        bits.push("still");
      }
      // Only meaningful for a file claiming to be a panorama already. A raw two-lens
      // file is whatever shape its sensors are, and calling its geometry wrong once the
      // projection has been declared is the opposite of the truth.
      const raw = format && format.projection && format.projection !== "equirect";
      if (raw) bits.push(format.projection === "dfisheye" ? "dual fisheye" : "fisheye");
      else if (!media.looks_equirectangular) bits.push("not 2:1 — geometry will be wrong");
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

async function openRecent(root) {
  try {
    const { project } = await api.post("/api/project/open", { path: root });
    applyProject(project);
  } catch (error) { report(error); }
}

// The whole project lives in one folder and saves itself: on open, on every settings
// change (debounced), and as each stage finishes. There is no Save button because there
// is nothing a save would capture that is not already on disk moments later.
let autosaveTimer = null;
function autosave() {
  if (!state.project) return;
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(async () => {
    if (!state.project) return;
    try {
      // Each stage contributes the settings it owns (Start: masking; Capture: rig etc.).
      const payload = {};
      for (const stage of Object.values(stages)) {
        if (stage.projectPayload) Object.assign(payload, stage.projectPayload());
      }
      const { project } = await api.post("/api/project/save", payload);
      state.project = project;
      topbar.setProject(project);
    } catch { /* a later save catches up; not worth interrupting the user */ }
  }, 900);
}

// ── recent menu ──────────────────────────────────────────────────────────

let closeRecent = null;
async function toggleRecent(menu, anchor) {
  if (!menu.hidden) { hideRecent(menu); return; }
  try {
    const { recent } = await api.get("/api/recent");
    menu.replaceChildren();
    if (!recent.length) {
      menu.append(el("div", { class: "menu__empty" }, "No recent projects"));
    }
    for (const entry of recent) {
      const row = el("button", {
        class: `menu__item${entry.exists ? "" : " menu__item--missing"}`,
        type: "button", role: "menuitem", title: entry.root,
        onclick: () => { hideRecent(menu); openRecent(entry.root); },
      },
        el("span", { class: "menu__title" }, entry.name || entry.root),
        el("span", { class: "menu__path" }, entry.exists ? entry.root : "missing"));
      menu.append(row);
    }
    menu.hidden = false;
    closeRecent = (event) => {
      if (!menu.contains(event.target) && event.target !== anchor
          && !anchor.contains(event.target)) hideRecent(menu);
    };
    setTimeout(() => document.addEventListener("click", closeRecent), 0);
  } catch (error) { report(error); }
}

function hideRecent(menu) {
  menu.hidden = true;
  if (closeRecent) { document.removeEventListener("click", closeRecent); closeRecent = null; }
}

export function applyProject(project, { keepMedia = false, keepStage = false } = {}) {
  state.project = project;
  topbar.setProject(project);
  for (const stage of Object.values(stages)) {
    if (stage.applyProject) stage.applyProject(project, { keepMedia });
  }
  refreshJobs();
  // Jump to the stage this project was last on (a freshly created one has none, so it
  // stays where the user is -- Start, mid-import). `keepStage` is for loading a source
  // *into* the tab you are on: picking a video is not the same as opening a project.
  const perProject = projectStageKey(project);
  const remembered = keepStage ? null
    : (perProject ? localStorage.getItem(perProject) : null);
  goTo(remembered || state.active);
}

// ── system status ──────────────────────────────────────────────────────

//: The tool names the server will store a path for, keyed by display name.
const TOOL_KEYS = { FFmpeg: "ffmpeg", COLMAP: "colmap", Brush: "brush",
                    SuperSplat: "supersplat" };

async function showSystem() {
  const dialog = document.getElementById("system-dialog");
  const body = dialog.querySelector(".dialog__body");
  body.replaceChildren(el("div", { class: "hint" }, "Checking…"));
  dialog.showModal();

  try {
    render(await api.get("/api/system"));
  } catch (error) {
    body.replaceChildren(el("div", { class: "hint hint--error" }, error.message));
  }

  function render({ tools, configured = {} }) {
    const fields = {};
    const rows = tools.map((tool) => {
      const key = TOOL_KEYS[tool.name];
      // Not every listed tool is one we know how to store a path for.
      const field = key
        ? el("input", { type: "text", value: configured[key] || "",
                        placeholder: tool.found ? tool.path : "path to the binary" })
        : null;
      if (field) fields[key] = field;
      return el("div", { class: "tool-row tool-row--settable" },
        el("span", { class: "tool-row__name" }, tool.name),
        StatusBadge(tool.found ? "done" : "error", tool.found ? "detected" : "missing"),
        el("span", { class: "tool-row__path" },
          tool.found ? `${tool.version || ""} ${tool.path}`.trim() : tool.detail),
        field || el("span", {}));
    });

    const status = el("span", { class: "hint", style: "flex:1" });
    rows.push(el("div", { class: "field", style: "margin:12px 0 0" },
      status,
      el("button", {
        class: "btn btn--primary", type: "button", style: "flex:0 0 auto",
        onclick: async () => {
          status.textContent = "Saving…";
          const values = {};
          for (const [key, field] of Object.entries(fields)) values[key] = field.value;
          try {
            // The answer is a fresh survey, so a path that holds no binary says so here
            // rather than at the start of a reconstruction twenty minutes later.
            const result = await api.post("/api/system/tools", { tools: values });
            render(result);
            flash("Tool paths saved.", { level: "info" });
          } catch (error) { status.textContent = error.message; }
        },
      }, "Save paths")));

    rows.push(el("p", { class: "hint" },
      "Leave a field empty to go back to searching PATH and the usual install "
      + "locations. THREESIXTY_FFMPEG and friends still win over anything set here."));
    body.replaceChildren(...rows);
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

function projectStageKey(project) {
  return project && project.root ? `stage:${project.root}` : null;
}

export function goTo(key) {
  state.active = key;
  localStorage.setItem("stage", key);
  // Remember per project, so reopening one lands back where it was left.
  const perProject = projectStageKey(state.project);
  if (perProject) localStorage.setItem(perProject, key);
  for (const stage of STAGES) {
    const panel = document.getElementById(`stage-panel-${stage.key}`);
    if (panel) panel.hidden = stage.key !== key;
  }
  pipeline.render({ jobs: state.jobs, readiness: state.readiness,
                    project: state.project, active: key });
  // A stage that animates (the point cloud, the splat) should stop when it is not on
  // screen; a requestAnimationFrame loop on a hidden canvas is pure waste.
  const left = stages[state.previous];
  if (left && left.onLeave && state.previous !== key) left.onLeave();
  state.previous = key;
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
    setSource: (media, format) => {
      state.media = media;
      if (format) state.sourceFormat = format;
      topbar.setSource(media, state.sourceFormat);
    },
    goTo, report, flash, pokeJobs,
    applyProject, autosave, openProject, openRecent,
  };

  stages.start = StartStage(context);
  stages.capture = CaptureStage(context);
  stages.reconstruct = ReconstructStage(context);
  stages.train = TrainStage(context);

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
    // applyProject navigates to the project's remembered stage; with nothing open, land
    // on Start, which owns project selection and import.
    if (project) applyProject(project);
    else goTo("start");
  } catch { goTo("start"); }

  refreshJobs();
}

boot();
