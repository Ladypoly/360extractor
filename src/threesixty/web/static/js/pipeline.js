// The pipeline navigator.
//
// It is the application's primary structure, not a tab strip: it shows where you are,
// what has finished, what is running *right now even on another stage*, and what
// cannot run yet and why.

import { el } from "./components.js";
import { icon } from "./icons.js";

export const STAGES = [
  { key: "capture",     label: "Capture",     iconName: "camera" },
  { key: "refine",      label: "Refine",      iconName: "refine" },
  { key: "reconstruct", label: "Reconstruct", iconName: "reconstruct" },
  { key: "train",       label: "Train",       iconName: "train" },
  { key: "inspect",     label: "Inspect",     iconName: "inspect" },
];

const STATE_ICON = {
  pending: "pending", ready: "ready", running: "running",
  done: "done", stale: "stale", error: "error", disabled: "disabled",
};

export function Pipeline({ onSelect }) {
  const buttons = new Map();

  const nav = el("nav", { class: "pipeline", role: "tablist",
                          "aria-label": "Processing pipeline" });

  STAGES.forEach((stage, index) => {
    const button = el("button", {
      class: "stage stage--pending",
      type: "button",
      role: "tab",
      id: `stage-tab-${stage.key}`,
      "aria-controls": `stage-panel-${stage.key}`,
      "aria-selected": "false",
      onclick: () => onSelect(stage.key),
    },
      el("span", { class: "stage__index" }, String(index + 1)),
      el("span", { class: "stage__icon", html: icon("pending", { size: 15 }) }),
      el("span", { class: "stage__label" }, stage.label));

    buttons.set(stage.key, button);
    nav.append(button);
  });

  // Left and right arrows move between stages, as a tablist should.
  nav.addEventListener("keydown", (event) => {
    const order = STAGES.map((s) => s.key);
    const current = order.findIndex((key) => buttons.get(key) === document.activeElement);
    if (current < 0) return;
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    const next = buttons.get(order[(current + step + order.length) % order.length]);
    next.focus();
  });

  /**
   * `jobs` is the per-stage job snapshot, `readiness` says what can run, `active` is
   * the stage on screen. Precedence matters: something running or failed outranks
   * "cannot run yet", because that is the more urgent thing to know.
   */
  function render({ jobs = {}, readiness = {}, project = null, active = "capture" }) {
    for (const stage of STAGES) {
      const button = buttons.get(stage.key);
      const job = jobs[stage.key] || {};
      const ready = readiness[stage.key] || { ready: false, reason: "" };
      const projectStage = project && project.stages ? project.stages[stageToProject(stage.key)] : null;

      let state = "pending";
      if (job.state === "running") state = "running";
      else if (job.state === "error") state = "error";
      else if (projectStage === "stale") state = "stale";
      else if (job.state === "done" || projectStage === "done") state = "done";
      else if (ready.ready) state = "ready";
      else state = "disabled";

      button.className = `stage stage--${state}`;
      button.setAttribute("aria-selected", String(stage.key === active));
      button.querySelector(".stage__icon").innerHTML =
        icon(STATE_ICON[state], { size: 15 });

      const blocked = !ready.ready && state !== "running" && state !== "error";
      button.disabled = blocked && stage.key !== active;
      button.title = blocked && ready.reason
        ? ready.reason
        : `${stage.label}${job.message ? ` — ${job.message}` : ""}`;

      const percent = job.state === "running" && job.fraction > 0
        ? ` ${Math.round(job.fraction * 100)}%` : "";
      button.querySelector(".stage__label").textContent = stage.label + percent;
    }
  }

  return { nav, render };
}

/** Pipeline stages and project stages are not quite the same set. */
function stageToProject(key) {
  return { capture: "extract", refine: "mask", reconstruct: "export",
           train: "train", inspect: "clean" }[key] || key;
}
