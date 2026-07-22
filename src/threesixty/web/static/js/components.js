// Shared interface pieces.
//
// Five stages need the same action bar, the same progress, the same log. Building
// those five times is how an application ends up feeling like several tools behind
// tabs, which is exactly what this redesign exists to undo.

import { icon } from "./icons.js";

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2).toLowerCase(), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function formatCount(value) {
  return Number(value).toLocaleString();
}

export function formatClock(seconds) {
  seconds = Math.max(0, Math.round(seconds));
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
  return `${Math.floor(seconds / 3600)}h ${String(Math.floor((seconds % 3600) / 60)).padStart(2, "0")}m`;
}

/** A small state pill. Always icon *and* colour, never colour alone. */
export function StatusBadge(state, label = "") {
  const icons = {
    pending: "pending", ready: "ready", running: "running",
    done: "done", stale: "stale", error: "error", disabled: "disabled",
  };
  return el("span", {
    class: `badge badge--${state}`,
    html: `${icon(icons[state] || "pending", { size: 12 })}<span>${label || state}</span>`,
  });
}

/** A collapsible inspector section, remembering whether it was open. */
export function InspectorSection(title, { note = "", open = true, id = "" } = {}) {
  const key = id ? `section:${id}` : "";
  const stored = key ? localStorage.getItem(key) : null;
  const isOpen = stored === null ? open : stored === "true";

  const body = el("div", { class: "section__body" });
  const noteEl = el("span", { class: "section__note" }, note);
  const head = el("button", {
    class: "section__head", type: "button", "aria-expanded": String(isOpen),
    html: `${icon("chevron", { size: 13, className: "section__chevron" })}<span>${title}</span>`,
  });
  head.append(noteEl);

  const section = el("section", { class: "section", "data-open": String(isOpen) }, head, body);
  head.addEventListener("click", () => {
    const next = section.dataset.open !== "true";
    section.dataset.open = String(next);
    head.setAttribute("aria-expanded", String(next));
    if (key) localStorage.setItem(key, String(next));
  });

  return { section, body, setNote: (text) => { noteEl.textContent = text; } };
}

/**
 * The bar along the bottom of every stage.
 *
 * While a job runs this becomes a live status display rather than a disabled button --
 * a bar sitting at zero with a greyed-out control is what made detection look dead.
 */
export function StageActionBar({ primaryLabel, primaryIcon = "play", onPrimary,
                                 onCancel, secondary = [] }) {
  const message = el("span", { class: "actionbar__message" }, "Ready");
  const detail = el("span", { class: "actionbar__detail" });
  const percent = el("span", { class: "actionbar__percent" });
  const fill = el("div", { class: "progress__fill" });
  const progress = el("div", { class: "progress", style: "display:none" }, fill);

  const cancel = el("button", {
    class: "btn btn--danger", type: "button", style: "display:none",
    html: `${icon("cancel", { size: 14 })}<span>Cancel</span>`,
    onclick: () => onCancel && onCancel(),
  });

  const primary = el("button", {
    class: "btn btn--primary", type: "button",
    html: `${icon(primaryIcon, { size: 14 })}<span>${primaryLabel}</span>`,
    onclick: () => onPrimary && onPrimary(),
  });

  const actions = el("div", { class: "actionbar__actions" },
    ...secondary, cancel, primary);

  const bar = el("div", { class: "actionbar" },
    el("div", { class: "actionbar__status" },
      el("div", { class: "actionbar__line" }, message, detail, percent),
      progress),
    actions);

  function render(job) {
    const running = job && job.state === "running";
    progress.style.display = running ? "" : "none";
    cancel.style.display = running && job.cancellable ? "" : "none";
    primary.disabled = Boolean(running);

    if (!job || job.state === "pending") {
      message.textContent = "Ready";
      detail.textContent = "";
      percent.textContent = "";
      return;
    }

    message.textContent = job.message || job.state;
    detail.textContent = job.detail || "";

    if (running) {
      const known = job.fraction > 0;
      progress.classList.toggle("progress--indeterminate", !known);
      fill.style.width = known ? `${Math.round(job.fraction * 100)}%` : "";
      percent.textContent = known ? `${Math.round(job.fraction * 100)}%` : "";
    } else if (job.state === "error") {
      percent.textContent = "";
      message.textContent = job.error || "failed";
    } else if (job.state === "done") {
      percent.textContent = job.elapsed ? formatClock(job.elapsed) : "";
    } else {
      percent.textContent = "";
    }
  }

  return { bar, render, primary, cancel,
           setPrimaryLabel: (text) => {
             primary.querySelector("span").textContent = text;
           } };
}

/** A monospace log with auto-scroll that yields when the user scrolls up. */
export function LogViewer({ title = "Log" } = {}) {
  const lines = el("div", { class: "log__lines", role: "log", "aria-live": "polite" });
  let follow = true;
  let rendered = 0;

  const followButton = el("button", {
    class: "btn btn--ghost btn--icon", type: "button", title: "Pause auto-scroll",
    html: icon("pause", { size: 13 }),
    onclick: () => {
      follow = !follow;
      followButton.innerHTML = icon(follow ? "pause" : "play", { size: 13 });
      followButton.title = follow ? "Pause auto-scroll" : "Resume auto-scroll";
      if (follow) lines.scrollTop = lines.scrollHeight;
    },
  });

  const copyButton = el("button", {
    class: "btn btn--ghost btn--icon", type: "button", title: "Copy log",
    html: icon("copy", { size: 13 }),
    onclick: async () => {
      try {
        await navigator.clipboard.writeText(lines.textContent);
        copyButton.innerHTML = icon("done", { size: 13 });
        setTimeout(() => { copyButton.innerHTML = icon("copy", { size: 13 }); }, 1200);
      } catch { /* clipboard refused; nothing useful to say */ }
    },
  });

  lines.addEventListener("scroll", () => {
    const atBottom = lines.scrollHeight - lines.scrollTop - lines.clientHeight < 24;
    if (!atBottom) follow = false;
  });

  const root = el("div", { class: "log" },
    el("div", { class: "log__bar" },
      el("span", {}, title),
      el("span", { style: "flex:1" }),
      followButton, copyButton),
    lines);

  function render(log = []) {
    if (log.length < rendered) {          // a new run: start the view again
      lines.replaceChildren();
      rendered = 0;
    }
    for (const line of log.slice(rendered)) {
      lines.append(el("div", { class: `log__line log__line--${line.level}` }, line.text));
    }
    rendered = log.length;
    if (follow) lines.scrollTop = lines.scrollHeight;
  }

  return { root, render, clear: () => { lines.replaceChildren(); rendered = 0; } };
}

/** Compact metric blocks. Deliberately not dashboard cards. */
export function MetricStrip(metrics = []) {
  const root = el("div", { class: "metrics" });
  function render(values) {
    root.replaceChildren(...values.map(({ label, value }) =>
      el("div", { class: "metric" },
        el("div", { class: "metric__label" }, label),
        el("div", { class: "metric__value" }, value))));
  }
  render(metrics);
  return { root, render };
}

/** What a stage shows before it has anything to show. */
export function EmptyState({ iconName = "info", title, body, actionLabel, onAction }) {
  const children = [
    el("div", { class: "empty__icon", html: icon(iconName, { size: 34 }) }),
    el("div", { class: "empty__title" }, title),
    el("div", { class: "empty__body" }, body),
  ];
  if (actionLabel && onAction) {
    children.push(el("button", { class: "btn btn--primary", type: "button",
                                 onclick: onAction }, actionLabel));
  }
  return el("div", { class: "empty" }, ...children);
}
