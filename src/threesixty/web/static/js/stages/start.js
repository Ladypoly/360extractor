// Start: the project + import hub. Pick or resume a project, choose a source and how it
// is sampled, then Process -- which extracts the frames and hands off to Capture for the
// rig. Segments and masking join this tab in a following step.

import { InspectorSection, StageActionBar, el } from "../components.js";
import { icon } from "../icons.js";

export function StartStage(ctx) {
  const local = { media: null };

  // ── workspace: recent projects + a drop target ─────────────────────────
  const recentList = el("div", { class: "landing__recent" });
  const dropZone = el("div", { class: "landing__drop" },
    el("div", { class: "landing__icon", html: icon("camera", { size: 40 }) }),
    el("div", { class: "landing__title" }, "Load a 360° video to begin"),
    el("div", { class: "landing__hint" }, "Drag a video here, or use Browse in the panel"),
    el("p", { class: "landing__note" },
      "A project folder is created next to the video; opening the same video later resumes it."));
  const workspace = el("div", { class: "start__workspace" },
    dropZone,
    el("div", { class: "landing__side" },
      el("div", { class: "landing__side-title" }, "Recent projects"),
      recentList));

  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault(); dropZone.classList.add("landing__drop--over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("landing__drop--over"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault(); dropZone.classList.remove("landing__drop--over");
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file && file.path) { pathField.value = file.path; loadSource(); }
    else { ctx.flash("A browser drop does not expose the file path — use Browse.",
                     { level: "info" }); browse(); }
  });

  // ── inspector: source + frame selection ────────────────────────────────
  const inspector = el("aside", { class: "inspector" });

  const source = InspectorSection("Source", { id: "start-source" });
  const pathField = el("input", { type: "text", readonly: true, placeholder: "no source loaded" });
  const mediaInfo = el("p", { class: "hint" });
  source.body.append(
    el("div", { class: "field field--stack" }, pathField),
    el("div", { class: "field" },
      el("button", { class: "btn btn--primary", type: "button", onclick: browse,
                     html: `${icon("folder", { size: 14 })}<span>Browse…</span>` }),
      el("button", { class: "btn", type: "button", onclick: () => ctx.openProject(),
                     html: `${icon("layers", { size: 14 })}<span>Open project…</span>` })),
    mediaInfo);

  const framesSection = InspectorSection("Frames", { id: "start-frames" });
  const frameMode = el("select", {},
    ...[["sharp", "sharpest per second"], ["fps", "every N per second"],
        ["every", "every Nth frame"], ["all", "all frames"]]
      .map(([value, label]) => el("option", { value }, label)));
  const frameValue = el("input", { type: "number", value: 2, step: 0.5, min: 0.1 });
  const estimate = el("p", { class: "hint" });
  framesSection.body.append(
    field("frames", frameMode), field("rate", frameValue), estimate);
  frameMode.addEventListener("change", () => {
    frameValue.disabled = frameMode.value === "all";
    frameValue.value = frameMode.value === "every" ? 10 : 2;
    updateEstimate(); ctx.autosave();
  });
  frameValue.addEventListener("change", () => { updateEstimate(); ctx.autosave(); });

  for (const part of [source, framesSection]) inspector.append(part.section);

  // ── process action ─────────────────────────────────────────────────────
  const actionBar = StageActionBar({
    primaryLabel: "Process",
    onPrimary: process,
    onCancel: () => ctx.api.jobs.cancel("capture").then(ctx.pokeJobs),
  });

  const panel = el("div", { class: "stage-panel start-panel" },
                    workspace, inspector, actionBar.bar);

  // ── helpers ─────────────────────────────────────────────────────────────
  function field(label, control) {
    return el("div", { class: "field" }, el("label", {}, label), control);
  }

  function updateEstimate() {
    const media = local.media;
    if (!media) { estimate.textContent = ""; return; }
    const mode = frameMode.value, value = parseFloat(frameValue.value) || 1;
    let frames = 1;
    if (media.is_video) {
      if (mode === "fps" || mode === "sharp") frames = Math.max(Math.floor(media.duration * value), 1);
      else if (mode === "every") frames = Math.max(Math.floor(media.frame_count / value), 1);
      else frames = media.frame_count;
    }
    estimate.textContent = `~${frames} frames`
      + (mode === "sharp" ? " (sharpest in each second)" : "");
  }

  function updateMediaInfo() {
    const media = local.media;
    if (!media) { mediaInfo.textContent = ""; return; }
    const bits = [`${media.width}×${media.height}`];
    if (media.is_video) bits.push(`${media.duration.toFixed(1)}s`, `${media.fps} fps`);
    else bits.push("still");
    if (!media.looks_equirectangular) bits.push("not 2:1 — geometry will be wrong");
    mediaInfo.textContent = bits.join("  ·  ");
  }

  async function browse() {
    try {
      const paths = await ctx.api.pick("open", "Select a 360 video or still", "media",
                                       local.media ? local.media.path : "");
      if (!paths.length) return;
      pathField.value = paths[0];
      await loadSource();
    } catch (error) { ctx.report(error); }
  }

  async function loadSource() {
    const path = pathField.value.trim();
    if (!path) return;
    try {
      const data = await ctx.api.post("/api/preview", { path, time: 0 });
      local.media = data.media;
      ctx.setSource(data.media);
      updateMediaInfo(); updateEstimate();
      // Opening a source is opening its project, created beside the video.
      const { project } = await ctx.api.post("/api/project/for-source", {
        path: data.media.path,
        frames: { mode: frameMode.value, value: parseFloat(frameValue.value) || 2 },
      });
      ctx.applyProject(project, { keepMedia: true });
    } catch (error) { ctx.report(error); }
  }

  let processing = false, lastState = null;
  async function process() {
    if (!local.media || !ctx.state.project) {
      ctx.flash("Load a source first.", { level: "warn" }); return;
    }
    try {
      await ctx.api.post("/api/frames/extract", {
        mode: frameMode.value, value: parseFloat(frameValue.value) || 2,
      });
      processing = true;
      ctx.pokeJobs();
    } catch (error) { ctx.report(error); }
  }

  async function refreshRecent() {
    try {
      const { recent } = await ctx.api.get("/api/recent");
      recentList.replaceChildren();
      if (!recent.length) {
        recentList.append(el("p", { class: "hint" }, "No recent projects yet."));
        return;
      }
      for (const entry of recent) {
        recentList.append(el("button", {
          class: `landing__recent-item${entry.exists ? "" : " landing__recent-item--missing"}`,
          type: "button", title: entry.root,
          onclick: entry.exists ? () => ctx.openRecent(entry.root) : undefined,
        },
          el("span", { class: "landing__recent-name" }, entry.name || entry.root),
          el("span", { class: "landing__recent-path" }, entry.exists ? entry.root : "missing")));
      }
    } catch { /* the list is a convenience */ }
  }

  return {
    panel,
    onEnter: () => refreshRecent(),
    onJobs: (_job, allJobs) => {
      const capture = allJobs.capture;
      actionBar.render(capture);
      // When our Process (frame extraction) finishes, move on to the rig.
      if (processing && capture && capture.state === "done" && lastState === "running") {
        processing = false;
        ctx.goTo("capture");
      }
      lastState = capture ? capture.state : null;
    },
    applyProject(project, { keepMedia } = {}) {
      if (!project) return;
      frameMode.value = project.frames.mode;
      frameValue.value = project.frames.value;
      updateEstimate();
      if (!keepMedia && project.sources && project.sources.length) {
        pathField.value = project.sources[0];
      }
    },
  };
}
