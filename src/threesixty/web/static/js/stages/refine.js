// Refine: find the occluders that move, and check the masks before trusting them.

import {
  EmptyState, InspectorSection, LogViewer, StageActionBar, el,
} from "../components.js";
import { icon } from "../icons.js";

const BACKEND_NOTE = {
  "sam2.1": "SAM 2.1 is promptable, not open-vocabulary — it has no concept of a "
          + "\"person\". YOLO finds what to mask, SAM sharpens exactly where.",
  yolo: "Fast and self-contained. Coarser outlines than SAM, which the grow setting "
      + "partly compensates for.",
};

const VIEW_MODES = [["overlay", "Overlay"], ["original", "Original"], ["mask", "Mask only"]];

export function RefineStage(ctx) {
  const local = { cameras: [], camera: null, frames: [], index: 0, mode: "overlay",
                  opacity: 0.55, available: false };

  const picture = el("img", { alt: "", style: "display:none" });
  const view = el("div", { class: "image-view" }, picture);
  const caption = el("div", { class: "log__bar" });
  const filmstrip = el("div", { class: "filmstrip" });
  const log = LogViewer({ title: "Detection log" });
  log.root.style.maxHeight = "180px";

  const workspace = el("div", { class: "workspace" }, view, caption, filmstrip, log.root);

  const inspector = el("aside", { class: "inspector" });

  const config = InspectorSection("Detection", { id: "ref-config" });
  const backend = el("select", {},
    el("option", { value: "sam2.1" }, "YOLO + SAM 2.1"),
    el("option", { value: "yolo" }, "YOLO only"));
  const note = el("p", { class: "hint" });
  const classes = el("input", { type: "text",
                                value: "person,car,bus,truck,motorcycle,bicycle" });
  const confidence = el("input", { type: "number", value: 0.25, step: 0.05, min: 0.01, max: 0.99 });
  const dilate = el("input", { type: "number", value: 6, step: 1, min: 0 });
  const fuse = el("select", {},
    el("option", { value: "1" }, "reconcile overlapping cameras"),
    el("option", { value: "0" }, "keep each camera independent"));

  config.body.append(
    field("find with", backend), note,
    field("classes", classes),
    el("div", { class: "pair" }, field("conf", confidence), field("grow", dilate)),
    field("sphere", fuse),
    el("p", { class: "hint" },
      "Masks are grown a few pixels: a sliver of leftover pedestrian is enough to seed "
      + "a floater."));

  const review = InspectorSection("Review", { id: "ref-review" });
  const cameraSelect = el("select", {});
  const opacity = el("input", { type: "range", min: 0, max: 100, value: 55 });
  const modeButtons = el("div", { class: "segmented" },
    ...VIEW_MODES.map(([value, label]) => el("button", {
      type: "button", "aria-pressed": String(value === "overlay"),
      onclick: () => setMode(value),
    }, label)));
  review.body.append(
    field("camera", cameraSelect), field("view", modeButtons),
    field("overlay", opacity),
    el("p", { class: "hint" },
      "Red is what will be excluded from training. Worth a look before committing to a "
      + "long reconstruction."));

  inspector.append(config.section, review.section);

  const actionBar = StageActionBar({
    primaryLabel: "Run Detection",
    onPrimary: runDetection,
    onCancel: () => ctx.api.jobs.cancel("refine").then(ctx.pokeJobs),
  });

  const panel = el("div", { class: "stage-panel" }, workspace, inspector, actionBar.bar);

  function field(label, control) {
    return el("div", { class: "field" }, el("label", {}, label), control);
  }

  function setMode(mode) {
    local.mode = mode;
    [...modeButtons.children].forEach((button, index) =>
      button.setAttribute("aria-pressed", String(VIEW_MODES[index][0] === mode)));
    showFrame();
  }

  backend.addEventListener("change", () => { note.textContent = BACKEND_NOTE[backend.value]; });
  note.textContent = BACKEND_NOTE["sam2.1"];

  opacity.addEventListener("input", () => {
    local.opacity = (parseInt(opacity.value, 10) || 0) / 100;
    showFrame();
  });

  cameraSelect.addEventListener("change", () => selectCamera(cameraSelect.value));

  async function refreshFrames() {
    if (!ctx.state.project) return;
    try {
      const data = await ctx.api.post("/api/detect/frames", {});
      local.cameras = data.cameras || [];
      cameraSelect.replaceChildren(...local.cameras.map((camera) =>
        el("option", { value: camera.name },
           `${camera.name} (${camera.frames.length})`)));
      review.setNote(data.masked ? `${data.masked} masks on disk` : "no masks yet");
      if (local.cameras.length) selectCamera(local.cameras[0].name);
      else view.replaceChildren(EmptyState({
        iconName: "film", title: "No extracted frames",
        body: "Detection runs on the images Capture produced.",
        actionLabel: "Go to Capture", onAction: () => ctx.goTo("capture"),
      }));
    } catch (error) { /* readiness already explains this */ }
  }

  function selectCamera(name) {
    const camera = local.cameras.find((entry) => entry.name === name);
    if (!camera) return;
    local.camera = camera;
    local.frames = camera.frames;
    local.index = 0;
    cameraSelect.value = name;

    filmstrip.replaceChildren(...camera.frames.slice(0, 200).map((frame, index) =>
      el("button", {
        type: "button", "aria-pressed": String(index === 0),
        onclick: () => { local.index = index; showFrame(); },
      }, String(frame))));
    showFrame();
  }

  let frameTimer = null;
  function showFrame() {
    const frame = local.frames[local.index];
    if (frame === undefined || !local.camera) return;
    caption.textContent = `Camera ${local.camera.name}    Frame ${local.index + 1} / `
      + `${local.frames.length}`;
    [...filmstrip.children].forEach((button, index) =>
      button.setAttribute("aria-pressed", String(index === local.index)));

    clearTimeout(frameTimer);
    frameTimer = setTimeout(async () => {
      try {
        const data = await ctx.api.post("/api/detect/preview", {
          camera: local.camera.name, frame,
          opacity: local.mode === "original" ? 0 : local.mode === "mask" ? 1 : local.opacity,
        });
        picture.src = data.url;
        picture.style.display = "block";
        if (!view.contains(picture)) view.replaceChildren(picture);
      } catch (error) { ctx.report(error); }
    }, 120);
  }

  async function runDetection() {
    if (!local.available) {
      ctx.flash('Dynamic masking needs the ML extra: pip install -e ".[ml]"',
                { level: "warn" });
      return;
    }
    try {
      await ctx.api.post("/api/detect/run", {
        backend: backend.value,
        classes: classes.value.split(",").map((s) => s.trim()).filter(Boolean),
        confidence: parseFloat(confidence.value) || 0.25,
        dilate: parseInt(dilate.value, 10) || 0,
        fuse: fuse.value === "1",
      });
      ctx.pokeJobs();
    } catch (error) { ctx.report(error); }
  }

  ctx.api.get("/api/detect/status").then((data) => {
    local.available = Boolean(data.available);
    config.setNote(local.available ? "ready" : "ML extra not installed");
  }).catch(() => {});

  let wasRunning = false;
  return {
    panel,
    onJobs(job) {
      actionBar.render(job);
      if (job) log.render(job.log || []);
      if (wasRunning && job && job.state !== "running") refreshFrames();
      wasRunning = Boolean(job && job.state === "running");
    },
    onEnter() {
      refreshFrames();
      ctx.api.jobs.status("refine").then((job) => log.render(job.log || [])).catch(() => {});
    },
  };
}
