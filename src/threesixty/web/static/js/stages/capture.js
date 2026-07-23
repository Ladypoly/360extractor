// Capture: the source, the grade, the rig, the occluder, and extraction.
//
// The panorama interaction is carried over deliberately unchanged in behaviour --
// footprints, dragging a camera to aim it, painting the occluder across the seam. It
// was the strongest part of the old interface and the redesign is about the structure
// around it, not about it.

import { footprint, occlusionFraction } from "../geometry.js";
import {
  EmptyState, InspectorSection, StageActionBar, el, formatCount,
} from "../components.js";
import { icon } from "../icons.js";

const PALETTE = ["#4da3ff", "#5ecb8b", "#e8b054", "#f06a6a", "#c58aff", "#4fd8d8",
                 "#ff8ac4", "#b7d24d", "#7f9cff", "#ffd166"];
const colourOf = (index) => PALETTE[index % PALETTE.length];

const GRADE_FIELDS = {
  exposure:   { label: "exposure", min: -2,   max: 2,   step: 0.05, neutral: 0 },
  brightness: { label: "bright",   min: -0.5, max: 0.5, step: 0.01, neutral: 0 },
  contrast:   { label: "contrast", min: 0.2,  max: 2.5, step: 0.01, neutral: 1 },
  saturation: { label: "satur.",   min: 0,    max: 2.5, step: 0.01, neutral: 1 },
  gamma:      { label: "gamma",    min: 0.3,  max: 3,   step: 0.01, neutral: 1 },
  black:      { label: "black",    min: -0.3, max: 0.3, step: 0.01, neutral: 0 },
};

const PROXY_WIDTH = 560;
const FULL_WIDTH = 1600;

export function CaptureStage(ctx) {
  const local = {
    rig: null, presets: {}, userPresets: new Set(), selected: 0, media: null,
    frames: [], frameIndex: 0, clip: null,
    nadir: 0, dragging: null, dragOffset: { yaw: 0, pitch: 0 }, image: null,
    paint: { mode: null, layer: null, brush: 40, drawing: false, last: null, dirty: false },
    coverage: {}, sizes: {},
  };

  // ── workspace ────────────────────────────────────────────────────────
  const canvas = el("canvas", { class: "panorama", width: 1600, height: 800 });
  const context2d = canvas.getContext("2d");
  const canvasHost = el("div", { class: "workspace__canvas" }, canvas);

  const timeSlider = el("input", { type: "range", min: 0, max: 0, step: 0.1, value: 0,
                                   style: "flex:1" });
  const timeLabel = el("span", { class: "actionbar__detail" }, "0.0s");
  const timeline = el("div", { class: "log__bar" },
    el("span", {}, "frame"), timeSlider, timeLabel);

  const workspace = el("div", { class: "workspace" }, canvasHost, timeline);

  // ── inspector ────────────────────────────────────────────────────────
  const inspector = el("aside", { class: "inspector" });

  const source = InspectorSection("Source", { id: "cap-source" });
  const pathField = el("input", { type: "text", readonly: true,
                                  placeholder: "no source loaded" });
  source.body.append(
    el("div", { class: "field field--stack" }, pathField),
    el("div", { class: "field", style: "margin-bottom:0" },
      el("button", { class: "btn btn--primary", type: "button", onclick: browse,
                     html: `${icon("folder", { size: 14 })}<span>Browse…</span>` })));

  const rigSection = InspectorSection("Rig", { id: "cap-rig" });
  const presetSelect = el("select", {});
  const presetName = el("input", { type: "text", placeholder: "save current rig as…" });
  const savePresetBtn = el("button", { class: "btn btn--ghost", type: "button",
                                       style: "flex:0 0 auto", onclick: savePreset,
                                       html: `${icon("save", { size: 14 })}<span>Save</span>` });
  const deletePresetBtn = el("button", { class: "btn btn--ghost", type: "button",
                                         style: "flex:0 0 auto", onclick: deletePreset,
                                         title: "Delete the selected saved preset",
                                         html: icon("trash", { size: 14 }) });
  // FOV and shape are one global rig property applied to every camera, not a per-camera
  // setting -- a mixed-FOV rig is not a case worth the interface it costs.
  const camFov = el("input", { type: "number", min: 20, max: 150, step: 1, value: 90 });
  const camShape = el("select", {},
    ...[["1.3333", "4:3"], ["1.5", "3:2"], ["1.7778", "16:9"], ["1", "square"]]
      .map(([value, label]) => el("option", { value }, label)));
  const camList = el("div", {});
  rigSection.body.append(
    el("div", { class: "field" }, presetSelect,
      el("button", { class: "btn", type: "button", style: "flex:0 0 auto",
                     onclick: applyPreset }, "Use"),
      deletePresetBtn),
    el("div", { class: "field" }, presetName, savePresetBtn),
    el("div", { class: "pair" }, field("fov", camFov), field("shape", camShape)),
    camList,
    el("div", { class: "field", style: "margin-bottom:0" },
      el("button", { class: "btn btn--ghost", type: "button", onclick: addCamera }, "Add"),
      el("button", { class: "btn btn--ghost", type: "button", onclick: duplicateCamera }, "Duplicate"),
      el("button", { class: "btn btn--ghost", type: "button", onclick: removeCamera }, "Remove")));
  presetSelect.addEventListener("change", updatePresetButtons);

  const image = InspectorSection("Image", { id: "cap-image", note: "unchanged" });
  const gradeInputs = {};
  for (const [key, spec] of Object.entries(GRADE_FIELDS)) {
    const control = slider(spec.min, spec.max, spec.step, spec.neutral);
    gradeInputs[key] = control;
    image.body.append(field(spec.label, control.root));
  }
  const gradeNotes = el("p", { class: "hint" });
  image.body.append(
    el("div", { class: "field", style: "margin-bottom:0" },
      el("button", { class: "btn btn--primary", type: "button", onclick: autoGrade,
                     html: `${icon("wand", { size: 14 })}<span>Auto</span>` }),
      el("button", { class: "btn btn--ghost", type: "button", onclick: resetGrade }, "Reset")),
    gradeNotes);

  // Masking: what to keep out of the splat. Runs when cameras are generated -- sky (a
  // cone for now, a semantic model later) plus object detection on the frames.
  const masking = InspectorSection("Masking", { id: "cap-masking" });
  const maskSky = el("input", { type: "checkbox", checked: true });
  const maskSkyMethod = el("select", {},
    ...[["auto", "auto (model or cone)"], ["cone", "cone only"], ["off", "off"]]
      .map(([value, label]) => el("option", { value }, label)));
  const maskSkyAngle = el("input", { type: "number", min: 0, max: 89, step: 1, value: 30 });
  const maskBackend = el("select", {},
    ...[["sam2.1", "YOLO + SAM 2.1"], ["yolo", "YOLO only"]]
      .map(([value, label]) => el("option", { value }, label)));
  const maskClasses = el("input", { type: "text",
                                    value: "person,car,bus,truck,motorcycle,bicycle" });
  const maskConfidence = el("input", { type: "number", min: 0.05, max: 0.95, step: 0.05, value: 0.25 });
  const maskDilate = el("input", { type: "number", min: 0, max: 40, step: 1, value: 6 });
  masking.body.append(
    el("div", { class: "field" }, el("label", {}, "exclude sky"), maskSky),
    field("sky via", maskSkyMethod),
    field("cone °", maskSkyAngle),
    el("p", { class: "hint" }, "Sky only ever seeds floaters, so it is masked by default; "
      + "the cone masks everything above the angle."),
    field("objects", maskBackend),
    field("classes", maskClasses),
    el("div", { class: "pair" }, field("confidence", maskConfidence), field("grow", maskDilate)),
    el("p", { class: "hint" }, "Object detection needs the ML extra and runs on the frames "
      + "when cameras are generated."));
  for (const control of [maskSkyMethod, maskSkyAngle, maskBackend, maskClasses,
                         maskConfidence, maskDilate]) {
    control.addEventListener("change", () => refresh());
  }
  maskSky.addEventListener("change", () => { updateMaskFields(); refresh(); });

  const occluder = InspectorSection("Occluder", { id: "cap-occluder", note: "no cone" });
  const nadirSlider = el("input", { type: "range", min: 0, max: 89, step: 1, value: 0 });
  const brushSlider = slider(4, 160, 2, 40);
  const paintButton = el("button", { class: "btn", type: "button",
                                     html: `${icon("brush", { size: 14 })}<span>Paint</span>` });
  const eraseButton = el("button", { class: "btn", type: "button",
                                     html: `${icon("eraser", { size: 14 })}<span>Erase</span>` });
  const maskMode = el("select", {},
    ...[["sidecar", "mask sidecars"], ["skip", "drop covered cameras"],
        ["burn", "burn into images"], ["none", "ignore"]]
      .map(([value, label]) => el("option", { value }, label)));
  occluder.body.append(
    field("nadir", nadirSlider), field("brush", brushSlider.root),
    el("div", { class: "field" }, paintButton, eraseButton,
      el("button", { class: "btn btn--ghost", type: "button", onclick: clearPaint,
                     html: icon("trash", { size: 14 }), title: "Clear painted occluder" })),
    field("handling", maskMode),
    el("p", { class: "hint" },
      "Painted once on the panorama: the rig is rigid, so one region covers every frame."));

  const output = InspectorSection("Output", { id: "cap-output" });
  const outDir = el("input", { type: "text", readonly: true, tabindex: -1,
                               placeholder: "a folder is created beside the source" });
  const outFormat = el("select", {}, el("option", {}, "jpg"), el("option", {}, "png"));
  const outQuality = el("input", { type: "number", value: 2, min: 1, max: 31 });
  const frameMode = el("select", {},
    ...[["sharp", "sharpest per second"], ["fps", "every N per second"],
        ["every", "every Nth frame"], ["all", "all frames"]]
      .map(([value, label]) => el("option", { value }, label)));
  const frameValue = el("input", { type: "number", value: 2, step: 0.5, min: 0.1 });
  const estimate = el("p", { class: "hint" });
  output.body.append(
    field("folder", outDir),
    el("p", { class: "hint" }, "Created automatically beside the source; everything the "
      + "project produces lands here and saves as you go."),
    el("div", { class: "pair" }, field("format", outFormat), field("quality", outQuality)));

  // ── segments ─────────────────────────────────────────────────────────
  // Split a long drive into independent projects: a kilometres-long clip reconstructs
  // far better as several short datasets than as one that COLMAP drifts across.
  const segments = InspectorSection("Segments", { id: "cap-segments", open: false });
  const segMode = el("select", {},
    ...[["off", "one project (no split)"], ["duration", "by duration"],
        ["motion-distance", "by distance (from video motion)"],
        ["motion-count", "by count (equal travel)"],
        ["gpx", "by distance (GPS track)"]]
      .map(([value, label]) => el("option", { value }, label)));
  const segSeconds = el("input", { type: "number", value: 60, min: 1, step: 5 });
  const segMeters = el("input", { type: "number", value: 500, min: 10, step: 50 });
  const segSpeed = el("input", { type: "number", value: 40, min: 1, step: 5 });
  const segCount = el("input", { type: "number", value: 4, min: 1, step: 1 });
  const segCreateBtn = el("button", { class: "btn btn--primary", type: "button",
                                      onclick: createSegments,
                                      html: `${icon("layers", { size: 14 })}<span>Create segments</span>` });
  const segResults = el("div", { class: "landing__recent" });
  const segFields = {
    duration: field("seconds", segSeconds),
    meters: field("metres", segMeters),
    speed: field("avg km/h", segSpeed),
    count: field("segments", segCount),
  };
  const segHint = el("p", { class: "hint" });
  segments.body.append(
    field("split", segMode),
    segFields.duration, segFields.meters, segFields.speed, segFields.count,
    segHint,
    el("div", { class: "field", style: "margin-bottom:0" }, segCreateBtn),
    segResults);
  segMode.addEventListener("change", updateSegFields);

  const orientation = InspectorSection("Rig orientation", { id: "cap-orient", open: false });
  const orientYaw = el("input", { type: "number", value: 0, step: 1 });
  const orientPitch = el("input", { type: "number", value: 0, step: 1 });
  orientation.body.append(
    el("div", { class: "pair" }, field("yaw", orientYaw), field("pitch", orientPitch)),
    el("p", { class: "hint" }, "Levels a tilted mount without editing every camera."));

  const previewSection = InspectorSection("Camera preview", { id: "cap-preview" });
  const previewImage = el("img", { style: "width:100%;border-radius:5px;display:none" });
  previewSection.body.append(previewImage);

  // Occluder and Rig-orientation sections are intentionally not mounted: masking now
  // lives in its own flow and every camera shares one global FOV/shape. Their element
  // objects stay constructed (referenced by legacy handlers with harmless defaults) but
  // are never shown.
  for (const part of [source, rigSection, image, masking, output,
                      segments, previewSection]) {
    inspector.append(part.section);
  }

  // ── action bar ───────────────────────────────────────────────────────
  const actionBar = StageActionBar({
    primaryLabel: "Extract frames",
    onPrimary: runCapture,
    onCancel: () => ctx.api.jobs.cancel("capture").then(ctx.pokeJobs),
  });

  // ── landing (empty state) ────────────────────────────────────────────
  // The way into the whole pipeline, shown until a source is loaded: a video (which
  // creates a project beside it) or an existing project. This is the app's front door.
  const recentList = el("div", { class: "landing__recent" });
  const dropZone = el("div", { class: "landing__drop" },
    el("div", { class: "landing__icon", html: icon("camera", { size: 40 }) }),
    el("div", { class: "landing__title" }, "Load a 360° video to begin"),
    el("div", { class: "landing__hint" }, "Drag a video here, or"),
    el("div", { class: "landing__actions" },
      el("button", { class: "btn btn--primary", type: "button", onclick: browse,
                     html: `${icon("folder", { size: 14 })}<span>Browse…</span>` }),
      el("button", { class: "btn", type: "button", onclick: () => ctx.openProject(),
                     html: `${icon("layers", { size: 14 })}<span>Open project…</span>` })),
    el("p", { class: "landing__note" },
      "A project folder is created next to the video; opening the same video later resumes it."));

  const landing = el("div", { class: "landing" },
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
    // Browsers hand over the bytes but not the path, and the server extracts by path --
    // so a drop cannot be resolved directly. `file.path` exists only in Electron-like
    // hosts; when it does not, fall back to the native picker so the drop still leads in.
    const path = file && file.path;
    if (path) { pathField.value = path; loadMedia(); }
    else {
      ctx.flash("A browser drop does not expose the file path — pick it in the dialog.",
                { level: "info" });
      browse();
    }
  });

  // Extract-frames dialog: opens on load (and from the primary button) to choose how the
  // source is sampled into the working set before the rig ever comes up.
  const dlgInfo = el("p", { class: "hint" });
  const framesDialog = el("dialog", { id: "frames-dialog", "aria-labelledby": "frames-title" },
    el("div", { class: "dialog__head", id: "frames-title" }, "Extract frames"),
    el("div", { class: "dialog__body" },
      dlgInfo,
      field("frames", frameMode),
      field("rate", frameValue),
      estimate),
    el("div", { class: "dialog__foot" },
      el("button", { class: "btn", type: "button", onclick: () => framesDialog.close() }, "Cancel"),
      el("button", { class: "btn btn--primary", type: "button", onclick: confirmFramesDialog,
                     html: `${icon("film", { size: 14 })}<span>Extract frames</span>` })));

  const panel = el("div", { class: "stage-panel" },
                    landing, workspace, inspector, actionBar.bar, framesDialog);

  function updateDialogInfo() {
    const media = local.media;
    if (!media) { dlgInfo.textContent = ""; return; }
    const bits = [`${media.width}×${media.height}`];
    if (media.is_video) bits.push(`${media.duration.toFixed(1)}s`, `${media.fps} fps`);
    dlgInfo.textContent = bits.join("  ·  ");
  }

  function openFramesDialog() {
    if (!local.media) { ctx.flash("Load a source first.", { level: "warn" }); return; }
    updateDialogInfo();
    estimateImages();
    if (!framesDialog.open) framesDialog.showModal();
  }

  async function confirmFramesDialog() {
    framesDialog.close();
    await extractFrames();
  }

  async function refreshLanding() {
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
    } catch { /* the list is a convenience; a failure just leaves it empty */ }
  }

  function updateLanding() {
    // The front door is for when nothing is open at all. Once a project exists you are
    // past it -- show the editor even before its first frame has finished decoding.
    const empty = !local.media && !ctx.state.project;
    panel.classList.toggle("stage-panel--empty", empty);
    if (empty) refreshLanding();
  }

  // ── helpers ──────────────────────────────────────────────────────────

  function field(label, control) {
    return el("div", { class: "field" }, el("label", {}, label), control);
  }

  function slider(min, max, step, value = min) {
    const input = el("input", { type: "range", min, max, step, value });
    const out = el("output", {}, String(value));
    input.addEventListener("input", () => { out.textContent = input.value; });
    return { input, output: out, root: el("div", { class: "slider" }, input, out),
             get value() { return parseFloat(input.value); },
             set value(v) { input.value = v; out.textContent = Number(v).toFixed(2); } };
  }

  const current = () => local.rig && local.rig.cameras[local.selected];

  // ── canvas ───────────────────────────────────────────────────────────

  function paintLayer() {
    const layer = local.paint.layer;
    if (layer && layer.width === canvas.width && layer.height === canvas.height) return layer;
    const fresh = document.createElement("canvas");
    fresh.width = canvas.width; fresh.height = canvas.height;
    if (layer) fresh.getContext("2d").drawImage(layer, 0, 0, fresh.width, fresh.height);
    local.paint.layer = fresh;
    return fresh;
  }

  function draw() {
    const W = canvas.width, H = canvas.height;
    context2d.clearRect(0, 0, W, H);

    if (local.image) context2d.drawImage(local.image, 0, 0, W, H);
    else {
      context2d.fillStyle = "#08090b"; context2d.fillRect(0, 0, W, H);
      context2d.fillStyle = "#4b535e"; context2d.font = "16px sans-serif";
      context2d.textAlign = "center";
      context2d.fillText("Load a 360 video or still to begin", W / 2, H / 2);
      context2d.textAlign = "left";
    }

    if (local.paint.layer) {
      context2d.save();
      context2d.globalAlpha = 0.55;
      context2d.drawImage(local.paint.layer, 0, 0, W, H);
      context2d.restore();
    }

    context2d.strokeStyle = "rgba(255,255,255,.2)"; context2d.lineWidth = 1;
    context2d.beginPath(); context2d.moveTo(0, H / 2); context2d.lineTo(W, H / 2);
    context2d.stroke();
    context2d.fillStyle = "rgba(255,255,255,.38)"; context2d.font = "11px sans-serif";
    for (let yaw = -180; yaw <= 180; yaw += 45) {
      const x = (yaw + 180) / 360 * W;
      context2d.beginPath(); context2d.moveTo(x, H / 2 - 5); context2d.lineTo(x, H / 2 + 5);
      context2d.stroke();
      context2d.fillText(`${yaw}°`, x + 3, H / 2 - 8);
    }

    if (local.nadir > 0) {
      const y = (90 + local.nadir) / 180 * H;
      context2d.fillStyle = "rgba(255,140,26,.18)";
      context2d.fillRect(0, y, W, H - y);
      context2d.strokeStyle = "rgba(255,140,26,.7)";
      context2d.setLineDash([6, 4]);
      context2d.beginPath(); context2d.moveTo(0, y); context2d.lineTo(W, y);
      context2d.stroke(); context2d.setLineDash([]);
    }

    if (!local.rig) return;
    local.rig.cameras.forEach((camera, index) => {
      if (!camera.enabled) return;
      const points = footprint(camera, local.rig.orientation, W, H);
      const colour = colourOf(index);
      const selected = index === local.selected;

      for (const offset of [-W, 0, W]) {
        context2d.beginPath();
        points.forEach((point, i) => {
          const x = point.x + offset;
          i ? context2d.lineTo(x, point.y) : context2d.moveTo(x, point.y);
        });
        context2d.closePath();
        context2d.fillStyle = colour + (selected ? "38" : "18");
        context2d.fill();
        context2d.strokeStyle = colour;
        context2d.lineWidth = selected ? 2.5 : 1.2;
        context2d.stroke();
      }

      const yaw = camera.yaw + local.rig.orientation.yaw;
      const pitch = camera.pitch + local.rig.orientation.pitch;
      const cx = ((((yaw + 180) % 360) + 360) % 360) / 360 * W;
      const cy = (90 - pitch) / 180 * H;
      context2d.fillStyle = colour;
      context2d.beginPath(); context2d.arc(cx, cy, selected ? 5 : 3.5, 0, 7);
      context2d.fill();
      context2d.font = `${selected ? "bold " : ""}12px sans-serif`;
      context2d.fillText(camera.name, cx + 9, cy - 8);
    });
  }

  function fitCanvas() {
    const aspect = local.media && local.media.aspect ? local.media.aspect : 2;
    const width = 1600;
    const height = Math.round(width / aspect);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height;
    }
    draw();
  }

  function canvasAngles(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      yaw: (event.clientX - rect.left) / rect.width * 360 - 180,
      pitch: 90 - (event.clientY - rect.top) / rect.height * 180,
    };
  }

  function canvasPixel(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / rect.width * canvas.width,
      y: (event.clientY - rect.top) / rect.height * canvas.height,
    };
  }

  function paintStroke(from, to) {
    const layer = paintLayer();
    const paint = layer.getContext("2d");
    paint.globalCompositeOperation =
      local.paint.mode === "erase" ? "destination-out" : "source-over";
    paint.strokeStyle = "#ff8c1a";
    paint.lineWidth = local.paint.brush;
    paint.lineCap = "round"; paint.lineJoin = "round";
    for (const offset of [0, -canvas.width, canvas.width]) {
      paint.beginPath();
      paint.moveTo(from.x + offset, from.y);
      paint.lineTo(to.x + offset, to.y);
      paint.stroke();
    }
    local.paint.dirty = true;
  }

  canvas.addEventListener("mousedown", (event) => {
    if (local.paint.mode) {
      local.paint.drawing = true;
      local.paint.last = canvasPixel(event);
      paintStroke(local.paint.last, local.paint.last);
      draw();
      return;
    }
    if (!local.rig) return;
    const at = canvasAngles(event);
    let best = -1, bestDistance = 1e9;
    local.rig.cameras.forEach((camera, index) => {
      if (!camera.enabled) return;
      const dy = Math.abs(((camera.yaw + local.rig.orientation.yaw - at.yaw + 540) % 360) - 180);
      const dp = Math.abs(camera.pitch + local.rig.orientation.pitch - at.pitch);
      const distance = Math.hypot(dy, dp);
      if (distance < bestDistance) { bestDistance = distance; best = index; }
    });
    if (best >= 0 && bestDistance < 60) {
      const camera = local.rig.cameras[best];
      // Remember where on the footprint the grab landed, so the camera moves with the
      // cursor by that offset rather than snapping its centre under the pointer.
      local.dragOffset = {
        yaw: ((at.yaw - (camera.yaw + local.rig.orientation.yaw) + 540) % 360) - 180,
        pitch: at.pitch - (camera.pitch + local.rig.orientation.pitch),
      };
      local.dragging = best;
      canvas.classList.add("is-dragging");
      select(best);
    }
  });

  window.addEventListener("mousemove", (event) => {
    if (local.paint.drawing) {
      const at = canvasPixel(event);
      paintStroke(local.paint.last, at);
      local.paint.last = at;
      draw();
      return;
    }
    if (local.dragging === null || !local.rig) return;
    const at = canvasAngles(event);
    const camera = local.rig.cameras[local.dragging];
    const offset = local.dragOffset || { yaw: 0, pitch: 0 };
    let yaw = at.yaw - local.rig.orientation.yaw - offset.yaw;
    yaw = ((yaw + 180) % 360 + 360) % 360 - 180;   // keep it in [-180, 180]
    camera.yaw = Math.round(yaw);
    camera.pitch = Math.max(-90, Math.min(90,
      Math.round(at.pitch - local.rig.orientation.pitch - offset.pitch)));
    renderCameras(); draw();
  });

  window.addEventListener("mouseup", () => {
    if (local.paint.drawing) {
      local.paint.drawing = false;
      syncPaintedOccluder();
      return;
    }
    if (local.dragging === null) return;
    local.dragging = null;
    canvas.classList.remove("is-dragging");
    refresh(); previewCamera();
  });

  // ── rig list ─────────────────────────────────────────────────────────

  function renderCameras() {
    if (!local.rig) return;
    camList.replaceChildren(...local.rig.cameras.map((camera, index) => {
      const share = camera.name in local.coverage
        ? local.coverage[camera.name]
        : occlusionFraction(camera, local.rig.orientation, local.nadir);

      const toggle = el("input", { type: "checkbox", checked: camera.enabled });
      toggle.addEventListener("click", (event) => event.stopPropagation());
      toggle.addEventListener("change", () => { camera.enabled = toggle.checked; refresh(); });

      // The name is editable in place -- there is no separate Selected-camera panel.
      const nameInput = el("input", {
        type: "text", value: camera.name, title: "Rename camera",
        style: "flex:1;min-width:0;background:none;border:0;border-bottom:1px solid transparent;"
             + "color:inherit;font:inherit;padding:1px 0;",
      });
      // Not wired to select(): selecting rebuilds the list and would drop focus mid-edit.
      nameInput.addEventListener("click", (event) => event.stopPropagation());
      nameInput.addEventListener("focus", () => {
        nameInput.style.borderBottomColor = "var(--border-strong)";
      });
      nameInput.addEventListener("blur", () => { nameInput.style.borderBottomColor = "transparent"; });
      nameInput.addEventListener("change", () => {
        const next = nameInput.value.trim();
        if (next && next !== camera.name) { camera.name = next; refresh(); }
      });

      const row = el("div", {
        class: `cam${index === local.selected ? " sel" : ""}${camera.enabled ? "" : " off"}`,
        style: "display:flex;align-items:center;gap:8px;padding:5px 7px;border-radius:5px;cursor:pointer;"
             + (index === local.selected ? "background:var(--accent-tint);" : ""),
        onclick: () => select(index),
      },
        el("span", { style: `width:10px;height:10px;border-radius:3px;flex:0 0 auto;background:${colourOf(index)}` }),
        nameInput,
        el("span", { class: "actionbar__detail" },
           `${Math.round(camera.yaw)}° / ${Math.round(camera.pitch)}°`),
        share > 0.005
          ? el("span", { class: "badge badge--stale" }, `${Math.round(share * 100)}%`)
          : null,
        toggle);
      return row;
    }));
    rigSection.setNote(`${local.rig.cameras.filter((c) => c.enabled).length} of ${local.rig.cameras.length} on`);
  }

  // Global FOV/shape controls reflect the rig (all cameras share them), so sync them
  // whenever the rig is replaced -- a preset, a loaded project, boot.
  function syncRigControls() {
    const camera = local.rig && local.rig.cameras[0];
    if (!camera) return;
    camFov.value = camera.h_fov;
    const ratio = camera.h_fov / camera.v_fov;
    let best = "1.3333", diff = 9;
    for (const option of camShape.options) {
      const delta = Math.abs(parseFloat(option.value) - ratio);
      if (delta < diff) { diff = delta; best = option.value; }
    }
    camShape.value = best;
  }

  // ── masking ──────────────────────────────────────────────────────────

  function updateMaskFields() {
    const on = maskSky.checked;
    maskSkyMethod.parentElement.hidden = !on;
    maskSkyAngle.parentElement.hidden = !on || maskSkyMethod.value === "off";
  }

  function readMasking() {
    return {
      exclude_sky: maskSky.checked,
      sky_method: maskSkyMethod.value,
      sky_cone_angle: parseFloat(maskSkyAngle.value) || 30,
      backend: maskBackend.value,
      classes: maskClasses.value.split(",").map((s) => s.trim()).filter(Boolean),
      confidence: parseFloat(maskConfidence.value) || 0.25,
      dilate: parseInt(maskDilate.value, 10) || 0,
    };
  }

  function writeMasking(detect) {
    if (!detect) return;
    maskSky.checked = detect.exclude_sky !== false;
    maskSkyMethod.value = detect.sky_method || "auto";
    maskSkyAngle.value = detect.sky_cone_angle != null ? detect.sky_cone_angle : 30;
    maskBackend.value = detect.backend || "sam2.1";
    maskClasses.value = (detect.classes || []).join(",");
    maskConfidence.value = detect.confidence != null ? detect.confidence : 0.25;
    maskDilate.value = detect.dilate != null ? detect.dilate : 6;
    updateMaskFields();
  }

  function applyGlobalFovShape() {
    if (!local.rig) return;
    const h = parseFloat(camFov.value) || 90;
    const ratio = parseFloat(camShape.value) || 4 / 3;
    for (const camera of local.rig.cameras) {
      camera.h_fov = h;
      camera.v_fov = Math.min(h / ratio, 179);
    }
    refresh();
  }

  function select(index) {
    local.selected = index;
    renderCameras(); draw(); previewCamera();
  }

  function addCamera() {
    if (!local.rig) return;
    local.rig.cameras.push({ name: `cam${local.rig.cameras.length + 1}`, yaw: 0, pitch: 0,
                             roll: 0, h_fov: 90, v_fov: 67.5, enabled: true });
    select(local.rig.cameras.length - 1); refresh();
  }

  function duplicateCamera() {
    const camera = current();
    if (!camera) return;
    const copy = structuredClone(camera);
    let n = 2;
    while (local.rig.cameras.some((c) => c.name === `${camera.name}_${n}`)) n++;
    copy.name = `${camera.name}_${n}`;
    local.rig.cameras.splice(local.selected + 1, 0, copy);
    select(local.selected + 1); refresh();
  }

  function removeCamera() {
    if (!local.rig || local.rig.cameras.length <= 1) {
      ctx.flash("A rig needs at least one camera.", { level: "warn" });
      return;
    }
    local.rig.cameras.splice(local.selected, 1);
    local.selected = Math.max(0, local.selected - 1);
    refresh(); previewCamera();
  }

  function applyPreset() {
    const chosen = local.presets[presetSelect.value];
    if (!chosen) return;
    const grade = local.rig ? local.rig.grade : null;
    local.rig = structuredClone(chosen);
    if (grade) local.rig.grade = grade;      // a preset changes cameras, not the look
    local.selected = 0;
    syncRigControls();
    refresh(); previewCamera();
  }

  function updatePresetButtons() {
    deletePresetBtn.disabled = !local.userPresets.has(presetSelect.value);
  }

  // Built-in and saved presets, grouped in the dropdown. `keep` is the option to
  // reselect after a rebuild -- the one just saved, or whatever was already showing.
  async function loadPresets(keep) {
    const data = await ctx.api.get("/api/presets");
    local.presets = data.presets;
    local.userPresets = new Set(data.user || []);
    const target = local.presets[keep] ? keep : (presetSelect.value || "ring");

    const builtin = el("optgroup", { label: "Built-in" });
    const saved = el("optgroup", { label: "Saved" });
    for (const name of Object.keys(data.presets)) {
      (local.userPresets.has(name) ? saved : builtin).append(el("option", { value: name }, name));
    }
    presetSelect.replaceChildren(builtin);
    if (saved.children.length) presetSelect.append(saved);
    presetSelect.value = local.presets[target] ? target : "ring";
    updatePresetButtons();
  }

  async function savePreset() {
    const name = presetName.value.trim();
    if (!name) { ctx.flash("Name the preset before saving.", { level: "warn" }); return; }
    readSettings();
    try {
      const data = await ctx.api.post("/api/preset/save", { name, rig: local.rig });
      local.presets = data.presets;
      local.userPresets = new Set(data.user || []);
      presetName.value = "";
      await loadPresets(name);
      ctx.flash(`Saved preset “${name}”.`);
    } catch (error) { ctx.report(error); }
  }

  async function deletePreset() {
    const name = presetSelect.value;
    if (!local.userPresets.has(name)) return;
    try {
      await ctx.api.post("/api/preset/delete", { name });
      await loadPresets("ring");
      ctx.flash(`Deleted preset “${name}”.`);
    } catch (error) { ctx.report(error); }
  }

  // ── grade ────────────────────────────────────────────────────────────

  function readGrade() {
    const grade = {};
    for (const [key, control] of Object.entries(gradeInputs)) grade[key] = control.value;
    return grade;
  }

  function writeGrade(grade) {
    for (const [key, spec] of Object.entries(GRADE_FIELDS)) {
      gradeInputs[key].value = grade && grade[key] !== undefined ? grade[key] : spec.neutral;
    }
    markGradeState();
  }

  function markGradeState() {
    const grade = readGrade();
    const identity = Object.entries(GRADE_FIELDS)
      .every(([key, spec]) => grade[key] === spec.neutral);
    image.setNote(identity ? "unchanged" : "graded");
    if (local.rig) local.rig.grade = grade;
  }

  let regradeBusy = false, regradePending = null;
  async function regrade(width) {
    if (!local.media) return;
    if (regradeBusy) { regradePending = width; return; }
    regradeBusy = true;
    try {
      const data = await ctx.api.post("/api/preview/grade",
                                      { grade: readGrade(), width });
      await new Promise((resolve) => {
        const img = new Image();
        img.onload = () => { local.image = img; draw(); resolve(); };
        img.onerror = resolve;
        img.src = data.url;
      });
    } catch (error) { ctx.report(error); }
    finally {
      regradeBusy = false;
      if (regradePending !== null) {
        const next = regradePending; regradePending = null; regrade(next);
      }
    }
  }

  for (const control of Object.values(gradeInputs)) {
    control.input.addEventListener("input", () => { markGradeState(); regrade(PROXY_WIDTH); });
    control.input.addEventListener("change", () => { refresh(); regrade(FULL_WIDTH); previewCamera(); });
  }

  async function autoGrade() {
    if (!local.media) { ctx.flash("Load a source first.", { level: "warn" }); return; }
    try {
      const data = await ctx.api.post("/api/grade/auto", {});
      writeGrade(data.grade);
      gradeNotes.innerHTML = data.notes.join("<br>");
      refresh(); regrade(FULL_WIDTH); previewCamera();
    } catch (error) { ctx.report(error); }
  }

  function resetGrade() {
    writeGrade(null); refresh(); regrade(FULL_WIDTH); previewCamera();
  }

  // ── occluder ─────────────────────────────────────────────────────────

  for (const [button, mode] of [[paintButton, "paint"], [eraseButton, "erase"]]) {
    button.addEventListener("click", () => {
      local.paint.mode = local.paint.mode === mode ? null : mode;
      paintButton.classList.toggle("btn--primary", local.paint.mode === "paint");
      eraseButton.classList.toggle("btn--primary", local.paint.mode === "erase");
      canvas.classList.toggle("is-painting", Boolean(local.paint.mode));
    });
  }

  function clearPaint() {
    local.paint.layer = null; local.paint.dirty = true;
    draw(); syncPaintedOccluder();
  }

  brushSlider.input.addEventListener("input", () => {
    local.paint.brush = brushSlider.value;
  });

  nadirSlider.addEventListener("input", () => {
    local.nadir = parseInt(nadirSlider.value, 10) || 0;
    occluder.setNote(local.nadir ? `below −${local.nadir}°` : "no cone");
    if (!local.rig) return;
    local.rig.occluders = (local.rig.occluders || []).filter((o) => o.type !== "nadir_cone");
    if (local.nadir) local.rig.occluders.push({ type: "nadir_cone", angle: local.nadir });
    renderCameras(); draw();
  });
  nadirSlider.addEventListener("change", refreshCoverage);

  let paintTimer = null;
  function syncPaintedOccluder() {
    if (!local.paint.dirty) return;
    clearTimeout(paintTimer);
    paintTimer = setTimeout(async () => {
      local.paint.dirty = false;
      const layer = paintLayer();
      const out = document.createElement("canvas");
      out.width = layer.width; out.height = layer.height;
      const octx = out.getContext("2d");
      octx.fillStyle = "#fff"; octx.fillRect(0, 0, out.width, out.height);
      octx.filter = "brightness(0)";        // painted colour becomes black = ignored
      octx.drawImage(layer, 0, 0);
      octx.filter = "none";
      try {
        const data = await ctx.api.post("/api/mask/paint",
                                        { image: out.toDataURL("image/png") });
        local.rig.occluders = (local.rig.occluders || [])
          .filter((o) => o.type !== "equirect_mask");
        if (data.path) local.rig.occluders.push({ type: "equirect_mask", path: data.path });
        await refreshCoverage();
      } catch (error) { ctx.report(error); }
    }, 350);
  }

  async function refreshCoverage() {
    if (!local.rig || !(local.rig.occluders || []).length) {
      local.coverage = {}; renderCameras(); return;
    }
    try {
      const data = await ctx.api.post("/api/mask/coverage", {
        rig: local.rig,
        source_width: local.media ? local.media.width : 0,
        source_height: local.media ? local.media.height : 0,
      });
      local.coverage = data.coverage || {};
      renderCameras();
    } catch (error) { ctx.report(error); }
  }

  // ── source ───────────────────────────────────────────────────────────

  async function browse() {
    try {
      const paths = await ctx.api.pick("open", "Select a 360 video or still", "media",
                                       local.media ? local.media.path : "");
      if (!paths.length) return;
      pathField.value = paths[0];
      await loadMedia();
      // Loading a source opens the extract prompt, unless this project already has frames.
      if (!local.frames.length) openFramesDialog();
    } catch (error) { ctx.report(error); }
  }

  // ── segments ─────────────────────────────────────────────────────────

  const SEG_SHOW = {
    off: [],
    duration: ["duration"],
    "motion-distance": ["meters", "speed"],
    "motion-count": ["count"],
    gpx: ["meters"],
  };
  const SEG_HINT = {
    off: "The whole clip becomes one project.",
    duration: "Cut every N seconds.",
    "motion-distance": "Estimates forward travel from the video (needs the ML extra). "
      + "Average speed turns motion into approximate metres.",
    "motion-count": "Splits into equal-travel pieces from video motion (needs ML). "
      + "No speed needed — distances are approximate.",
    gpx: "Cuts by true metres along a <clip>.gpx track placed beside the video.",
  };

  function updateSegFields() {
    const show = new Set(SEG_SHOW[segMode.value] || []);
    for (const [key, node] of Object.entries(segFields)) node.hidden = !show.has(key);
    segHint.textContent = SEG_HINT[segMode.value] || "";
    segCreateBtn.hidden = segMode.value === "off";
  }

  async function createSegments() {
    if (!local.media) { ctx.flash("Load a source first.", { level: "warn" }); return; }
    const mode = segMode.value;
    const payload = { path: local.media.path };
    if (mode === "duration") {
      payload.mode = "duration"; payload.seconds = parseFloat(segSeconds.value) || 60;
    } else if (mode === "motion-distance") {
      const speed = parseFloat(segSpeed.value) || 0;
      if (!speed) { ctx.flash("Enter an average speed for metre-based segments.",
                              { level: "warn" }); return; }
      payload.mode = "motion";
      payload.meters = parseFloat(segMeters.value) || 500;
      payload.speed_kph = speed;
    } else if (mode === "motion-count") {
      payload.mode = "motion"; payload.count = parseInt(segCount.value, 10) || 2;
    } else if (mode === "gpx") {
      payload.mode = "gpx"; payload.meters = parseFloat(segMeters.value) || 500;
    } else { return; }

    segCreateBtn.disabled = true;
    segCreateBtn.querySelector("span").textContent = "Analysing…";
    try {
      const { segments: made } = await ctx.api.post("/api/segment", payload);
      renderSegments(made);
      ctx.flash(`Created ${made.length} segment project${made.length === 1 ? "" : "s"}.`);
    } catch (error) { ctx.report(error); }
    finally {
      segCreateBtn.disabled = false;
      segCreateBtn.querySelector("span").textContent = "Create segments";
    }
  }

  function renderSegments(made) {
    segResults.replaceChildren();
    for (const seg of made) {
      const span = `${seg.start.toFixed(1)}–${seg.end.toFixed(1)}s`;
      const dist = seg.distance != null
        ? `  ·  ${seg.approximate ? "≈" : ""}${Math.round(seg.distance)} m` : "";
      segResults.append(el("button", {
        class: "landing__recent-item", type: "button", title: seg.root,
        onclick: () => ctx.openRecent(seg.root),
      },
        el("span", { class: "landing__recent-name" }, seg.name),
        el("span", { class: "landing__recent-path" }, span + dist)));
    }
  }

  async function loadMedia() {
    const path = pathField.value.trim();
    if (!path) return;
    try {
      const data = await ctx.api.post("/api/preview", {
        path, time: parseFloat(timeSlider.value) || 0, grade: readGrade(),
      });
      local.media = data.media;
      ctx.setSource(data.media);
      timeSlider.max = data.media.is_video
        ? Math.max(data.media.duration - 0.1, 0) : 0;

      // Opening a source is opening its project. This resolves the output folder too,
      // so there is nothing for the user to pick.
      await ensureProject(data.media.path);

      const img = new Image();
      img.onload = () => { local.image = img; fitCanvas(); };
      img.src = data.url;

      updateLanding();
      refresh(); previewCamera(); refreshCoverage();
      // If this project already has extracted frames, switch the canvas to them and set
      // the button to Generate cameras; otherwise the preview stays and it reads Extract.
      await refreshFrames();
    } catch (error) { ctx.report(error); }
  }

  function samePath(a, b) {
    const norm = (p) => String(p).replace(/\\/g, "/").toLowerCase().replace(/\/+$/, "");
    return norm(a) === norm(b);
  }

  async function ensureProject(path) {
    // Create a project in a folder beside the video, or resume one already there. Skip
    // if the open project already owns this source, so re-seeking does not re-create it.
    const project = ctx.state.project;
    if (project && (project.sources || []).some((s) => samePath(s, path))) return;
    try {
      const { project: ensured } = await ctx.api.post("/api/project/for-source", {
        path, rig: local.rig,
        frames: { mode: frameMode.value, value: parseFloat(frameValue.value) || 2 },
        output: { mask_mode: maskMode.value },
      });
      ctx.applyProject(ensured, { keepMedia: true });
    } catch (error) { ctx.report(error); }
  }

  timeSlider.addEventListener("input", () => {
    if (local.frames.length) {
      const i = Math.round(parseFloat(timeSlider.value) || 0);
      timeLabel.textContent = `frame ${i + 1} / ${local.frames.length}`;
    } else {
      timeLabel.textContent = `${(parseFloat(timeSlider.value) || 0).toFixed(1)}s`;
    }
  });
  timeSlider.addEventListener("change", () => {
    if (local.frames.length) loadFrame(Math.round(parseFloat(timeSlider.value) || 0));
    else loadMedia();
  });

  let previewTimer = null;
  function previewCamera() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
      const camera = current();
      if (!local.media || !camera || !camera.enabled) {
        previewImage.style.display = "none"; return;
      }
      try {
        const data = await ctx.api.post("/api/camera-preview", {
          path: local.media.path, rig: local.rig, camera: camera.name,
          time: parseFloat(timeSlider.value) || 0, width: 420,
        });
        previewImage.src = data.url;
        previewImage.style.display = "block";
      } catch { previewImage.style.display = "none"; }
    }, 200);
  }

  // ── settings and extraction ──────────────────────────────────────────

  function readSettings() {
    if (!local.rig) return;
    local.rig.grade = readGrade();
    local.rig.output.format = outFormat.value;
    local.rig.output.quality = parseInt(outQuality.value, 10) || 2;
    local.rig.output.auto = true;
    local.rig.orientation.yaw = parseFloat(orientYaw.value) || 0;
    local.rig.orientation.pitch = parseFloat(orientPitch.value) || 0;
  }

  function estimateImages() {
    if (!local.media || !local.rig) { estimate.textContent = ""; return; }
    const enabled = local.rig.cameras.filter((c) => c.enabled).length;
    const mode = frameMode.value, value = parseFloat(frameValue.value) || 1;
    let frames = 1;
    if (local.media.is_video) {
      if (mode === "fps" || mode === "sharp") frames = Math.max(Math.floor(local.media.duration * value), 1);
      else if (mode === "every") frames = Math.max(Math.floor(local.media.frame_count / value), 1);
      else frames = local.media.frame_count;
    }
    estimate.textContent = `~${formatCount(frames)} frames × ${enabled} cameras = `
      + `~${formatCount(frames * enabled)} images`
      + (mode === "sharp" ? " (sharpest in each window)" : "");
  }

  let validateTimer = null;
  function refresh(revalidate = true) {
    readSettings();
    renderCameras(); draw(); estimateImages();
    if (!revalidate) return;
    ctx.autosave();  // settings changed -> fold it into project.json (debounced)
    clearTimeout(validateTimer);
    validateTimer = setTimeout(async () => {
      try {
        const data = await ctx.api.post("/api/rig/validate", {
          rig: local.rig, source_width: local.media ? local.media.width : 0,
        });
        local.sizes = data.sizes || {};
      } catch (error) { /* shown when the user acts on it */ }
    }, 220);
  }

  // Capture is two stages sharing one button: first extract equirect frames into the
  // working set, then -- once they exist -- project them through the rig into camera
  // tiles. The primary action label follows which step is next.
  async function runCapture() {
    if (!local.media) { ctx.flash("Load a source first.", { level: "warn" }); return; }
    readSettings();
    if (!local.frames.length) return openFramesDialog();
    return generateCameras();
  }

  async function extractFrames() {
    try {
      await ctx.api.post("/api/frames/extract", {
        mode: frameMode.value, value: parseFloat(frameValue.value) || 2,
      });
      ctx.pokeJobs();
    } catch (error) { ctx.report(error); }
  }

  async function generateCameras() {
    try {
      await ctx.api.post("/api/cameras/generate", { rig: local.rig });
      ctx.pokeJobs();
    } catch (error) { ctx.report(error); }
  }

  async function refreshFrames() {
    try {
      const { clip, frames } = await ctx.api.get("/api/frames/list");
      local.clip = clip;
      local.frames = frames || [];
      actionBar.setPrimaryLabel(local.frames.length ? "Generate cameras" : "Extract frames");
      if (local.frames.length) {
        timeSlider.max = local.frames.length - 1;
        loadFrame(Math.min(local.frameIndex, local.frames.length - 1));
      }
    } catch { /* leave the viewer as it is */ }
  }

  function loadFrame(index) {
    if (!local.frames.length || !local.clip) return;
    local.frameIndex = Math.max(0, Math.min(index, local.frames.length - 1));
    timeSlider.value = local.frameIndex;
    const img = new Image();
    img.onload = () => { local.image = img; fitCanvas(); };
    img.src = `/frames/${local.clip}/${local.frames[local.frameIndex]}`;
  }

  // FOV/shape are global: a change rewrites every camera, not just the selected one.
  camFov.addEventListener("input", applyGlobalFovShape);
  camFov.addEventListener("change", previewCamera);
  camShape.addEventListener("change", () => { applyGlobalFovShape(); previewCamera(); });
  for (const control of [outFormat, outQuality, orientYaw, orientPitch, frameValue]) {
    control.addEventListener("change", () => refresh());
  }
  frameMode.addEventListener("change", () => {
    frameValue.disabled = frameMode.value === "all";
    frameValue.value = frameMode.value === "every" ? 10 : 2;
    refresh();
  });
  window.addEventListener("resize", fitCanvas);

  // ── stage interface ──────────────────────────────────────────────────

  (async function init() {
    try {
      await loadPresets("ring");
      local.rig = structuredClone(local.presets.ring);
      outFormat.value = local.rig.output.format;
      outQuality.value = local.rig.output.quality;
      writeGrade(local.rig.grade);
      syncRigControls();
      refresh(false);
      fitCanvas();
      updateLanding();
      updateSegFields();
      updateMaskFields();
    } catch (error) { ctx.report(error); }
  })();

  let lastCaptureState = null;
  return {
    panel,
    onJobs: (job) => {
      actionBar.render(job);
      // When a capture job finishes, the working set changed: reload the frame list so
      // the button flips Extract frames -> Generate cameras and the viewer updates.
      if (job && job.state === "done" && lastCaptureState === "running") refreshFrames();
      lastCaptureState = job ? job.state : null;
    },
    onEnter: () => { updateLanding(); refreshFrames(); fitCanvas(); },
    applyProject(project, { keepMedia } = {}) {
      if (!project) return;
      local.rig = project.rig;
      local.selected = 0;
      outFormat.value = local.rig.output.format;
      outQuality.value = local.rig.output.quality;
      orientYaw.value = local.rig.orientation.yaw;
      orientPitch.value = local.rig.orientation.pitch;
      writeGrade(local.rig.grade);
      frameMode.value = project.frames.mode;
      frameValue.value = project.frames.value;
      maskMode.value = project.output.mask_mode;
      writeMasking(project.detect);
      outDir.value = project.root;

      const cone = (local.rig.occluders || []).find((o) => o.type === "nadir_cone");
      local.nadir = cone ? cone.angle : 0;
      nadirSlider.value = local.nadir;
      occluder.setNote(local.nadir ? `below −${local.nadir}°` : "no cone");

      syncRigControls();
      refresh(); refreshCoverage(); updateLanding();
      if (!keepMedia && project.sources && project.sources.length) {
        pathField.value = project.sources[0];
        loadMedia();
      }
    },
    projectPayload() {
      readSettings();
      return {
        rig: local.rig,
        sources: local.media ? [local.media.path] : [],
        frames: { mode: frameMode.value, value: parseFloat(frameValue.value) || 2 },
        output: { mask_mode: maskMode.value },
        detect: readMasking(),
      };
    },
  };
}
