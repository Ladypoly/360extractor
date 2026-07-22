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
    rig: null, presets: {}, selected: 0, media: null,
    nadir: 0, dragging: null, image: null,
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
  const camList = el("div", {});
  rigSection.body.append(
    el("div", { class: "field" }, presetSelect,
      el("button", { class: "btn", type: "button", style: "flex:0 0 auto",
                     onclick: applyPreset }, "Use")),
    camList,
    el("div", { class: "field", style: "margin-bottom:0" },
      el("button", { class: "btn btn--ghost", type: "button", onclick: addCamera }, "Add"),
      el("button", { class: "btn btn--ghost", type: "button", onclick: duplicateCamera }, "Duplicate"),
      el("button", { class: "btn btn--ghost", type: "button", onclick: removeCamera }, "Remove")));

  const editor = InspectorSection("Selected camera", { id: "cap-camera" });
  const camName = el("input", { type: "text" });
  const camYaw = slider(-180, 180, 1);
  const camPitch = slider(-90, 90, 1);
  const camFov = slider(20, 150, 1);
  const camShape = el("select", {},
    ...[["1.3333", "4:3"], ["1.5", "3:2"], ["1.7778", "16:9"], ["1", "square"]]
      .map(([value, label]) => el("option", { value }, label)));
  const camSize = el("p", { class: "hint" });
  editor.body.append(
    field("name", camName), field("yaw", camYaw.root), field("pitch", camPitch.root),
    field("fov", camFov.root), field("shape", camShape), camSize);

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
  const outDir = el("input", { type: "text", placeholder: "chosen with the source" });
  const outFormat = el("select", {}, el("option", {}, "jpg"), el("option", {}, "png"));
  const outQuality = el("input", { type: "number", value: 2, min: 1, max: 31 });
  const frameMode = el("select", {},
    ...[["sharp", "sharpest per second"], ["fps", "every N per second"],
        ["every", "every Nth frame"], ["all", "all frames"]]
      .map(([value, label]) => el("option", { value }, label)));
  const frameValue = el("input", { type: "number", value: 2, step: 0.5, min: 0.1 });
  const estimate = el("p", { class: "hint" });
  output.body.append(
    el("div", { class: "field" }, el("label", {}, "folder"), outDir,
      el("button", { class: "btn btn--ghost", type: "button", style: "flex:0 0 auto",
                     onclick: browseOut, html: "…" })),
    el("div", { class: "pair" }, field("format", outFormat), field("quality", outQuality)),
    field("frames", frameMode), field("rate", frameValue), estimate);

  const orientation = InspectorSection("Rig orientation", { id: "cap-orient", open: false });
  const orientYaw = el("input", { type: "number", value: 0, step: 1 });
  const orientPitch = el("input", { type: "number", value: 0, step: 1 });
  orientation.body.append(
    el("div", { class: "pair" }, field("yaw", orientYaw), field("pitch", orientPitch)),
    el("p", { class: "hint" }, "Levels a tilted mount without editing every camera."));

  const previewSection = InspectorSection("Camera preview", { id: "cap-preview" });
  const previewImage = el("img", { style: "width:100%;border-radius:5px;display:none" });
  previewSection.body.append(previewImage);

  for (const part of [source, rigSection, editor, image, occluder, output,
                      orientation, previewSection]) {
    inspector.append(part.section);
  }

  // ── action bar ───────────────────────────────────────────────────────
  const actionBar = StageActionBar({
    primaryLabel: "Extract Frames",
    onPrimary: extract,
    onCancel: () => ctx.api.jobs.cancel("capture").then(ctx.pokeJobs),
  });

  const panel = el("div", { class: "stage-panel" }, workspace, inspector, actionBar.bar);

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
    camera.yaw = Math.round(at.yaw - local.rig.orientation.yaw);
    camera.pitch = Math.max(-90, Math.min(90,
      Math.round(at.pitch - local.rig.orientation.pitch)));
    renderCameras(); renderEditor(); draw();
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

      const row = el("div", {
        class: `cam${index === local.selected ? " sel" : ""}${camera.enabled ? "" : " off"}`,
        style: "display:flex;align-items:center;gap:8px;padding:5px 7px;border-radius:5px;cursor:pointer;"
             + (index === local.selected ? "background:var(--accent-tint);" : ""),
        onclick: () => select(index),
      },
        el("span", { style: `width:10px;height:10px;border-radius:3px;background:${colourOf(index)}` }),
        el("span", { style: "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" },
           camera.name),
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

  function renderEditor() {
    const camera = current();
    if (!camera) return;
    camName.value = camera.name;
    camYaw.value = camera.yaw;
    camPitch.value = camera.pitch;
    camFov.value = camera.h_fov;

    const ratio = camera.h_fov / camera.v_fov;
    let best = "1.3333", diff = 9;
    for (const option of camShape.options) {
      const delta = Math.abs(parseFloat(option.value) - ratio);
      if (delta < diff) { diff = delta; best = option.value; }
    }
    camShape.value = best;

    const size = local.sizes[camera.name];
    camSize.textContent = size
      ? `writes ${size[0]}×${size[1]} — native detail for a ${Math.round(camera.h_fov)}° view`
      : "";
  }

  function select(index) {
    local.selected = index;
    renderCameras(); renderEditor(); draw(); previewCamera();
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
    const grade = local.rig ? local.rig.grade : null;
    local.rig = structuredClone(local.presets[presetSelect.value]);
    if (grade) local.rig.grade = grade;      // a preset changes cameras, not the look
    local.selected = 0;
    refresh(); previewCamera();
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
    } catch (error) { ctx.report(error); }
  }

  async function browseOut() {
    try {
      const paths = await ctx.api.pick("directory", "Choose an output folder", "media",
                                       outDir.value);
      if (paths.length) outDir.value = paths[0];
    } catch (error) { ctx.report(error); }
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

      if (!outDir.value) {
        const separator = data.media.path.includes("\\") ? "\\" : "/";
        outDir.value = data.media.path.slice(0, data.media.path.lastIndexOf(separator))
          + separator + "dataset";
      }

      const img = new Image();
      img.onload = () => { local.image = img; fitCanvas(); };
      img.src = data.url;

      refresh(); previewCamera(); refreshCoverage();
    } catch (error) { ctx.report(error); }
  }

  timeSlider.addEventListener("input", () => {
    timeLabel.textContent = `${(parseFloat(timeSlider.value) || 0).toFixed(1)}s`;
  });
  timeSlider.addEventListener("change", loadMedia);

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
    renderCameras(); renderEditor(); draw(); estimateImages();
    if (!revalidate) return;
    clearTimeout(validateTimer);
    validateTimer = setTimeout(async () => {
      try {
        const data = await ctx.api.post("/api/rig/validate", {
          rig: local.rig, source_width: local.media ? local.media.width : 0,
        });
        local.sizes = data.sizes || {};
        renderEditor();
      } catch (error) { /* shown when the user acts on it */ }
    }, 220);
  }

  async function extract() {
    if (!local.media) { ctx.flash("Load a source first.", { level: "warn" }); return; }
    readSettings();
    const root = outDir.value.trim();
    if (!root) { ctx.flash("Choose an output folder.", { level: "warn" }); return; }
    try {
      await ctx.api.post("/api/extract", {
        sources: [local.media.path], rig: local.rig,
        mode: frameMode.value, value: parseFloat(frameValue.value) || 2,
        output_dir: root, resume: true, mask_mode: maskMode.value,
      });
      ctx.pokeJobs();
    } catch (error) { ctx.report(error); }
  }

  camName.addEventListener("change", () => {
    const camera = current(); if (camera) { camera.name = camName.value; refresh(); }
  });
  for (const [control, key] of [[camYaw, "yaw"], [camPitch, "pitch"], [camFov, "h_fov"]]) {
    control.input.addEventListener("input", () => {
      const camera = current(); if (!camera) return;
      camera[key] = control.value;
      if (key === "h_fov") {
        camera.v_fov = Math.min(control.value / (parseFloat(camShape.value) || 4 / 3), 179);
      }
      refresh();
    });
    control.input.addEventListener("change", previewCamera);
  }
  camShape.addEventListener("change", () => {
    const camera = current(); if (!camera) return;
    camera.v_fov = Math.min(camera.h_fov / parseFloat(camShape.value), 179);
    refresh(); previewCamera();
  });
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
      const data = await ctx.api.get("/api/presets");
      local.presets = data.presets;
      for (const name of Object.keys(data.presets)) {
        presetSelect.append(el("option", { value: name }, name));
      }
      presetSelect.value = "ring";
      local.rig = structuredClone(data.presets.ring);
      outFormat.value = local.rig.output.format;
      outQuality.value = local.rig.output.quality;
      writeGrade(local.rig.grade);
      refresh(false);
      fitCanvas();
    } catch (error) { ctx.report(error); }
  })();

  return {
    panel,
    onJobs: (job) => actionBar.render(job),
    onEnter: () => fitCanvas(),
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
      outDir.value = project.root;

      const cone = (local.rig.occluders || []).find((o) => o.type === "nadir_cone");
      local.nadir = cone ? cone.angle : 0;
      nadirSlider.value = local.nadir;
      occluder.setNote(local.nadir ? `below −${local.nadir}°` : "no cone");

      refresh(); refreshCoverage();
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
      };
    },
  };
}
