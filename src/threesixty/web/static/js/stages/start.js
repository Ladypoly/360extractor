// Start: the project + import hub. Pick or resume a project, choose a source and how it
// is sampled, then Process -- which extracts the frames and hands off to Capture for the
// rig. Segments and masking join this tab in a following step.

import { InspectorSection, StageActionBar, el } from "../components.js";
import { icon } from "../icons.js";

export function StartStage(ctx) {
  const local = { media: null, imported: 0 };

  // ── workspace: the loaded panorama (with a scrubber) or the project hub ─
  const recentList = el("div", { class: "landing__recent" });
  const dropZone = el("div", { class: "landing__drop" },
    el("div", { class: "landing__icon", html: icon("camera", { size: 40 }) }),
    el("div", { class: "landing__title" }, "Load a 360° video to begin"),
    el("div", { class: "landing__hint" }, "Drag a video here, or use Browse in the panel"),
    el("p", { class: "landing__note" },
      "A project folder is created next to the video; opening the same video later resumes it."));

  // Once a source is loaded, the middle shows the panorama at the scrubbed time, with
  // the mask overlay tinted on top so a frame can be picked and checked before Process.
  const previewImg = el("img", { class: "start-preview__img" });
  const previewTime = el("input", { type: "range", min: 0, max: 0, step: 0.1, value: 0,
                                    style: "flex:1" });
  const previewLabel = el("span", { class: "actionbar__detail" }, "0.0s");
  const previewPane = el("div", { class: "start-preview", hidden: true },
    el("div", { class: "start-preview__frame" }, previewImg),
    el("div", { class: "log__bar" },
      el("span", {}, "frame"), previewTime, previewLabel));

  const workspace = el("div", { class: "start__workspace" },
    el("div", { class: "start__main" }, dropZone, previewPane),
    el("div", { class: "landing__side" },
      el("div", { class: "landing__side-title" }, "Recent projects"),
      recentList));

  previewTime.addEventListener("input", () => {
    previewLabel.textContent = `${(parseFloat(previewTime.value) || 0).toFixed(1)}s`;
  });
  previewTime.addEventListener("change", refreshPreview);

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
  // Cameras write two files: the stitched 360, and the raw one straight off both lenses.
  // The raw file has been resampled by nothing, so it is the better source -- as long as
  // we are told that is what it is, because it is 2:1 like a panorama and looks like one.
  const projection = el("select", { id: "start-projection" },
    ...[["equirect", "equirectangular (stitched 360)"],
        ["dfisheye", "dual fisheye (raw, both lenses)"],
        ["fisheye", "fisheye (one lens)"]]
      .map(([value, label]) => el("option", { value }, label)));
  const lensFov = el("input", { id: "start-lens-fov", type: "number", value: 190,
                               min: 1, max: 360, step: 1 });
  // The spec sheet rounds the lens figure and the difference shows up as a jump across
  // the stitch line, so it can be measured instead of guessed.
  const fitBtn = el("button", { id: "start-fit-fov", class: "btn", type: "button",
                                onclick: fitLensFov, title: "Measure it from the stitch",
                                html: "<span>Fit</span>" });
  const lensField = el("div", { class: "field" },
    el("label", {}, "lens FOV°"), lensFov, fitBtn);
  // Where the lenses are. A camera that gives each lens its own video stream (the
  // QooCam 8K) looks like an ordinary square video until you ask for the second track.
  const lensLayout = el("select", { id: "start-lens-layout" },
    ...[["sbs", "side by side in one frame"], ["streams", "one video stream per lens"]]
      .map(([value, label]) => el("option", { value }, label)));
  const layoutField = field("lenses", lensLayout);
  // Sensors are mounted however the body needed them; opposed is the common case.
  const lensRotate = el("select", { id: "start-lens-rotate" },
    ...[["0,0", "upright"], ["90,-90", "90° opposed (QooCam 8K)"],
        ["-90,90", "90° opposed, other way"], ["90,90", "90° both"],
        ["180,180", "180° both"]]
      .map(([value, label]) => el("option", { value }, label)));
  const rotateField = field("rotation", lensRotate);
  // The circular trim: v360 hands one lens over to the other at exactly 90°, which is
  // the softest, worst-stitched part of a fisheye. This cuts that rim out of training.
  const lensTrim = el("input", { id: "start-lens-trim", type: "number", value: 0,
                                 min: 0, max: 45, step: 1 });
  const trimField = field("trim edges°", lensTrim);
  const trimHint = el("p", { class: "hint" });
  // Which picture to show: the lenses as shot, or the panorama they project to.
  const sourceView = el("select", { id: "start-view" },
    ...[["lenses", "the lenses, side by side"], ["panorama", "the panorama they make"]]
      .map(([value, label]) => el("option", { value }, label)));
  const viewField = field("show", sourceView);
  source.body.append(
    el("div", { class: "field field--stack" }, pathField),
    el("div", { class: "field" },
      el("button", { class: "btn btn--primary", type: "button", onclick: browse,
                     html: `${icon("folder", { size: 14 })}<span>Browse…</span>` }),
      el("button", { class: "btn", type: "button", onclick: () => ctx.openProject(),
                     html: `${icon("layers", { size: 14 })}<span>Open project…</span>` })),
    field("projection", projection), layoutField, rotateField, lensField,
    trimField, trimHint, viewField,
    mediaInfo);
  for (const control of [projection, lensLayout, lensRotate, lensFov, lensTrim]) {
    control.addEventListener("change", () => {
      syncProjection(); updateMediaInfo(); ctx.autosave(); refreshPreview();
    });
  }
  // Start's choice is the app's choice: Capture places the rig on the same picture.
  sourceView.addEventListener("change", () => {
    ctx.setSourceView(sourceView.value);
    refreshPreview();
  });

  const framesSection = InspectorSection("Frames", { id: "start-frames" });
  const frameMode = el("select", { id: "start-frame-mode" },
    ...[["sharp", "sharpest frame every N seconds"], ["fps", "every N per second"],
        ["every", "every Nth frame"], ["all", "all frames"]]
      .map(([value, label]) => el("option", { value }, label)));
  const frameValue = el("input", { id: "start-frame-value", type: "number", value: 0.5,
                                   step: 0.1, min: 0.05 });
  const estimate = el("p", { class: "hint" });
  // The unit changes with the mode: `sharp` states a window in seconds, `fps` a rate.
  const frameValueLabel = el("label", {}, "seconds");
  const frameValueField = el("div", { class: "field" }, frameValueLabel, frameValue);
  // Re-decoding an 8K source takes tens of minutes, so a project that has already been
  // imported says so and Process leaves it alone unless this is ticked.
  const reextract = el("input", { type: "checkbox" });
  const importedNote = el("label", { class: "imported", hidden: true },
    el("span", { class: "imported__icon", html: icon("done", { size: 14 }) }),
    el("span", { class: "imported__text" }),
    el("span", { class: "imported__redo" }, reextract, "re-extract"));
  framesSection.body.append(
    importedNote, field("frames", frameMode), frameValueField, estimate);
  reextract.addEventListener("change", syncImported);
  frameMode.addEventListener("change", () => {
    frameValue.disabled = frameMode.value === "all";
    frameValue.value = { every: 10, fps: 2, sharp: 0.5 }[frameMode.value] ?? 2;
    syncFrameUnit();
    updateEstimate(); ctx.autosave();
  });

  function syncFrameUnit() {
    frameValueLabel.textContent =
      { sharp: "seconds", fps: "per second", every: "every Nth" }[frameMode.value] || "value";
    frameValueField.hidden = frameMode.value === "all";
  }
  frameValue.addEventListener("change", () => { updateEstimate(); ctx.autosave(); });

  // ── segments ─────────────────────────────────────────────────────────
  const segments = InspectorSection("Segments", { id: "start-segments", open: false });
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
    duration: field("seconds", segSeconds), meters: field("metres", segMeters),
    speed: field("avg km/h", segSpeed), count: field("segments", segCount),
  };
  const segHint = el("p", { class: "hint" });
  segments.body.append(
    field("split", segMode),
    segFields.duration, segFields.meters, segFields.speed, segFields.count,
    segHint,
    el("div", { class: "field", style: "margin-bottom:0" }, segCreateBtn),
    segResults);
  segMode.addEventListener("change", updateSegFields);

  // ── masking ──────────────────────────────────────────────────────────
  const masking = InspectorSection("Masking", { id: "start-masking", open: false });
  const maskBackend = el("select", {},
    ...[["sam-world", "YOLO-World + SAM (any class, incl. sky)"],
        ["yolo-world", "YOLO-World (any class, incl. sky)"],
        ["sam2.1", "YOLO + SAM 2.1 (COCO classes)"], ["yolo", "YOLO only (COCO classes)"]]
      .map(([value, label]) => el("option", { value }, label)));
  const maskClasses = el("input", { type: "text",
                                    value: "sky,person,car,bus,truck,motorcycle,bicycle" });
  const maskConfidence = el("input", { type: "number", min: 0.05, max: 0.95, step: 0.05, value: 0.1 });
  const maskDilate = el("input", { type: "number", min: 0, max: 40, step: 1, value: 6 });
  const previewMaskBtn = el("button", { class: "btn", type: "button", onclick: runMaskPreview,
    html: `${icon("inspect", { size: 14 })}<span>Preview masking</span>` });
  masking.body.append(
    field("detector", maskBackend), field("classes", maskClasses),
    el("div", { class: "pair" }, field("confidence", maskConfidence), field("grow", maskDilate)),
    el("p", { class: "hint" }, "Anything you name is masked out of the splat. A YOLO-World "
      + "detector takes any words (sky, trees, water); the COCO ones only know a fixed set. "
      + "Preview runs it on the current frame (first run downloads weights)."),
    el("div", { class: "field", style: "margin-bottom:0" }, previewMaskBtn));
  for (const control of [maskBackend, maskClasses, maskConfidence, maskDilate]) {
    control.addEventListener("change", () => ctx.autosave());
  }

  for (const part of [source, framesSection, segments, masking]) inspector.append(part.section);

  // ── process action ─────────────────────────────────────────────────────
  const actionBar = StageActionBar({
    primaryLabel: "Process",
    onPrimary: process,
    onCancel: () => ctx.api.jobs.cancel("start").then(ctx.pokeJobs),
  });

  const panel = el("div", { class: "stage-panel start-panel" },
                    workspace, inspector, actionBar.bar);

  // ── helpers ─────────────────────────────────────────────────────────────
  function field(label, control) {
    return el("div", { class: "field" }, el("label", {}, label), control);
  }

  function readSourceFormat() {
    const raw = projection.value !== "equirect";
    return {
      projection: projection.value || "equirect",
      lens_fov: parseFloat(lensFov.value) || 190,
      layout: projection.value === "dfisheye" ? lensLayout.value : "single",
      rotate: raw ? lensRotate.value.split(",").map(Number) : [],
      trim: raw ? (parseFloat(lensTrim.value) || 0) : 0,
    };
  }

  function writeSourceFormat(format) {
    projection.value = format.projection || "equirect";
    if (format.lens_fov) lensFov.value = format.lens_fov;
    if (format.layout && format.layout !== "single") lensLayout.value = format.layout;
    const rotate = format.rotate && format.rotate.length
      ? format.rotate.map((r) => String(Math.round(r))).join(",") : "0,0";
    lensRotate.value = [...lensRotate.options].some((o) => o.value === rotate)
      ? rotate : "0,0";
    lensTrim.value = format.trim || 0;
    syncProjection();
  }

  //: What the projection was last time, so a *change* can move the view with it
  //: without overriding a choice the user made afterwards.
  let lastProjection = null;

  function syncProjection() {
    const raw = projection.value !== "equirect";
    if (lastProjection !== null && lastProjection !== projection.value) {
      // Declaring raw footage means you want to look at it; going back to a stitched
      // panorama leaves nothing else to look at.
      ctx.setSourceView(raw ? "lenses" : "panorama");
    } else if (!raw && ctx.state.sourceView === "lenses") {
      ctx.setSourceView("panorama");
    }
    lastProjection = projection.value;
    lensField.hidden = !raw;
    rotateField.hidden = !raw;
    trimField.hidden = !raw;
    viewField.hidden = !raw;
    layoutField.hidden = projection.value !== "dfisheye";
    trimHint.hidden = !raw || !(parseFloat(lensTrim.value) > 0);
    const trim = parseFloat(lensTrim.value) || 0;
    trimHint.textContent = trim > 0
      ? `Ignores a ${trim}° band either side of the stitch line — and, because that `
        + "line runs through both poles, the sky straight up and the ground straight "
        + "down with it."
      : "";
  }

  // What the panorama will come out as. Mirrors source.py: two lenses share the width,
  // one spends all of it, and the result is rounded to a size an encoder will take.
  function equirectSize(media, format) {
    if (!media || format.projection === "equirect") return null;
    const merged = format.layout === "streams" && media.video_streams > 1
      ? media.width * 2 : media.width;
    const lens = format.projection === "dfisheye" ? merged / 2 : merged;
    const width = Math.round(360 * (lens / (format.lens_fov || 190)) / 4) * 4;
    return [width, width / 2];
  }

  function updateEstimate() {
    const media = local.media;
    if (!media) { estimate.textContent = ""; return; }
    const mode = frameMode.value, value = parseFloat(frameValue.value) || 1;
    let frames = 1;
    if (media.is_video) {
      if (mode === "sharp") frames = Math.max(Math.floor(media.duration / (value || 1)), 1);
      else if (mode === "fps") frames = Math.max(Math.floor(media.duration * value), 1);
      else if (mode === "every") frames = Math.max(Math.floor(media.frame_count / value), 1);
      else frames = media.frame_count;
    }
    estimate.textContent = `~${frames} frames`
      + (mode === "sharp" ? ` (sharpest in each ${value}s)` : "");
  }

  function updateMediaInfo() {
    const media = local.media;
    if (!media) { mediaInfo.textContent = ""; return; }
    const format = readSourceFormat();
    const bits = [`${media.width}×${media.height}`];
    if (media.video_streams > 1) bits.push(`${media.video_streams} video streams`);
    if (media.is_video) bits.push(`${media.duration.toFixed(1)}s`, `${media.fps} fps`);
    else bits.push("still");
    const equirect = equirectSize(media, format);
    if (equirect) bits.push(`→ ${equirect[0]}×${equirect[1]} equirectangular`);
    // Only meaningful for a source claiming to be a panorama already: a raw two-lens
    // file is 2:1 as well, and that says nothing about whether it is right.
    else if (!media.looks_equirectangular) bits.push("not 2:1 — geometry will be wrong");
    mediaInfo.textContent = bits.join("  ·  ");
    // The top bar carries the same warning, so it needs to know what this file is too.
    ctx.setSource(media, format);
  }

  async function fitLensFov() {
    if (!local.media) { ctx.flash("Load a source first.", { level: "warn" }); return; }
    if (projection.value !== "dfisheye") {
      ctx.flash("Fitting compares the two lenses, so it needs a dual-fisheye source.",
                { level: "warn" });
      return;
    }
    fitBtn.disabled = true;
    fitBtn.querySelector("span").textContent = "Measuring…";
    try {
      const result = await ctx.api.post("/api/source/fit",
        { path: local.media.path, source_format: readSourceFormat() });
      lensFov.value = result.lens_fov;
      const rotate = (result.rotate || []).map((r) => String(Math.round(r))).join(",");
      const known = [...lensRotate.options].some((o) => o.value === rotate);
      if (known) lensRotate.value = rotate;
      updateMediaInfo(); ctx.autosave(); refreshPreview();
      ctx.flash(`Lenses line up best at ${result.lens_fov}°`
        + (known && rotate !== "0,0" ? `, mounted ${rotate}°.` : ".")
        + " Check the panorama view — a mounting turned over stitches just as well.");
    } catch (error) { ctx.report(error); }
    finally {
      fitBtn.disabled = false;
      fitBtn.querySelector("span").textContent = "Fit";
    }
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
      let data = await ctx.api.post("/api/preview",
        { path, time: 0, source_format: readSourceFormat(), view: sourceView.value });

      // A file the container gives away -- two video streams is a lens each -- is set
      // up for the user rather than left to be discovered by a wrong-looking panorama.
      const suggested = data.media && data.media.suggested_source;
      if (suggested && projection.value === "equirect") {
        writeSourceFormat(suggested);
        // Raw footage is shown as it was shot, here and in Capture.
        ctx.setSourceView("lenses");
        ctx.flash(`Looks like ${suggested.projection === "dfisheye"
          ? "a raw two-lens file" : "a fisheye file"} — set the source to match. `
          + "The lens figures are a guess: press Fit to measure them, and check "
          + "the panorama view before processing.", { level: "info" });
        data = await ctx.api.post("/api/preview",
          { path, time: 0, source_format: readSourceFormat(), view: sourceView.value });
      }
      local.media = data.media;
      ctx.setSource(data.media, readSourceFormat());
      updateMediaInfo(); updateEstimate();

      // Show the panorama in the middle with a scrubber; hide the drop target.
      previewTime.max = data.media.is_video ? Math.max(data.media.duration - 0.1, 0) : 0;
      previewTime.value = 0;
      previewLabel.textContent = "0.0s";
      dropZone.hidden = true;
      previewPane.hidden = false;

      // Opening a source is opening its project, created beside the video. Loading is
      // not importing: nothing is decoded, masked or extracted until Process is pressed,
      // and the user stays here to set that up -- so no jump to the project's last stage.
      const { project } = await ctx.api.post("/api/project/for-source", {
        path: data.media.path,
        frames: { mode: frameMode.value, value: parseFloat(frameValue.value) || 2 },
        source_format: readSourceFormat(),
      });
      ctx.applyProject(project, { keepMedia: true, keepStage: true });
      refreshPreview();
    } catch (error) { ctx.report(error); }
  }

  // ── already imported ─────────────────────────────────────────────────
  async function refreshImported() {
    try {
      const { frames } = await ctx.api.get("/api/frames/list");
      local.imported = (frames || []).length;
    } catch { local.imported = 0; }
    syncImported();
  }

  function syncImported() {
    const count = local.imported || 0;
    importedNote.hidden = count === 0;
    if (count) {
      importedNote.querySelector(".imported__text").textContent =
        `${count} frames already extracted`;
    }
    // The sampling controls only mean something for a run that is going to happen.
    const skipping = count > 0 && !reextract.checked;
    frameMode.disabled = skipping;
    frameValue.disabled = skipping || frameMode.value === "all";
    actionBar.setPrimaryLabel(skipping ? "Continue" : "Process");
  }

  let processing = false, lastState = null;
  async function process() {
    if (!ctx.state.project) {
      ctx.flash("Load a source first.", { level: "warn" }); return;
    }
    // Already imported and not asked to redo it: this is a "carry on", not a re-decode.
    if (local.imported && !reextract.checked) { ctx.goTo("capture"); return; }
    if (!local.media) {
      ctx.flash("Load a source first.", { level: "warn" }); return;
    }
    try {
      await ctx.api.post("/api/frames/extract", {
        mode: frameMode.value, value: parseFloat(frameValue.value) || 2,
        source_format: readSourceFormat(),
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

  // ── segments logic ───────────────────────────────────────────────────
  const SEG_SHOW = {
    off: [], duration: ["duration"], "motion-distance": ["meters", "speed"],
    "motion-count": ["count"], gpx: ["meters"],
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
    const payload = { path: local.media.path, source_format: readSourceFormat() };
    if (mode === "duration") {
      payload.mode = "duration"; payload.seconds = parseFloat(segSeconds.value) || 60;
    } else if (mode === "motion-distance") {
      const speed = parseFloat(segSpeed.value) || 0;
      if (!speed) { ctx.flash("Enter an average speed for metre-based segments.",
                              { level: "warn" }); return; }
      payload.mode = "motion";
      payload.meters = parseFloat(segMeters.value) || 500; payload.speed_kph = speed;
    } else if (mode === "motion-count") {
      payload.mode = "motion"; payload.count = parseInt(segCount.value, 10) || 2;
    } else if (mode === "gpx") {
      payload.mode = "gpx"; payload.meters = parseFloat(segMeters.value) || 500;
    } else { return; }

    segCreateBtn.disabled = true;
    segCreateBtn.querySelector("span").textContent = "Analysing…";
    try {
      const { segments: made } = await ctx.api.post("/api/segment", payload);
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
      ctx.flash(`Created ${made.length} segment project${made.length === 1 ? "" : "s"}.`);
    } catch (error) { ctx.report(error); }
    finally {
      segCreateBtn.disabled = false;
      segCreateBtn.querySelector("span").textContent = "Create segments";
    }
  }

  // ── masking logic ────────────────────────────────────────────────────
  function readMasking() {
    return {
      backend: maskBackend.value,
      classes: maskClasses.value.split(",").map((s) => s.trim()).filter(Boolean),
      confidence: parseFloat(maskConfidence.value) || 0.1,
      dilate: parseInt(maskDilate.value, 10) || 0,
      exclude_sky: false,   // sky is handled as a detection class now, not the cone
    };
  }

  function writeMasking(detect) {
    if (!detect) return;
    maskBackend.value = detect.backend || "sam-world";
    maskClasses.value = (detect.classes || []).join(",");
    maskConfidence.value = detect.confidence != null ? detect.confidence : 0.1;
    maskDilate.value = detect.dilate != null ? detect.dilate : 6;
  }

  // The plain panorama at the scrubbed time. Detection is expensive, so the mask overlay
  // is on-demand behind the Preview button rather than on every scrub.
  let previewTimer = null;
  function refreshPreview() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
      if (!local.media) return;
      try {
        const { url } = await ctx.api.post("/api/preview",
          { path: local.media.path, time: parseFloat(previewTime.value) || 0,
            source_format: readSourceFormat(), view: sourceView.value });
        previewImg.src = url;
      } catch { /* keep the frame that is showing */ }
    }, 250);
  }

  // Run detection on the current frame, on demand, and tint the result over it.
  async function runMaskPreview() {
    if (!local.media) { ctx.flash("Load a source first.", { level: "warn" }); return; }
    previewMaskBtn.disabled = true;
    previewMaskBtn.querySelector("span").textContent = "Running…";
    try {
      const { url } = await ctx.api.post("/api/mask/preview", {
        path: local.media.path, time: parseFloat(previewTime.value) || 0,
        objects: true, detect: readMasking(), source_format: readSourceFormat(),
        view: sourceView.value,
      });
      previewImg.src = url;
    } catch (error) { ctx.report(error); }
    finally {
      previewMaskBtn.disabled = false;
      previewMaskBtn.querySelector("span").textContent = "Preview masking";
    }
  }

  updateSegFields();
  syncFrameUnit();
  syncProjection();

  return {
    panel,
    projectPayload: () => {
      const payload = {
        detect: readMasking(),
        frames: { mode: frameMode.value, value: parseFloat(frameValue.value) || 2 },
        source_format: readSourceFormat(),
      };
      if (local.media) payload.sources = [local.media.path];
      return payload;
    },
    onEnter: () => { refreshRecent(); refreshImported(); },
    onJobs: (_job, allJobs) => {
      const importing = allJobs.start;
      actionBar.render(importing);
      // When our Process (frame extraction) finishes, move on to the rig.
      if (processing && importing && importing.state === "done" && lastState === "running") {
        processing = false;
        reextract.checked = false;
        refreshImported();
        ctx.goTo("capture");
      }
      lastState = importing ? importing.state : null;
    },
    applyView(view) {
      sourceView.value = view;
    },
    applyProject(project) {
      if (!project) return;
      frameMode.value = project.frames.mode;
      frameValue.value = project.frames.value;
      syncFrameUnit();
      writeSourceFormat(project.source_format || {});
      sourceView.value = ctx.state.sourceView || "lenses";
      updateMediaInfo();
      writeMasking(project.detect);
      updateEstimate();
      if (project.sources && project.sources.length) pathField.value = project.sources[0];
      reextract.checked = false;
      refreshImported();
    },
  };
}
