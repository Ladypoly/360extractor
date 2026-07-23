// Start: the project + import hub. Pick or resume a project, choose a source and how it
// is sampled, then Process -- which extracts the frames and hands off to Capture for the
// rig. Segments and masking join this tab in a following step.

import { InspectorSection, StageActionBar, el } from "../components.js";
import { icon } from "../icons.js";

export function StartStage(ctx) {
  const local = { media: null };

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
  const previewMaskBtn = el("button", { class: "btn", type: "button", onclick: runMaskPreview,
    html: `${icon("inspect", { size: 14 })}<span>Preview masking</span>` });
  masking.body.append(
    el("div", { class: "field" }, el("label", {}, "exclude sky"), maskSky),
    field("sky via", maskSkyMethod), field("cone °", maskSkyAngle),
    el("p", { class: "hint" }, "Sky only seeds floaters, so it is masked by default; the "
      + "cone masks everything above the angle. Red on the panorama is what gets masked out."),
    field("objects", maskBackend), field("classes", maskClasses),
    el("div", { class: "pair" }, field("confidence", maskConfidence), field("grow", maskDilate)),
    el("p", { class: "hint" }, "Object detection needs the ML extra. Preview runs it on the "
      + "current frame (the first run downloads model weights)."),
    el("div", { class: "field", style: "margin-bottom:0" }, previewMaskBtn));
  for (const control of [maskBackend, maskClasses, maskConfidence, maskDilate]) {
    control.addEventListener("change", () => ctx.autosave());
  }
  for (const control of [maskSkyMethod, maskSkyAngle]) {
    control.addEventListener("change", () => { ctx.autosave(); refreshPreview(); });
  }
  maskSkyAngle.addEventListener("input", refreshPreview);
  maskSky.addEventListener("change", () => { updateMaskFields(); ctx.autosave(); refreshPreview(); });

  for (const part of [source, framesSection, segments, masking]) inspector.append(part.section);

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

      // Show the panorama in the middle with a scrubber; hide the drop target.
      previewTime.max = data.media.is_video ? Math.max(data.media.duration - 0.1, 0) : 0;
      previewTime.value = 0;
      previewLabel.textContent = "0.0s";
      dropZone.hidden = true;
      previewPane.hidden = false;

      // Opening a source is opening its project, created beside the video.
      const { project } = await ctx.api.post("/api/project/for-source", {
        path: data.media.path,
        frames: { mode: frameMode.value, value: parseFloat(frameValue.value) || 2 },
      });
      ctx.applyProject(project, { keepMedia: true });
      refreshPreview();
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
    const payload = { path: local.media.path };
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
  function updateMaskFields() {
    const on = maskSky.checked;
    maskSkyMethod.parentElement.hidden = !on;
    maskSkyAngle.parentElement.hidden = !on || maskSkyMethod.value === "off";
  }

  function readMasking() {
    return {
      exclude_sky: maskSky.checked, sky_method: maskSkyMethod.value,
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

  // The panorama at the scrubbed time, with the mask tinted on top when sky exclusion is
  // on, so a frame can be picked and the masking checked before Process runs.
  let previewTimer = null;
  function refreshPreview() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
      if (!local.media) return;
      const time = parseFloat(previewTime.value) || 0;
      const masked = maskSky.checked && maskSkyMethod.value !== "off";
      try {
        const { url } = masked
          ? await ctx.api.post("/api/mask/preview", {
              path: local.media.path, time,
              sky_cone_angle: parseFloat(maskSkyAngle.value) || 30 })
          : await ctx.api.post("/api/preview", { path: local.media.path, time });
        previewImg.src = url;
      } catch { /* keep the frame that is showing */ }
    }, 250);
  }

  // Run the full masking (sky + object detection) on the current frame, on demand.
  async function runMaskPreview() {
    if (!local.media) { ctx.flash("Load a source first.", { level: "warn" }); return; }
    const skyOn = maskSky.checked && maskSkyMethod.value !== "off";
    previewMaskBtn.disabled = true;
    previewMaskBtn.querySelector("span").textContent = "Running…";
    try {
      const { url } = await ctx.api.post("/api/mask/preview", {
        path: local.media.path, time: parseFloat(previewTime.value) || 0,
        objects: true, detect: readMasking(),
        sky_cone_angle: skyOn ? (parseFloat(maskSkyAngle.value) || 30) : null,
      });
      previewImg.src = url;
    } catch (error) { ctx.report(error); }
    finally {
      previewMaskBtn.disabled = false;
      previewMaskBtn.querySelector("span").textContent = "Preview masking";
    }
  }

  updateSegFields();
  updateMaskFields();

  return {
    panel,
    projectPayload: () => {
      const payload = {
        detect: readMasking(),
        frames: { mode: frameMode.value, value: parseFloat(frameValue.value) || 2 },
      };
      if (local.media) payload.sources = [local.media.path];
      return payload;
    },
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
    applyProject(project) {
      if (!project) return;
      frameMode.value = project.frames.mode;
      frameValue.value = project.frames.value;
      writeMasking(project.detect);
      updateEstimate();
      if (project.sources && project.sources.length) pathField.value = project.sources[0];
    },
  };
}
