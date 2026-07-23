# 360extract — handoff

For a fresh instance picking this up. Read this, then `README.md` for the user-facing
story and `docs/ui-redesign.md` + `docs/360extract UI Redesign.pdf` for where the UI is
going. Everything below is current as of commit `0b4bee0`.

---

## What this is

A standalone, open-source (Apache-2.0) Python tool that turns 360° equirectangular video
into a trained 3D Gaussian splat, keeping the camera operator and the vehicle out of the
result.

```
video ─▶ grade ─▶ rig ─▶ EXTRACT ─▶ MASK ─▶ COLMAP ─▶ BRUSH ─▶ CLEAN ─▶ view
```

It is a CLI **and** a local web UI. It is **not** a Blender add-on, despite living inside
Blender's extensions directory — that location was the user's choice; Blender ignores it
because there is no `blender_manifest.toml`. Do not add `bpy` code.

The whole pipeline has been run end to end on real footage (8K village drive, roof-mounted
car): COLMAP registered 240/240 images at 0.54 px, Brush trained 318k gaussians, cleanup
removed the trail of floaters along the driven path. This is not theoretical.

- **Repo:** https://github.com/Ladypoly/360extractor (public, `main` is pushed and clean)
- **Package name:** `threesixty` (import), `360extract` / `threesixty` (CLI entry points)
- **611 tests passing** at the last commit.

---

## This machine (Windows) — the tools are all installed

Discovery is deliberate everywhere: never trust PATH order, probe candidates, report in
`doctor`. Reuse the existing resolvers rather than shelling out to a bare name.

| tool | where | resolver |
|---|---|---|
| ffmpeg **7.1** | `C:\Users\Elin\Documents\ffmpeg-7.1-full_build\bin` | `ffmpeg.py:resolve_ffmpeg` |
| COLMAP **4.1.1 CUDA** | `C:\Tools\colmap\colmap-x64-windows-cuda\bin\colmap.exe` | `colmap/locate.py:resolve` |
| Brush **0.3.0** | `C:\Tools\brush-app-x86_64-pc-windows-msvc\brush_app.exe` | `tools.py:find_brush` |
| SuperSplat **2.28.1** (built) | `C:\Tools\supersplat\dist` | `tools.py:find_supersplat` |

**PATH hazard, still true:** `which ffmpeg` resolves to miniconda's 4.3.1, which has no
usable `v360`. Always resolve explicitly.

Env / flag overrides exist for all four: `--ffmpeg` / `THREESIXTY_FFMPEG`, `--colmap` /
`THREESIXTY_COLMAP`, `THREESIXTY_BRUSH`, `THREESIXTY_SUPERSPLAT`.

### Running things

- venv is at `.venv/` in the repo root. Use `.\.venv\Scripts\python.exe`.
- **Run the UI:** `360extract-ui.bat --no-browser --port 8420 --project <dir>` (or without
  `--project`). It creates the venv on first run and refuses to open onto a broken ffmpeg.
- **Tests:** `.venv\Scripts\python.exe -m pytest -q`. Markers: `ffmpeg` (needs ffmpeg —
  most integration tests), `slow` (real YOLO/SAM, downloads weights), `colmap` (real
  COLMAP), `ui` (Playwright + Chromium). Skip heavy ones with `-m "not slow and not ui"`.
- **PowerShell tests take a long time**; run them backgrounded with `Start-Job` +
  `Wait-Job -Timeout`, not a foreground call that hits the 2-minute wall.

### A working scratch project already exists

`…\scratchpad\village\` (under the session temp dir, path in the shell) is a real project:
the user's 8K clip extracted to 6 cameras, a full COLMAP `sparse/0`, and a trained
`splat/*.ply`. Point the UI at it to exercise Reconstruct/Train/Inspect without waiting
for an extraction. The user's source videos are on the Desktop (`Q360_*.mp4`).

---

## ⚠️ The one trap that has bitten twice — Windows text encoding

**Never round-trip a UTF-8 file through PowerShell `Get-Content`/`Set-Content`.** It reads
as Windows-1252 and rewrites double-encoded, turning every `°`, `—`, `×` into mojibake.
It has corrupted the README once and a checker script once. Edit text files with the
Edit/Write tools or with Python (`pathlib` + explicit `encoding="utf-8"`). All JSON readers
already use `utf-8-sig` so hand-edited files with a BOM still load — keep that.

---

## Architecture

### Backend (`src/threesixty/`)

Pure library, no framework. Each module is a plain function of its inputs so it can be
tested without a server.

```
rig.py          Camera/Rig/Grade/Output/Orientation dataclasses; native_size, output_size
plan.py         one ffmpeg decode → N cameras; FrameSelection; camera_size; filter graph
extract.py      runs the passes, resume markers, sidecar masks
ffmpeg.py       binary discovery + probe (the pattern every other resolver copies)
sharp.py        blurdetect-based sharpest-frame selection
autograde.py    measures a frame, proposes a conservative Grade  (tuned for 360, see below)
project.py      project.json: settings + per-stage fingerprints (done/stale detection)
mask/
  geometric.py  static occluders → per-camera masks via the *same* v360 call as the image
  apply.py      sidecar | skip | burn | none
  dynamic.py    ML detection over tiles; on_fraction + should_cancel hooks
  fuse.py       sphere reconciliation, hand-written inverse projection (see gotcha)
  ml.py         YOLO / SAM 2.1 backends behind one interface  ([ml] extra)
colmap/
  export.py     rig_config.json, intrinsics, command list; the OpenCV/mirror conversion
  model.py      read/write sparse models (bin + txt); C = -R^T t
  locate.py     COLMAP discovery + version gate (rig support needs 3.12+)
  batches.py    long-capture chunking + model_merger  (UNVERIFIED — see below)
splat/
  ply.py        header-preserving INRIA PLY read/filter/write; refuses packed variant
  clean.py      sphere-per-rig-position removal; --floor; trajectory_from_model
gps.py          EXIF + GPX → per-frame coords → geo_registration.txt
tools.py        Brush + SuperSplat discovery; survey() for the System dialog
web/
  server.py     stdlib http.server; ONE handler, routes by path. No framework.
  jobs.py       Job + JobRegistry: per-stage state/fraction/log/cancel
  runner.py     run external binaries, stream stdout to a job log, parse progress by regex
  stages.py     the work behind Reconstruct/Train/Inspect + readiness() gating
  picker.py     native file dialog in a subprocess (Tk needs the main thread)
```

### Frontend (`src/threesixty/web/static/`)

Plain ES modules, no build step, no framework (the redesign brief forbids introducing
one). Served by `server.py` with correct MIME types — a module served as `text/plain` is
refused by the browser, which is why `CONTENT_TYPES` exists.

```
index.html          a shell: #panels mount point + a <dialog>. Everything else is built in JS.
css/tokens.css      every colour/space/radius. Do not scatter literals.
css/app.css         the shell, controls, log, metrics, steps, empty states
js/app.js           boot: top bar, pipeline, one job poller for all stages
js/pipeline.js      the five-stage navigator (tablist); reads readiness + project fingerprints
js/components.js    el(), StageActionBar, LogViewer, MetricStrip, InspectorSection, StatusBadge, EmptyState
js/api.js           fetch wrapper; ApiError models 409 AlreadyRunning with runningStage
js/icons.js         36 vendored Lucide icons, inline SVG (regenerate: scripts/vendor_icons.py)
js/geometry.js      spherical math for the footprint overlay — ALSO run under node by a test
js/stages/*.js      capture, refine, reconstruct, train, inspect — one module each
```

**Stage module contract:** each returns `{ panel, onJobs?(job, allJobs), onEnter?() }`.
`app.js` mounts `panel`, calls `onEnter` when the stage is shown, and feeds `onJobs` the
polled job snapshot. `context` (passed in) gives `api`, `state`, `goTo`, `report`, `flash`,
`pokeJobs`, `setSource`, `applyProject`.

**Job model is the spine.** Every long task runs through `web/jobs.py:Job.start`, reports a
real `fraction` (a bar at 0 reads as a hang — that was the original bug report), streams a
log, and survives navigation. `runner.py:run` / `run_steps` wrap external binaries; give it
a `ProgressPattern` regex and it drives the bar from the tool's own stdout (COLMAP's
`Registering image #N (M)`, Brush's step counter).

---

## Non-obvious things that are easy to get wrong

These each cost real debugging. Do not "simplify" them away.

1. **v360 yaw is clamped to [-180, 180].** Ring layouts naturally produce 240; it errors.
   `rig.wrap180` handles it — everything passes through `normalized_cameras()`.
2. **Our equirect world is mirrored** (det = -1) relative to a right-handed world; yaw
   increases clockwise from above. Verified against ffmpeg. It cancels because COLMAP only
   ever gets *relative* rotations — see `colmap/export.py:camera_axes_in_rig` and the long
   comment there. Do not "fix" the determinant.
3. **COLMAP is OpenCV: +X right, +Y down, +Z forward.** The middle row of `cam_from_rig` is
   `-up`. Getting the sign wrong flips every camera and looks plausible. Tested.
4. **COLMAP groups frames by matching filenames across camera folders.** That is why the
   `brush` layout writes `<clip>/<camera>/00001.jpg` (camera in the folder, not the name).
   The `flat` layout keeps the old distinct names. Break this and the rig silently never
   forms — you get one camera per frame.
5. **`--Mapper.ba_refine_sensor_from_rig 0`** is what makes the rig honoured. Proof it
   worked: within-frame camera spread comes back as *exactly 0* (all cameras share an
   optical centre). `clean.py` surfaces that as a sanity metric.
6. **v360's `alpha_mask` does NOT mark the field of view.** It was the obvious way to do
   sphere fusion and it is wrong (measured: 60° camera, true coverage 0.05, alpha 0.72).
   The inverse projection in `fuse.py` is hand-written for this reason; pinned against
   v360's forward projection by `test_fuse.py::TestRoundTrip`.
7. **Mask polarity: white keeps, black ignores.** Brush, COLMAP and nerfstudio agree
   (read from Brush's source). Backwards trains on *only* the car.
8. **The `burn` mask filter needs `shortest=1`** (the `-loop 1` mask never ends) and must
   run in **RGB** (YUV multiply tints the whole frame). Both cost a hang / wrong colours.
9. **Auto-grade is calibrated for 360, not photos.** 30–40 % of an equirect frame is sky,
   so the 1–99 percentile span is always wide; contrast is judged on the 10–90 band and
   **only ever raised**. Corrections too small to see snap to neutral — because a
   non-identity grade changes the rig fingerprint and marks the dataset stale.
10. **Static occluders happen BEFORE extraction, dynamic AFTER.** Static are rigid to the
    rig (paint once, same region every frame); dynamic detectors are trained on rectilinear
    images and need the tiles. Different pipeline positions on purpose.

---

## Where the UI redesign stands

`docs/360extract UI Redesign.pdf` is the full 18-page brief; `docs/ui-redesign.md` is the
plan. Done: five-stage shell, per-stage jobs with fraction/log/cancel, pipeline navigator
with seven states (icon **and** colour), Reconstruct running real COLMAP, Train driving
Brush, Inspect embedding SuperSplat, design tokens, vendored Lucide icons, 25 Playwright
tests (`tests/test_ui.py`).

**Not yet done from the brief** (roughly in priority order):

- **Resizable panels with persisted sizes.** Sections collapse and remember (localStorage),
  but inspector/log widths are fixed. Brief wants drag-resize stored locally.
- **Empty states are incomplete** — Refine and Inspect have them; Capture/Reconstruct/Train
  do not. Brief wants an intentional empty state per stage.
- **Accessibility pass** beyond roles + keyboard tablist nav (brief asks for an a11y check
  of the shell and primary controls).
- **Refine filmstrip silently caps at 200 thumbnails** — should paginate or say so.
- **Container queries** for inspector-panel adaptation (brief mentions; only media queries
  exist now).
- **SuperSplat "Comparison" view** (removed points overlaid on the result) — only Result
  and Removed are wired. Also: the viewer loads a `.ply` via a `?load=` URL param; confirm
  that param name against the installed SuperSplat build (it worked in manual testing but
  is not covered by an automated test because the iframe is cross-document).

**Known stubs / unverified:**

- **`colmap/batches.py` (long-capture merge) is UNVERIFIED.** It generates commands and a
  `model_merger` chain but has never been run against a real capture. Labelled as such in
  its own output. GLOMAP drifts on rigid rigs (issue #229) — target COLMAP.
- **Temporal/inpainting nadir fill** was designed (M6) and never built; needs GPS odometry.
- **Auto-detecting the car occluder from temporal variance** was tried and rejected —
  textureless moving road is indistinguishable from a static car by variance. Would need
  optical flow. Not in the codebase; a note for if the user asks again.

---

## Working style the user expects

Established over the whole build; worth continuing:

- **Verify against the real tool, not your reading of its format.** Every format claim here
  (Brush PLY, COLMAP rig config, mask polarity) was checked by running the actual binary or
  reading its source. Doing this found real bugs — a wrong focal length COLMAP accepted
  silently, the filename-grouping failure.
- **Push back with a reason, don't just comply.** The user picked a plain-sphere cleanup;
  the honest answer was "that eats the road" with a measured table, and a `--floor` flag.
- **Report failures plainly.** When a run's output dir was empty, or a test was wrong rather
  than the code, say so. Several "failures" here were test artifacts chased to ground rather
  than assertions loosened.
- **Tests pin the non-obvious behaviour and are mutation-checked** where it matters (the
  yaw-sign flip, the up-vector inversion). A test that would pass with the bug present is
  worthless.
- **Commit style:** long, explanatory bodies that say *why*. End with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Commit/push only
  when the work is verified; `main` is the working branch and is pushed.
- Global response style is terse ("smart caveman" — drop articles/filler). Prose in code
  and docs stays normal.

---

## First moves for the next instance

1. `git log --oneline` and skim `README.md` → this file → `docs/ui-redesign.md`.
2. Start the UI against the `village` scratch project and click through all five stages to
   see the current state for yourself.
3. `pytest -m "not slow and not ui"` for a fast green baseline (~2 min); add `ui` when you
   touch the frontend, `colmap`/`slow` when you touch those paths.
4. Pick up the redesign remainder above, or whatever the user asks next.
