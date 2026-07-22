# 360extract — what it is, what the UI is now, and what it should become

Written after a real end-to-end run exposed the seams: extraction and masking are solid,
but the UI stops being a pipeline halfway through and hands you a shell script.

---

## 1. What the app does

Turns 360° equirectangular footage into a trained Gaussian splat, without the camera
operator or the vehicle ending up in the result.

```
 source video ─▶ grade ─▶ rig ─▶ EXTRACT ─▶ MASK ─▶ COLMAP ─▶ BRUSH ─▶ CLEAN ─▶ view
   8K equirect            tiles   occluders   poses    splat   floaters
```

| stage | what happens | state today |
|---|---|---|
| **Grade** | exposure / contrast / saturation on the panorama, once, before the split | done |
| **Rig** | which directions to extract, at what fov and size | done |
| **Extract** | one ffmpeg decode fans out to every camera via `v360` | done |
| **Mask (static)** | nadir cone or painted region → per-camera masks, no ML | done |
| **Mask (dynamic)** | YOLO / SAM 2.1 on the tiles, reconciled through the sphere | done, CLI + partial UI |
| **Reconstruct** | COLMAP with our exact rig and intrinsics | **CLI only — UI writes a script** |
| **Train** | Brush CLI | **not in the UI at all** |
| **Clean** | delete gaussians where the rig itself was | done, UI panel exists |
| **View** | inspect the result | **missing** |

Verified end to end on real footage (8K village drive): 240/240 images registered by
COLMAP at 0.54 px, rig honoured exactly, Brush trained 318k gaussians, cleanup removed
5,942 along the path.

---

## 2. The UI as it stands

Three tabs, one shared sidebar, one **global footer**.

```
┌───────────────────────────────────────────────────────────────────────┐
│ 360extract  [ path……… ] [Browse…]  8K · 55s   │ Capture │Refine│Export│
├──────────────────────────────────────────┬────────────────────────────┤
│                                          │  Project   open/save/snap  │
│                                          │  Rig       preset, cameras │
│         PANORAMA CANVAS                  │  Image     grade sliders   │
│         camera footprints                │  Occluder  cone + brush    │
│         drag to aim, paint occluders     │  Output    format, frames  │
│                                          │  Orientation               │
│                                          │  Preview   selected camera │
├──────────────────────────────────────────┴────────────────────────────┤
│ frame ▭▬▭  12.4s   [progress bar]  status…      [Cancel] [Extract]    │  ← always
└───────────────────────────────────────────────────────────────────────┘
```

- **Capture** — everything above. Works well; the canvas interaction is the strong part.
- **Refine** — detection settings + a frame reviewer with the mask tinted red.
- **Export** — writes `rig_config.json` and a `run_colmap.sh`, plus the splat-cleanup panel.

### What's wrong with it

**The footer is global but only serves Capture.** `Extract` and `Cancel` sit there on
every tab, so on Refine you are offered the wrong verb entirely. The user reported
exactly this.

**Detection looks dead while it runs.** Reproduced: the job *does* start and *does*
report `"detecting in c00 (40 frames)"`, but

```json
{"state": "running", "message": "detecting in c00 (40 frames)", "fraction": 0.0}
```

`fraction` never moves — `mask/dynamic.py:run()` only emits text, never a completion
ratio. So the progress bar sits at 0 for the entire run. On CPU torch with SAM 2.1 that
is minutes of apparently nothing, so you click again and get **"something is already
running"**, which is technically correct and completely unhelpful.

**One `Job` for the whole session.** Extraction and detection share a single state
machine, so they cannot run independently, cannot be told apart in the UI, and a stuck
one blocks everything. Detection also has no cancel, though extraction does.

**The pipeline stops at Export.** The tab writes a shell script and tells you to run it.
COLMAP is installed and discovered — `doctor` finds it — but the UI never invokes it.
Brush is installed too and is not referenced anywhere in the UI.

**No way to see the result.** The whole point is a splat, and the app cannot show you one.

**Tabs are not stages.** "Export" mixes writing a COLMAP project with cleaning a trained
splat, which are the first and last things you do.

---

## 3. Proposed restructure

Make the tabs *be* the pipeline, in order, each owning its own action and its own
progress. Nothing global that belongs to one stage.

```
┌───────────────────────────────────────────────────────────────────────┐
│ 360extract   project: village ▾                                       │
│  ① Capture   ② Refine   ③ Reconstruct   ④ Train   ⑤ Inspect          │
│     done        done        running…       ·          ·               │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│                      STAGE WORKSPACE                                  │
│         (canvas / filmstrip / log / viewer, per stage)                │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│  [████████████░░░░░░]  62%  registering frame 24/40      [Cancel] [▶] │  ← this stage
└───────────────────────────────────────────────────────────────────────┘
```

The tab strip doubles as pipeline status, reading from the project's existing stage
fingerprints — `done` / `stale` / `pending` are already computed in `project.py`, they
are simply not surfaced.

### ① Capture
As today. Move `Extract` out of the global footer into this stage's action bar.

### ② Refine
As today, plus: real fraction, a cancel button, and a running log. Detection should
report `frames done / frames total`, which `dynamic.run` already knows.

### ③ Reconstruct *(replaces Export)*
Runs COLMAP rather than describing it.

- Buttons for the four steps, and one **Run all**:
  `feature_extractor → rig_configurator → matcher → mapper`
- Live log pane with the ffmpeg-style streaming we already do for extraction.
- Progress from COLMAP's own stdout — it prints `Registering image #N (M)`, which is
  enough for a bar.
- Optional `--gpx` geo-registration step (`model_aligner`), which is also what makes the
  cleanup radius mean metres.
- Afterwards, show what came back: images registered, points, mean reprojection error,
  and the within-frame rig spread — the number that proves the rig was honoured.
- Keep "write the script" as a secondary button for people who want to run it themselves.

### ④ Train *(new)*
Drives `brush_app.exe`.

- Settings: total steps, max resolution, export every / export path, eval split.
- Live log and a step counter; Brush prints per-step progress.
- `--with-viewer` as an option for people who want Brush's own window.
- On completion, hand the exported `.ply` straight to ⑤ and to the cleanup.

### ⑤ Inspect *(new)*
Two things that belong together:

- **SuperSplat viewer.** `C:\Tools\supersplat\dist` is a built PlayCanvas web app
  (v2.28.1, `index.html` + `index.js`). Serve it from our own server and embed it in an
  iframe, loading the trained `.ply`. No new dependency, no packaging.
- **Splat cleanup**, moved here from Export, because it operates on a *trained* splat.
  Radius / floor / up, dry-run counts, and — the useful part — load `_removed.ply` into
  the viewer to see exactly what would go before committing.

---

## 4. What has to change underneath

**A real job model.** Replace the single `Session.job` with a small registry keyed by
stage, each with `state / fraction / message / log / cancel`. Requirements:

- more than one kind of job, distinguishable by the UI;
- every long job reports a *fraction*, not just text;
- every long job can be cancelled, including detection;
- a log buffer per job, so the Reconstruct and Train tabs have something to show.

**A process runner.** COLMAP and Brush are external binaries with long lifetimes and
useful stdout. We already stream ffmpeg's `-progress`; generalise that into one helper
that runs a command, tails its output into a job's log, and parses progress with a
supplied regex.

**Stage gating.** Reconstruct should be unavailable until extraction exists, Train until
a sparse model exists, Inspect until a `.ply` exists. The project already knows all of
this; the UI just needs to read it and say *why* something is disabled.

**Binary discovery for Brush**, matching what `colmap/locate.py` does for COLMAP and
`ffmpeg.py` does for ffmpeg — probe candidates, report in `doctor`, allow an override.

---

## 5. Bugs to fix on the way

1. **Detection progress fraction never advances** (`mask/dynamic.py:run`) — the reported
   symptom. Emit `done/total` per camera and per frame.
2. **No cancel for detection** — `job.cancel` exists and is only honoured by extraction.
3. **"Something is already running" with nothing on screen** — once fraction and log are
   live this stops being mysterious, but the message should also name the running stage
   and offer to cancel it.
4. **Global footer shows Extract on every tab** — move actions into their stage.
5. **Shared job blocks unrelated work** — per-stage jobs.

---

## 6. Suggested order

1. Job registry + process runner + detection fraction and cancel. *(fixes the reported
   bug and unblocks everything else)*
2. Restructure the tabs and move the footer actions into stages.
3. Reconstruct tab running COLMAP for real.
4. Train tab running Brush.
5. Inspect tab: SuperSplat embed, then move cleanup into it.

Steps 1 and 2 are worth doing together — the second is mostly moving markup, and the
first is what makes any of the later tabs honest about what they are doing.
