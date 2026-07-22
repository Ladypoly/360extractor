# 360extract

Turn 360° equirectangular footage into perspective image sets for photogrammetry and
3D Gaussian Splatting — with precise control over **which directions get extracted**, so the
person holding the camera or the car it was mounted on never reaches your dataset.

> **Status: pre-1.0 and under active development.** Extraction, masking and the rig editor
> work and are covered by 365 tests. Export to COLMAP and Brush is not built yet — see the
> milestone table at the bottom. Interfaces may still change without notice.

```bash
360extract rig new ring --count 8 -o rigs/ring8.json
360extract extract CLIP.mp4 --rig rigs/ring8.json --fps 2 -o dataset/
```

## How it works

The extraction engine is ffmpeg's `v360` filter. 360extract decodes the source **once** and fans
it out to every camera in a single pass:

```
[0:v]fps=2,split=8[s0]...[s7];
[s0]v360=e:rectilinear:yaw=0:pitch=-5:h_fov=90:v_fov=67.5:w=1920:h=1440[o0];
...
```

Running ffmpeg once per camera would pay the decode cost N times. There is a test
(`test_single_pass_matches_separate_runs_byte_for_byte`) asserting the batched output is
byte-identical to the naive version, so the optimisation can never silently change results.

## Requirements

- Python 3.10+
- ffmpeg **5.0 or newer**, with the `v360` filter

Most machines have several ffmpeg builds installed, and the first one on `PATH` is often an old
one. 360extract probes every candidate and picks the newest usable build rather than trusting
`PATH` order. Run `360extract doctor` to see exactly what it found and what it chose.

```bash
360extract doctor
```

Override discovery with `--ffmpeg /path/to/ffmpeg` or the `THREESIXTY_FFMPEG` environment
variable. When you name a binary explicitly it is used or the run fails — 360extract never
silently substitutes a different one.

## Install

```bash
pip install -e ".[dev]"     # dev install with tests
```

## The rig editor

Double-click **`360extract-ui.bat`** (Windows) or run `360extract ui`. It creates the
virtualenv on first run, checks ffmpeg, and opens a local page on `127.0.0.1:8360`.

**Browse…** opens a real file dialog. A browser can never hand a server a filesystem path —
`<input type=file>` deliberately hides it — so the dialog is raised by the app itself.

The panorama fills the window with every camera's footprint drawn on it, so you can see at a
glance whether the car hood or the person holding the stick falls inside a camera. Drag a
footprint to aim it; the camera list stays a quiet summary and editing happens in one panel
below it, rather than in a grid of inline number boxes. The selected camera's real extracted
view is previewed underneath.

The **occluder guide** slider shades everything below a chosen angle and reports what
percentage of each camera falls inside it — the number to watch when deciding whether to
re-aim a camera or drop it.

The overlay is not decoration: `tests/test_overlay_geometry.py` runs the shipped
`geometry.js` under node, then places markers just inside and just outside each drawn edge
and checks against real ffmpeg extractions. A UI that draws coverage it does not have is
worse than no UI, because it hides exactly the occluder you were trying to exclude.

Node is only needed to run those tests, never to use the tool.

## Projects

A project is one `project.json` at the root of the dataset it describes, beside `images/` and
`masks/`. That makes the folder self-describing: move it, hand it to someone else, come back in
a month, and the settings arrive with the pixels.

```bash
360extract project new dataset/ --source CLIP.mp4 --rig car-forward
360extract run dataset/                 # extract, then mask; skips what is current
360extract project show dataset/
```

It records **what has already been done**, and each stage stores a fingerprint of the settings
that produced it — so the tool distinguishes "already extracted" from "extracted, but you have
since changed the rig":

```
  stages:
    extract  stale  at 2026-07-22T14:02:20+00:00  images=100, cameras=5
             settings changed since this ran; re-run to update
    mask     pending
```

The fingerprints are per stage, so changing the detector does not force a re-extract, while
changing the rig does. Redoing a stage clears the later ones, which would otherwise claim to be
current while describing images that no longer exist.

Snapshots are cheap insurance before a big change — settings only, no images:

```bash
360extract project snapshot dataset/ --label before-retilt
360extract project snapshot dataset/ --restore before-retilt
```

The painted occluder lives in `assets/` inside the project, so it survives a reboot.

`360extract ui --project dataset/` opens straight into one.

## Rigs

A rig is a JSON file listing the cameras to extract. It is a plain, diffable artifact you can
version-control, share, and generate from scripts.

```json
{
  "version": 1,
  "name": "car-forward",
  "orientation": { "yaw": 0, "pitch": 0, "roll": 0 },
  "output": { "width": 1920, "height": 1440, "format": "jpg", "quality": 2, "interp": "line" },
  "cameras": [
    { "name": "fwd",   "yaw": 0,   "pitch": -5, "roll": 0, "h_fov": 90, "v_fov": 67.5, "enabled": true },
    { "name": "left",  "yaw": -90, "pitch": -5, "roll": 0, "h_fov": 90, "v_fov": 67.5, "enabled": true }
  ],
  "occluders": [{ "type": "nadir_cone", "angle": 40 }]
}
```

`orientation` is applied on top of every camera, which is how you level a tilted capture (a rig
bolted to a car roof at an angle) without editing each camera individually.

### Presets

| Preset | What it is for |
|---|---|
| `ring` | N cameras around the horizon. The photogrammetry workhorse. |
| `cube` | Six 90° faces. Complete spherical coverage, no overlap. |
| `dome` | Horizon ring + upper ring + zenith. Everything except the ground. |
| `car-forward` | Roof-mounted vehicle capture. Forward and sides, tilted down, no rear. |
| `handheld` | Walking capture on a stick, tilted up to keep the operator out of frame. |

```bash
360extract rig list
360extract rig show car-forward
360extract rig new dome --count 8 -o rigs/dome8.json
```

Presets are accepted anywhere a rig file is, so `--rig ring` works without writing a file first.

## Keeping the operator and the vehicle out of the dataset

This is the whole point of the tool, and there are two halves to it.

### Static occluders — the stick, the tripod, the car roof

Rigid relative to the rig, so they sit in the *same region of every single frame*. That is what
makes them cheap to deal with, and it is why this happens **before** extraction rather than
after.

The cheapest fix is rig layout: `dome` and `handheld` never point a camera at the ground, and
`car-forward` omits the rear where the mount usually is. Beyond that:

```bash
360extract extract CLIP.mp4 --rig car-forward --nadir 40 --mask sidecar -o dataset/
```

`--nadir 40` masks everything more than 40° below the horizon. For anything that is not a neat
cone — a hood, a mount arm, a wing mirror — paint it directly onto the panorama in the UI.

Paint once, and the same region is pushed through the *identical* `v360` call used for the
picture, so the per-camera mask is aligned pixel for pixel by construction. One render per
camera, reused for every frame, rather than one per frame.

`--mask` chooses what to do about it:

| mode | effect |
|---|---|
| `sidecar` | a mask beside every image. No pixels lost, the trainer decides. **Default.** |
| `skip` | drop cameras more than two thirds occluder — not worth extracting, let alone training |
| `burn` | paint it black into the images. For tools that cannot read masks. Irreversible |
| `none` | record the occluders in the rig, mask nothing |

Cameras the occluder never reaches get no mask file at all: an all-white mask changes nothing
but still costs a file per frame and needlessly switches that camera into masked handling.

**Mask polarity: white keeps, black is ignored.** Brush, COLMAP and nerfstudio all agree —
Brush copies mask luma straight into the image's alpha (`pixel[3] = mask_pixel[0]`) and treats
alpha 0 as "do not train here". Getting this backwards silently trains on *only* the car.

### Dynamic occluders — people, passing cars, faces and plates

These move, so no painted region can catch them. Detection runs *after* extraction, on the
rectilinear tiles rather than the panorama: detectors are trained on ordinary photographs and
equirectangular distortion wrecks their recall away from the equator.

```bash
pip install -e ".[ml]"
360extract mask dataset/ --rig rigs/car.json --backend sam2.1
```

| backend | what it does |
|---|---|
| `yolo` | finds objects by class on its own. Fast, weights self-download |
| `sam2.1` | YOLO supplies the prompts, SAM 2.1 sharpens the outlines. **Default** |

**SAM 2.1 has no concept of a "person".** It segments what it is pointed at — it is promptable,
not open-vocabulary. So the `sam2.1` backend is YOLO finding *what* to mask and SAM refining
*exactly where*, which is what it is genuinely better at.

Masks are grown by a few pixels (`--dilate`, default 6): segmentation edges sit slightly inside
the object, and a sliver of leftover pedestrian is enough to seed a floater.

#### Why detections are reconciled through the sphere

A pedestrian caught by camera A and *missed* by overlapping camera B gives inconsistent
supervision, and a splat trainer happily bakes the ghost in from B. So every camera's tile mask
is projected back onto the sphere, unioned there, and re-projected. Tile-space accuracy,
sphere-wide consistency. `--no-fuse` turns it off.

That reverse projection is done in numpy rather than by ffmpeg, and the reason is worth
recording: `v360` can map `flat` back to `e`, but it clamps the tile's border pixels outward
across the whole sphere, so one black pixel at a tile edge would mark half the panorama as
ignored. Its `alpha_mask` option looks like the fix and is not — measured against analytically
computed frustum coverage it disagrees completely (60°×60° camera: true coverage 0.052, alpha
0.725; at yaw 90 it reports nothing at all). So the inverse is written out explicitly and
`tests/test_fuse.py::TestRoundTrip` pins it against `v360`'s forward projection across five
camera configurations, including the seam and steep pitch.

#### GPU

`pip install torch` gives a CPU build on Windows. For a CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
360extract mask dataset/ --rig rigs/car.json --device cuda:0
```

## Frame selection

```bash
--sharp 2        # 2 per second, keeping the SHARPEST frame in each window
--fps 2          # 2 frames per second, whatever lands on the tick (default)
--every 10       # every 10th source frame
--all-frames     # everything
--start 5 --end 30   # only this time window
```

Sampling uniformly in *time* is the right basis for photogrammetry: a capture that pauses
does not then flood the dataset with near-duplicates from wherever the operator stopped
walking.

**`--sharp` is usually the better choice.** Uniform sampling is blind to motion blur — it
takes whatever frame the tick lands on, and on a walking or driving capture a good share of
those are smeared. Blurred frames are worse than useless: they contribute no matchable
features and drag the reconstruction down. `--sharp 2` asks for the same two frames per
second but keeps the *sharpest* frame in each half-second window instead.

Sharpness comes from ffmpeg's own `blurdetect` filter, so there is no extra dependency.
It costs one analysis decode of the source before extraction begins, and reports what it did:

```
big360.mp4: analysing sharpness…
  picked 20 of 300 frames, mean blur 4.45 vs 4.52 across all frames (lower is sharper)
```

The idea is [Florian Bruggisser's sharp-frame-extractor](https://github.com/cansik/sharp-frame-extractor).

## Output size

By default each camera is written at the source's own pixel density: an equirectangular frame
carries `width` pixels across 360°, so a 90° camera gets exactly `width / 4` pixels across.
A 3840-wide source with a 90°×67.5° camera yields 960×720.

Anything smaller throws away detail the capture paid for; anything larger invents it and
inflates the dataset without adding a single real feature to match. Sizing is per-camera, so
a 45° camera in a mixed rig is not padded out to match a 90° one.

Pass `--width`/`--height` to override with a fixed size for every camera.

## Output

```
dataset/
  images/clip/fwd/   clip_fwd_00001.jpg  clip_fwd_00002.jpg  ...
  images/clip/left/  clip_left_00001.jpg ...
  masks/ clip/fwd/   clip_fwd_00001.png  ...
```

This is the layout Brush and COLMAP both read. Brush pairs an image with its mask by mirroring
the subpath — `images/a/b/x.jpg` to `masks/a/b/x.png` — and **requires the nested directories to
match**, which is why masks mirror the image tree rather than sitting in one folder.

Use `--layout flat` for the older shape. Sequence numbers are consistent across cameras —
the same number always means the same instant, because every camera receives the identical frame
set from one split.

Completed cameras are marked done, so re-running resumes rather than redoing work. `--no-resume`
forces a full redo. A pass that fails marks nothing, so a half-written camera never looks
finished.

## Status

| Milestone | State |
|---|---|
| M1 — rig format, ffmpeg discovery, extraction | **done** |
| M2 — nadir cones and painted equirect masks | **done** — no ML dependency |
| M3 — ML masking (YOLO, SAM 2.1) + sphere fusion | **done** — optional `[ml]` extra |
| M4 — Brush/COLMAP rig export | not started |
| M5 — rig editor UI | **done** |
| M6 — inpainting | not started |

## Tests

```bash
pytest                    # everything
pytest -m "not ffmpeg"    # unit tests only, no ffmpeg needed
pytest -m "not slow"      # skip the tests that run real detection models
```

Detection is tested at two levels: the pipeline against a stub detector, so it is
deterministic and needs no weights; and the real backends against ultralytics' own sample
photographs, which genuinely contain people and a bus. The latter skip rather than silently
pass when weights are unavailable.

## License

Apache-2.0. Model weights for the masking milestone are never vendored — they carry their own
licenses and are downloaded separately.
