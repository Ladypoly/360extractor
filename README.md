# 360extract

A 360° video is not a photogrammetry dataset. It's one wide-angle recording of a rig driving
or walking through a scene, with the rig itself, its mount, and whoever is holding it baked
into every frame. 360extract is an app that turns that video into a trained 3D Gaussian splat:
it cuts the panorama into per-camera tiles, keeps the rig and the operator out of what gets
trained, runs COLMAP and Brush, and shows you the result — all from one window, without hand-running
five separate command-line tools.

```
video ──► frames ──► camera tiles + masks ──► COLMAP ──► Brush ──► splat
```

> **Status: pre-1.0 and under active development.** The whole pipeline has been run end to end
> on real footage — see [Verified on real footage](#verified-on-real-footage) — and is covered
> by over 790 tests, including tests that drive real COLMAP. Interfaces may still change
> without notice.

---

## Contents

- [Install](#install)
- [**Using the app**](#using-the-app) — Start, Capture, Reconstruct, Train
- [Projects](#projects) · [Rigs](#rigs)
- [How it works](#how-it-works) — frame selection, projection, grading, masking, reconstruction, output layout, all with the evidence behind them
- [Verified on real footage](#verified-on-real-footage) · [Status](#status)
- [**CLI reference**](#cli-reference) — scripting the pipeline instead of using the app
- [Tests](#tests) · [License](#license)

---

## Install

Requirements: **Python 3.10+** and **ffmpeg 5.0+** with the `v360` filter.

```bash
pip install -e ".[dev]"     # with tests
pip install -e ".[ml]"      # optional: detection-based masking (torch, ultralytics)
```

Then either double-click **`360extract-ui.bat`** (Windows) or run `360extract ui`. It creates
the virtualenv on first run, checks ffmpeg, and opens the app at `127.0.0.1:8360`.

### Where the other tools are

None of ffmpeg, COLMAP, [Brush](https://github.com/ArthurBrussee/brush) or
[SuperSplat](https://github.com/playcanvas/supersplat) is a Python package, so none is a
dependency — each is discovered, in this order:

1. an explicit `--ffmpeg` / `--colmap` / `--brush` / `--supersplat` argument (CLI only)
2. `THREESIXTY_FFMPEG` / `THREESIXTY_COLMAP` / `THREESIXTY_BRUSH` / `THREESIXTY_SUPERSPLAT`
3. **a path set in the app's System status dialog**, saved to `~/.threesixty/tools.json`
4. `PATH`
5. the usual install locations

Name a binary explicitly and it is used or the run fails — never silently substituted. Setting
a path in System status re-runs the search immediately, so a path that holds no binary says so
there rather than at the start of a reconstruction twenty minutes later. Run `360extract doctor`
any time to see what was found and why the others were rejected — most machines have several
ffmpeg builds, and the first on `PATH` is often an old one without `v360`.

They are not bundled on purpose: ffmpeg is probed for a build that actually has `v360`; COLMAP's
rig support and CUDA feature extraction are what this whole approach depends on, and the
`pycolmap` wheels ship neither; Brush is a Rust binary from GitHub releases; SuperSplat is a
static web build, embedded directly as the Train tab's viewer (see [Train](#train)).

---

## Using the app

Four stages, in order, each one tab: **Start → Capture → Reconstruct → Train**. Everything in
between — running ffmpeg, COLMAP, Brush, and the SuperSplat viewer — happens inside the app; you
never leave the window or touch a terminal.

`360extract project show dataset/` (CLI) reports the same status the app shows: which stages are
done, and which are stale because a setting changed since they last ran — see [Projects](#projects).

### Start — import and pick what gets extracted

Drag a video onto the window, or click **Browse…**. That opens a real native file dialog raised
by the app itself — a browser `<input type=file>` deliberately hides the filesystem path from
the page, so a plain web upload control couldn't hand the server one.

**Source.** Tell it what the footage is: a stitched **equirectangular** file (the default), a
raw **dual fisheye** file straight off the two lenses, or a single **fisheye**. Picking dual
fisheye is worth doing when your camera offers both files for the same shot — it avoids
resampling the image twice (once in the camera's stitch, once here). See
[Source projection](#source-projection-dual-fisheye) for why, and the measured picture quality.

**Frame selection.** Four modes, picked from a dropdown, with a live estimate of how many
frames each one will produce as you adjust the value:

- **Sharpest frame every N seconds** (the default, `0.5`) — ffmpeg's own `blurdetect` filter
  scores every frame, and the sharpest one in each window is kept. This is the one to reach
  for: uniform sampling is blind to motion blur, and a blurred frame contributes no matchable
  features while still costing a full camera's worth of extraction and training time. See
  [Frame selection](#frame-selection) for the measured effect.
- **N per second** — a fixed rate, whatever lands on the tick.
- **Every Nth frame** — a fixed stride through the source.
- **All frames.**

**Segments.** For a long drive or walk, split the source into overlapping sub-projects up
front — by duration, by distance (given an average speed), or by a fixed count — so
reconstruction can run in manageable chunks instead of one enormous COLMAP job. (The
`model_merger` chain that reassembles chunked reconstructions is not yet verified against a
real capture — see [`batches`](#batches).)

**Masking, and testing it before committing.** Pick a detector backend, type in the classes to
remove (the default list is `sky, person, car, bus, truck, motorcycle, bicycle`), and set
confidence and how far to grow each mask. Then hit **Preview masking** — it runs the detector on
the current frame only and shows you what it would remove, so you can tune classes and
confidence against real footage before running detection on the whole capture. (First run
downloads the model weights.) See [Masking](#masking-keeping-the-rig-out-of-the-dataset) for
what each backend can and can't see.

**Process** kicks off extraction and masking together, with a live log and a cancel button.
Re-running after a settings change only redoes what's now stale.

### Capture — place the cameras on the panorama, by hand

The panorama fills the canvas with every camera's field-of-view drawn on it as a footprint, so
you can see at a glance whether the car's hood or the person holding the stick falls inside a
camera. **Drag a footprint to re-aim it** — yaw and pitch follow the mouse — instead of typing
angles into a form. The rig's per-camera FOV, format and interpolation are editable alongside it.

The same panorama carries a **mask overlay**, tinting exactly what extraction would exclude —
static occluders and, once run, the dynamic ones — so you're looking at real coverage, not a
guess. That overlay is pixel-accurate on purpose: `tests/test_overlay_geometry.py` runs the
shipped `geometry.js` under node, places markers just inside and just outside each drawn edge,
and checks them against real ffmpeg extractions. A UI that draws coverage it doesn't actually
have is worse than no UI, because it hides exactly the occluder you were trying to exclude.

**Painting occluders.** A nadir cone handles the ground below the rig, but a hood, a mount arm,
or a wing mirror isn't a neat cone — paint it directly onto the panorama instead. Paint once, and
the same region is pushed through the identical `v360` call used for the picture itself, so the
mask lands aligned to the image by construction, for every camera, for free.

**Image properties (grading).** Sliders for exposure, brightness, contrast, saturation, gamma
and black point, with the panorama re-grading live as you drag — about 50 ms per update on an
8K source, because the server keeps the decoded frame around instead of re-seeking the video.
Press **Auto** to grade the frame you're looking at automatically: it measures for what a splat
trainer needs, not what looks good to a person looking at a photo — see [Grading](#grading) for
why those are different and the measured effect on a real drive. Grading is stored in the rig,
so it survives saving and reloading a preset.

**Presets.** Save the current rig — cameras and grade together — under a name, reload it later,
delete the ones you don't want, on top of the built-in starting points:

| Preset | What it is for |
|---|---|
| `ring` | N cameras around the horizon. The photogrammetry workhorse. |
| `cube` | Six 90° faces. Complete spherical coverage, no overlap. |
| `dome` | Horizon ring + upper ring + zenith. Everything except the ground. |
| `car-forward` | Roof-mounted vehicle capture. Forward and sides, tilted down, no rear. |
| `handheld` | Walking capture on a stick, tilted up to keep the operator out of frame. |

**Generate cameras** locks the rig in and cuts the tiles.

### Reconstruct — COLMAP, watched as it runs

One button runs COLMAP end to end — feature extraction, rig configuration, matching, mapping —
against the tiles Capture produced. The sparse point cloud is drawn as COLMAP builds it, with
the raw log available underneath rather than in front, so you're watching the reconstruction
take shape rather than reading text scroll by. See
[Reconstructing, training and cleaning](#reconstructing-training-and-cleaning) for why the
cameras are handed to COLMAP as *known* geometry rather than left for it to estimate.

### Train — Brush and SuperSplat, in the same window

Runs Brush against the COLMAP output, and loads each export straight into an embedded
**SuperSplat** viewer as it lands — the same viewer used to inspect the result, not a separate
app you'd otherwise have to open and re-import into. Training options (steps, export interval,
eval split) are set right there.

**Splat cleanup** runs from the same tab: `clean-splat` deletes the gaussians a trainer invents
in the volume the rig itself occupied — no camera ever saw that space, so nothing contradicts
whatever the trainer puts there. The viewer gets a **Result / Removed** toggle so you can look at
exactly what was deleted rather than take it on trust. See
[Removing floaters where the rig was](#removing-floaters-where-the-rig-was) for the measured
before/after.

---

## Projects

A project is one `project.json` at the root of the dataset it describes. That makes the folder
self-describing: move it, hand it to someone else, come back in a month, and the settings arrive
with the pixels.

It records **what has already been done**, and each stage stores a fingerprint of the settings
that produced it — so the app (and the CLI) distinguishes "already extracted" from "extracted,
but you've since changed the rig":

```
  stages:
    extract  stale  at 2026-07-22T14:02:20+00:00  images=100, cameras=5
             settings changed since this ran; re-run to update
    mask     pending
    export   pending
    train    pending
    clean    pending
```

Fingerprints are per stage, so changing the detector doesn't force a re-extract while changing
the rig does. Redoing a stage clears the ones after it, which would otherwise claim to be
current while describing images that no longer exist.

Snapshots are cheap insurance before a big change — settings only, no images:
`360extract project snapshot dataset/ --label before-retilt`, restored with `--restore`.

The painted occluder lives in `assets/` inside the project, so it survives a reboot.

## Rigs

A rig is a JSON file listing the cameras to extract: a plain, diffable artifact you can
version-control, share, and generate from scripts — the same file Capture's presets save and
load.

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

---

## How it works

The mechanics behind each part of the app above, with the evidence that backs each claim —
useful when a default needs overriding, or when something needs trusting rather than taking on
faith.

### The single-decode extraction

The extraction engine is ffmpeg's `v360` filter. 360extract decodes the source **once** and fans
it out to every camera in a single pass:

```
[0:v]fps=2,split=8[s0]...[s7];
[s0]v360=e:rectilinear:yaw=0:pitch=-5:h_fov=90:v_fov=67.5:w=1920:h=1440[o0];
...
```

Running ffmpeg once per camera would pay the decode cost N times. A test
(`test_single_pass_matches_separate_runs_byte_for_byte`) asserts the batched output is
byte-identical to the naive version, so the optimisation can never silently change results.

### Frame selection

```bash
--sharp 0.5      # the SHARPEST frame of every half second  (default)
--fps 2          # 2 per second, whatever lands on the tick
--every 10       # every 10th source frame
--all-frames     # everything
--start 5 --end 30
```

`--sharp` states a **window in seconds**, not a rate: `1` keeps one frame per second, `0.5` one
per half second. Smaller means more frames.

**It is usually the better choice.** Uniform sampling is blind to motion blur — it takes
whatever frame the tick lands on, and on a walking or driving capture a good share of those are
smeared. Blurred frames are worse than useless: they contribute no matchable features and drag
the reconstruction down.

Sharpness comes from ffmpeg's own `blurdetect` filter, so there is no extra dependency. It costs
one analysis decode before extraction, and reports what it did:

```
big360.mp4: analysing sharpness…
  picked 20 of 300 frames, mean blur 4.45 vs 4.52 across all frames (lower is sharper)
```

The idea is [Florian Bruggisser's sharp-frame-extractor](https://github.com/cansik/sharp-frame-extractor).

Sampling uniformly in *time* is the right basis for photogrammetry: a capture that pauses does
not then flood the dataset with near-duplicates from wherever the operator stopped walking.

### Source projection (dual fisheye)

Most 360 cameras write **two** files for the same shot: a stitched equirectangular one, and the
raw one straight off the two lenses — two circular images side by side. 360extract reads either.

In the app it's the **projection** dropdown in Start ▸ Source, beside the file. Pick
`dual fisheye` and the panorama appears on the canvas; the lens field of view sits under it
(190° is the usual figure — check the camera's specs).

| Projection | The source is |
|---|---|
| `equirect` | a stitched 360 file, 2:1 (the default) |
| `dfisheye` | a raw file with both lenses side by side |
| `fisheye` | one lens, covering less than the sphere |

**Why bother with the raw file.** The camera's stitch is a resample, and often a lossy re-encode
on top of it. Feeding the raw file means the pixels are resampled exactly once — by the same
`v360` that cuts the camera tiles — instead of twice. Tiles come out of a dual-fisheye source
essentially identical to tiles cut from the stitched panorama of the same footage: the front
camera measures ~36 dB PSNR against its stitched counterpart on the synthetic clip in
`tests/test_extract_integration.py` (a wrong projection would land near 10), and 74 dB on a
lossless still, where the h.264 in that clip is not part of the difference.

**You have to say so.** A raw two-lens file is 2:1 as well, so nothing about its dimensions gives
it away. Left as `equirect` it extracts happily and produces a dataset of the wrong
directions — which COLMAP will try, and fail, to reconstruct.

**What it costs.** The two lenses share the frame's width, so a 5760-wide dual fisheye at 190°
carries the same detail as a 5456-wide panorama, and that is the size the working set and the
automatic tile sizes are computed from. Stitching seams and blending are the camera's job, not
this tool's: `v360` maps each lens's own circle, and the small overlap between them is where a
real stitcher would blend. For most photogrammetry that is fine — features near the seam are
covered by the neighbouring camera anyway — but if your footage has heavy parallax right at the
seam, the camera's own stitch may match better there.

### Grading

Flat or dark footage can be corrected before it is cut into tiles. `--exposure` is in stops and
acts on the light; `--brightness`, `--contrast`, `--gamma`, `--saturation` and `--black` act on
the encoded values afterwards, which is the order a photographer expects.

**It is applied once to the panorama, before the split.** That is not just cheaper: two
overlapping cameras must agree about exposure, or feature matching sees two different pictures
of the same wall and the trained splat carries the seam. A test asserts two cameras aimed the
same way come out byte-identical. The default grade is the identity and emits no filter at all,
so an ungraded extraction is byte-for-byte what it was before this feature existed.

**Auto grading** grades **for a splat trainer, not for a viewer**, and those want different
things. 3DGS fits what it is given: a clipped highlight has no gradient to descend, so gaussians
land on it flat and white; a crushed shadow takes the features COLMAP was going to match with;
boosted chroma is 4:2:0 subsampling baked into per-gaussian colour. Three ideas follow.

**Measure the part that gets trained.** A third of an equirectangular frame is sky, and the sky
is masked out of the splat. Statistics over the whole panorama are therefore statistics of
pixels that will be thrown away — a bright overcast drags the median up and darkens the street to
compensate. Only the band between −60° and +40° elevation is measured. On one real drive that
moved the median from 0.27 (whole frame) to 0.20 (street level).

**Simulate before committing.** Every correction is applied to the measured pixels and checked
for clipping, then backed off until it stops, with a 1% budget at each end. Sweeping that budget
showed 1% is the knee: going to 4% bought about 0.2 stops and took clipped pixels from ~0.5% to
3%.

**Never add saturation.** It is a ceiling, not a target: it comes down for a lurid camera profile
and is otherwise exactly 1.0. Black is only pulled when the picture is both lifted and
flat — haze — since crushing shadows costs COLMAP its features.

Measured before and after on a real drive, clipping inside the trained band:

| frame | before | after | saturation |
|---|---|---|---|
| t=5s | 13.9% | **0.0%** | 1.40 → 1.00 |
| t=60s | 17.8% | **0.3%** | 1.40 → 1.00 |
| t=120s | 17.8% | **0.2%** | 1.40 → 1.00 |

Corrections too small to see snap to exactly neutral. That matters more than it sounds: a grade
that is not quite the identity changes the rig, which marks an already-extracted dataset stale
and invites re-running everything for a tenth of a stop nobody can see.

### Output size

By default each camera is written at the source's own pixel density: an equirectangular frame
carries `width` pixels across 360°, so a 90° camera gets exactly `width / 4` pixels across. A
3840-wide source with a 90°×67.5° camera yields 960×720.

Anything smaller throws away detail the capture paid for; anything larger invents it and inflates
the dataset without adding a single real feature to match. Sizing is per-camera, so a 45° camera
in a mixed rig is not padded out to match a 90° one. `--width`/`--height` override with a fixed
size for every camera.

### Masking — keeping the rig out of the dataset

This is the whole point of the tool, and there are two halves to it.

#### Static occluders — the stick, the tripod, the car roof

Rigid relative to the rig, so they sit in the *same region of every single frame*. That is what
makes them cheap to deal with, and why this happens **before** extraction rather than after.

The cheapest fix is rig layout: `dome` and `handheld` never point a camera at the ground, and
`car-forward` omits the rear where the mount usually is. Beyond that, a nadir cone masks
everything more than N° below the horizon, and anything that isn't a neat cone — a hood, a mount
arm, a wing mirror — gets painted directly onto the panorama in Capture.

Paint once, and the same region is pushed through the *identical* `v360` call used for the
picture, so the per-camera mask is aligned pixel for pixel by construction. One render per
camera, reused for every frame.

| `--mask` mode | effect |
|---|---|
| `sidecar` | a mask beside every image. No pixels lost, the trainer decides. **Default.** |
| `skip` | drop cameras more than two thirds occluder — not worth extracting, let alone training |
| `burn` | paint it black into the images. For tools that cannot read masks. Irreversible |
| `none` | record the occluders in the rig, mask nothing |

**Mask polarity: white keeps, black is ignored.** Brush, COLMAP and nerfstudio all agree — Brush
copies mask luma straight into the image's alpha (`pixel[3] = mask_pixel[0]`) and treats alpha 0
as "do not train here". Getting this backwards silently trains on *only* the car.

#### Dynamic occluders — people, passing cars, the sky

These move, so no painted region catches them; a detector runs per frame instead
(`pip install -e ".[ml]"`).

| backend | what it does |
|---|---|
| `sam-world` | YOLO-World finds it, SAM 2.1 outlines it. Any class, including sky. **Default** |
| `yolo-world` | YOLO-World alone. Any class, box-shaped masks |
| `sam2.1` | YOLO supplies prompts, SAM refines. COCO's 80 classes only |
| `yolo` | YOLO alone. COCO's 80 classes only |

**SAM has no concept of a "person".** It segments what it is pointed at — promptable, not
open-vocabulary. So the `sam-` backends are a detector finding *what* to mask and SAM refining
*exactly where*, which is what it is genuinely better at.

**Sky needs an open-vocabulary detector.** COCO's class list has no `sky`, so `yolo` and
`sam2.1` cannot mask it at any confidence. This matters because sky seeds floaters above the
drive that the geometric cleanup structurally cannot remove — it only deletes gaussians *near*
the rig, and sky is the farthest geometry there is.

The default confidence is **0.1**, deliberately low: a missed object becomes a floater nothing
downstream can remove, while a false positive costs a few masked pixels. Masks are grown by a
few pixels (`--dilate`, default 6) because segmentation edges sit slightly inside the object, and
a sliver of leftover pedestrian is enough to seed a floater.

**Why detections are reconciled through the sphere.** A pedestrian caught by camera A and
*missed* by overlapping camera B gives inconsistent supervision, and a splat trainer happily
bakes the ghost in from B. So every camera's tile mask is projected back onto the sphere, unioned
there, and re-projected. Tile-space accuracy, sphere-wide consistency. `--no-fuse` turns it off.

That reverse projection is done in numpy rather than by ffmpeg, and the reason is worth
recording: `v360` can map `flat` back to `e`, but it clamps the tile's border pixels outward
across the whole sphere, so one black pixel at a tile edge would mark half the panorama as
ignored. Its `alpha_mask` option looks like the fix and is not — measured against analytically
computed frustum coverage it disagrees completely (60°×60° camera: true coverage 0.052, alpha
0.725; at yaw 90 it reports nothing at all). So the inverse is written out explicitly, and
`tests/test_fuse.py::TestRoundTrip` pins it against `v360`'s forward projection across five
camera configurations, including the seam and steep pitch.

**GPU.** `pip install torch` gives a CPU build on Windows. For CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Reconstructing, training and cleaning

Because the cameras are synthetic, their relative poses and intrinsics are **known exactly**
rather than estimated. `rig_config.json` hands COLMAP those known values with
`--Mapper.ba_refine_sensor_from_rig 0`, so it only has to solve the rig trajectory — which is
what stops panoramic tile sets drifting.

Two details that are easy to get wrong and are pinned by tests: COLMAP groups images into frames
by **matching filenames across camera folders**, which is why files are named `00001.jpg` inside
`images/<camera>/images/` rather than carrying the camera name; and the rig must be configured
*before* matching, because sequential matching pairs images by frame.

Verified against real COLMAP rather than against our reading of the format: `rig_configurator`
accepts the file, produces **one rig containing every camera**, gathers **all cameras into each
frame**, and adopts our exact intrinsics with `prior_focal_length = 1`. Those tests run whenever
COLMAP is installed and skip when it is not — `pytest -m colmap`.

A GPX track is worth supplying (`--gpx`): it geo-registers the model through `model_aligner`,
and that similarity transform carries a **uniform scale** — the only thing that makes a cleanup
radius mean metres rather than arbitrary units.

#### Removing floaters where the rig was

Masking keeps the vehicle out of the *images*. It cannot stop a trainer putting gaussians **where
the rig was**: no camera sees that volume from any distance, so anything placed there explains
away residual error and nothing contradicts it. On a drive-through they form a continuous trail
down the middle of the street.

This works because **Brush does not move the world** — its COLMAP loader inverts world-to-cam
and uses the translation as-is — so camera centres and splat coordinates share a frame with no
alignment step.

Non-destructive: it writes `trained_cleaned.ply` *and* `trained_removed.ply`, so what was deleted
can be loaded and looked at rather than taken on trust. `--dry-run` reports counts without
writing.

**Use `--floor`.** A sphere centred on a roof-mounted rig also reaches the road beneath it, and
that road is real data — the tarmac under the vehicle at time *t* is observed from *t ± Δ*.
Measured on a synthetic street:

| setting | floaters removed | road destroyed |
|---|---|---|
| radius 2.5, no floor | 100% | **19.9%** |
| radius 2.5, floor 1.5 | 99.4% | **0%** |
| radius 4.0, no floor | 100% | **53.3%** |
| radius 4.0, floor 1.5 | 99.4% | **0%** |

The floor needs to know which way is up, and a **straight** capture cannot reveal that — a line
is symmetric about its own axis. So on a straight drive it asks for `--up` rather than guessing.
After geo-registering with `--alignment_type enu` the answer is exactly `--up enu`.

### Output layout

```
dataset/                          <- the project folder
  project.json
  frames/                         00001.jpg  00002.jpg  ...   (panoramas, app flow only)
  images/fwd/images/              00001.jpg  00002.jpg  ...   (the tiles)
  images/fwd/masks/               00001.png  00002.png  ...
  images/left/images/             ...
  image_list.txt  rig_config.json  colmap_cameras.txt  run_colmap.sh
  sparse/0/                       the COLMAP model
  splat/                          splat_05000.ply ...
  .threesixty/colmap_masks/       hard links, for COLMAP only
```

A camera owns one folder holding its images and the masks that go with them.

**Brush reads the masks where they are** — its documentation asks for "a folder of images called
`masks`", and `images/<camera>/masks/` is one, beside the images it belongs to.

**COLMAP cannot.** `--ImageReader.mask_path` is a *root* that must mirror each image's subpath
below `--image_path`, so for `fwd/images/00001.jpg` it will only ever look for
`<root>/fwd/images/00001.png`. No single folder satisfies that and keeps the masks beside their
images, so a hard-linked mirror lives under `.threesixty/` — directory entries, not a second
copy. (Measured against COLMAP 4.0.2, it accepts `x.png` or `x.jpg.png`; both names are
written.)

**`image_list.txt` is not optional.** COLMAP's feature extractor scans `--image_path`
recursively, so left alone it reads each camera's `masks/` folder as more photographs —
measured, that doubled the camera count and fed all-white images into the reconstruction.

**Filenames stay identical across camera folders**, and that is load-bearing: `rig_configurator`
groups images into *frames* by what is left of the path once a camera's `image_prefix` is
stripped, so the prefixes are `fwd/images/` and the frame is `00001.jpg`. Put the camera in the
filename and every image becomes its own frame, dissolving the rig constraint that keeps a
panoramic tile set from drifting. (COLMAP 4.0 has no `image_suffix` to strip it back off.) See
`dataset.py`.

**Every image has a mask.** Where masking found nothing, a blank one is written — so an absent
mask never has to stand for both "nothing needed masking" and "masking failed". The app reports
frames where detection found nothing at all and offers to drop them.

Sequence numbers are consistent across cameras: the same number always means the same instant,
because every camera receives the identical frame set from one split. `--layout flat` gives the
older shape. Older datasets — a repeated clip folder, or images directly in `images/<camera>/` —
still open; lookups follow whatever is on disk.

---

## Verified on real footage

A roof-mounted 360 camera on a car driving through a village: 8K equirectangular (7680×3840)
HEVC, 55 seconds. A 20-second window, 6 cameras at pitch −10°, sharpest frame of every half
second.

| step | result |
|---|---|
| extract | 240 tiles at 1920×1440 from 8K, **31 s** |
| mask | 25° nadir cone, 25% of each camera |
| COLMAP `rig_configurator` | one rig, 6 cameras, **40 frames × 6** |
| COLMAP `mapper` | **240/240 images registered**, 35,934 points, **0.54 px** mean reprojection error |
| rig honoured | within-frame camera spread **0.000000** |
| Brush | 6,000 steps, 318,343 gaussians, 61 s |
| `clean-splat` | 5,942 removed along the path, 10,920 spared by the floor |

The within-frame spread is the one to look at: all six cameras of a frame came back sharing an
optical centre *exactly*, which is `--Mapper.ba_refine_sensor_from_rig 0` honouring the rig we
handed COLMAP rather than re-solving it.

The floor spared nearly twice what it removed — that is road surface which would otherwise have
been deleted.

## Status

| Milestone | State |
|---|---|
| M1 — rig format, ffmpeg discovery, extraction | **done** |
| M2 — nadir cones and painted equirect masks | **done** — no ML dependency |
| M3 — ML masking + sphere fusion | **done** — optional `[ml]` extra |
| M4 — COLMAP rig export, GPS, splat cleanup | **done** — verified end to end on real footage |
| M5 — app: capture, reconstruct, train | **done** |
| M6 — semantic sky segmentation | in progress — see below |
| M7 — inpainting | not started |

The current sky masking is YOLO-World + SAM prompted with the word "sky". A benchmark of
alternatives on real footage recommends a segmentation model on overlapping perspective views
instead; the open-vocabulary detector under-masks at the silhouette.

---

# CLI reference

*Everything above runs through the app. This section is for scripting the same pipeline —
batch jobs, CI, or driving 360extract from another tool — without opening a browser.*

Every command takes `--ffmpeg PATH` and `--colmap PATH` to override tool discovery.

```bash
360extract project new dataset/ --source CLIP.mp4 --rig car-forward --sharp 0.5
360extract run    dataset/          # extract the tiles, then mask them
360extract export dataset/          # rig_config.json, image_list.txt, run_colmap.sh
sh dataset/run_colmap.sh            # COLMAP: features -> rig -> match -> map
brush dataset/ --total-steps 30000 --export-path dataset/splat
```

| Command | What it does |
|---|---|
| [`doctor`](#doctor) | check ffmpeg / COLMAP discovery and capabilities |
| [`probe`](#probe) | report dimensions, frame rate and duration of a source |
| [`rig`](#rig) | create and inspect camera rigs |
| [`project`](#project) | create and inspect projects |
| [`run`](#run) | extract and mask a project, skipping what is current |
| [`extract`](#extract) | extract tiles without a project |
| [`mask`](#mask) | detect and mask moving occluders in an extracted dataset |
| [`export`](#export) | write the COLMAP project |
| [`clean-splat`](#clean-splat) | delete gaussians where the rig was |
| [`batches`](#batches) | plan a batched reconstruction for a long capture |
| [`ui`](#ui) | open the app in a browser |

## Quick start

```bash
# 1. A project: source, rig, and how to sample it
360extract project new dataset/ \
    --source Q360_0001.mp4 \
    --rig car-forward \
    --sharp 0.5 \
    --classes sky person car bus truck motorcycle bicycle

# 2. Extract the tiles and mask them (skips whatever is already current)
360extract run dataset/

# 3. Write the COLMAP project
360extract export dataset/ --gpx track.gpx

# 4. Reconstruct, then train
sh dataset/run_colmap.sh
brush dataset/ --total-steps 30000 --export-every 5000 \
      --export-path dataset/splat --export-name "splat_{iter}.ply"

# 5. Optional: remove the floaters where the rig was
360extract clean-splat dataset/splat/splat_30000.ply \
    --sparse dataset/sparse/0 --radius 2.5 --floor 1.5 --up enu
```

## `doctor`

```bash
360extract doctor
```

Lists every ffmpeg it found, marks the one it would use, and says why the others were rejected.
Then the same for COLMAP, including whether that build has `rig_configurator` — without it,
reconstruction cannot use the rig and the whole approach falls back to ordinary
structure-from-motion.

## `probe`

```bash
360extract probe CLIP.mp4 OTHER.mp4
```

Dimensions, aspect, codec, frame rate, duration, estimated frame count. Warns when a source is
not 2:1 — extraction will still run, but the geometry will be wrong. A camera's *raw* two-lens
file is 2:1 as well, so the warning says nothing about it; declare that one with
`--projection dfisheye` (see [Source projection](#source-projection-dual-fisheye)).

## `rig`

```bash
360extract rig list                             # the presets
360extract rig show car-forward                 # cameras, fov, orientation, occluders
360extract rig new dome --count 8 -o rigs/dome8.json
360extract rig new ring --count 6 --pitch -10 --h-fov 90 -o rigs/ring6.json
```

`rig new` options: `--count` (cameras per ring), `--pitch` (negative looks down), `--h-fov`,
`--width`/`--height` (fixed output size; omit to derive per camera), `--format`, `--quality`,
`--interp`, `--name`, `-o/--output-file` (stdout if omitted).

Presets are accepted anywhere a rig file is, so `--rig ring` works without writing a file.

## `project`

```bash
360extract project new dataset/ --source CLIP.mp4 --rig car-forward --sharp 0.5
360extract project show dataset/
360extract project snapshot dataset/ --label before-retilt
360extract project snapshot dataset/ --restore before-retilt
```

`project new`:

| Option | Meaning |
|---|---|
| `--source PATH` | a 360 video or still; repeat for several |
| `--rig RIG` | rig file or preset name (default `ring`) |
| `--sharp SECONDS` | keep the sharpest frame of every SECONDS (default, at `0.5`) |
| `--fps N` | N frames per second, whatever lands on the tick |
| `--every N` | every Nth source frame |
| `--all-frames` | everything |
| `--start SEC` / `--end SEC` | limit to a time window |
| `--projection {equirect,dfisheye,fisheye}` | what the footage is (default `equirect`) |
| `--lens-fov DEG` | field of view of one lens, for the fisheye projections (default 190) |
| `--classes ...` | what masking removes; an empty list disables masking |
| `--name`, `--force` | project name; replace an existing `project.json` |

A source outside the project folder is stored as an absolute path; one inside stays relative, so
a dataset carrying its own footage is still movable.

## `run`

```bash
360extract run dataset/            # extract, then mask
360extract run dataset/ --no-mask  # stop after extraction
360extract run dataset/ --force    # redo stages already done
```

Reads every setting from `project.json` and skips stages whose fingerprint still matches. This
is the command to reach for; `extract` and `mask` below are the project-less versions.

## `extract`

```bash
360extract extract CLIP.mp4 --rig car-forward --sharp 0.5 -o dataset/
360extract extract CLIP.mp4 --rig ring --nadir 40 --mask sidecar --auto-grade -o dataset/
360extract extract CLIP.mp4 --rig ring --dry-run       # print the ffmpeg commands
```

Frame selection: `--sharp SECONDS` · `--fps N` · `--every N` · `--all-frames` ·
`--start SEC` · `--end SEC`.
Source: `--projection {equirect,dfisheye,fisheye}` · `--lens-fov DEG`.
Output: `-o/--output-dir` · `--width` / `--height` · `--layout {brush,flat}` ·
`--max-streams` (cameras per ffmpeg pass, default 8).
Occluders: `--nadir DEG` · `--mask {sidecar,skip,burn,none}`.
Grading: `--auto-grade` · `--exposure` · `--black` · `--brightness` · `--contrast` ·
`--saturation` · `--gamma`.
Resume: completed cameras are marked done and skipped; `--no-resume` forces a full redo.

## `mask`

Needs `pip install -e ".[ml]"`.

```bash
360extract mask dataset/ --rig rigs/car.json
360extract mask dataset/ --rig car-forward --classes sky,person,car --device cuda:0
360extract mask dataset/ --rig car-forward --backend yolo-world --confidence 0.1
```

| Option | Meaning |
|---|---|
| `--backend` | `sam-world` (default), `yolo-world`, `sam2.1`, `yolo` |
| `--classes` | comma-separated class names (default: sky and the usual traffic) |
| `--confidence` | detection threshold (default 0.1 — see [Masking](#masking-keeping-the-rig-out-of-the-dataset)) |
| `--dilate` | grow masks by N pixels (default 6) |
| `--device` | torch device, e.g. `cuda:0` |
| `--no-fuse` | do not reconcile overlapping cameras through the sphere |
| `--no-static` | do not merge in the rig's own occluders |

**Only the `-world` backends can see sky.** The COCO ones know a fixed 80-class list that does
not contain it.

## `export`

```bash
360extract export dataset/
360extract export dataset/ --gpx track.gpx
```

Writes `rig_config.json` (the known rig, for `rig_configurator`), `colmap_cameras.txt`
(intrinsics), `image_list.txt` (the images, and nothing else — see
[Output layout](#output-layout)), `run_colmap.sh`, and with `--gpx` a `geo_registration.txt`.

## `clean-splat`

```bash
360extract clean-splat trained.ply --sparse dataset/sparse/0 \
    --radius 2.5 --floor 1.5 --up enu --dry-run
360extract clean-splat trained.ply --sparse dataset/sparse/0 --radius-in-spacings 1.5
```

| Option | Meaning |
|---|---|
| `--sparse DIR` | the COLMAP model, for the camera trajectory (required) |
| `--radius R` | removal radius in model units (metres once geo-registered) |
| `--radius-in-spacings N` | radius as N × the median gap between frames, when there is no real scale |
| `--floor D` | spare anything more than D below the rig |
| `--up DIR` | `enu`, `y`, `z`, or `X,Y,Z` |
| `-o`, `--no-removed`, `--dry-run` | output path; skip the removed file; report without writing |

See [Removing floaters](#removing-floaters-where-the-rig-was) for why `--floor` matters.

## `batches`

```bash
360extract batches dataset/ --chunk 300 --overlap 40
```

Splits a long trajectory into overlapping chunks and emits per-chunk commands plus a
`model_merger` chain. The overlap is the mechanism: `model_merger` aligns neighbours using the
images they share. **Not yet verified against a real capture** — the commands are generated, the
merge is untested.

## `ui`

```bash
360extract ui
360extract ui --project dataset/ --port 8360 --no-browser
```

---

## Tests

```bash
pytest                       # everything
pytest -m "not ffmpeg"       # unit tests only, no ffmpeg needed
pytest -m "not slow"         # skip the tests that run real detection models
pytest -m ui                 # browser tests (needs playwright + chromium)
pytest -m colmap             # the ones that drive real COLMAP
```

Detection is tested at two levels: the pipeline against a stub detector, so it is deterministic
and needs no weights; and the real backends against ultralytics' own sample photographs, which
genuinely contain people and a bus. The latter skip rather than silently pass when weights are
unavailable.

## License

Apache-2.0. Model weights are never vendored — they carry their own licenses and are downloaded
separately.
