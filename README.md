# 360extract

Turn 360Â° equirectangular footage into perspective image sets for photogrammetry and
3D Gaussian Splatting â€” with precise control over **which directions get extracted**, so the
person holding the camera or the car it was mounted on never reaches your dataset.

> **Status: pre-1.0 and under active development.** The whole pipeline has been run end to
> end on real footage â€” see [Verified on real footage](#verified-on-real-footage) â€” and is
> covered by 525 tests, including tests that drive real COLMAP. Interfaces may still change
> without notice.

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

## Verified on real footage

A roof-mounted 360 camera on a car driving through a village: 8K equirectangular
(7680Ã—3840) HEVC, 55 seconds. A 20-second window, 6 cameras at pitch âˆ’10Â°, sharpest frame
of each half second.

| step | result |
|---|---|
| extract | 240 tiles at 1920Ã—1440 from 8K, **31 s** |
| mask | 25Â° nadir cone, 25% of each camera |
| COLMAP `rig_configurator` | one rig, 6 cameras, **40 frames Ã— 6** |
| COLMAP `mapper` | **240/240 images registered**, 35,934 points, **0.54 px** mean reprojection error |
| rig honoured | within-frame camera spread **0.000000** |
| Brush | 6,000 steps, 318,343 gaussians, 61 s |
| `clean-splat` | 5,942 removed along the path, 10,920 spared by the floor |

The within-frame spread is the one to look at: all six cameras of a frame came back sharing
an optical centre *exactly*, which is `--Mapper.ba_refine_sensor_from_rig 0` honouring the
rig we handed COLMAP rather than re-solving it.

The floor spared nearly twice what it removed â€” that is road surface which would otherwise
have been deleted.

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
variable. When you name a binary explicitly it is used or the run fails â€” 360extract never
silently substitutes a different one.

## Install

```bash
pip install -e ".[dev]"     # dev install with tests
```

## The rig editor

Double-click **`360extract-ui.bat`** (Windows) or run `360extract ui`. It creates the
virtualenv on first run, checks ffmpeg, and opens a local page on `127.0.0.1:8360`.

**Browseâ€¦** opens a real file dialog. A browser can never hand a server a filesystem path â€”
`<input type=file>` deliberately hides it â€” so the dialog is raised by the app itself.

The panorama fills the window with every camera's footprint drawn on it, so you can see at a
glance whether the car hood or the person holding the stick falls inside a camera. Drag a
footprint to aim it; the camera list stays a quiet summary and editing happens in one panel
below it, rather than in a grid of inline number boxes. The selected camera's real extracted
view is previewed underneath.

The **occluder guide** slider shades everything below a chosen angle and reports what
percentage of each camera falls inside it â€” the number to watch when deciding whether to
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
that produced it â€” so the tool distinguishes "already extracted" from "extracted, but you have
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

Snapshots are cheap insurance before a big change â€” settings only, no images:

```bash
360extract project snapshot dataset/ --label before-retilt
360extract project snapshot dataset/ --restore before-retilt
```

The painted occluder lives in `assets/` inside the project, so it survives a reboot.

`360extract ui --project dataset/` opens straight into one.

## Reconstructing and training

```bash
360extract export dataset/ --gpx track.gpx    # rig_config.json, intrinsics, commands
sh dataset/run_colmap.sh                      # COLMAP, then Brush
```

Because the cameras are synthetic, their relative poses and intrinsics are **known exactly**
rather than estimated. `rig_config.json` hands COLMAP those known values with
`--Mapper.ba_refine_sensor_from_rig 0`, so it only has to solve the rig trajectory â€” which is
what stops panoramic tile sets drifting.

Two details that are easy to get wrong and are pinned by tests: COLMAP groups images into
frames by **matching filenames across camera folders**, which is why files are named
`00001.jpg` inside `<clip>/<camera>/` rather than carrying the camera name; and the rig must be
configured *before* matching, because sequential matching pairs images by frame.

Verified against COLMAP 4.1.1 itself, not just against our reading of the format:
`rig_configurator` accepts the file, produces **one rig containing every camera**, gathers
**all cameras into each frame**, and adopts our exact intrinsics with
`prior_focal_length = 1`. Those tests run whenever COLMAP is installed and skip when it is
not â€” `pytest -m colmap`.

`360extract doctor` reports which COLMAP it found and whether that build has rig support;
`--colmap PATH` or `THREESIXTY_COLMAP` override discovery, and the generated script uses the
binary that was actually found.

A `--gpx` track is worth supplying. It geo-registers the model through `model_aligner`, and
that similarity transform carries a **uniform scale** â€” which is the only thing that makes a
cleanup radius mean metres rather than arbitrary units.

### Removing floaters where the rig was

Masking keeps the vehicle out of the *images*. It cannot stop a trainer putting gaussians
**where the rig was**: no camera sees that volume from any distance, so anything placed there
explains away residual error and nothing contradicts it. On a drive-through they form a
continuous trail down the middle of the street.

```bash
360extract clean-splat trained.ply --sparse dataset/sparse/0 \
    --radius 2.5 --floor 1.5 --up enu --dry-run
```

This works because **Brush does not move the world** â€” its COLMAP loader inverts world-to-cam
and uses the translation as-is â€” so camera centres and splat coordinates share a frame with no
alignment step.

Non-destructive: it writes `trained_cleaned.ply` *and* `trained_removed.ply`, so what was
deleted can be loaded and looked at rather than taken on trust. `--dry-run` reports counts
without writing.

**Use `--floor`.** A sphere centred on a roof-mounted rig also reaches the road beneath it, and
that road is real data â€” the tarmac under the vehicle at time *t* is observed from *t Â± Î”*.
Measured on a synthetic street:

| setting | floaters removed | road destroyed |
|---|---|---|
| radius 2.5, no floor | 100% | **19.9%** |
| radius 2.5, floor 1.5 | 99.4% | **0%** |
| radius 4.0, no floor | 100% | **53.3%** |
| radius 4.0, floor 1.5 | 99.4% | **0%** |

The floor needs to know which way is up, and a **straight** capture cannot reveal that â€” a line
is symmetric about its own axis. So on a straight drive it asks for `--up` rather than guessing.
After geo-registering with `--alignment_type enu` the answer is exactly `--up enu`.

### Long captures

```bash
360extract batches dataset/ --chunk 300 --overlap 40
```

Splits the trajectory into overlapping chunks and emits per-chunk commands plus a
`model_merger` chain. The overlap is the mechanism: `model_merger` aligns neighbours using the
images they share. **Not yet verified against a real capture** â€” the commands are generated,
the merge is untested.

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
| `cube` | Six 90Â° faces. Complete spherical coverage, no overlap. |
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

### Static occluders â€” the stick, the tripod, the car roof

Rigid relative to the rig, so they sit in the *same region of every single frame*. That is what
makes them cheap to deal with, and it is why this happens **before** extraction rather than
after.

The cheapest fix is rig layout: `dome` and `handheld` never point a camera at the ground, and
`car-forward` omits the rear where the mount usually is. Beyond that:

```bash
360extract extract CLIP.mp4 --rig car-forward --nadir 40 --mask sidecar -o dataset/
```

`--nadir 40` masks everything more than 40Â° below the horizon. For anything that is not a neat
cone â€” a hood, a mount arm, a wing mirror â€” paint it directly onto the panorama in the UI.

Paint once, and the same region is pushed through the *identical* `v360` call used for the
picture, so the per-camera mask is aligned pixel for pixel by construction. One render per
camera, reused for every frame, rather than one per frame.

`--mask` chooses what to do about it:

| mode | effect |
|---|---|
| `sidecar` | a mask beside every image. No pixels lost, the trainer decides. **Default.** |
| `skip` | drop cameras more than two thirds occluder â€” not worth extracting, let alone training |
| `burn` | paint it black into the images. For tools that cannot read masks. Irreversible |
| `none` | record the occluders in the rig, mask nothing |

Cameras the occluder never reaches get no mask file at all: an all-white mask changes nothing
but still costs a file per frame and needlessly switches that camera into masked handling.

**Mask polarity: white keeps, black is ignored.** Brush, COLMAP and nerfstudio all agree â€”
Brush copies mask luma straight into the image's alpha (`pixel[3] = mask_pixel[0]`) and treats
alpha 0 as "do not train here". Getting this backwards silently trains on *only* the car.

### Dynamic occluders â€” people, passing cars, faces and plates

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

**SAM 2.1 has no concept of a "person".** It segments what it is pointed at â€” it is promptable,
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
ignored. Its `alpha_mask` option looks like the fix and is not â€” measured against analytically
computed frustum coverage it disagrees completely (60Â°Ã—60Â° camera: true coverage 0.052, alpha
0.725; at yaw 90 it reports nothing at all). So the inverse is written out explicitly and
`tests/test_fuse.py::TestRoundTrip` pins it against `v360`'s forward projection across five
camera configurations, including the seam and steep pitch.

#### Reviewing the masks

The **Refine tab** in the UI drives all of this: pick the backend and classes, run it, then
step through frames with the mask tinted red over the picture. The preview is composited from
the mask file that will actually be handed to the trainer, not an approximation â€” worth a look
before committing to a long reconstruction.

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

**`--sharp` is usually the better choice.** Uniform sampling is blind to motion blur â€” it
takes whatever frame the tick lands on, and on a walking or driving capture a good share of
those are smeared. Blurred frames are worse than useless: they contribute no matchable
features and drag the reconstruction down. `--sharp 2` asks for the same two frames per
second but keeps the *sharpest* frame in each half-second window instead.

Sharpness comes from ffmpeg's own `blurdetect` filter, so there is no extra dependency.
It costs one analysis decode of the source before extraction begins, and reports what it did:

```
big360.mp4: analysing sharpnessâ€¦
  picked 20 of 300 frames, mean blur 4.45 vs 4.52 across all frames (lower is sharper)
```

The idea is [Florian Bruggisser's sharp-frame-extractor](https://github.com/cansik/sharp-frame-extractor).

## Exposure, brightness and contrast

Flat or dark 360 footage can be corrected before it is cut into tiles:

```bash
360extract extract CLIP.mp4 --rig ring --exposure 0.7 --contrast 1.15 --saturation 1.12
```

`--exposure` is in stops and acts on the light; `--brightness`, `--contrast`, `--gamma`,
`--saturation` and `--black` act on the encoded values afterwards, which is the order a
photographer expects. The Capture tab has sliders for all of them, and the panorama on the
canvas is re-rendered with the grade applied, so what you see is what gets written.

**It is applied once to the panorama, before the split.** That is not just cheaper: two
overlapping cameras must agree about exposure, or feature matching sees two different
pictures of the same wall and the trained splat carries the seam. A test asserts two cameras
aimed the same way come out byte-identical.

The default grade is the identity and emits **no filter at all** â€” an ungraded extraction is
byte-for-byte what it was before this feature existed, which is also asserted by a test.

Grading is stored in the rig, so it travels with `rig save` and survives switching presets.

## Output size

By default each camera is written at the source's own pixel density: an equirectangular frame
carries `width` pixels across 360Â°, so a 90Â° camera gets exactly `width / 4` pixels across.
A 3840-wide source with a 90Â°Ã—67.5Â° camera yields 960Ã—720.

Anything smaller throws away detail the capture paid for; anything larger invents it and
inflates the dataset without adding a single real feature to match. Sizing is per-camera, so
a 45Â° camera in a mixed rig is not padded out to match a 90Â° one.

Pass `--width`/`--height` to override with a fixed size for every camera.

## Output

```
dataset/
  images/clip/fwd/   clip_fwd_00001.jpg  clip_fwd_00002.jpg  ...
  images/clip/left/  clip_left_00001.jpg ...
  masks/ clip/fwd/   clip_fwd_00001.png  ...
```

This is the layout Brush and COLMAP both read. Brush pairs an image with its mask by mirroring
the subpath â€” `images/a/b/x.jpg` to `masks/a/b/x.png` â€” and **requires the nested directories to
match**, which is why masks mirror the image tree rather than sitting in one folder.

Use `--layout flat` for the older shape. Sequence numbers are consistent across cameras â€”
the same number always means the same instant, because every camera receives the identical frame
set from one split.

Completed cameras are marked done, so re-running resumes rather than redoing work. `--no-resume`
forces a full redo. A pass that fails marks nothing, so a half-written camera never looks
finished.

## Status

| Milestone | State |
|---|---|
| M1 â€” rig format, ffmpeg discovery, extraction | **done** |
| M2 â€” nadir cones and painted equirect masks | **done** â€” no ML dependency |
| M3 â€” ML masking (YOLO, SAM 2.1) + sphere fusion | **done** â€” optional `[ml]` extra |
| M4 â€” COLMAP rig export, GPS, splat cleanup | **done** â€” verified end to end on real footage |
| M5 â€” rig editor UI | **done** |
| M6 â€” inpainting | not started |

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

Apache-2.0. Model weights for the masking milestone are never vendored â€” they carry their own
licenses and are downloaded separately.
