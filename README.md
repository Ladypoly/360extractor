# 360extract

Turn 360° equirectangular footage into perspective image sets for photogrammetry and
3D Gaussian Splatting — with precise control over **which directions get extracted**, so the
person holding the camera or the car it was mounted on never reaches your dataset.

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

**Static occluders — the stick, the tripod, the car roof.** These are rigid relative to the rig,
so they occupy the same region of the frame in every single frame. The cheapest fix is rig
layout: `dome` and `handheld` simply never point a camera at the ground, and `car-forward` omits
the rear where the mount usually is. `pitch` and `orientation` handle the rest.

**Dynamic occluders — the operator walking, passing cars, faces and plates.** These move, so
layout cannot exclude them. That is the masking milestone (below).

The `occluders` block in a rig is already parsed and round-trips safely; it is consumed by the
masking stage, which is not implemented yet.

## Frame selection

```bash
--fps 2          # 2 frames per second (default)
--every 10       # every 10th source frame
--all-frames     # everything
--start 5 --end 30   # only this time window
```

`--fps` is the right default for photogrammetry: it samples uniformly in *time*, so a capture
that pauses does not flood the dataset with near-duplicate frames from wherever the operator
stopped walking.

## Output

```
dataset/
  clip/
    fwd/   clip_fwd_00001.jpg  clip_fwd_00002.jpg  ...
    left/  clip_left_00001.jpg ...
```

Use `--flat` to put every camera in one folder. Sequence numbers are consistent across cameras —
the same number always means the same instant, because every camera receives the identical frame
set from one split.

Completed cameras are marked done, so re-running resumes rather than redoing work. `--no-resume`
forces a full redo. A pass that fails marks nothing, so a half-written camera never looks
finished.

## Status

| Milestone | State |
|---|---|
| M1 — rig format, ffmpeg discovery, extraction | **done** |
| M2 — nadir cones and painted equirect masks | not started |
| M3 — ML masking (YOLO, SAM 2.1, SAM 3) | not started |
| M4 — GPS/GPX and COLMAP export | not started |
| M5 — web UI | not started |
| M6 — inpainting | not started |

## Tests

```bash
pytest                    # everything
pytest -m "not ffmpeg"    # unit tests only, no ffmpeg needed
```

## License

Apache-2.0. Model weights for the masking milestone are never vendored — they carry their own
licenses and are downloaded separately.
