"""Static occluder masking, done in equirectangular space.

The camera operator's stick, a tripod, the roof of the car: all rigid relative to the
rig, so they sit in the *same region of every single frame*. That is what makes this
cheap. Paint the occluder once in equirect space, then push that one image through the
identical `v360` call used for the picture, and the per-camera mask is aligned pixel
for pixel by construction -- no reprojection maths of our own to get wrong, and one
render per camera rather than one per frame.

Polarity follows Brush, COLMAP and nerfstudio, which agree: **white keeps, black is
ignored**. Brush copies the mask's luma straight into the image's alpha channel
(`pixel[3] = mask_pixel[0]`) and treats alpha 0 as "do not train here".
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..ffmpeg import FFmpegError, FFmpegInfo
from ..rig import Camera, Rig

#: Masks are hard-edged by nature, so nearest-neighbour resampling keeps them crisp.
#: Bilinear would produce grey fringes that are neither kept nor ignored.
MASK_INTERP = "near"

KEEP = "white"
IGNORE = "black"


class MaskError(RuntimeError):
    """An occluder definition could not be turned into a mask."""


def _run(argv: list[str]) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise MaskError(f"mask render failed:\n{proc.stderr.strip()}")


@dataclass(frozen=True)
class Occluder:
    """One entry from a rig's `occluders` list."""

    kind: str
    angle: float = 0.0
    path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Occluder":
        kind = data.get("type", "")
        if kind == "nadir_cone":
            return cls(kind=kind, angle=float(data.get("angle", 0)))
        if kind == "zenith_cone":
            return cls(kind=kind, angle=float(data.get("angle", 0)))
        if kind == "equirect_mask":
            path = data.get("path")
            if not path:
                raise MaskError("equirect_mask occluder has no 'path'")
            return cls(kind=kind, path=str(path))
        if kind == "ml":
            # Handled by the dynamic masking stage, not here.
            return cls(kind=kind)
        raise MaskError(f"unknown occluder type {kind!r}")


def occluders_of(rig: Rig) -> list[Occluder]:
    """Static occluders only. `ml` entries belong to the dynamic stage."""
    return [o for o in (Occluder.from_dict(d) for d in rig.occluders) if o.kind != "ml"]


def build_equirect_mask(ffmpeg: FFmpegInfo, occluders: Sequence[Occluder],
                        width: int, height: int, output: Path) -> Path:
    """Combine every static occluder into a single equirect mask.

    Starts fully white (keep everything) and darkens. Combining with `darken` means an
    occluder can only ever remove coverage, never restore it, so the order of the list
    cannot change the result.
    """
    if width <= 0 or height <= 0:
        raise MaskError(f"mask size must be positive, got {width}x{height}")
    output.parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = ["-f", "lavfi", "-i", f"color={KEEP}:size={width}x{height}"]
    chain: list[str] = []
    label = "[0:v]"

    # Cones are drawn directly onto the base; equirect is linear in elevation, so a
    # cone below -angle is simply everything under that row.
    boxes = []
    for occluder in occluders:
        if occluder.kind == "nadir_cone" and occluder.angle > 0:
            y = int(round((90.0 + occluder.angle) / 180.0 * height))
            boxes.append(f"drawbox=x=0:y={y}:w={width}:h={height - y}:color={IGNORE}:t=fill")
        elif occluder.kind == "zenith_cone" and occluder.angle > 0:
            y = int(round((90.0 - occluder.angle) / 180.0 * height))
            boxes.append(f"drawbox=x=0:y=0:w={width}:h={y}:color={IGNORE}:t=fill")

    if boxes:
        chain.append(f"{label}{','.join(boxes)}[base]")
        label = "[base]"

    index = 1
    for occluder in occluders:
        if occluder.kind != "equirect_mask":
            continue
        painted = Path(occluder.path)
        if not painted.exists():
            raise MaskError(f"painted occluder not found: {painted}")
        inputs += ["-i", str(painted)]
        chain.append(f"[{index}:v]scale={width}:{height},format=gray[m{index}]")
        chain.append(f"{label}[m{index}]blend=all_mode=darken[b{index}]")
        label = f"[b{index}]"
        index += 1

    chain.append(f"{label}format=gray[out]")
    argv = [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y", *inputs,
            "-filter_complex", ";".join(chain), "-map", "[out]",
            "-frames:v", "1", str(output)]
    _run(argv)
    return output


def render_camera_mask(ffmpeg: FFmpegInfo, equirect_mask: Path, camera: Camera,
                       width: int, height: int, output: Path) -> Path:
    """Project the equirect mask through one camera.

    Uses exactly the parameters the picture is extracted with, which is the whole point:
    alignment is guaranteed rather than computed.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    params = ":".join([
        "e", "rectilinear",
        f"yaw={camera.yaw:g}", f"pitch={camera.pitch:g}", f"roll={camera.roll:g}",
        f"h_fov={camera.h_fov:g}", f"v_fov={camera.v_fov:g}",
        f"w={width}", f"h={height}", f"interp={MASK_INTERP}",
    ])
    _run([str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
          "-i", str(equirect_mask), "-vf", f"v360={params},format=gray",
          "-frames:v", "1", str(output)])
    return output


def ignored_fraction(ffmpeg: FFmpegInfo, mask: Path) -> float:
    """How much of a mask is black, i.e. how much of that camera is thrown away.

    Measured from the rendered mask rather than estimated from geometry, so it accounts
    for a hand-painted occluder of any shape.
    """
    raw = subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-i", str(mask),
         "-vf", "scale=64:64:flags=area,format=gray", "-f", "rawvideo",
         "-pix_fmt", "gray", "-"],
        capture_output=True).stdout
    if not raw:
        raise MaskError(f"could not measure mask {mask}")
    values = raw[:64 * 64]
    return 1.0 - (sum(values) / (len(values) * 255.0))


def nadir_cone_rig(angle: float) -> list[dict[str, Any]]:
    """The occluder list for a plain nadir cone, for callers building rigs."""
    return [{"type": "nadir_cone", "angle": angle}] if angle > 0 else []
