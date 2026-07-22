"""Making overlapping cameras agree about what to mask.

Detection runs per tile, because detectors are trained on ordinary photographs and
equirectangular distortion wrecks their recall away from the equator. But per-tile
detection has a failure mode that matters specifically for Gaussian splatting: a
pedestrian caught in camera A and *missed* in overlapping camera B gives inconsistent
supervision, and the ghost is happily baked in from B.

So the tile masks are projected back onto the sphere, unioned there, and re-projected.
Tile-space accuracy, sphere-wide consistency.

**Why the reverse projection is done here rather than by ffmpeg.** `v360` can map
`flat` back to `e`, but it clamps the tile's border pixels outward across the entire
rest of the sphere -- so a single black pixel at a tile edge would mark half the
panorama as "ignore". Its `alpha_mask` option looks like the answer and is not: measured
against analytically computed frustum coverage it disagrees completely (for a 60x60
camera, true coverage 0.052 versus alpha 0.725, and for a camera at yaw 90 it reports
nothing at all). It marks something other than visibility.

So the inverse mapping is written out explicitly below, and
`test_fuse.py::TestRoundTrip` pins it directly against `v360`'s forward projection --
the direction the whole tool is already verified on. As a bonus this removes a
subprocess per camera per frame from the fusion loop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from ..ffmpeg import FFmpegInfo
from ..rig import Camera
from .geometric import MaskError

#: Below this luma a mask pixel counts as "ignore". Masks are nominally binary, but
#: resampling leaves soft edges.
IGNORE_BELOW = 128


def _numpy():
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise MaskError(
            'sphere fusion needs numpy. Install the ML extra: pip install -e ".[ml]"'
        ) from exc
    return numpy


def sphere_directions(width: int, height: int):
    """Unit direction for the centre of every equirect pixel.

    Matches the convention the rest of the tool is pinned to:
    x = (yaw + 180) / 360 * W, y = (90 - pitch) / 180 * H, and
    dir(yaw, pitch) = (sin y cos p, sin p, cos y cos p).
    """
    numpy = _numpy()
    yaws = numpy.radians((numpy.arange(width) + 0.5) / width * 360.0 - 180.0)
    pitches = numpy.radians(90.0 - (numpy.arange(height) + 0.5) / height * 180.0)
    yaw_grid, pitch_grid = numpy.meshgrid(yaws, pitches)
    return numpy.stack([
        numpy.sin(yaw_grid) * numpy.cos(pitch_grid),
        numpy.sin(pitch_grid),
        numpy.cos(yaw_grid) * numpy.cos(pitch_grid),
    ], axis=-1)


def camera_basis(camera: Camera):
    """Forward, right and up for a camera, identical to the overlay's `basisOf`."""
    numpy = _numpy()
    yaw, pitch = numpy.radians(camera.yaw), numpy.radians(camera.pitch)
    forward = numpy.array([numpy.sin(yaw) * numpy.cos(pitch),
                           numpy.sin(pitch),
                           numpy.cos(yaw) * numpy.cos(pitch)])
    # Taken straight from the bearing so it never degenerates looking straight down.
    right = numpy.array([numpy.cos(yaw), 0.0, -numpy.sin(yaw)])
    up = numpy.cross(forward, right)
    up = up / (numpy.linalg.norm(up) or 1.0)

    roll = numpy.radians(camera.roll or 0.0)
    if roll:
        c, s = numpy.cos(roll), numpy.sin(roll)
        right, up = c * right + s * up, c * up - s * right
    return forward, right, up


def image_plane_coords(directions, camera: Camera):
    """Where each direction lands on the camera's image plane.

    Returns `(nx, ny, visible)` with nx and ny in [-1, 1] across the frame, and
    `visible` true only where the direction is in front of the camera and inside its
    field of view.
    """
    numpy = _numpy()
    forward, right, up = camera_basis(camera)

    z = directions @ forward
    x = directions @ right
    y = directions @ up

    in_front = z > 1e-6
    safe = numpy.where(in_front, z, 1.0)
    nx = (x / safe) / numpy.tan(numpy.radians(camera.h_fov / 2.0))
    ny = (y / safe) / numpy.tan(numpy.radians(camera.v_fov / 2.0))

    visible = in_front & (numpy.abs(nx) <= 1.0) & (numpy.abs(ny) <= 1.0)
    return nx, ny, visible


def project_to_sphere(tile, camera: Camera, width: int, height: int,
                      directions=None):
    """Scatter one camera's tile mask onto the sphere.

    Returns `(mask, visible)`: the tile's values sampled at every direction the camera
    could actually see, and a boolean of where that is. Outside `visible` the mask is
    left at 255 (keep), so a camera never votes about what it did not look at.
    """
    numpy = _numpy()
    tile = numpy.asarray(tile)
    if tile.ndim != 2:
        raise MaskError(f"tile mask must be 2-D, got shape {tile.shape}")

    if directions is None:
        directions = sphere_directions(width, height)
    nx, ny, visible = image_plane_coords(directions, camera)

    tile_h, tile_w = tile.shape
    # +ny is up on the image plane, and row 0 is the top of the image.
    columns = numpy.clip(((nx + 1.0) * 0.5 * (tile_w - 1)).astype(numpy.int32), 0, tile_w - 1)
    rows = numpy.clip(((1.0 - ny) * 0.5 * (tile_h - 1)).astype(numpy.int32), 0, tile_h - 1)

    sampled = tile[rows, columns]
    mask = numpy.where(visible, sampled, 255).astype(numpy.uint8)
    return mask, visible


def fuse(ffmpeg: FFmpegInfo, tile_masks: Sequence[tuple[Camera, Path]],
         width: int, height: int, output: Path) -> Path:
    """Union every camera's opinion into one equirect mask.

    A direction is ignored if *any* camera that could see it says so. Cameras vote only
    within their own field of view.
    """
    numpy = _numpy()
    if not tile_masks:
        raise MaskError("fuse() needs at least one camera mask")

    directions = sphere_directions(width, height)
    ignored = numpy.zeros((height, width), dtype=bool)

    for camera, tile_path in tile_masks:
        tile = read_gray_native(ffmpeg, tile_path)
        mask, visible = project_to_sphere(tile, camera, width, height, directions)
        ignored |= visible & (mask < IGNORE_BELOW)

    sphere = numpy.where(ignored, 0, 255).astype(numpy.uint8)
    write_gray(ffmpeg, sphere, output)
    return output


def write_gray(ffmpeg: FFmpegInfo, array, output: Path) -> Path:
    """Write an HxW uint8 array as a PNG, without an image library."""
    height, width = array.shape
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}", "-i", "-",
         "-frames:v", "1", str(output)],
        input=bytes(array), capture_output=True)
    if proc.returncode != 0 or not output.exists():
        raise MaskError(f"writing {output.name} failed: "
                        f"{proc.stderr.decode(errors='replace').strip()}")
    return output


def image_size(ffmpeg: FFmpegInfo, path: Path) -> tuple[int, int]:
    from ..ffmpeg import ffprobe_for

    out = subprocess.run(
        [str(ffprobe_for(ffmpeg)), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    try:
        width, height = (int(v) for v in out.split(",")[:2])
    except ValueError as exc:
        raise MaskError(f"could not read the size of {path}") from exc
    return width, height


def read_gray_native(ffmpeg: FFmpegInfo, path: Path):
    """Read an image at its own resolution as an HxW uint8 luma array."""
    width, height = image_size(ffmpeg, path)
    return read_gray(ffmpeg, path, width, height)


def read_gray(ffmpeg: FFmpegInfo, path: Path, width: int, height: int):
    """Read an image as an HxW uint8 luma array, resampled if asked."""
    numpy = _numpy()
    proc = subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"scale={width}:{height}:flags=neighbor,format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True)
    expected = width * height
    if len(proc.stdout) < expected:
        raise MaskError(f"could not read {path}")
    return numpy.frombuffer(proc.stdout[:expected], dtype=numpy.uint8).reshape(height, width)


def coverage_of(mask_array) -> float:
    """Share of a mask that is ignored, for reporting."""
    return float((mask_array < IGNORE_BELOW).mean())
