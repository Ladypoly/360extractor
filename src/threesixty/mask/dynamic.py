"""Running dynamic masking over an already-extracted dataset.

Walks `images/<clip>/<camera>/`, detects moving occluders in each camera's frame
sequence, optionally reconciles overlapping cameras through the sphere, and writes the
result to `masks/<clip>/<camera>/` where Brush and COLMAP will find it.

Runs after extraction, on the thinned set of frames that were actually kept, rather
than over every frame of the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..ffmpeg import FFmpegInfo
from ..rig import Camera, Rig
from . import fuse as fuse_module
from . import geometric
from .geometric import MaskError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_TRAILING_NUMBER = re.compile(r"(\d+)(?!.*\d)")

#: Sphere resolution used when reconciling cameras. Masks are blobby, so this does not
#: need the source's full resolution -- and every camera is re-projected from it, so
#: keeping it modest keeps fusion from dominating the run.
FUSION_WIDTH = 2048


@dataclass
class CameraImages:
    """One camera's extracted frames, keyed by their sequence number."""

    camera: Camera
    directory: Path
    mask_directory: Path
    frames: dict[int, Path] = field(default_factory=dict)


@dataclass
class DynamicReport:
    """What dynamic masking did."""

    cameras: int = 0
    images: int = 0
    detections: int = 0
    masks_written: int = 0
    fused: bool = False
    by_label: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.images:
            return "no images found"
        labels = ", ".join(f"{name} x{count}" for name, count
                           in sorted(self.by_label.items(), key=lambda kv: -kv[1]))
        return (f"{self.detections} detections across {self.images} images "
                f"from {self.cameras} cameras"
                + (f" ({labels})" if labels else "")
                + (", reconciled through the sphere" if self.fused else ""))


def frame_number(path: Path) -> int:
    """Sequence number from a filename like clip_fwd_00042.jpg.

    Every camera receives the identical frame set from one split, so the same number
    always means the same instant -- which is what makes cross-camera fusion valid.
    """
    match = _TRAILING_NUMBER.search(path.stem)
    if not match:
        raise MaskError(f"cannot read a frame number from {path.name}")
    return int(match.group(1))


def discover(root: Path, rig: Rig) -> list[CameraImages]:
    """Find each rig camera's extracted images under `root`."""
    images_root = root / "images"
    if not images_root.exists():
        raise MaskError(
            f"{images_root} does not exist. Dynamic masking runs on an extracted "
            f"dataset -- run `360extract extract` first."
        )

    by_name = {camera.name: camera for camera in rig.normalized_cameras()}
    found: list[CameraImages] = []

    for directory in sorted(p for p in images_root.rglob("*") if p.is_dir()):
        camera = by_name.get(directory.name)
        if camera is None:
            continue
        frames = {}
        for image in sorted(directory.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                frames[frame_number(image)] = image
        if frames:
            relative = directory.relative_to(images_root)
            found.append(CameraImages(
                camera=camera,
                directory=directory,
                mask_directory=root / "masks" / relative,
                frames=frames,
            ))
    return found


def run(ffmpeg: FFmpegInfo, root: Path, rig: Rig, backend,
        fuse: bool = True, static: bool = True,
        on_progress=None, on_fraction=None, should_cancel=None) -> DynamicReport:
    """Detect, optionally reconcile, and write masks for an extracted dataset.

    `on_fraction(done, total, message)` reports genuine progress. Without it the UI can
    only show a bar sitting at zero for the whole run, which reads as a hang -- the
    exact complaint that prompted this. `should_cancel()` is checked between frames, so
    a long detection can be stopped without killing the process.
    """
    def cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    def report(done: int, total: int, message: str) -> None:
        if on_fraction:
            on_fraction(done, total, message)
    numpy = fuse_module._numpy()
    root = Path(root)
    cameras = discover(root, rig)
    if not cameras:
        raise MaskError(f"no images for any rig camera under {root / 'images'}")

    report = DynamicReport(cameras=len(cameras), fused=fuse)
    workdir = root / ".threesixty" / "dynamic"
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Detect per camera, writing tile masks out so fusion can work frame by frame
    #    without holding every mask for every camera in memory at once.
    tiles: dict[str, dict[int, Path]] = {}
    # Detection is roughly three quarters of the work; fusion and writing share the
    # rest. Splitting it this way keeps the bar moving at a believable rate throughout.
    total_images = sum(len(entry.frames) for entry in cameras)
    done_images = 0

    for entry in cameras:
        if cancelled():
            return report
        ordered = [entry.frames[n] for n in sorted(entry.frames)]
        if on_progress:
            on_progress(f"detecting in {entry.camera.name} ({len(ordered)} frames)")

        camera_dir = workdir / entry.camera.name
        camera_dir.mkdir(parents=True, exist_ok=True)
        tiles[entry.camera.name] = {}

        # One image at a time so progress and cancellation are per frame rather than
        # per camera; a camera can be hundreds of frames.
        for position, image in enumerate(ordered):
            if cancelled():
                return report
            frame = backend.detect([image])[0]

            report.images += 1
            report.detections += frame.found
            for detection in frame.detections:
                report.by_label[detection.label] = report.by_label.get(detection.label, 0) + 1

            number = frame_number(image)
            target = camera_dir / f"{number:06d}.png"
            fuse_module.write_gray(ffmpeg, frame.mask, target)
            tiles[entry.camera.name][number] = target

            done_images += 1
            report_message = (f"{entry.camera.name}  frame {position + 1} / "
                              f"{len(ordered)}  ·  overall {done_images} / {total_images}")
            report_fraction = 0.75 * done_images / max(total_images, 1)
            if on_fraction:
                on_fraction(report_fraction, 1.0, report_message)

    # 2. Reconcile overlapping cameras, if asked. A pedestrian seen by two cameras and
    #    caught by only one would otherwise be masked in one and trained on in the other.
    if fuse and len(cameras) > 1:
        numbers = sorted(set().union(*(set(t) for t in tiles.values())))
        height = FUSION_WIDTH // 2
        for position, number in enumerate(numbers):
            if cancelled():
                return report
            if on_progress and position % 10 == 0:
                on_progress(f"reconciling frame {position + 1}/{len(numbers)}")
            if on_fraction:
                on_fraction(0.75 + 0.15 * (position / max(len(numbers), 1)), 1.0,
                            f"reconciling cameras  ·  frame {position + 1} / "
                            f"{len(numbers)}")

            present = [(entry.camera, tiles[entry.camera.name][number])
                       for entry in cameras if number in tiles[entry.camera.name]]
            if len(present) < 2:
                continue

            sphere = fuse_module.fuse(ffmpeg, present, FUSION_WIDTH, height,
                                      workdir / f"sphere_{number:06d}.png")
            # Re-project the reconciled sphere back over each camera's tile mask, so
            # every camera now carries the union rather than just its own detections.
            for entry in cameras:
                if number not in tiles[entry.camera.name]:
                    continue
                width, height = _size_of(ffmpeg, entry.frames[number])
                geometric.render_camera_mask(ffmpeg, sphere, entry.camera, width, height,
                                             tiles[entry.camera.name][number])

    # 3. Write the final masks, combining with the static occluder where present.
    static_masks = _static_masks(ffmpeg, rig, cameras, root) if static else {}
    written = 0
    for entry in cameras:
        if cancelled():
            return report
        entry.mask_directory.mkdir(parents=True, exist_ok=True)
        for number, image in sorted(entry.frames.items()):
            written += 1
            if on_fraction and written % 5 == 0:
                on_fraction(0.90 + 0.10 * (written / max(total_images, 1)), 1.0,
                            f"writing masks  ·  {written} / {total_images}")
            tile = tiles[entry.camera.name].get(number)
            if tile is None:
                continue
            width, height = _size_of(ffmpeg, image)
            dynamic = fuse_module.read_gray(ffmpeg, tile, width, height)

            stationary = static_masks.get(entry.camera.name)
            if stationary is not None:
                # Both are "white keeps": the stricter of the two wins.
                dynamic = numpy.minimum(dynamic, stationary)

            fuse_module.write_gray(ffmpeg, dynamic,
                                   entry.mask_directory / f"{image.stem}.png")
            report.masks_written += 1

    return report


def _size_of(ffmpeg: FFmpegInfo, image: Path) -> tuple[int, int]:
    """Pixel size of an extracted image, straight from ffprobe."""
    import subprocess

    from ..ffmpeg import ffprobe_for

    out = subprocess.run(
        [str(ffprobe_for(ffmpeg)), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(image)],
        capture_output=True, text=True).stdout.strip()
    try:
        width, height = (int(v) for v in out.split(",")[:2])
    except ValueError as exc:
        raise MaskError(f"could not read the size of {image}") from exc
    return width, height


def _static_masks(ffmpeg: FFmpegInfo, rig: Rig, cameras: list[CameraImages],
                  root: Path) -> dict:
    """Render the rig's static occluders once per camera, to merge in at the end."""
    occluders = geometric.occluders_of(rig)
    if not occluders:
        return {}

    workdir = root / ".threesixty" / "dynamic"
    equirect = geometric.build_equirect_mask(
        ffmpeg, occluders, FUSION_WIDTH, FUSION_WIDTH // 2, workdir / "static_equirect.png")

    rendered = {}
    for entry in cameras:
        sample = next(iter(sorted(entry.frames.values())), None)
        if sample is None:
            continue
        width, height = _size_of(ffmpeg, sample)
        path = geometric.render_camera_mask(
            ffmpeg, equirect, entry.camera, width, height,
            workdir / f"static_{entry.camera.name}.png")
        rendered[entry.camera.name] = fuse_module.read_gray(ffmpeg, path, width, height)
    return rendered
