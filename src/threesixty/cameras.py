"""Stage B of capture: project extracted equirect frames into camera tiles.

Stage A left a neutral working set of panorama frames in ``frames/<clip>/``. This is the
second half: read that image sequence, apply the grade, and fan it out to one rectilinear
tile per camera with the same one-decode->split->v360 graph the single-pass extractor
used -- only the input is the frame sequence, not the video. Every camera writes the same
frame numbers so COLMAP still groups a frame's tiles across camera folders by filename.

Masks are projected the same way (a following change), by running the identical v360 over
the per-frame equirect mask sequence with nearest-neighbour sampling.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from dataclasses import replace as dc_replace

from .ffmpeg import FFmpegError, FFmpegInfo, probe_media
from .mask import geometric
from .mask.apply import link_sidecars
from .plan import build_filter_graph, camera_size, safe_stem
from .rig import Grade, Rig


@dataclass
class CamerasResult:
    directories: list[Path] = field(default_factory=list)
    images_written: int = 0
    masks_written: int = 0
    cancelled: bool = False


def _project_masks(ffmpeg: FFmpegInfo, rig: Rig, cameras, sizes, directories,
                   sample, output_root: Path, clip: str,
                   sky_cone_angle: float | None) -> int:
    """Project static occluders and the sky cone into per-camera sidecar masks.

    Static occluders are rigid to the rig, so one equirect mask projects to one mask per
    camera and links beside every frame -- the same guaranteed-alignment trick the
    single-pass extractor used. Sky exclusion rides along as a zenith cone here; the
    per-frame semantic path (detection on the equirect frames) layers on later.
    """
    raw = list(rig.occluders)
    if sky_cone_angle and sky_cone_angle > 0:
        raw.append({"type": "zenith_cone", "angle": float(sky_cone_angle)})
    occluders = [o for o in (geometric.Occluder.from_dict(d) for d in raw)
                 if o.kind != "ml"]
    if not occluders:
        return 0

    work = output_root / ".threesixty" / "masks"
    equirect = geometric.build_equirect_mask(
        ffmpeg, occluders, sample.width, sample.height or sample.width // 2,
        work / "equirect.png")
    total = 0
    for camera, (width, height), image_dir in zip(cameras, sizes, directories):
        camera_mask = geometric.render_camera_mask(
            ffmpeg, equirect, camera, width, height, work / f"{camera.name}.png")
        total += link_sidecars(camera_mask, image_dir,
                               output_root / "masks" / clip / camera.name)
    return total


def _sequence(frames_directory: Path) -> tuple[str, int]:
    """The image2 pattern and start number for an extracted frame folder."""
    files = sorted(frames_directory.glob("*.jpg"))
    if not files:
        raise FFmpegError(
            f"no frames in {frames_directory}; extract frames before generating cameras")
    digits = len(files[0].stem)
    return f"%0{digits}d.jpg", int(files[0].stem)


def _detect_equirect_masks(ffmpeg, frames_directory: Path, rig: Rig, detect,
                           sky_cone_angle, work_dir: Path, on_progress, should_cancel):
    """Detect on each equirect frame and write a per-frame ignore-mask; return its dir.

    Detection runs on the panorama frames themselves (the two-stage choice), each combined
    with the static occluders / sky cone. Returns None when there is nothing to detect
    (no classes or no ML), so the caller falls back to the static-only path.
    """
    classes = list(getattr(detect, "classes", []) or [])
    if not classes:
        return None
    from .mask import ml
    if not ml.available():
        return None
    import cv2
    import numpy as np

    backend = ml.make_backend(
        getattr(detect, "backend", "sam2.1"), classes=classes,
        confidence=getattr(detect, "confidence", 0.25),
        dilate=getattr(detect, "dilate", 6), device=getattr(detect, "device", None))

    frames = sorted(frames_directory.glob("*.jpg"))
    if not frames:
        return None
    sample = cv2.imread(str(frames[0]))
    height, width = sample.shape[:2]

    raw = list(rig.occluders)
    if sky_cone_angle and sky_cone_angle > 0:
        raw.append({"type": "zenith_cone", "angle": float(sky_cone_angle)})
    occluders = [o for o in (geometric.Occluder.from_dict(d) for d in raw)
                 if o.kind != "ml"]
    static = None
    if occluders:
        static_path = geometric.build_equirect_mask(
            ffmpeg, occluders, width, height, work_dir / "static.png")
        static = cv2.imread(str(static_path), cv2.IMREAD_GRAYSCALE)

    out_dir = work_dir / "equirect_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(frames)
    for index, frame in enumerate(frames):
        if should_cancel and should_cancel():
            break
        mask = np.asarray(backend.detect([frame])[0].mask)   # white keeps, black ignores
        if static is not None:
            stationary = static if static.shape == mask.shape else \
                cv2.resize(static, (mask.shape[1], mask.shape[0]))
            mask = np.minimum(mask, stationary)              # the stricter of the two wins
        cv2.imwrite(str(out_dir / f"{frame.stem}.png"), mask)
        if on_progress is not None:
            on_progress((index + 1) / total, index + 1, 0.0)
    return out_dir


def _project_mask_sequence(ffmpeg, mask_sequence_dir: Path, rig: Rig, cameras, sizes,
                           output_root: Path, clip: str, start_number: int,
                           pattern: str) -> int:
    """Project the per-frame equirect mask sequence into per-camera mask sidecars."""
    mask_pattern = pattern.replace(".jpg", ".png")
    # Neutral grade and nearest sampling: a mask must not be graded or interpolated.
    mask_rig = dc_replace(rig, grade=Grade(),
                          output=dc_replace(rig.output, interp="near", format="png"))
    graph, labels = build_filter_graph(cameras, mask_rig, "", sizes=sizes, burn=False)

    argv = [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-start_number", str(start_number),
            "-i", str(mask_sequence_dir / mask_pattern), "-filter_complex", graph]
    directories = []
    for label, camera in zip(labels, cameras):
        mask_dir = output_root / "masks" / clip / camera.name
        mask_dir.mkdir(parents=True, exist_ok=True)
        directories.append(mask_dir)
        argv += ["-map", f"[{label}]", "-start_number", str(start_number),
                 str(mask_dir / mask_pattern)]
    result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise FFmpegError(f"mask projection failed: {result.stderr.strip()}")
    return sum(len(list(directory.glob("*.png"))) for directory in directories)


def generate_cameras(ffmpeg: FFmpegInfo, frames_directory: str | Path, rig: Rig,
                     output_root: str | Path, clip: str | None = None,
                     sky_cone_angle: float | None = None, detect=None,
                     on_progress=None, on_mask_progress=None, should_cancel=None,
                     overwrite: bool = True) -> CamerasResult:
    """Project every extracted frame through the rig into images/<clip>/<camera>/.

    Also writes the matching per-camera mask sidecars so the result is training-ready:
    per-frame detection on the panorama frames when `detect` has classes and ML is
    installed, otherwise just the static occluders / sky cone.
    """
    rig.validate()
    frames_directory = Path(frames_directory)
    pattern, start_number = _sequence(frames_directory)
    clip = clip or safe_stem(frames_directory.name)

    # The equirect frame's own dimensions drive the auto output size, exactly as the
    # source video's did in the single-pass path.
    sample = probe_media(sorted(frames_directory.glob("*.jpg"))[0], ffmpeg)
    cameras = rig.normalized_cameras()
    sizes = [camera_size(camera, rig, sample) for camera in cameras]
    # prefix "" -- no thinning here (frames are already the kept set); build_filter_graph
    # folds in the grade itself.
    graph, labels = build_filter_graph(cameras, rig, "", sizes=sizes, burn=False)

    argv = [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-y" if overwrite else "-n", "-progress", "pipe:1",
            "-start_number", str(start_number), "-i", str(frames_directory / pattern),
            "-filter_complex", graph]

    root = Path(output_root)
    directories: list[Path] = []
    for label, camera in zip(labels, cameras):
        camera_dir = root / "images" / clip / camera.name
        camera_dir.mkdir(parents=True, exist_ok=True)
        directories.append(camera_dir)
        argv += ["-map", f"[{label}]", "-start_number", str(start_number),
                 "-q:v", str(rig.output.quality), str(camera_dir / pattern)]

    expected = len(list(frames_directory.glob("*.jpg")))
    started = time.monotonic()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    cancelled = False
    try:
        for line in process.stdout:
            if should_cancel and should_cancel():
                process.terminate()
                cancelled = True
                break
            key, _, value = line.strip().partition("=")
            if key == "frame" and on_progress is not None:
                try:
                    frame = int(value)
                except ValueError:
                    continue
                on_progress(min(frame / max(expected, 1), 1.0), frame,
                            time.monotonic() - started)
    finally:
        error = process.stderr.read() if process.stderr else ""
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
        code = process.wait()

    if not cancelled and code not in (0, None):
        raise FFmpegError(f"camera generation failed: {error.strip()}")

    written = sum(len(list(directory.glob("*.jpg"))) for directory in directories)
    masks_written = 0
    if not cancelled:
        root = Path(output_root)
        equirect_masks = _detect_equirect_masks(
            ffmpeg, frames_directory, rig, detect, sky_cone_angle,
            root / ".threesixty" / "masks", on_mask_progress, should_cancel)
        if equirect_masks is not None:
            # Per-frame masks (detection): project the whole sequence.
            masks_written = _project_mask_sequence(
                ffmpeg, equirect_masks, rig, cameras, sizes, root, clip,
                start_number, pattern)
        else:
            # Static only: one rigid mask per camera, linked beside every frame.
            masks_written = _project_masks(ffmpeg, rig, cameras, sizes, directories,
                                           sample, root, clip, sky_cone_angle)
    return CamerasResult(directories=directories, images_written=written,
                         masks_written=masks_written, cancelled=cancelled)
