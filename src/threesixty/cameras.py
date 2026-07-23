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

from .ffmpeg import FFmpegError, FFmpegInfo, probe_media
from .plan import build_filter_graph, camera_size, safe_stem
from .rig import Rig


@dataclass
class CamerasResult:
    directories: list[Path] = field(default_factory=list)
    images_written: int = 0
    cancelled: bool = False


def _sequence(frames_directory: Path) -> tuple[str, int]:
    """The image2 pattern and start number for an extracted frame folder."""
    files = sorted(frames_directory.glob("*.jpg"))
    if not files:
        raise FFmpegError(
            f"no frames in {frames_directory}; extract frames before generating cameras")
    digits = len(files[0].stem)
    return f"%0{digits}d.jpg", int(files[0].stem)


def generate_cameras(ffmpeg: FFmpegInfo, frames_directory: str | Path, rig: Rig,
                     output_root: str | Path, clip: str | None = None,
                     on_progress=None, should_cancel=None,
                     overwrite: bool = True) -> CamerasResult:
    """Project every extracted frame through the rig into images/<clip>/<camera>/."""
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
    return CamerasResult(directories=directories, images_written=written,
                         cancelled=cancelled)
