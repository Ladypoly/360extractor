"""Forward-motion estimation from the video itself, for GPS-free segmentation.

The question segmentation asks is "how far has the vehicle travelled?", and a dashcam-ish
360 clip answers it without GPS: when the rig moves forward the panorama flows outward
from the point straight ahead, and the amount of that flow tracks speed. Sampling a few
low-resolution frames a second and measuring optical flow across the equatorial band
gives a per-interval motion magnitude -- near zero at a red light, larger the faster the
drive. `segment.segment_by_motion` turns that series into cuts.

This is a proxy, not odometry: monocular flow has no absolute scale, and rotation adds
flow too. It is enough to tell moving from stopped and to space segments evenly by
travel; true metres still need an average speed or a GPX track. Needs OpenCV (ships with
the ``[ml]`` extra); `available()` says whether it is importable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .ffmpeg import FFmpegError


def available() -> bool:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def forward_motion(ffmpeg, source: str | Path, sample_fps: float = 2.0,
                   width: int = 320, on_progress=None, should_cancel=None
                   ) -> list[tuple[float, float]]:
    """Per-interval forward-motion magnitude, as `(time_seconds, magnitude)`.

    Decodes the clip downscaled to `width`x`width/2` grayscale at `sample_fps` and runs
    Farneback optical flow between consecutive frames, reporting the mean flow magnitude
    over the equatorial band (the poles are ignored -- equirect distortion there is all
    projection, not motion). The first sample is `(0.0, 0.0)`: there is nothing before it
    to compare against.
    """
    import cv2
    import numpy as np

    if not available():
        raise FFmpegError("forward-motion needs OpenCV: pip install -e \".[ml]\"")

    height = width // 2
    frame_bytes = width * height
    argv = [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", f"fps={sample_fps},scale={width}:{height},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    samples: list[tuple[float, float]] = []
    previous = None
    index = 0
    step = 1.0 / sample_fps
    band = slice(height // 4, height - height // 4)
    try:
        while True:
            if should_cancel and should_cancel():
                break
            buffer = _read_exact(process.stdout, frame_bytes)
            if buffer is None:
                break
            frame = np.frombuffer(buffer, np.uint8).reshape(height, width)
            time = index * step
            if previous is None:
                samples.append((time, 0.0))
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    previous, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude = np.hypot(flow[..., 0], flow[..., 1])
                samples.append((time, float(magnitude[band].mean())))
            previous = frame
            index += 1
            if on_progress and index % 20 == 0:
                on_progress(f"analysing motion: {index} frames")
    finally:
        process.stdout.close()
        error = process.stderr.read().decode("utf-8", "replace")
        process.stderr.close()
        code = process.wait()

    if not samples and code not in (0, None):
        raise FFmpegError(f"motion analysis failed: {error.strip()}")
    return samples


def _read_exact(stream, count: int) -> bytes | None:
    """Read exactly `count` bytes, or None at a clean end of stream."""
    chunks, remaining = [], count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None  # short read at the tail -- an incomplete final frame
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
