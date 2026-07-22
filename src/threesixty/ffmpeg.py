"""Locating, probing and driving ffmpeg.

The one rule in this module: never trust PATH order. On a typical workstation several
ffmpeg builds are installed and the first one found is frequently an old one that
predates the ``v360`` filter. :func:`resolve_ffmpeg` enumerates every candidate and
picks the best, rather than taking the first.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# v360 gained rectilinear output in ffmpeg 4.2, but rotation order and the fov
# handling only settled down in 5.x. Below this we refuse rather than produce
# subtly wrong angles.
MIN_MAJOR = 5

_VERSION_RE = re.compile(r"ffmpeg version (?:n)?(\d+)\.(\d+)")


class FFmpegError(RuntimeError):
    """ffmpeg is missing, too old, or failed on a specific invocation."""


@dataclass(frozen=True)
class FFmpegInfo:
    """A single ffmpeg binary we found and interrogated."""

    path: Path
    version: str
    major: int
    minor: int
    has_v360: bool
    source: str  # how we found it, for `doctor` output

    @property
    def usable(self) -> bool:
        return self.has_v360 and self.major >= MIN_MAJOR

    @property
    def problem(self) -> str | None:
        if not self.has_v360:
            return "no v360 filter"
        if self.major < MIN_MAJOR:
            return f"version {self.major}.{self.minor} is older than the required {MIN_MAJOR}.0"
        return None


def _run(argv: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run a short-lived helper command.

    Always an argv list, never ``shell=True`` -- filter graphs contain ``[0:v]`` and
    other characters that shells (MSYS in particular) mangle beyond recognition.
    """
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def _candidate_paths(explicit: str | os.PathLike[str] | None) -> list[tuple[Path, str]]:
    """Every ffmpeg worth probing, in preference order, tagged with its origin."""
    found: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add(raw: str | os.PathLike[str] | None, source: str) -> None:
        if not raw:
            return
        path = Path(raw)
        if path.is_dir():
            path = path / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        try:
            key = path.resolve()
        except OSError:
            return
        if key in seen or not path.exists():
            return
        seen.add(key)
        found.append((path, source))

    add(explicit, "--ffmpeg")
    add(os.environ.get("THREESIXTY_FFMPEG"), "THREESIXTY_FFMPEG")
    add(Path(__file__).parent / "bin", "bundled")

    # Every ffmpeg on PATH, not just the first: the first is often the oldest.
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry:
            add(Path(entry) / exe, "PATH")
    add(shutil.which("ffmpeg"), "PATH")

    return found


def inspect_ffmpeg(path: Path, source: str = "explicit") -> FFmpegInfo | None:
    """Probe one binary. Returns None if it is not a working ffmpeg at all."""
    try:
        proc = _run([str(path), "-hide_banner", "-version"])
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None

    first = proc.stdout.splitlines()[0] if proc.stdout else ""
    match = _VERSION_RE.search(first)
    major, minor = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    version = first.split(" Copyright")[0].removeprefix("ffmpeg version ").strip() or "unknown"

    has_v360 = False
    try:
        # `-h filter=v360` exits 0 and prints the option table when the filter exists.
        help_proc = _run([str(path), "-hide_banner", "-h", "filter=v360"])
        has_v360 = "AVOptions" in help_proc.stdout and "v360" in help_proc.stdout
    except (OSError, subprocess.SubprocessError):
        pass

    return FFmpegInfo(path=path, version=version, major=major, minor=minor,
                      has_v360=has_v360, source=source)


def survey_ffmpeg(explicit: str | os.PathLike[str] | None = None) -> list[FFmpegInfo]:
    """Probe every candidate. Used by `doctor` to show what is really installed."""
    results = []
    for path, source in _candidate_paths(explicit):
        info = inspect_ffmpeg(path, source)
        if info is not None:
            results.append(info)
    return results


def resolve_ffmpeg(explicit: str | os.PathLike[str] | None = None) -> FFmpegInfo:
    """Pick the ffmpeg we will actually use.

    An explicitly requested binary is used or the call fails -- we never silently
    substitute a different one when the user named a path. Otherwise the newest
    usable build wins, which is what makes an old PATH-shadowing ffmpeg harmless.
    """
    candidates = survey_ffmpeg(explicit)
    if not candidates:
        raise FFmpegError(
            "no ffmpeg found. Install ffmpeg 5.0+ and put it on PATH, or pass --ffmpeg "
            "/path/to/ffmpeg, or set THREESIXTY_FFMPEG."
        )

    if explicit is not None:
        chosen = candidates[0]
        if not chosen.usable:
            raise FFmpegError(f"{chosen.path} is unusable: {chosen.problem}")
        return chosen

    usable = [c for c in candidates if c.usable]
    if not usable:
        lines = [f"  {c.path}  ({c.version}) -- {c.problem}" for c in candidates]
        raise FFmpegError(
            "found ffmpeg, but no usable build. Need version "
            f"{MIN_MAJOR}.0+ with the v360 filter.\n" + "\n".join(lines)
        )
    # Newest wins; ties broken by candidate order, which is preference order.
    return max(usable, key=lambda c: (c.major, c.minor))


def ffprobe_for(ffmpeg: FFmpegInfo) -> Path:
    """The ffprobe that ships beside a given ffmpeg, falling back to PATH."""
    sibling = ffmpeg.path.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    if sibling.exists():
        return sibling
    found = shutil.which("ffprobe")
    if found:
        return Path(found)
    raise FFmpegError(f"no ffprobe found next to {ffmpeg.path} or on PATH")


@dataclass(frozen=True)
class MediaInfo:
    """What we need to know about a source file to plan an extraction."""

    path: Path
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    codec: str
    is_video: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def looks_equirectangular(self) -> bool:
        # Equirect is 2:1 by definition. Allow a little slack for odd encoder padding.
        return 1.9 <= self.aspect <= 2.1


def _parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            denominator = float(den)
            return float(num) / denominator if denominator else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe_media(path: str | os.PathLike[str], ffmpeg: FFmpegInfo | None = None) -> MediaInfo:
    """Read dimensions, frame rate and duration from a video or still."""
    media_path = Path(path)
    if not media_path.exists():
        raise FFmpegError(f"no such file: {media_path}")

    ffmpeg = ffmpeg or resolve_ffmpeg()
    argv = [
        str(ffprobe_for(ffmpeg)), "-hide_banner", "-loglevel", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(media_path),
    ]
    proc = _run(argv, timeout=60)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {media_path}:\n{proc.stderr.strip()}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"could not parse ffprobe output for {media_path}") from exc

    streams = data.get("streams") or []
    if not streams:
        raise FFmpegError(f"{media_path} contains no video stream")
    stream = streams[0]

    fps = _parse_fraction(stream.get("r_frame_rate"))
    duration = _parse_fraction(stream.get("duration")) or _parse_fraction(
        (data.get("format") or {}).get("duration")
    )

    try:
        frame_count = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0
    if not frame_count and fps and duration:
        frame_count = int(round(fps * duration))

    codec = str(stream.get("codec_name") or "unknown")
    # Stills decode as a single frame regardless of codec; ffprobe reports a nominal
    # fps for them, so frame count is the only reliable discriminator.
    is_video = frame_count > 1

    return MediaInfo(
        path=media_path,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=fps,
        duration=duration,
        frame_count=max(frame_count, 1),
        codec=codec,
        is_video=is_video,
    )
