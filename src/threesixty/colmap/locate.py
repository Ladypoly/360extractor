"""Finding a usable COLMAP.

Same discipline as `ffmpeg.py`: probe candidates rather than trusting whatever is first
on PATH, and report what was found. Rig support arrived in COLMAP 3.12, and without it
the whole export is pointless -- the version check is the difference between "this will
work" and "this will silently reconstruct each camera as its own rig".
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Rigs and frames were introduced in 3.12; earlier builds ignore rig_config.json.
MIN_VERSION = (3, 12)

#: Places Windows installers and archives tend to land.
COMMON_LOCATIONS = (
    r"C:\Tools\colmap\colmap-x64-windows-cuda\bin",
    r"C:\Tools\colmap\colmap-x64-windows-nocuda\bin",
    r"C:\Program Files\COLMAP\bin",
    r"C:\COLMAP\bin",
)

_VERSION = re.compile(r"COLMAP\s+(\d+)\.(\d+)(?:\.(\d+))?")


@dataclass(frozen=True)
class ColmapInfo:
    path: Path
    version: str
    major: int
    minor: int
    has_cuda: bool
    commands: frozenset[str]
    source: str

    @property
    def has_rig_support(self) -> bool:
        return ((self.major, self.minor) >= MIN_VERSION
                and "rig_configurator" in self.commands)

    @property
    def problem(self) -> str | None:
        if (self.major, self.minor) < MIN_VERSION:
            return (f"version {self.major}.{self.minor} predates rig support "
                    f"(needs {MIN_VERSION[0]}.{MIN_VERSION[1]}+)")
        if "rig_configurator" not in self.commands:
            return "this build has no rig_configurator"
        return None

    @property
    def usable(self) -> bool:
        return self.problem is None


def _candidates(explicit: str | os.PathLike[str] | None) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add(raw, source: str) -> None:
        if not raw:
            return
        path = Path(raw)
        if path.is_dir():
            path = path / ("colmap.exe" if os.name == "nt" else "colmap")
        try:
            key = path.resolve()
        except OSError:
            return
        if key in seen or not path.exists():
            return
        seen.add(key)
        found.append((path, source))

    add(explicit, "--colmap")
    add(os.environ.get("THREESIXTY_COLMAP"), "THREESIXTY_COLMAP")
    add(shutil.which("colmap"), "PATH")
    for location in COMMON_LOCATIONS:
        add(location, "common location")
    return found


def inspect(path: Path, source: str = "explicit") -> ColmapInfo | None:
    try:
        version_out = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace")
        help_out = subprocess.run(
            [str(path), "help"], capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None

    text = (version_out.stdout or "") + (version_out.stderr or "")
    match = _VERSION.search(text)
    if not match:
        return None
    major, minor = int(match.group(1)), int(match.group(2))

    line = next((l.strip() for l in text.splitlines() if "COLMAP" in l), text.strip())
    commands = set(re.findall(r"^\s{2,}(\w+)\s*$",
                              (help_out.stdout or "") + (help_out.stderr or ""),
                              re.M))

    return ColmapInfo(path=path, version=line, major=major, minor=minor,
                      has_cuda="CUDA" in text, commands=frozenset(commands),
                      source=source)


def survey(explicit=None) -> list[ColmapInfo]:
    return [info for info in (inspect(path, source) for path, source in _candidates(explicit))
            if info is not None]


def resolve(explicit=None) -> ColmapInfo | None:
    """The COLMAP to use, or None. Never raises: COLMAP is optional."""
    usable = [info for info in survey(explicit) if info.usable]
    if not usable:
        return None
    return max(usable, key=lambda info: (info.major, info.minor, info.has_cuda))
