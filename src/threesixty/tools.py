"""Finding the external pieces: Brush, and the SuperSplat viewer build.

Same discipline as `ffmpeg.py` and `colmap/locate.py` -- probe candidates, never trust
PATH order alone, report what was found so `doctor` and the system-status dialog can
show it, and allow an explicit override.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

BRUSH_LOCATIONS = (
    r"C:\Tools\brush-app-x86_64-pc-windows-msvc",
    r"C:\Tools\brush",
    r"C:\Program Files\brush",
)

SUPERSPLAT_LOCATIONS = (
    r"C:\Tools\supersplat\dist",
    r"C:\Tools\supersplat",
    r"C:\Program Files\supersplat\dist",
)

BRUSH_NAMES = ("brush_app.exe", "brush.exe", "brush_app", "brush")


@dataclass(frozen=True)
class Tool:
    """One discovered dependency."""

    name: str
    path: Path | None
    version: str = ""
    source: str = ""
    detail: str = ""

    @property
    def found(self) -> bool:
        return self.path is not None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "found": self.found,
            "path": str(self.path) if self.path else "",
            "version": self.version,
            "source": self.source,
            "detail": self.detail,
        }


def find_brush(explicit: str | os.PathLike[str] | None = None) -> Tool:
    """Locate the Brush trainer."""
    candidates: list[tuple[Path, str]] = []

    def add(raw, source: str) -> None:
        if not raw:
            return
        path = Path(raw)
        if path.is_dir():
            for name in BRUSH_NAMES:
                if (path / name).exists():
                    candidates.append((path / name, source))
                    return
            return
        if path.exists():
            candidates.append((path, source))

    add(explicit, "--brush")
    add(os.environ.get("THREESIXTY_BRUSH"), "THREESIXTY_BRUSH")
    for name in BRUSH_NAMES:
        found = shutil.which(name)
        if found:
            add(found, "PATH")
            break
    for location in BRUSH_LOCATIONS:
        add(location, "common location")

    for path, source in candidates:
        version = ""
        try:
            proc = subprocess.run([str(path), "--version"], capture_output=True,
                                  text=True, timeout=30, errors="replace")
            version = (proc.stdout or proc.stderr or "").strip().splitlines()[0][:80]
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
        return Tool("Brush", path, version or "installed", source)

    return Tool("Brush", None, detail="needed to train a splat")


def find_supersplat(explicit: str | os.PathLike[str] | None = None) -> Tool:
    """Locate a *built* SuperSplat, which is what can actually be served.

    A source checkout without `index.html` is no use to us, so the check is for the
    built entry point rather than for the folder.
    """
    candidates: list[tuple[Path, str]] = []

    def add(raw, source: str) -> None:
        if raw:
            candidates.append((Path(raw), source))

    add(explicit, "--supersplat")
    add(os.environ.get("THREESIXTY_SUPERSPLAT"), "THREESIXTY_SUPERSPLAT")
    for location in SUPERSPLAT_LOCATIONS:
        add(location, "common location")

    for path, source in candidates:
        for root in (path, path / "dist"):
            if (root / "index.html").exists():
                version = ""
                package = root.parent / "package.json"
                if package.exists():
                    import json
                    try:
                        version = json.loads(package.read_text(encoding="utf-8-sig")
                                             ).get("version", "")
                    except (ValueError, OSError):
                        pass
                return Tool("SuperSplat", root, version, source)

    return Tool("SuperSplat", None,
                detail="a built copy (with index.html) is needed for the viewer")


def survey(ffmpeg_path=None, colmap_path=None, brush_path=None,
           supersplat_path=None) -> list[dict]:
    """Everything the system-status dialog shows, in pipeline order."""
    from .colmap import locate as colmap_locate
    from .ffmpeg import resolve_ffmpeg, FFmpegError

    tools: list[Tool] = []

    try:
        info = resolve_ffmpeg(ffmpeg_path)
        tools.append(Tool("FFmpeg", info.path, info.version, info.source,
                          "v360 filter present"))
    except FFmpegError as exc:
        tools.append(Tool("FFmpeg", None, detail=str(exc)))

    colmap = colmap_locate.resolve(colmap_path)
    if colmap:
        tools.append(Tool("COLMAP", colmap.path, colmap.version, colmap.source,
                          "rig support present" +
                          (", CUDA" if colmap.has_cuda else "")))
    else:
        tools.append(Tool("COLMAP", None,
                          detail="needed to reconstruct camera poses (3.12+)"))

    tools.append(find_brush(brush_path))
    tools.append(find_supersplat(supersplat_path))
    return [tool.describe() for tool in tools]
