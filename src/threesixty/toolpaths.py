"""Where the user says the external tools are.

ffmpeg, COLMAP, Brush and SuperSplat are all found by probing: an explicit flag, then a
`THREESIXTY_*` environment variable, then `PATH`, then a list of places they are commonly
installed. That covers the normal cases and none of the awkward ones -- a build in a
folder nobody would guess had to be re-stated on every run, or exported into the
environment before starting the app at all.

So there is one more step in the chain, between the environment and `PATH`: a path the
user set once and expects to stay set. It lives beside the recent-projects list in
`~/.threesixty`, is per user rather than per project (a tool is a property of the machine),
and is written the same careful way -- atomic replace, `utf-8-sig` on read, a write failure
swallowed rather than allowed to break what the user was actually doing.

The environment still wins over it, because exporting a variable for one run is a
deliberate override of the settled answer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: The tools that can be configured. Names match the `THREESIXTY_<NAME>` variables.
TOOLS = ("ffmpeg", "colmap", "brush", "supersplat")


def _store() -> Path:
    override = os.environ.get("THREESIXTY_STATE_DIR")
    base = Path(override) if override else Path.home() / ".threesixty"
    return base / "tools.json"


def _read() -> dict:
    path = _store()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(paths: dict) -> None:
    try:
        path = _store()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def stored() -> dict:
    """Every configured path, as a tool name -> path string mapping."""
    return {name: str(value) for name, value in _read().items()
            if name in TOOLS and str(value).strip()}


def get(tool: str) -> str:
    """The configured path for one tool, or an empty string."""
    return stored().get(tool, "")


def save(tool: str, path: str | os.PathLike[str] | None) -> None:
    """Set or, with an empty path, clear one tool's location."""
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}; expected one of {TOOLS}")
    paths = stored()
    text = str(path).strip() if path else ""
    if text:
        paths[tool] = text
    else:
        paths.pop(tool, None)
    _write(paths)


def save_many(values: dict) -> dict:
    """Apply several at once, ignoring names that are not tools. Returns what is set."""
    paths = stored()
    for tool, value in values.items():
        if tool not in TOOLS:
            continue
        text = str(value).strip() if value else ""
        if text:
            paths[tool] = text
        else:
            paths.pop(tool, None)
    _write(paths)
    return paths


# -- common install locations ------------------------------------------------
#
# Evaluated on each call rather than frozen at import, so `~` and `%LOCALAPPDATA%`
# resolve for whoever is running -- and so a test can point HOME somewhere else.


def _expand(candidates) -> list[Path]:
    return [Path(os.path.expandvars(os.path.expanduser(raw))) for raw in candidates]


def colmap_locations() -> list[Path]:
    return _expand((
        r"%LOCALAPPDATA%\Programs\COLMAP\bin",
        r"C:\Program Files\COLMAP\bin",
        r"C:\COLMAP\bin",
        "/usr/local/bin", "/usr/bin", "/opt/homebrew/bin", "/snap/bin",
        "~/COLMAP/bin",
        # Last, and deliberately: a personal convention, not a standard location.
        r"C:\Tools\colmap\colmap-x64-windows-cuda\bin",
        r"C:\Tools\colmap\colmap-x64-windows-nocuda\bin",
        r"C:\Tools\colmap\bin",
    ))


def brush_locations() -> list[Path]:
    return _expand((
        r"%LOCALAPPDATA%\Programs\brush",
        r"C:\Program Files\brush",
        # Brush is Rust, so `cargo install` is a real way to have it.
        "~/.cargo/bin",
        "/usr/local/bin", "/usr/bin", "/opt/homebrew/bin",
        "~/brush",
        r"C:\Tools\brush-app-x86_64-pc-windows-msvc",
        r"C:\Tools\brush",
    ))


def supersplat_locations() -> list[Path]:
    # A web build, so there is no PATH to search: it is this list or a configured path.
    return _expand((
        r"%LOCALAPPDATA%\Programs\supersplat\dist",
        r"%LOCALAPPDATA%\Programs\supersplat",
        r"C:\Program Files\supersplat\dist",
        "/usr/local/share/supersplat/dist", "/usr/local/share/supersplat",
        "/opt/supersplat/dist", "/opt/supersplat",
        "~/supersplat/dist", "~/supersplat",
        r"C:\Tools\supersplat\dist",
        r"C:\Tools\supersplat",
    ))
