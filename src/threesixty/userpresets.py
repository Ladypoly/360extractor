"""User-defined rig presets, saved globally so they show up in every project.

Kept beside the recent-projects list in the user's ~/.threesixty state directory (see
[[canonical-repo-location]] for where that resolves): a rig worked out once should be
one dropdown pick away in the next project, not re-derived by hand. The built-in presets
in `rig.PRESETS` are code; these are the user's own, and the only ones that can be
overwritten or deleted.

Written the careful way the rest of the project writes JSON -- atomic replace,
`utf-8-sig` on read -- and, like the recents list, a write failure is swallowed rather
than allowed to break the save the user asked for.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _store() -> Path:
    override = os.environ.get("THREESIXTY_STATE_DIR")
    base = Path(override) if override else Path.home() / ".threesixty"
    return base / "presets.json"


def _read() -> dict:
    path = _store()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(presets: dict) -> None:
    try:
        path = _store()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(presets, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def stored() -> dict:
    """Every saved preset, as a name -> rig-dict mapping."""
    return _read()


def save(name: str, rig: dict) -> None:
    presets = _read()
    presets[name] = rig
    _write(presets)


def delete(name: str) -> None:
    presets = _read()
    if presets.pop(name, None) is not None:
        _write(presets)
