"""Recently opened projects, so the UI can offer a list instead of a folder dialog.

A tiny JSON file in the user's home, separate from any single project: it has to
outlive every project and be there before one is open. Written the same careful way
as `project.json` -- atomic replace, `utf-8-sig` on read so a hand-edited or
BOM-carrying file still loads -- because the same Windows encoding trap applies.

Entries are ordered most-recent-first and carry an ``exists`` flag on read rather than
being pruned: a project on a drive that is currently unplugged should still be listed,
greyed, not silently forgotten.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

MAX_ENTRIES = 15


def _store() -> Path:
    """Where the list lives. Overridable so tests do not touch the real home."""
    override = os.environ.get("THREESIXTY_STATE_DIR")
    base = Path(override) if override else Path.home() / ".threesixty"
    return base / "recent.json"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_raw() -> list[dict]:
    path = _store()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def _write(entries: list[dict]) -> None:
    # A recents list is a convenience; failing to write it must never break the open
    # or save that triggered it. Swallow filesystem trouble rather than surface a 500.
    try:
        path = _store()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def record(root: str | os.PathLike[str], name: str) -> None:
    """Move a project to the front of the list, de-duplicated by resolved root."""
    key = str(Path(root).resolve())
    entries = [e for e in _read_raw() if e.get("root") != key]
    entries.insert(0, {"root": key, "name": name, "opened_at": _now()})
    _write(entries[:MAX_ENTRIES])


def remove(root: str | os.PathLike[str]) -> None:
    key = str(Path(root).resolve())
    _write([e for e in _read_raw() if e.get("root") != key])


def entries() -> list[dict]:
    """The list, newest first, each tagged with whether its folder still exists."""
    result = []
    for entry in _read_raw():
        root = entry.get("root", "")
        result.append({**entry, "exists": bool(root) and Path(root).exists()})
    return result
