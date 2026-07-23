"""Projects: everything needed to reproduce a dataset, and what has been done so far.

Until now the state of a job lived in several places at once -- a rig file here, a
painted occluder in a temp directory there, the frame rate and mask mode only in
whichever command line happened to be typed. Close the window and most of it was gone.

A project is one `project.json` sitting at the root of the dataset it describes, beside
`images/` and `masks/`. That makes the folder self-describing: move it, hand it to
someone else, come back to it in a month, and the settings arrive with the pixels.

It also records **what has already been done**. Each stage stores a fingerprint of the
settings that produced it, so the tool can tell the difference between "already
extracted" and "extracted, but you have since changed the rig".
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .rig import Rig, RigError, ring

SCHEMA_VERSION = 1
PROJECT_FILENAME = "project.json"

STAGES = ("extract", "mask", "export")


class ProjectError(ValueError):
    """A project file is missing, malformed, or from a newer build."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class FrameSettings:
    """How frames are chosen from each source."""

    mode: str = "sharp"
    value: float = 2.0
    start: float | None = None
    end: float | None = None


@dataclass
class OutputSettings:
    """Where and how the images land."""

    layout: str = "brush"
    mask_mode: str = "sidecar"


@dataclass
class DetectSettings:
    """Dynamic occluder detection, and the always-on sky exclusion."""

    backend: str = "sam2.1"
    classes: list[str] = field(default_factory=lambda: [
        "person", "car", "bus", "truck", "motorcycle", "bicycle"])
    confidence: float = 0.25
    dilate: int = 6
    device: str | None = None
    fuse: bool = True
    #: The sky cone: a geometric fallback that masks everything above `sky_cone_angle`
    #: degrees of elevation. Off by default now that sky is handled as a detection class
    #: (an open-vocabulary "sky" via YOLO-World follows the real horizon); kept for a
    #: dependency-free option when ML is unavailable.
    exclude_sky: bool = False
    sky_method: str = "cone"
    sky_cone_angle: float = 30.0


@dataclass
class Stage:
    """A completed pipeline step."""

    done_at: str = ""
    fingerprint: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    """A dataset plus the settings that produced it."""

    root: Path
    name: str = "project"
    sources: list[str] = field(default_factory=list)
    rig: Rig = field(default_factory=lambda: ring(8))
    frames: FrameSettings = field(default_factory=FrameSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    detect: DetectSettings = field(default_factory=DetectSettings)
    stages: dict[str, Stage] = field(default_factory=dict)
    created: str = ""
    modified: str = ""

    # -- locations ----------------------------------------------------------

    @property
    def file(self) -> Path:
        return self.root / PROJECT_FILENAME

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def assets_dir(self) -> Path:
        """Where the project's own files live -- the painted occluder, mainly.

        Kept inside the project rather than in a temp directory, which is what used to
        happen: a painted mask that disappears on reboot takes the rig's occluder
        reference down with it.
        """
        return self.root / "assets"

    def relative(self, path: str | os.PathLike[str]) -> str:
        """Store paths inside the project relative, so the folder stays portable."""
        target = Path(path)
        try:
            return target.resolve().relative_to(self.root.resolve()).as_posix()
        except (ValueError, OSError):
            return str(target)

    def absolute(self, stored: str) -> Path:
        """Resolve a stored path back, relative to the project root."""
        path = Path(stored)
        return path if path.is_absolute() else (self.root / path)

    def resolved_sources(self) -> list[Path]:
        return [self.absolute(source) for source in self.sources]

    def missing_sources(self) -> list[Path]:
        """Sources that no longer exist. Reported rather than raised on load.

        A project whose footage lives on a drive that is not plugged in should still
        open, so the rig can be looked at and the settings edited.
        """
        return [path for path in self.resolved_sources() if not path.exists()]

    # -- fingerprints -------------------------------------------------------

    def fingerprint(self, stage: str) -> str:
        """Hash of everything a stage's output depends on.

        Changing the rig invalidates extraction; changing detection settings
        invalidates masking but not extraction. This is what lets `is_stale` say
        something more useful than "files exist".
        """
        if stage == "extract":
            payload = {
                "sources": sorted(self.sources),
                "rig": self.rig.to_dict(),
                "frames": asdict(self.frames),
                "layout": self.output.layout,
                "mask_mode": self.output.mask_mode,
            }
        elif stage == "mask":
            payload = {
                "extract": self.fingerprint("extract"),
                "detect": asdict(self.detect),
                "occluders": self.rig.occluders,
            }
        elif stage == "export":
            payload = {"mask": self.fingerprint("mask")}
        else:
            raise ProjectError(f"unknown stage {stage!r}; expected one of {STAGES}")

        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def is_done(self, stage: str) -> bool:
        return stage in self.stages and bool(self.stages[stage].fingerprint)

    def is_stale(self, stage: str) -> bool:
        """Done, but with settings that have since changed."""
        if not self.is_done(stage):
            return False
        return self.stages[stage].fingerprint != self.fingerprint(stage)

    def status(self, stage: str) -> str:
        if not self.is_done(stage):
            return "pending"
        return "stale" if self.is_stale(stage) else "done"

    def mark_done(self, stage: str, **details: Any) -> Stage:
        if stage not in STAGES:
            raise ProjectError(f"unknown stage {stage!r}; expected one of {STAGES}")
        record = Stage(done_at=_now(), fingerprint=self.fingerprint(stage), details=details)
        self.stages[stage] = record
        # A redone stage invalidates the ones after it, which would otherwise claim to
        # be current while describing images that no longer exist.
        for later in STAGES[STAGES.index(stage) + 1:]:
            self.stages.pop(later, None)
        return record

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "name": self.name,
            "created": self.created or _now(),
            "modified": _now(),
            "sources": self.sources,
            "rig": self.rig.to_dict(),
            "frames": asdict(self.frames),
            "output": asdict(self.output),
            "detect": asdict(self.detect),
            "stages": {name: asdict(stage) for name, stage in self.stages.items()},
        }

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        self.created = payload["created"]
        self.modified = payload["modified"]
        # Written via a temporary file: a half-written project.json after a crash
        # would lose every setting, not just the last edit.
        temporary = self.file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.file)
        return self.file

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path) -> "Project":
        version = data.get("version", SCHEMA_VERSION)
        if version > SCHEMA_VERSION:
            raise ProjectError(
                f"project uses schema version {version}, this build understands up to "
                f"{SCHEMA_VERSION} -- upgrade 360extract"
            )
        try:
            rig = Rig.from_dict(data["rig"]) if "rig" in data else ring(8)
        except (RigError, KeyError) as exc:
            raise ProjectError(f"project contains an invalid rig: {exc}") from exc

        try:
            project = cls(
                root=root,
                name=data.get("name", root.name),
                sources=list(data.get("sources", [])),
                rig=rig,
                frames=FrameSettings(**data.get("frames", {})),
                output=OutputSettings(**data.get("output", {})),
                detect=DetectSettings(**data.get("detect", {})),
                created=data.get("created", ""),
                modified=data.get("modified", ""),
            )
        except TypeError as exc:
            raise ProjectError(f"unrecognized field in project: {exc}") from exc

        for name, raw in (data.get("stages") or {}).items():
            if name in STAGES:
                project.stages[name] = Stage(**raw)
        return project

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Project":
        """Open a project from its file, or from the folder containing one."""
        target = Path(path)
        if target.is_dir():
            target = target / PROJECT_FILENAME
        if not target.exists():
            raise ProjectError(
                f"no project at {target}. Create one with `360extract project new`."
            )
        try:
            # utf-8-sig, not utf-8: Notepad and PowerShell's Set-Content both write a
            # BOM, and a hand-edited project would otherwise fail to open with a
            # message about byte order marks.
            data = json.loads(target.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ProjectError(f"{target} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ProjectError(f"{target} must contain a JSON object")
        return cls.from_dict(data, target.parent)

    @classmethod
    def create(cls, root: str | os.PathLike[str], sources: list[str] | None = None,
               rig: Rig | None = None, name: str | None = None,
               overwrite: bool = False) -> "Project":
        directory = Path(root)
        project = cls(root=directory, name=name or directory.name, rig=rig or ring(8))
        if project.file.exists() and not overwrite:
            raise ProjectError(
                f"{project.file} already exists. Use --force to replace it."
            )
        project.sources = [project.relative(s) for s in (sources or [])]
        project.created = _now()
        project.save()
        return project

    # -- snapshots ----------------------------------------------------------

    @property
    def snapshots_dir(self) -> Path:
        return self.root / ".threesixty" / "snapshots"

    def snapshot(self, label: str) -> Path:
        """Keep a named copy of the current settings.

        Cheap insurance for the moment before a big change: it stores settings only,
        not the images, so it costs a few kilobytes.
        """
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label).strip("_")
        if not safe:
            raise ProjectError("snapshot needs a name")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        target = self.snapshots_dir / f"{safe}.json"
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def snapshots(self) -> list[str]:
        if not self.snapshots_dir.exists():
            return []
        return sorted(p.stem for p in self.snapshots_dir.glob("*.json"))

    def restore(self, label: str) -> "Project":
        """Load a snapshot's settings back over this project, without saving."""
        target = self.snapshots_dir / f"{label}.json"
        if not target.exists():
            available = ", ".join(self.snapshots()) or "none"
            raise ProjectError(f"no snapshot named {label!r}. Available: {available}")
        data = json.loads(target.read_text(encoding="utf-8-sig"))
        return Project.from_dict(data, self.root)


def find(start: str | os.PathLike[str]) -> Path | None:
    """Look for a project in a folder or any of its parents."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / PROJECT_FILENAME).exists():
            return candidate / PROJECT_FILENAME
    return None
