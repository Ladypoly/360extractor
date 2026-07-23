"""The work behind Reconstruct, Train and Inspect.

Kept out of the request handler so each stage is a plain function of
(job, project, settings) and can be tested without a server. Every one of them runs on
a job from `jobs.py`, so they report a real fraction, stream a log, and stop when asked.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..colmap import export as colmap_export
from ..colmap import locate as colmap_locate
from ..colmap.model import read_model
from ..plan import safe_stem
from ..project import Project
from ..tools import find_brush
from .jobs import Job
from .runner import ProgressPattern, Step, run, run_steps

#: COLMAP prints `Registering image #24 (40)` while mapping, which is the only honest
#: progress signal the whole reconstruction offers.
COLMAP_REGISTERING = ProgressPattern(
    re.compile(r"Registering image #\d+\s*\((\d+)\)"), message="registering images")

#: ...and this while extracting features.
COLMAP_FEATURES = ProgressPattern(
    re.compile(r"Processed file \[(\d+)/(\d+)\]"), message="extracting features")

#: Brush prints its step counter; accept a few plausible shapes rather than betting on
#: one, since this is cosmetic and a miss only costs a smooth bar.
BRUSH_STEPS = ProgressPattern(
    re.compile(r"(?:step|iter\w*)\D{0,4}(\d+)\s*/\s*(\d+)", re.I), message="training")


class StageError(RuntimeError):
    """A stage cannot run yet, and the message says what is missing."""


# -- readiness --------------------------------------------------------------


def sparse_model_dir(project: Project) -> Path | None:
    """The sparse model COLMAP produced, if there is one."""
    for candidate in (project.root / "sparse" / "0", project.root / "sparse"):
        if (candidate / "images.bin").exists() or (candidate / "images.txt").exists():
            return candidate
    return None


def trained_splats(project: Project) -> list[Path]:
    """Every .ply that looks like a trained splat, newest first.

    Cleaned and removed outputs are skipped: offering the user the result of a previous
    cleanup as the thing to clean is a trap.
    """
    splat_dir = project.root / "splat"
    if not splat_dir.exists():
        return []
    found = [p for p in splat_dir.glob("*.ply")
             if not p.stem.endswith(("_cleaned", "_removed"))]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def readiness(project: Project | None) -> dict[str, dict]:
    """Which stages can run, and -- when they cannot -- exactly why.

    The UI shows disabled stages rather than hiding them, so the reason matters as much
    as the flag.
    """
    if project is None:
        blocked = {"ready": False, "reason": "Open or create a project first."}
        states = {stage: dict(blocked) for stage in
                  ("capture", "reconstruct", "train", "inspect")}
        states["start"] = {"ready": True, "reason": ""}   # the entry point is always open
        return states

    images_root = project.root / "images"
    has_images = images_root.exists() and any(
        next(images_root.rglob(pattern), None) is not None
        for pattern in ("*.jpg", "*.jpeg", "*.png"))
    sparse = sparse_model_dir(project)
    splats = trained_splats(project)

    return {
        "start": {"ready": True, "reason": ""},
        "capture": {
            "ready": bool(project.sources),
            "reason": "" if project.sources else "Load a 360 video or still first.",
        },
        "reconstruct": {
            "ready": has_images,
            "reason": "" if has_images else "Extract images before reconstruction can begin.",
        },
        "train": {
            "ready": sparse is not None,
            "reason": "" if sparse else "Run reconstruction before training.",
        },
        "inspect": {
            "ready": bool(splats),
            "reason": "" if splats else "No trained .ply file is available yet.",
        },
    }


# -- reconstruct ------------------------------------------------------------


def reconstruction_steps(project: Project, colmap: Path, clip: str,
                         geo: bool = False, sequential: bool = True) -> list[Step]:
    """The COLMAP pipeline for this project, in the order it must run.

    The rig has to be configured *before* matching: sequential matching pairs images by
    frame, and without the rig there are no frames to pair by.
    """
    root = project.root
    database = root / "database.db"
    masks = root / "masks"

    extract = [str(colmap), "feature_extractor",
               "--image_path", str(root / "images"),
               "--database_path", str(database),
               "--ImageReader.single_camera_per_folder", "1"]
    if masks.exists():
        extract += ["--ImageReader.mask_path", str(masks)]

    steps = [
        Step("features", "Feature extraction", extract, COLMAP_FEATURES),
        Step("rig", "Rig configuration",
             [str(colmap), "rig_configurator",
              "--database_path", str(database),
              "--rig_config_path", str(root / "rig_config.json")]),
        Step("match", "Feature matching",
             [str(colmap), f"{'sequential' if sequential else 'exhaustive'}_matcher",
              "--database_path", str(database)]),
        Step("map", "Mapping",
             [str(colmap), "mapper",
              "--database_path", str(database),
              "--image_path", str(root / "images"),
              "--output_path", str(root / "sparse"),
              "--Mapper.ba_refine_sensor_from_rig", "0",
              # Snapshots let the point cloud be shown as it is built, not only at the end.
              "--Mapper.snapshot_path", str(root / "sparse" / "snapshots"),
              "--Mapper.snapshot_frames_freq", "20"],
             COLMAP_REGISTERING),
    ]

    if geo and (root / "geo_registration.txt").exists():
        steps.append(Step(
            "geo", "Geo alignment",
            [str(colmap), "model_aligner",
             "--input_path", str(root / "sparse" / "0"),
             "--output_path", str(root / "sparse" / "aligned"),
             "--ref_images_path", str(root / "geo_registration.txt"),
             "--ref_is_gps", "1", "--alignment_type", "enu",
             "--robust_alignment_max_error", "3.0"],
            optional=True))

    return steps


def reconstruction_metrics(project: Project) -> dict:
    """What the reconstruction actually achieved, read back from the model.

    The rig spread is the one worth showing: every camera in a frame shares an optical
    centre by construction, so a non-zero spread means COLMAP did not honour the rig.
    """
    sparse = sparse_model_dir(project)
    if sparse is None:
        return {}

    try:
        model = read_model(sparse)
    except Exception:                                     # noqa: BLE001
        return {}

    from ..splat.clean import trajectory_from_model
    metrics = {
        "registered_images": len(model.images),
        "cameras": len(model.cameras),
    }
    try:
        trajectory = trajectory_from_model(model)
        metrics["frames"] = len(trajectory)
        metrics["rig_spread"] = round(trajectory.spread, 6)
        metrics["path_length"] = round(trajectory.length, 3)
    except Exception:                                     # noqa: BLE001
        pass
    return metrics


def run_reconstruction(job: Job, project: Project, settings: dict) -> dict:
    """Run COLMAP for real, step by step."""
    colmap = colmap_locate.resolve(settings.get("colmap"))
    if colmap is None:
        raise StageError(
            "No COLMAP with rig support was found. Install COLMAP 3.12 or newer, "
            "or set its path in System status.")

    sources = project.resolved_sources()
    if not sources:
        raise StageError("This project has no source video.")
    clip = safe_stem(sources[0].stem)

    if not (project.root / "rig_config.json").exists():
        from ..ffmpeg import probe_media, resolve_ffmpeg
        width = probe_media(sources[0], resolve_ffmpeg()).width
        colmap_export.export(project.root, project.rig, clip, width,
                             has_masks=(project.root / "masks").exists(),
                             geo_registration=bool(settings.get("geo")),
                             colmap=str(colmap.path))
        job.log("wrote rig_config.json", "info")

    only = settings.get("only")
    steps = reconstruction_steps(project, colmap.path, clip,
                                 geo=bool(settings.get("geo")),
                                 sequential=settings.get("matcher", "sequential") == "sequential")
    if only:
        steps = [s for s in steps if s.key == only]
        if not steps:
            raise StageError(f"unknown reconstruction step {only!r}")

    states: dict[str, dict] = {}

    def note(key: str, state: str, seconds: float) -> None:
        states[key] = {"state": state, "seconds": round(seconds, 1)}
        job.update(result={"steps": states})

    (project.root / "sparse").mkdir(parents=True, exist_ok=True)
    (project.root / "sparse" / "snapshots").mkdir(parents=True, exist_ok=True)
    records = run_steps(job, steps, on_step=note)

    metrics = reconstruction_metrics(project)
    project.mark_done("export", **metrics)
    project.save()

    return {"steps": states, "records": records, "metrics": metrics,
            "summary": f"{metrics.get('registered_images', 0)} images registered"}


# -- train ------------------------------------------------------------------


def run_training(job: Job, project: Project, settings: dict) -> dict:
    """Drive the Brush CLI."""
    brush = find_brush(settings.get("brush"))
    if not brush.found:
        raise StageError(
            "Brush was not found. Install it, or set its path in System status.")

    if sparse_model_dir(project) is None:
        raise StageError("Run reconstruction before training.")

    output = project.root / "splat"
    output.mkdir(parents=True, exist_ok=True)
    steps = int(settings.get("total_steps", 30000))

    argv = [str(brush.path), str(project.root),
            "--total-steps", str(steps),
            "--export-every", str(int(settings.get("export_every", 5000))),
            "--export-path", str(output),
            "--export-name", "splat_{iter}.ply",
            "--max-resolution", str(int(settings.get("max_resolution", 1920)))]
    if settings.get("eval_split_every"):
        argv += ["--eval-split-every", str(int(settings["eval_split_every"]))]
    if settings.get("with_viewer"):
        argv.append("--with-viewer")

    started = time.time()

    def estimate(line: str) -> None:
        # Only show a remaining time once it is worth trusting.
        if job.fraction > 0.05:
            elapsed = time.time() - started
            remaining = elapsed / job.fraction - elapsed
            job.update(detail=f"elapsed {_clock(elapsed)} · about {_clock(remaining)} left")

    result = run(job, argv, progress=ProgressPattern(BRUSH_STEPS.pattern, total=steps,
                                                     message="training"),
                 on_line=estimate, label="training")
    if not result.ok:
        raise StageError(f"Brush exited with code {result.returncode}:\n{result.tail(12)}")

    produced = trained_splats(project)
    project.mark_done("train", steps=steps,
                      splat=str(produced[0]) if produced else "")
    project.save()

    return {
        "splat": str(produced[0]) if produced else "",
        "splats": [str(p) for p in produced],
        "summary": f"trained {steps:,} steps",
    }


def _clock(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


# -- inspect ----------------------------------------------------------------


def run_cleanup(job: Job, project: Project, settings: dict) -> dict:
    """Preview or apply the splat cleanup, on a job so big files do not block."""
    from ..splat import clean as splat_clean
    from ..splat import ply

    splat_path = Path(settings.get("splat") or "")
    if not splat_path.exists():
        found = trained_splats(project)
        if not found:
            raise StageError("No trained .ply file is available yet.")
        splat_path = found[0]

    sparse = sparse_model_dir(project)
    if sparse is None:
        raise StageError("A sparse reconstruction is needed to know where the rig was.")

    job.progress(0.1, "reading the reconstruction")
    trajectory = splat_clean.trajectory_from_model(read_model(sparse))

    job.progress(0.3, "reading the splat")
    splats = ply.read(splat_path)

    named = {"enu": [0.0, 0.0, 1.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    up = named.get(str(settings.get("up", "")).lower())

    job.progress(0.6, "measuring the cleanup volume")
    kept, removed, report = splat_clean.clean(
        splats, trajectory, float(settings.get("radius", 2.5)),
        settings.get("floor"), up)

    result = {
        "before": report.total,
        "would_remove": report.removed,
        "remaining": report.kept,
        "notes": report.lines(),
        "splat": str(splat_path),
        "summary": f"{report.removed:,} of {report.total:,} inside the volume",
    }

    if settings.get("apply"):
        job.progress(0.8, "writing the cleaned splat")
        cleaned = splat_path.with_name(splat_path.stem + "_cleaned.ply")
        removed_path = splat_path.with_name(splat_path.stem + "_removed.ply")
        ply.write(kept, cleaned)
        ply.write(removed, removed_path)
        result["cleaned"] = str(cleaned)
        result["removed_file"] = str(removed_path)
    else:
        # Preview still writes the removed points, because seeing them is the point.
        job.progress(0.8, "writing the preview of removed points")
        preview = splat_path.with_name(splat_path.stem + "_removed.ply")
        ply.write(removed, preview)
        result["removed_file"] = str(preview)

    job.progress(1.0, result["summary"])
    return result
