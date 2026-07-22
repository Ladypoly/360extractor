"""Command line interface.

    360extract doctor
    360extract probe CLIP.mp4
    360extract rig new ring --count 8 -o rigs/ring8.json
    360extract extract CLIP.mp4 --rig rigs/ring8.json --fps 2 -o dataset/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .extract import (
    ExtractResult,
    clear_markers,
    finish_progress,
    run_extraction,
    terminal_progress,
)
from .ffmpeg import (
    FFmpegError,
    MIN_MAJOR,
    probe_media,
    resolve_ffmpeg,
    survey_ffmpeg,
)
from .mask import apply as mask_apply
from .plan import DEFAULT_MAX_STREAMS, FrameSelection, plan_extraction, safe_stem
from .project import STAGES, Project, ProjectError
from .rig import PRESETS, Output, Rig, RigError, cube, dome, handheld, ring, car_forward


def _err(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def ml_defaults() -> tuple[str, ...]:
    """Default occluder classes, without importing torch just to build the parser."""
    return ("person", "car", "bus", "truck", "motorcycle", "bicycle")


# -- doctor -----------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report which ffmpeg builds exist and which one we would use."""
    print(f"360extract {__version__}")
    print(f"python     {sys.version.split()[0]}")
    print()

    candidates = survey_ffmpeg(args.ffmpeg)
    if not candidates:
        _err("no ffmpeg found at all")
        print("\nInstall ffmpeg 5.0+ and put it on PATH, or set THREESIXTY_FFMPEG.")
        return 1

    try:
        chosen = resolve_ffmpeg(args.ffmpeg)
    except FFmpegError as exc:
        chosen = None
        _err(str(exc))

    print(f"ffmpeg candidates ({len(candidates)} found):")
    for info in candidates:
        mark = "->" if chosen and info.path == chosen.path else "  "
        note = "" if info.usable else f"  [unusable: {info.problem}]"
        print(f" {mark} {info.version:<24} {info.path}  (via {info.source}){note}")

    if chosen is None:
        return 1

    print(f"\nusing: {chosen.path}")
    print(f"       version {chosen.version}, v360 filter present")

    shadowed = [c for c in candidates if c.source == "PATH" and not c.usable]
    if shadowed and chosen.source == "PATH":
        print(
            f"\nnote: {len(shadowed)} older ffmpeg build(s) appear earlier on PATH and would "
            f"be picked by `which ffmpeg`. 360extract ignores them and uses the newest usable one."
        )

    from .colmap import locate as colmap_locate

    print("\nCOLMAP (optional, needed to reconstruct):")
    found = colmap_locate.survey(args.colmap)
    if not found:
        print("    not found. Extraction and masking work without it; reconstruction "
              "does not.")
    else:
        chosen_colmap = colmap_locate.resolve(args.colmap)
        for info in found:
            mark = "->" if chosen_colmap and info.path == chosen_colmap.path else "  "
            note = "" if info.usable else f"  [{info.problem}]"
            cuda = " CUDA" if info.has_cuda else ""
            print(f" {mark} {info.version}{cuda}\n       {info.path} (via {info.source}){note}")
        if chosen_colmap:
            print(f"\n    rig support: yes (rig_configurator present)")
    return 0


# -- probe ------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    status = 0
    for target in args.media:
        try:
            info = probe_media(target, ffmpeg)
        except FFmpegError as exc:
            _err(str(exc))
            status = 1
            continue

        kind = "video" if info.is_video else "still"
        print(f"{info.path.name}")
        print(f"  {info.width}x{info.height}  aspect {info.aspect:.3f}  {kind}  {info.codec}")
        if info.is_video:
            print(f"  {info.fps:g} fps  {info.duration:.2f}s  ~{info.frame_count} frames")
        if not info.looks_equirectangular:
            print(
                "  warning: not a 2:1 image -- equirectangular sources are 2:1. "
                "Extraction will run but the geometry will be wrong."
            )
    return status


# -- rig --------------------------------------------------------------------


def _output_from_args(args: argparse.Namespace) -> Output:
    # Asking for an explicit size is itself the request to stop deriving one.
    fixed = args.width is not None or args.height is not None
    return Output(
        width=args.width or 1920,
        height=args.height or 1440,
        format=args.format,
        quality=args.quality,
        interp=args.interp,
        auto=not fixed,
    )


def cmd_rig_new(args: argparse.Namespace) -> int:
    output = _output_from_args(args)
    preset = args.preset

    if preset == "ring":
        rig = ring(count=args.count, pitch=args.pitch, h_fov=args.h_fov, output=output)
    elif preset == "cube":
        rig = cube(output=output)
    elif preset == "dome":
        rig = dome(ring_count=args.count, h_fov=args.h_fov, output=output)
    elif preset == "car-forward":
        rig = car_forward(h_fov=args.h_fov, pitch=args.pitch, output=output)
    elif preset == "handheld":
        rig = handheld(ring_count=args.count, pitch=args.pitch, h_fov=args.h_fov, output=output)
    else:  # pragma: no cover - argparse restricts choices
        _err(f"unknown preset {preset!r}")
        return 2

    if args.name:
        rig.name = args.name

    if args.output_file:
        path = rig.save(args.output_file)
        print(f"wrote {path}  ({len(rig.enabled_cameras)} cameras)")
    else:
        sys.stdout.write(rig.to_json())

    for warning in rig.warnings():
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def cmd_rig_show(args: argparse.Namespace) -> int:
    rig = load_rig(args.rig)
    out = rig.output
    print(f"{rig.name}  ({len(rig.enabled_cameras)}/{len(rig.cameras)} cameras enabled)")
    size = "native (from source and fov)" if out.auto else f"{out.width}x{out.height}"
    print(f"  output      {size} {out.format} q{out.quality} interp={out.interp}")
    orientation = rig.orientation
    if (orientation.yaw, orientation.pitch, orientation.roll) != (0.0, 0.0, 0.0):
        print(f"  orientation yaw={orientation.yaw:g} pitch={orientation.pitch:g} "
              f"roll={orientation.roll:g}")
    if rig.occluders:
        print(f"  occluders   {len(rig.occluders)} defined (applied by the masking stage)")
    print()
    print(f"  {'name':<12} {'yaw':>8} {'pitch':>8} {'roll':>7} {'h_fov':>7} {'v_fov':>7}")
    for camera in rig.cameras:
        flag = " " if camera.enabled else "x"
        print(f"{flag} {camera.name:<12} {camera.yaw:>8.2f} {camera.pitch:>8.2f} "
              f"{camera.roll:>7.2f} {camera.h_fov:>7.2f} {camera.v_fov:>7.2f}")

    for warning in rig.warnings():
        print(f"\nwarning: {warning}", file=sys.stderr)
    return 0


def cmd_rig_list(args: argparse.Namespace) -> int:
    print("presets:")
    for name, factory in PRESETS.items():
        summary = (factory.__doc__ or "").strip().splitlines()[0]
        print(f"  {name:<14} {summary}")
    return 0


def load_rig(value: str) -> Rig:
    """Accept either a path to a rig file or a bare preset name."""
    path = Path(value)
    if path.exists():
        return Rig.load(path)
    if value in PRESETS:
        return PRESETS[value]()
    raise RigError(
        f"{value!r} is neither an existing rig file nor a preset "
        f"({', '.join(PRESETS)}). Run `360extract rig list`."
    )


# -- extract ----------------------------------------------------------------


def _selection_from_args(args: argparse.Namespace) -> FrameSelection:
    if args.sharp is not None:
        return FrameSelection("sharp", args.sharp, args.start, args.end)
    if args.every is not None:
        return FrameSelection("every", float(args.every), args.start, args.end)
    if args.all_frames:
        return FrameSelection("all", 0.0, args.start, args.end)
    return FrameSelection("fps", args.fps, args.start, args.end)


def cmd_export(args: argparse.Namespace) -> int:
    """Write the COLMAP project for an extracted dataset."""
    from .colmap import export as colmap_export
    from .ffmpeg import probe_media as _probe

    project = Project.load(args.directory)
    ffmpeg = resolve_ffmpeg(args.ffmpeg)

    sources = project.resolved_sources()
    if not sources:
        _err("this project has no sources, so there is nothing to describe")
        return 1
    clip = safe_stem(sources[0].stem)
    source_width = _probe(sources[0], ffmpeg).width

    geo = None
    if args.gpx:
        geo = _write_geo_registration(project, clip, args.gpx, ffmpeg)

    from .colmap import locate as colmap_locate
    found = colmap_locate.resolve(args.colmap)
    if found is None:
        print("note: no usable COLMAP found; the commands will just say `colmap`")

    paths = colmap_export.export(
        project.root, project.rig, clip, source_width,
        has_masks=(project.root / "masks").exists(),
        geo_registration=geo is not None,
        colmap=str(found.path) if found else "colmap",
    )
    print(f"wrote {paths.rig_config.name}, {paths.cameras.name}, {paths.commands.name}")
    if geo:
        print(f"wrote {geo.name} ({args.gpx})")
    print(f"\nRun the pipeline with:\n  sh {paths.commands}")

    project.mark_done("export", rig_config=str(paths.rig_config))
    project.save()
    return 0


def _write_geo_registration(project, clip: str, gpx_path: str, ffmpeg):
    """Turn a GPX track into COLMAP's geo-registration reference file."""
    from . import gps
    from .colmap import export as colmap_export

    fixes = gps.read_gpx(gpx_path)
    images_root = project.root / "images" / clip
    if not images_root.exists():
        raise ValueError(f"{images_root} does not exist; extract before exporting")

    selection = FrameSelection(mode=project.frames.mode, value=project.frames.value,
                              start=project.frames.start, end=project.frames.end)
    per_second = selection.value if selection.mode in {"fps", "sharp"} else 1.0
    start = selection.start or 0.0

    entries = {}
    for camera_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for image in sorted(camera_dir.iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            number = dynamic_frame_number(image)
            offset = start + (number - 1) / max(per_second, 1e-6)
            name = f"{clip}/{camera_dir.name}/{image.name}"
            entries[name] = gps.interpolate(fixes, fixes[0].time + offset)

    return gps.write_geo_registration(entries, project.root / "geo_registration.txt")


def dynamic_frame_number(path: Path) -> int:
    from .mask.dynamic import frame_number
    return frame_number(path)


def cmd_clean_splat(args: argparse.Namespace) -> int:
    """Remove floaters from the volume the rig itself occupied."""
    from .colmap.model import read_model
    from .splat import clean as splat_clean
    from .splat import ply

    model = read_model(args.sparse)
    trajectory = splat_clean.trajectory_from_model(model)
    splats = ply.read(args.splat)
    print(ply.describe(splats))

    radius = args.radius
    if args.radius_in_spacings:
        radius = args.radius_in_spacings * trajectory.median_spacing
        print(f"radius {radius:.3f} = {args.radius_in_spacings} x median frame spacing")

    up = None
    if args.up:
        # After `model_aligner --alignment_type enu` the model is East-North-Up, so up
        # is exactly +Z. That is the one case where the answer is known rather than
        # guessed, and it is the usual one when a GPX was supplied.
        named = {"enu": [0.0, 0.0, 1.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
        if args.up.lower() in named:
            up = named[args.up.lower()]
        else:
            try:
                parts = [float(v) for v in args.up.replace(" ", "").split(",")]
            except ValueError:
                parts = []
            if len(parts) != 3:
                _err(f"--up takes 'enu', 'y', 'z', or three numbers like 0,1,0; "
                     f"got {args.up!r}")
                return 1
            up = parts

    kept, removed, report = splat_clean.clean(splats, trajectory, radius,
                                              args.floor, up)
    for line in report.lines():
        print(f"  {line}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    output = Path(args.output) if args.output else Path(args.splat).with_name(
        Path(args.splat).stem + "_cleaned.ply")
    ply.write(kept, output)
    print(f"\nwrote {output}")

    if not args.no_removed:
        removed_path = output.with_name(output.stem.replace("_cleaned", "") + "_removed.ply")
        ply.write(removed, removed_path)
        print(f"wrote {removed_path} -- load it to see exactly what was taken out")
    return 0


def cmd_batches(args: argparse.Namespace) -> int:
    """Plan a batched reconstruction for a long capture."""
    from .colmap import batches as batch_module
    from .mask.dynamic import discover

    project = Project.load(args.directory)
    found = discover(project.root, project.rig)
    if not found:
        _err("no extracted images found; run `360extract run` first")
        return 1

    frames = sorted(found[0].frames)
    plan = batch_module.plan_batches(frames, args.chunk, args.overlap)
    print(plan.summary())

    clip = found[0].directory.parent.name
    cameras = [entry.camera.name for entry in found]
    written = batch_module.write_image_lists(plan, project.root, clip, cameras,
                                             extension=project.rig.output.format)
    commands = project.root / "batches" / "run_batches.sh"
    commands.write_text(batch_module.build_commands(plan, project.root), encoding="utf-8")

    print(f"wrote {len(written)} image lists and {commands}")
    print("note: the merge step is untested against a real capture -- see the README")
    return 0


def cmd_project_new(args: argparse.Namespace) -> int:
    rig = load_rig(args.rig) if args.rig else None
    project = Project.create(args.directory, sources=args.source, rig=rig,
                             name=args.name, overwrite=args.force)
    print(f"created {project.file}")
    print(f"  rig     {project.rig.name} ({len(project.rig.enabled_cameras)} cameras)")
    print(f"  sources {len(project.sources) or 'none yet'}")
    return 0


def cmd_project_show(args: argparse.Namespace) -> int:
    project = Project.load(args.directory)
    print(f"{project.name}  ({project.file})")
    print(f"  created  {project.created or 'unknown'}")
    print(f"  modified {project.modified or 'unknown'}")

    print(f"\n  sources ({len(project.sources)}):")
    missing = {str(p) for p in project.missing_sources()}
    for source in project.resolved_sources():
        flag = "  MISSING" if str(source) in missing else ""
        print(f"    {source}{flag}")

    print(f"\n  rig      {project.rig.name}, "
          f"{len(project.rig.enabled_cameras)}/{len(project.rig.cameras)} cameras")
    print(f"  frames   {project.frames.mode} {project.frames.value:g}")
    print(f"  output   layout={project.output.layout} mask={project.output.mask_mode}")
    print(f"  detect   {project.detect.backend}, {', '.join(project.detect.classes)}")

    print("\n  stages:")
    for stage in STAGES:
        status = project.status(stage)
        record = project.stages.get(stage)
        detail = ""
        if record and record.details:
            detail = "  " + ", ".join(f"{k}={v}" for k, v in record.details.items())
        when = f"  at {record.done_at}" if record and record.done_at else ""
        print(f"    {stage:<8} {status}{when}{detail}")
        if status == "stale":
            print(f"             settings changed since this ran; re-run to update")

    snapshots = project.snapshots()
    if snapshots:
        print(f"\n  snapshots: {', '.join(snapshots)}")
    return 0


def cmd_project_snapshot(args: argparse.Namespace) -> int:
    project = Project.load(args.directory)
    if args.restore:
        restored = project.restore(args.restore)
        restored.save()
        print(f"restored {args.restore}")
        return 0
    path = project.snapshot(args.label)
    print(f"saved {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the whole pipeline for a project, skipping what is already current."""
    project = Project.load(args.directory)
    ffmpeg = resolve_ffmpeg(args.ffmpeg)

    missing = project.missing_sources()
    if missing:
        _err("these sources are missing:\n  " + "\n  ".join(str(p) for p in missing))
        return 1
    if not project.sources:
        _err("this project has no sources. Add one with `project new --source ...`")
        return 1

    if project.status("extract") == "done" and not args.force:
        print(f"extract: already current ({project.stages['extract'].details})")
    else:
        if project.is_stale("extract"):
            print("extract: settings changed since the last run, redoing")
        totals = _extract_project(project, ffmpeg, args)
        project.mark_done("extract", images=totals.images_written,
                          cameras=totals.cameras_done)
        project.save()

    if args.no_mask:
        return 0

    if project.status("mask") == "done" and not args.force:
        print(f"mask: already current ({project.stages['mask'].details})")
        return 0

    from .mask import dynamic, ml
    if not ml.available():
        print('mask: skipped, the ML extra is not installed (pip install -e ".[ml]")')
        return 0

    backend = ml.make_backend(
        project.detect.backend, classes=project.detect.classes,
        confidence=project.detect.confidence, dilate=project.detect.dilate,
        device=project.detect.device)
    print(f"mask: {backend.name}, looking for {', '.join(project.detect.classes)}")
    report = dynamic.run(ffmpeg, project.root, project.rig, backend,
                         fuse=project.detect.fuse,
                         on_progress=lambda note: print(f"  {note}", flush=True))
    print(f"  {report.summary()}")
    project.mark_done("mask", masks=report.masks_written,
                      detections=report.detections)
    project.save()
    return 0


def _extract_project(project: Project, ffmpeg, args) -> ExtractResult:
    """Extraction driven entirely by a project's stored settings."""
    selection = FrameSelection(
        mode=project.frames.mode, value=project.frames.value,
        start=project.frames.start, end=project.frames.end)

    totals = ExtractResult()
    for source in project.resolved_sources():
        media = probe_media(source, ffmpeg)
        if selection.mode == "sharp" and media.is_video:
            print(f"{media.path.name}: analysing sharpness…", flush=True)

        plan = plan_extraction(
            media=media, rig=project.rig, selection=selection,
            output_root=project.root, layout=project.output.layout,
            resume=False, ffmpeg=ffmpeg,
            on_analysis=lambda note: print(f"  {note}"),
            mask_mode=project.output.mask_mode,
        )
        for line in mask_apply.summarize(plan.mask_plan) if plan.mask_plan else []:
            print(f"  {line}")
        print(f"{media.path.name}: {plan.total_cameras} cameras, "
              f"~{plan.estimated_images} images")

        result = run_extraction(plan, ffmpeg, on_progress=terminal_progress())
        finish_progress()
        totals.images_written += result.images_written
        totals.masks_written += result.masks_written
        totals.cameras_done += result.cameras_done
    return totals


def cmd_mask(args: argparse.Namespace) -> int:
    """Detect and mask dynamic occluders in an already-extracted dataset."""
    from .mask import dynamic, ml

    if not ml.available():
        _err('dynamic masking needs the ML extra: pip install -e ".[ml]"')
        return 1

    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    rig = load_rig(args.rig)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    backend = ml.make_backend(
        args.backend, classes=classes, confidence=args.confidence,
        dilate=args.dilate, device=args.device,
        yolo_model=args.yolo_model, sam_model=args.sam_model,
    )
    print(f"backend: {backend.name}, looking for {', '.join(classes)}")

    report = dynamic.run(
        ffmpeg, Path(args.dataset), rig, backend,
        fuse=not args.no_fuse, static=not args.no_static,
        on_progress=lambda note: print(f"  {note}", flush=True),
    )
    print(f"\n{report.summary()}")
    print(f"{report.masks_written} masks written to {Path(args.dataset) / 'masks'}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Serve the local rig editor."""
    from .web.server import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser,
          ffmpeg_path=args.ffmpeg, project_path=args.project)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    rig = load_rig(args.rig)

    if args.nadir:
        # Replaces any nadir cone the rig already carried, rather than stacking.
        rig.occluders = [o for o in rig.occluders if o.get("type") != "nadir_cone"]
        rig.occluders.append({"type": "nadir_cone", "angle": args.nadir})

    if args.width or args.height:
        rig.output.width = args.width or rig.output.width
        rig.output.height = args.height or rig.output.height
        rig.output.auto = False  # an explicit size overrides automatic sizing
        rig.output.validate()

    for warning in rig.warnings():
        print(f"warning: {warning}", file=sys.stderr)

    selection = _selection_from_args(args)
    root = Path(args.output_dir)

    if args.no_resume:
        removed = clear_markers(root)
        if removed:
            print(f"cleared {removed} resume marker(s)")

    totals = ExtractResult()
    for target in args.media:
        media = probe_media(target, ffmpeg)
        if not media.looks_equirectangular:
            print(
                f"warning: {media.path.name} is {media.width}x{media.height} "
                f"(aspect {media.aspect:.3f}), not the 2:1 of an equirectangular source",
                file=sys.stderr,
            )

        if selection.mode == "sharp" and media.is_video:
            print(f"{media.path.name}: analysing sharpness…", flush=True)

        plan = plan_extraction(
            media=media,
            rig=rig,
            selection=selection,
            output_root=root,
            max_streams=args.max_streams,
            layout=args.layout,
            resume=not args.no_resume,
            ffmpeg=ffmpeg,
            on_analysis=lambda note: print(f"  {note}"),
            mask_mode=args.mask,
        )

        for line in mask_apply.summarize(plan.mask_plan) if plan.mask_plan else []:
            print(f"  {line}")

        if not plan.passes:
            print(f"{media.path.name}: already extracted ({len(plan.skipped)} cameras), skipping")
            totals.cameras_skipped += len(plan.skipped)
            continue

        print(
            f"{media.path.name}: {plan.total_cameras} cameras in {len(plan.passes)} pass(es), "
            f"~{plan.estimated_frames} frames each, ~{plan.estimated_images} images"
            + (f", {len(plan.skipped)} cameras already done" if plan.skipped else "")
        )

        result = run_extraction(
            plan,
            ffmpeg,
            on_progress=terminal_progress(),
            dry_run=args.dry_run,
            overwrite=True,
        )
        finish_progress()

        totals.images_written += result.images_written
        totals.masks_written += result.masks_written
        totals.cameras_done += result.cameras_done
        totals.cameras_skipped += result.cameras_skipped
        totals.passes_run += result.passes_run
        totals.elapsed += result.elapsed

        if result.cancelled:
            print("\ncancelled. Completed cameras are marked done; re-run to continue.")
            return 130

    if args.dry_run:
        return 0

    rate = totals.images_written / totals.elapsed if totals.elapsed else 0.0
    print(
        f"\n{totals.images_written} images from {totals.cameras_done} cameras "
        f"in {totals.elapsed:.1f}s ({rate:.0f} img/s)"
        + (f", {totals.masks_written} masks" if totals.masks_written else "")
        + (f", {totals.cameras_skipped} skipped" if totals.cameras_skipped else "")
    )
    return 0


# -- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="360extract",
        description="Extract perspective image sets from 360 equirectangular footage.",
    )
    parser.add_argument("--version", action="version", version=f"360extract {__version__}")
    parser.add_argument(
        "--colmap", metavar="PATH",
        help="COLMAP binary to use (needs 3.12+ for rig support). Overrides "
             "THREESIXTY_COLMAP and PATH discovery.")
    parser.add_argument(
        "--ffmpeg", metavar="PATH",
        help=f"ffmpeg binary to use (must be {MIN_MAJOR}.0+ with the v360 filter). "
             "Overrides THREESIXTY_FFMPEG and PATH discovery.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check ffmpeg discovery and capabilities")
    doctor.set_defaults(func=cmd_doctor)

    probe = sub.add_parser("probe", help="report dimensions, frame rate and duration of sources")
    probe.add_argument("media", nargs="+")
    probe.set_defaults(func=cmd_probe)

    rig_parser = sub.add_parser("rig", help="create and inspect camera rigs")
    rig_sub = rig_parser.add_subparsers(dest="rig_command", required=True)

    rig_list = rig_sub.add_parser("list", help="list built-in presets")
    rig_list.set_defaults(func=cmd_rig_list)

    rig_new = rig_sub.add_parser("new", help="generate a rig from a preset")
    rig_new.add_argument("preset", choices=sorted(PRESETS))
    rig_new.add_argument("-o", "--output-file", metavar="FILE",
                         help="write here instead of stdout")
    rig_new.add_argument("--name", help="rig name")
    rig_new.add_argument("--count", type=int, default=8, help="cameras per ring (default 8)")
    rig_new.add_argument("--pitch", type=float, default=0.0,
                         help="camera pitch in degrees, negative looks down (default 0)")
    rig_new.add_argument("--h-fov", dest="h_fov", type=float, default=90.0,
                         help="horizontal field of view (default 90)")
    rig_new.add_argument("--width", type=int,
                         help="fixed output width; omit to size each camera from the "
                              "source resolution and its own field of view")
    rig_new.add_argument("--height", type=int, help="fixed output height")
    rig_new.add_argument("--format", choices=["jpg", "png"], default="jpg")
    rig_new.add_argument("--quality", type=int, default=2,
                         help="jpeg quality 1-31, lower is better (default 2)")
    rig_new.add_argument("--interp", default="line",
                         help="v360 interpolation: line, cubic, lanczos, spline16 (default line)")
    rig_new.set_defaults(func=cmd_rig_new)

    rig_show = rig_sub.add_parser("show", help="print a rig as a table")
    rig_show.add_argument("rig", help="rig file or preset name")
    rig_show.set_defaults(func=cmd_rig_show)

    project_parser = sub.add_parser(
        "project", help="create and inspect projects (settings plus what has been done)")
    project_sub = project_parser.add_subparsers(dest="project_command", required=True)

    project_new = project_sub.add_parser("new", help="start a project in a folder")
    project_new.add_argument("directory", help="the dataset folder; project.json goes here")
    project_new.add_argument("--source", action="append", default=[],
                             help="a 360 video or still; repeat for several")
    project_new.add_argument("--rig", help="rig file or preset name (default ring)")
    project_new.add_argument("--name")
    project_new.add_argument("--force", action="store_true",
                             help="replace an existing project.json")
    project_new.set_defaults(func=cmd_project_new)

    project_show = project_sub.add_parser("show", help="print settings and stage status")
    project_show.add_argument("directory", nargs="?", default=".")
    project_show.set_defaults(func=cmd_project_show)

    project_snap = project_sub.add_parser(
        "snapshot", help="save or restore a named copy of the settings")
    project_snap.add_argument("directory", nargs="?", default=".")
    project_snap.add_argument("--label", default="",
                              help="name for the snapshot being saved")
    project_snap.add_argument("--restore", metavar="LABEL",
                              help="restore this snapshot instead of saving one")
    project_snap.set_defaults(func=cmd_project_snapshot)

    export_parser = sub.add_parser(
        "export", help="write the COLMAP project (rig, intrinsics, commands)")
    export_parser.add_argument("directory", nargs="?", default=".")
    export_parser.add_argument("--gpx", metavar="FILE",
                               help="GPX track for the capture; writes COLMAP's "
                                    "geo-registration file, which gives the model a real "
                                    "scale and makes cleanup radii mean metres")
    export_parser.set_defaults(func=cmd_export)

    clean = sub.add_parser(
        "clean-splat",
        help="delete gaussians at the recorded camera positions, where floaters collect")
    clean.add_argument("splat", help="the trained .ply")
    clean.add_argument("--sparse", required=True,
                       help="COLMAP sparse model directory, e.g. dataset/sparse/0")
    radius_group = clean.add_mutually_exclusive_group(required=True)
    radius_group.add_argument("--radius", type=float,
                              help="removal radius in model units (metres once the "
                                   "model is geo-registered)")
    radius_group.add_argument("--radius-in-spacings", type=float, metavar="N",
                              help="removal radius as N x the median distance between "
                                   "frames; use when the model has no real scale")
    clean.add_argument("--floor", type=float, metavar="D",
                       help="spare anything more than D below the rig, so the road "
                            "under the vehicle survives")
    clean.add_argument("--up", metavar="DIR",
                       help="which way is up: 'enu' after geo-registering with "
                            "--alignment_type enu, otherwise 'y', 'z', or X,Y,Z. "
                            "Needed for --floor on a straight capture, since a line "
                            "cannot reveal its own vertical")
    clean.add_argument("-o", "--output", help="output .ply (default <name>_cleaned.ply)")
    clean.add_argument("--no-removed", action="store_true",
                       help="do not also write the removed gaussians")
    clean.add_argument("--dry-run", action="store_true",
                       help="report what would be removed and write nothing")
    clean.set_defaults(func=cmd_clean_splat)

    batches = sub.add_parser(
        "batches", help="plan a batched reconstruction for a long capture")
    batches.add_argument("directory", nargs="?", default=".")
    batches.add_argument("--chunk", type=int, default=300, help="frames per chunk")
    batches.add_argument("--overlap", type=int, default=40,
                         help="frames shared with the next chunk; this is what lets "
                              "model_merger align them")
    batches.set_defaults(func=cmd_batches)

    run = sub.add_parser(
        "run", help="extract and mask a project, skipping what is already current")
    run.add_argument("directory", nargs="?", default=".")
    run.add_argument("--force", action="store_true", help="redo stages already done")
    run.add_argument("--no-mask", action="store_true", help="stop after extraction")
    run.set_defaults(func=cmd_run)

    mask = sub.add_parser(
        "mask", help="mask moving occluders (people, cars) in an extracted dataset")
    mask.add_argument("dataset", help="the folder `extract` wrote, containing images/")
    mask.add_argument("--rig", required=True, help="the rig used for the extraction")
    mask.add_argument("--backend", choices=["yolo", "sam2.1"], default="sam2.1",
                      help="yolo finds objects by class; sam2.1 uses YOLO for prompts "
                           "and refines the outlines (default)")
    mask.add_argument("--classes", default=",".join(ml_defaults()),
                      help="comma-separated COCO class names to mask")
    mask.add_argument("--confidence", type=float, default=0.25)
    mask.add_argument("--dilate", type=int, default=6,
                      help="grow masks by this many pixels; a sliver of leftover "
                           "pedestrian is enough to seed a floater (default 6)")
    mask.add_argument("--device", help="torch device, e.g. cuda:0 or cpu")
    mask.add_argument("--yolo-model", default="yolo11n-seg.pt")
    mask.add_argument("--sam-model", default="sam2.1_t.pt")
    mask.add_argument("--no-fuse", action="store_true",
                      help="skip reconciling overlapping cameras through the sphere")
    mask.add_argument("--no-static", action="store_true",
                      help="do not merge in the rig's painted or cone occluders")
    mask.set_defaults(func=cmd_mask)

    ui = sub.add_parser("ui", help="open the rig editor in a browser")
    ui.add_argument("--host", default="127.0.0.1", help="bind address (default localhost only)")
    ui.add_argument("--port", type=int, default=8360)
    ui.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ui.add_argument("--project", help="open this project on startup")
    ui.set_defaults(func=cmd_ui)

    extract = sub.add_parser("extract", help="extract perspective images from 360 sources")
    extract.add_argument("media", nargs="+", help="360 video or still files")
    extract.add_argument("--rig", required=True, help="rig file or preset name")
    extract.add_argument("-o", "--output-dir", default="dataset", help="output root (default dataset)")

    frames = extract.add_mutually_exclusive_group()
    frames.add_argument("--fps", type=float, default=2.0,
                        help="sample this many frames per second (default 2)")
    frames.add_argument("--sharp", type=float, metavar="FPS",
                        help="like --fps, but keep the sharpest frame in each window "
                             "instead of whatever lands on the tick (skips motion blur)")
    frames.add_argument("--every", type=int, metavar="N", help="take every Nth source frame")
    frames.add_argument("--all-frames", action="store_true", help="extract every source frame")

    extract.add_argument("--start", type=float, metavar="SEC", help="skip to this timestamp")
    extract.add_argument("--end", type=float, metavar="SEC", help="stop at this timestamp")
    extract.add_argument("--width", type=int, help="override rig output width")
    extract.add_argument("--height", type=int, help="override rig output height")
    extract.add_argument("--max-streams", type=int, default=DEFAULT_MAX_STREAMS,
                         help=f"cameras per ffmpeg pass (default {DEFAULT_MAX_STREAMS})")
    extract.add_argument("--layout", choices=["brush", "flat"], default="brush",
                         help="brush: <out>/images/<clip>/<camera>/ with masks/ mirroring it, "
                              "which Brush and COLMAP both read (default). flat: the older shape")
    extract.add_argument("--mask", choices=["sidecar", "skip", "burn", "none"],
                         default="sidecar",
                         help="what to do about the rig's occluders. sidecar: write a mask "
                              "beside every image, losing no pixels (default). skip: drop "
                              "cameras that are mostly occluder. burn: black it into the "
                              "images. none: ignore them")
    extract.add_argument("--nadir", type=float, metavar="DEG",
                         help="add a nadir cone occluder of this many degrees, covering the "
                              "tripod, stick or car roof directly below the rig")
    extract.add_argument("--no-resume", action="store_true",
                         help="ignore and clear resume markers, redo everything")
    extract.add_argument("--dry-run", action="store_true",
                         help="print the ffmpeg commands instead of running them")
    extract.set_defaults(func=cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FFmpegError, RigError, ProjectError, ValueError) as exc:
        _err(str(exc))
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
