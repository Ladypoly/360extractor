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
from .plan import DEFAULT_MAX_STREAMS, FrameSelection, plan_extraction
from .rig import PRESETS, Output, Rig, RigError, cube, dome, handheld, ring, car_forward


def _err(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


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


def cmd_ui(args: argparse.Namespace) -> int:
    """Serve the local rig editor."""
    from .web.server import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser,
          ffmpeg_path=args.ffmpeg)
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

    ui = sub.add_parser("ui", help="open the rig editor in a browser")
    ui.add_argument("--host", default="127.0.0.1", help="bind address (default localhost only)")
    ui.add_argument("--port", type=int, default=8360)
    ui.add_argument("--no-browser", action="store_true", help="do not open a browser window")
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
    except (FFmpegError, RigError, ValueError) as exc:
        _err(str(exc))
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
