"""Turning a rig plus a source file into concrete ffmpeg invocations.

The planner exists to make one thing cheap: decoding the source exactly once and
fanning it out to every camera. A naive implementation runs ffmpeg once per camera
and pays the decode cost N times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import sharp
from .ffmpeg import FFmpegInfo, MediaInfo
from .rig import Camera, Rig, native_size, output_size

#: `brush` writes <out>/images/<clip>/<camera>/, which both Brush and COLMAP read and
#: which lets <out>/masks/<clip>/<camera>/ mirror it exactly -- Brush requires nested
#: mask directories to match their image directories. `flat` is the older shape.
LAYOUTS = {"brush", "flat"}

#: Beyond this the filtergraph goes into a script file instead of the command line.
#: Sharp selection produces one `eq(n,N)` term per kept frame, which on a long clip
#: runs to tens of kilobytes -- well past what Windows accepts in a command line.
GRAPH_INLINE_LIMIT = 3000

#: How many v360 chains to drive from a single decode. Each one is a full-frame
#: resample plus an mjpeg encoder, so past roughly this many the passes stop scaling
#: and start thrashing. Tunable per machine via --max-streams.
DEFAULT_MAX_STREAMS = 8

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_stem(value: str) -> str:
    """Make a string safe to use as a path component."""
    cleaned = _UNSAFE.sub("_", value).strip("._")
    return cleaned or "unnamed"


@dataclass(frozen=True)
class FrameSelection:
    """Which frames of the source to extract.

    ``fps`` is the right default for photogrammetry: it samples uniformly in *time*,
    so a capture that slows down does not flood the dataset with near-duplicate
    frames from wherever the operator stopped walking.
    """

    mode: str = "fps"  # fps | sharp | every | all
    value: float = 2.0
    start: float | None = None  # seconds
    end: float | None = None  # seconds
    #: Frame numbers chosen by sharpness analysis, filled in by plan_extraction.
    #: Empty until then, which is why `sharp` needs a planning pass with ffmpeg.
    frames: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.mode not in {"fps", "sharp", "every", "all"}:
            raise ValueError(
                f"frame selection mode must be fps|sharp|every|all, got {self.mode!r}")
        if self.mode in {"fps", "sharp"} and not self.value > 0:
            raise ValueError(f"--{self.mode} must be positive, got {self.value}")
        if self.mode == "every" and not (self.value >= 1 and float(self.value).is_integer()):
            raise ValueError(f"--every must be a positive whole number, got {self.value}")
        if self.start is not None and self.start < 0:
            raise ValueError("--start must not be negative")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError(f"--end ({self.end}) must be greater than --start ({self.start})")

    def filter_prefix(self, media: MediaInfo) -> str:
        """The filter applied once, before the split, to thin the frame stream."""
        if not media.is_video or self.mode == "all":
            return ""
        if self.mode == "sharp":
            # Picked ahead of time by blurdetect; see sharp.py.
            return sharp.select_expression(list(self.frames))
        if self.mode == "fps":
            return f"fps={self.value:g}"
        # Commas inside a filter argument have to be escaped or ffmpeg reads them as
        # a filter separator and the graph fails to parse.
        return rf"select='not(mod(n\,{int(self.value)}))'"

    def estimate_frames(self, media: MediaInfo) -> int:
        """Roughly how many frames each camera will produce. For progress and warnings."""
        if not media.is_video:
            return 1
        duration = media.duration or 0.0
        if self.start is not None:
            duration -= self.start
        if self.end is not None:
            duration = min(duration, self.end - (self.start or 0.0))
        duration = max(duration, 0.0)

        if self.mode == "sharp":
            return len(self.frames) if self.frames else max(int(duration * self.value), 1)
        if self.mode == "fps":
            return max(int(duration * self.value), 1)
        source_frames = int(duration * media.fps) if media.fps else media.frame_count
        if self.mode == "every":
            return max(source_frames // int(self.value), 1)
        return max(source_frames, 1)


def camera_size(camera: Camera, rig: Rig, media: MediaInfo) -> tuple[int, int]:
    """Output size for one camera.

    With `output.auto` the size follows the source resolution and this camera's own
    field of view, so a 45-degree camera is not padded out to the same pixel count as
    a 90-degree one. Otherwise the rig's fixed width and height are used for all.
    """
    return output_size(rig.output, camera, media.width)


@dataclass(frozen=True)
class CameraJob:
    """One camera's share of one pass."""

    camera: Camera
    directory: Path
    pattern: str  # ffmpeg image2 pattern, e.g. clip_fwd_%05d.jpg
    width: int = 0
    height: int = 0
    #: Where this camera's mask sidecars go. Mirrors `directory` with images/ swapped
    #: for masks/, because Brush matches nested mask paths to nested image paths.
    mask_directory: Path | None = None
    #: Resume marker. Kept out of the image folder: COLMAP's feature extractor scans
    #: that folder, and stray files there are at best noise.
    marker_path: Path | None = None

    @property
    def output_pattern(self) -> Path:
        return self.directory / self.pattern

    @property
    def marker(self) -> Path:
        """Written on success so a re-run can skip this camera."""
        if self.marker_path is not None:
            return self.marker_path
        return self.directory / f".{self.camera.name}.done"


@dataclass
class Pass:
    """A single ffmpeg invocation covering up to max_streams cameras."""

    index: int
    jobs: list[CameraJob]

    @property
    def cameras(self) -> list[Camera]:
        return [job.camera for job in self.jobs]


@dataclass
class ExtractPlan:
    """Everything needed to extract one source file with one rig."""

    media: MediaInfo
    rig: Rig
    selection: FrameSelection
    output_root: Path
    passes: list[Pass] = field(default_factory=list)
    skipped: list[CameraJob] = field(default_factory=list)
    #: Equirect occluder mask multiplied into the source before the split, when the
    #: `burn` mask mode is in use. None for every other mode.
    burn_mask: Path | None = None
    #: Populated when static occluders were resolved; carries per-camera coverage and
    #: the rendered masks that `sidecar` links beside the images.
    mask_plan: "object | None" = None

    @property
    def total_cameras(self) -> int:
        return sum(len(p.jobs) for p in self.passes)

    @property
    def estimated_frames(self) -> int:
        return self.selection.estimate_frames(self.media)

    @property
    def estimated_images(self) -> int:
        return self.estimated_frames * self.total_cameras


def build_filter_graph(cameras: list[Camera], rig: Rig, prefix: str,
                       sizes: list[tuple[int, int]] | None = None,
                       burn: bool = False,
                       source_size: tuple[int, int] | None = None) -> tuple[str, list[str]]:
    """Build the filter_complex string and the output label for each camera.

    Shape::

        [0:v]fps=2,split=3[s0][s1][s2];
        [s0]v360=e:rectilinear:yaw=0:...[o0];
        [s1]v360=...[o1];
        [s2]v360=...[o2]

    Cameras arrive here already normalized -- yaw wrapped into [-180, 180] and rig
    orientation folded in. Passing an unwrapped 240 makes ffmpeg abort.
    """
    if not cameras:
        raise ValueError("cannot build a filter graph with no cameras")

    out = rig.output
    count = len(cameras)
    labels = [f"o{i}" for i in range(count)]
    chains: list[str] = []

    source = "[0:v]"
    if burn:
        # Multiply the panorama by the occluder mask once, before the split: cheaper
        # than blacking out every tile afterwards, and the cameras cannot disagree
        # about where the occluder was. Done in RGB -- multiplying in YUV would scale
        # the chroma planes towards 128 and tint the whole image.
        chains.append(f"[0:v]{prefix + ',' if prefix else ''}format=gbrp[bsrc]")
        # blend refuses mismatched sizes, so the mask is forced to the source's own
        # dimensions rather than assumed to be exactly 2:1.
        scale = f"scale={source_size[0]}:{source_size[1]}," if source_size else ""
        chains.append(f"[1:v]{scale}format=gbrp[bmask]")
        # shortest=1 is load-bearing: the mask is fed with -loop 1 and never ends on
        # its own, so without it the blend runs forever and the extraction hangs.
        chains.append("[bsrc][bmask]blend=all_mode=multiply:shortest=1[burned]")
        source, prefix = "[burned]", ""

    if count == 1:
        head = f"{source}{prefix}," if prefix else source
        source_labels = [head]
    else:
        split_labels = "".join(f"[s{i}]" for i in range(count))
        head = f"{source}{prefix},split={count}{split_labels}" if prefix \
            else f"{source}split={count}{split_labels}"
        chains.append(head)
        source_labels = [f"[s{i}]" for i in range(count)]

    for index, camera in enumerate(cameras):
        width, height = sizes[index] if sizes else (out.width, out.height)
        params = ":".join([
            "e", "rectilinear",
            f"yaw={camera.yaw:g}",
            f"pitch={camera.pitch:g}",
            f"roll={camera.roll:g}",
            f"h_fov={camera.h_fov:g}",
            f"v_fov={camera.v_fov:g}",
            f"w={width}",
            f"h={height}",
            f"interp={out.interp}",
        ])
        chains.append(f"{source_labels[index]}v360={params}[{labels[index]}]")

    return ";".join(chains), labels


def build_pass_argv(
    ffmpeg_path: Path,
    plan: ExtractPlan,
    single_pass: Pass,
    overwrite: bool = True,
    graph_path: Path | None = None,
) -> list[str]:
    """The complete argv for one ffmpeg run.

    Returned as a list and executed without a shell -- filter graphs contain brackets
    and quotes that shells mangle.
    """
    rig = plan.rig
    out = rig.output
    selection = plan.selection

    argv: list[str] = [str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-nostdin"]
    argv.append("-y" if overwrite else "-n")

    # Input-side seeking: fast, and keeps the decoder from touching skipped frames.
    if selection.start is not None:
        argv += ["-ss", f"{selection.start:g}"]
    if selection.end is not None:
        argv += ["-to", f"{selection.end:g}"]
    argv += ["-i", str(plan.media.path)]

    burn = plan.burn_mask is not None
    if burn:
        # -loop 1 so the single mask frame lasts as long as the video.
        argv += ["-loop", "1", "-i", str(plan.burn_mask)]

    graph, labels = build_filter_graph(
        single_pass.cameras, rig, selection.filter_prefix(plan.media),
        sizes=[(job.width, job.height) for job in single_pass.jobs],
        burn=burn,
        source_size=(plan.media.width, plan.media.height) if burn else None,
    )
    if graph_path is not None and len(graph) > GRAPH_INLINE_LIMIT:
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(graph, encoding="utf-8")
        argv += ["-/filter_complex", str(graph_path)]
    else:
        argv += ["-filter_complex", graph]

    for label, job in zip(labels, single_pass.jobs):
        argv += ["-map", f"[{label}]"]
        if out.format == "jpg":
            argv += ["-q:v", str(out.quality)]
        else:
            argv += ["-compression_level", "6"]
        # Every camera in a pass receives the identical frame set from the split, so
        # the same sequence number always means the same instant across cameras.
        argv += ["-start_number", "1", "-fps_mode", "passthrough", str(job.output_pattern)]

    argv += ["-progress", "pipe:1", "-nostats"]
    return argv


def plan_extraction(
    media: MediaInfo,
    rig: Rig,
    selection: FrameSelection,
    output_root: str | Path,
    max_streams: int = DEFAULT_MAX_STREAMS,
    layout: str = "brush",
    resume: bool = False,
    ffmpeg: FFmpegInfo | None = None,
    on_analysis: "callable | None" = None,
    mask_mode: str = "sidecar",
) -> ExtractPlan:
    """Work out the passes needed to extract `media` with `rig`.

    Sharp selection needs a decode pass over the source before anything can be
    planned, so `ffmpeg` is required for that mode and ignored for the others.
    """
    selection.validate()
    rig.validate()
    if max_streams < 1:
        raise ValueError(f"--max-streams must be at least 1, got {max_streams}")

    if selection.mode == "sharp" and media.is_video and not selection.frames:
        if ffmpeg is None:
            raise ValueError("sharp frame selection needs ffmpeg; pass ffmpeg=...")
        scores = sharp.analyze(ffmpeg, media, selection.start, selection.end)
        frames = sharp.choose(scores, media.fps, selection.value)
        if on_analysis is not None:
            on_analysis(sharp.summarize(scores, frames))
        selection = replace(selection, frames=tuple(frames))

    root = Path(output_root)
    clip = safe_stem(media.path.stem)
    digits = max(5, len(str(selection.estimate_frames(media))) + 1)
    extension = rig.output.format

    # Static occluders are resolved before any frame is extracted, so `skip` can drop
    # a camera before it costs anything and `burn` can join the filter graph.
    mask_plan = None
    if ffmpeg is not None and mask_mode != "none" and rig.occluders:
        from .mask import apply as mask_apply
        sizes_by_name = {c.name: camera_size(c, rig, media) for c in rig.normalized_cameras()}
        mask_plan = mask_apply.prepare(
            ffmpeg, rig, sizes_by_name, root / ".threesixty" / "masks",
            mode=mask_mode, source_width=media.width or 4096,
            source_height=media.height or 0,
        )

    jobs: list[CameraJob] = []
    skipped: list[CameraJob] = []
    if layout not in LAYOUTS:
        raise ValueError(f"--layout must be one of {sorted(LAYOUTS)}, got {layout!r}")

    dropped = set(mask_plan.skipped) if mask_plan else set()

    for camera in rig.normalized_cameras():
        if camera.name in dropped:
            continue
        # `brush` puts everything under images/, which is what both Brush and COLMAP
        # expect, and lets masks/ mirror the same subpaths exactly.
        directory = root / "images" / clip / camera.name if layout == "brush" \
            else root / clip / camera.name
        # COLMAP groups images into frames by matching filenames *across* camera
        # folders, so in the brush layout every camera's frame N must be called the
        # same thing; the camera is identified by its folder. The flat layout puts
        # every camera in one directory, where names have to stay distinct instead.
        pattern = f"%0{digits}d.{extension}" if layout == "brush" \
            else f"{clip}_{camera.name}_%0{digits}d.{extension}"
        width, height = camera_size(camera, rig, media)
        mask_directory = (root / "masks" / clip / camera.name) if layout == "brush" \
            else (root / "masks" / clip / camera.name)
        job = CameraJob(camera=camera, directory=directory, pattern=pattern,
                        width=width, height=height, mask_directory=mask_directory,
                        marker_path=root / ".threesixty" / "markers"
                        / f"{clip}_{camera.name}.done")
        if resume and job.marker.exists():
            skipped.append(job)
        else:
            jobs.append(job)

    passes = [
        Pass(index=i, jobs=jobs[start:start + max_streams])
        for i, start in enumerate(range(0, len(jobs), max_streams))
    ]

    return ExtractPlan(
        media=media,
        rig=rig,
        selection=selection,
        output_root=root,
        passes=passes,
        skipped=skipped,
        burn_mask=(mask_plan.equirect_mask
                   if mask_plan and mask_plan.mode == "burn" else None),
        mask_plan=mask_plan,
    )
