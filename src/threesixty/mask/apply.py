"""Deciding what to actually do about an occluder.

Four answers, and they are genuinely different tools:

`sidecar`  write a mask beside every image. No pixels are lost, the trainer decides.
           This is the right default: masked reconstruction beats deleted data.
`skip`     drop cameras that are mostly occluder anyway. A camera pointing 80% into
           the car roof is not worth extracting, let alone training on.
`burn`     paint the occluder black into the images themselves. For tools that cannot
           read masks. Irreversible, so not the default.
`none`     leave everything alone; the occluders remain documentation.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..ffmpeg import FFmpegInfo
from ..rig import Rig
from . import geometric

MODES = {"sidecar", "skip", "burn", "none"}

#: A camera with more than this share of its view inside the occluder is dropped by
#: `skip`. Two thirds gone leaves too little to match against to be worth the frames.
DEFAULT_SKIP_THRESHOLD = 0.66

#: Below this share, a mask is treated as touching nothing and is not written at all.
#: Guards against a stray pixel of rounding turning into thousands of no-op files.
NEGLIGIBLE = 0.001


@dataclass
class MaskPlan:
    """What masking will do, worked out before any frames are extracted."""

    mode: str
    equirect_mask: Path | None = None
    #: camera name -> fraction of that camera hidden by the occluder
    coverage: dict[str, float] = field(default_factory=dict)
    #: camera name -> rendered mask matching that camera's output size
    camera_masks: dict[str, Path] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.mode != "none" and self.equirect_mask is not None


def prepare(ffmpeg: FFmpegInfo, rig: Rig, sizes: dict[str, tuple[int, int]],
            workdir: Path, mode: str = "sidecar", source_width: int = 4096,
            source_height: int = 0,
            skip_threshold: float = DEFAULT_SKIP_THRESHOLD) -> MaskPlan:
    """Build the equirect mask and every per-camera mask, and measure coverage.

    Done once per extraction, before any frame is touched. `sizes` maps camera name to
    the output size that camera will be written at, so each mask matches its images
    exactly instead of relying on the trainer to rescale it.
    """
    if mode not in MODES:
        raise ValueError(f"mask mode must be one of {sorted(MODES)}, got {mode!r}")

    occluders = geometric.occluders_of(rig)
    if mode == "none" or not occluders:
        return MaskPlan(mode="none")

    workdir.mkdir(parents=True, exist_ok=True)
    # Built at the source's own resolution: projecting it never has to invent detail
    # the occluder outline did not have, and `burn` blends it against the source
    # directly, which refuses mismatched sizes.
    height = max(source_height or source_width // 2, 2)
    equirect = geometric.build_equirect_mask(
        ffmpeg, occluders, source_width, height, workdir / "occluder_equirect.png")

    plan = MaskPlan(mode=mode, equirect_mask=equirect)
    for camera in rig.normalized_cameras():
        width, camera_height = sizes.get(camera.name, (1024, 768))
        rendered = geometric.render_camera_mask(
            ffmpeg, equirect, camera, width, camera_height,
            workdir / f"mask_{camera.name}.png")
        share = geometric.ignored_fraction(ffmpeg, rendered)
        plan.coverage[camera.name] = share
        # A camera the occluder never reaches gets no mask at all. An all-white mask
        # would be a no-op, but it still costs a file per frame and, worse, makes Brush
        # switch that camera into masked alpha handling for no reason.
        if share > NEGLIGIBLE:
            plan.camera_masks[camera.name] = rendered

    if mode == "skip":
        plan.skipped = [name for name, share in plan.coverage.items()
                        if share >= skip_threshold]

    return plan


def burn_filter(equirect_mask: Path) -> str:
    """Filter fragment that multiplies the source by the mask before the split.

    Applying it in equirect space, once per frame, is both cheaper and more consistent
    than blacking out every extracted tile afterwards: one blend covers every camera,
    and they cannot disagree about where the occluder was.
    """
    return f"[1:v]format=gray,scale=rw:rh[bmask];[src][bmask]blend=all_mode=multiply"


def link_sidecars(mask: Path, image_directory: Path, mask_directory: Path,
                  extension: str = ".png") -> int:
    """Give every extracted image a mask file of its own.

    Brush pairs `images/a/b/x.jpg` with `masks/a/b/x.png`, so the mask has to exist once
    per image even though a static occluder produces the same mask every time. Hard
    links keep that honest: thousands of directory entries, one copy of the pixels.
    Falls back to copying where links are unavailable.
    """
    if not image_directory.exists():
        return 0
    mask_directory.mkdir(parents=True, exist_ok=True)

    written = 0
    for image in sorted(image_directory.iterdir()):
        if not image.is_file() or image.name.startswith("."):
            continue
        target = mask_directory / (image.stem + extension)
        if target.exists():
            target.unlink()
        try:
            os.link(mask, target)
        except OSError:
            shutil.copyfile(mask, target)
        written += 1
    return written


def summarize(plan: MaskPlan) -> list[str]:
    """Lines describing what masking did, for the CLI and the UI."""
    if not plan.active:
        return []
    lines = []
    for name, share in sorted(plan.coverage.items(), key=lambda kv: -kv[1]):
        if share <= 0.001:
            continue
        note = "  (skipped)" if name in plan.skipped else ""
        lines.append(f"{name}: {share * 100:.0f}% occluded{note}")
    return lines
