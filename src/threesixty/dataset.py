"""The dataset as it lands on disk: the exported layout, and pruning frames from it.

Camera generation produces two views of the same pixels, and they cannot be one tree.

The **working set** is what COLMAP and Brush read: `images/<clip>/<camera>/00001.jpg`
beside `masks/<clip>/<camera>/00001.png`. Its filenames are deliberately identical
across camera folders, because COLMAP's `rig_configurator` groups images into *frames*
by whatever is left of the path once a camera's `image_prefix` is stripped. Put the view
in the filename and every image becomes its own frame -- which dissolves the rig
constraint that is the entire reason a panoramic tile set does not drift. (COLMAP 4.0
has no `image_suffix` to strip it back off; only `image_prefix` exists.)

The **export** is the layout other tools want, with the view spelled out per file:

    RC_Dataset/
      view_00/
        .geometry/frame_000001_v00.jpg
        .mask/    frame_000001_v00.png
      view_01/
        ...

It is built out of hard links, so it costs directory entries rather than a second copy
of the dataset.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

#: The exported tree, at the project root beside `images/` and `masks/`.
EXPORT_DIRNAME = "RC_Dataset"

#: The subfolders inside each view. Dot-prefixed, which is what the consuming tool
#: expects -- and conveniently keeps them out of naive image scans.
GEOMETRY_DIRNAME = ".geometry"
MASK_DIRNAME = ".mask"


# -- where things live ------------------------------------------------------
#
# A project folder is already named after its clip, so the dataset used to read
# `Q360_.../images/Q360_.../c01/` -- the clip spelled twice for no reason. The clip level
# is gone; a project holds one clip and its own name says which.
#
# Projects written before that still have it, and re-extracting an 8K source to move some
# folders is not a reasonable thing to ask, so every lookup falls back to the old shape
# when it is what is on disk. Writing is always the new shape.


def _tree(root: str | Path, name: str, clip: str | None) -> Path:
    """`<root>/<name>`, or the old `<root>/<name>/<clip>` when that is what exists.

    A project that already has the clip level keeps it -- reads *and* writes -- so it
    stays internally consistent rather than ending up half in each shape.
    """
    current = Path(root) / name
    legacy = current / clip if clip else None
    return legacy if legacy is not None and legacy.is_dir() else current


def frames_dir(root: str | Path, clip: str | None = None) -> Path:
    """Where the extracted equirect frames live."""
    return _tree(root, "frames", clip)


def images_dir(root: str | Path, clip: str | None = None) -> Path:
    """The parent of the per-camera image folders."""
    return _tree(root, "images", clip)


def masks_dir(root: str | Path, clip: str | None = None) -> Path:
    """The parent of the per-camera mask folders."""
    return _tree(root, "masks", clip)


@dataclass
class ExportResult:
    views: list[Path] = field(default_factory=list)
    images: int = 0
    masks: int = 0


@dataclass
class RemovalResult:
    frames: int = 0
    images: int = 0
    masks: int = 0
    exported: int = 0


def frame_number(stem: str) -> int:
    """The frame index behind an extracted frame's name (`00042` -> 42)."""
    digits = "".join(character for character in stem if character.isdigit())
    return int(digits) if digits else 0


def export_name(number: int, view: int, suffix: str) -> str:
    """`frame_000001_v00.jpg` -- the view is in the name, once per file."""
    return f"frame_{number:06d}_v{view:02d}{suffix}"


def view_dir(root: str | Path, view: int) -> Path:
    return Path(root) / EXPORT_DIRNAME / f"view_{view:02d}"


def export_dataset(root: str | Path, clip: str, camera_names, on_progress=None
                   ) -> ExportResult:
    """Mirror the working set into `RC_Dataset/view_NN/{.geometry,.mask}/`.

    Rebuilt from scratch every time rather than merged: a re-run with fewer frames must
    not leave the previous run's images behind, silently exporting a dataset that no
    longer matches the reconstruction.
    """
    root = Path(root)
    names = list(camera_names)
    result = ExportResult()

    for index, name in enumerate(names):
        images = images_dir(root, clip) / name
        masks = masks_dir(root, clip) / name
        geometry_dir = _reset(view_dir(root, index) / GEOMETRY_DIRNAME)
        mask_dir = _reset(view_dir(root, index) / MASK_DIRNAME)
        if not images.is_dir():
            continue
        result.views.append(view_dir(root, index))

        for image in sorted(images.iterdir()):
            if not image.is_file() or image.name.startswith("."):
                continue
            number = frame_number(image.stem)
            result.images += _link(
                image, geometry_dir / export_name(number, index, image.suffix))
            mask = masks / f"{image.stem}.png"
            if mask.exists():
                result.masks += _link(
                    mask, mask_dir / export_name(number, index, ".png"))
        if on_progress is not None:
            on_progress((index + 1) / max(len(names), 1))

    return result


def remove_frames(root: str | Path, clip: str, stems) -> RemovalResult:
    """Delete these frames from the working set, the export, and the frame store.

    Removal is from `frames/` too, so regenerating cameras does not quietly bring them
    back -- the point of dropping a frame is that it never reaches the reconstruction.
    """
    root = Path(root)
    wanted = sorted({Path(str(stem)).stem for stem in stems if str(stem).strip()})
    removed = RemovalResult()

    for stem in wanted:
        for path in _glob(frames_dir(root, clip), f"{stem}.*"):
            removed.frames += _unlink(path)
        for directory in _subdirs(images_dir(root, clip)):
            for path in _glob(directory, f"{stem}.*"):
                removed.images += _unlink(path)
        for directory in _subdirs(masks_dir(root, clip)):
            for path in _glob(directory, f"{stem}.*"):
                removed.masks += _unlink(path)
        _unlink(root / ".threesixty" / "masks" / "equirect_masks" / f"{stem}.png")

        number = frame_number(stem)
        for view in _glob(root / EXPORT_DIRNAME, "view_*"):
            index = frame_number(view.name)
            for sub in (view / GEOMETRY_DIRNAME, view / MASK_DIRNAME):
                for path in _glob(sub, export_name(number, index, ".*")):
                    removed.exported += _unlink(path)

    return removed


def blank_masks(root: str | Path, clip: str) -> list[str]:
    """Frames whose equirect mask ignores nothing -- i.e. nothing was detected.

    Read back from disk so the warning survives a page reload, and so a dataset masked
    by an earlier run can still be checked.
    """
    directory = Path(root) / ".threesixty" / "masks" / "equirect_masks"
    if not directory.is_dir():
        return []
    try:
        import cv2
    except ImportError:
        return []

    found = []
    for path in sorted(directory.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None and int(image.min()) >= 255:
            found.append(path.stem)
    return found


# -- plumbing ---------------------------------------------------------------


def _reset(directory: Path) -> Path:
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _link(source: Path, target: Path) -> int:
    """Hard-link, falling back to a copy where links are unavailable."""
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        try:
            shutil.copyfile(source, target)
        except OSError:
            return 0
    return 1


def _unlink(path: Path) -> int:
    try:
        path.unlink()
    except OSError:
        return 0
    return 1


def _glob(directory: Path, pattern: str):
    return sorted(directory.glob(pattern)) if directory.is_dir() else []


def _subdirs(directory: Path):
    return [p for p in _glob(directory, "*") if p.is_dir()]
