"""Where the dataset lives on disk, and what has to be true about it.

One layout, not two. Each camera owns a folder holding its images and the masks that go
with them, side by side, so a camera is a self-contained thing you can look at:

    <project>/
      frames/                    00001.jpg ...          the extracted panoramas
      images/
        c00/
          .geometry/             00001.jpg ...          the tiles
          .mask/                 00001.png ...          white keeps, black is ignored
        c01/
          ...
      masks/                                            hard links, for the trainers
        c00/.geometry/           00001.png  00001.jpg.png

**Filenames stay identical across camera folders.** COLMAP's `rig_configurator` groups
images into *frames* by whatever is left of the path once a camera's `image_prefix` is
stripped, so `c00/.geometry/` and `c01/.geometry/` are the prefixes and `00001.jpg` is
the frame. Putting the camera in the filename would give every image its own frame and
dissolve the rig constraint that is the entire reason a panoramic tile set does not
drift. (COLMAP 4.0 has no `image_suffix` to strip it back off; only `image_prefix`.)

**`masks/` is a mirror, not a copy.** Both trainers insist on finding masks in a root
that mirrors the image tree, and neither can be pointed elsewhere: Brush pairs
`images/<sub>/x.jpg` with `masks/<sub>/x.png`, and COLMAP's `--ImageReader.mask_path`
wants `<sub>/x.jpg.png` -- note the doubled extension, which is COLMAP's documented
convention and easy to get silently wrong, because a mask it cannot find is simply not
applied. Both names are written, as hard links, so the mirror costs directory entries
rather than a second copy of the dataset.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

#: The two folders inside a camera. Dot-prefixed, which keeps them out of naive image
#: scans and is what the tools reading this layout expect.
GEOMETRY_DIRNAME = ".geometry"
MASK_DIRNAME = ".mask"


# -- where things live ------------------------------------------------------
#
# Two older shapes exist on disk and both still open. A project folder is already named
# after its clip, so `Q360_.../images/Q360_.../c01/` spelled it twice; and before that
# images sat directly in `images/<camera>/` with masks in a separate `masks/<camera>/`.
# Re-extracting an 8K source to move some folders is not a reasonable thing to ask, so
# lookups follow whatever is actually there. Writing is always the current shape.


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
    """The parent of the per-camera folders."""
    return _tree(root, "images", clip)


def masks_dir(root: str | Path, clip: str | None = None) -> Path:
    """The root of the mask mirror the trainers read."""
    return _tree(root, "masks", clip)


def camera_dir(root: str | Path, clip: str | None, camera: str) -> Path:
    return images_dir(root, clip) / camera


def geometry_dir(root: str | Path, clip: str | None, camera: str) -> Path:
    """A camera's images. Falls back to the flat folder on a dataset written before
    images and masks moved in together."""
    inside = camera_dir(root, clip, camera)
    nested = inside / GEOMETRY_DIRNAME
    if nested.is_dir() or not _has_images(inside):
        return nested
    return inside


def mask_dir(root: str | Path, clip: str | None, camera: str) -> Path:
    """A camera's masks, beside its images."""
    inside = camera_dir(root, clip, camera)
    nested = inside / MASK_DIRNAME
    if nested.is_dir() or not _has_images(inside):
        return nested
    return masks_dir(root, clip) / camera        # the separate tree, as it used to be


#: What the trainers see below `images/` and below `masks/`, for one camera.
def relative_subpath(root: str | Path, clip: str | None, camera: str) -> str:
    directory = geometry_dir(root, clip, camera)
    return directory.relative_to(images_dir(root, clip)).as_posix()


@dataclass
class MirrorResult:
    cameras: list[str] = field(default_factory=list)
    masks: int = 0
    #: (camera, stem) pairs that had no mask and were given a blank one.
    filled: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class RemovalResult:
    frames: int = 0
    images: int = 0
    masks: int = 0
    mirrored: int = 0


def frame_number(stem: str) -> int:
    """The frame index behind an extracted frame's name (`00042` -> 42)."""
    digits = "".join(character for character in stem if character.isdigit())
    return int(digits) if digits else 0


def fill_missing_masks(root: str | Path, clip: str, camera_names) -> list[tuple[str, str]]:
    """Give every image a mask, inventing a blank one where masking left none.

    An image with no mask at all is the worst of the three states: the trainers treat
    that camera inconsistently frame to frame, and nothing downstream can tell "nothing
    needed masking here" apart from "masking failed here". A blank (all-white) mask says
    the first out loud; the caller reports the list so the user can decide about the
    second.
    """
    root = Path(root)
    filled: list[tuple[str, str]] = []
    try:
        import cv2
        import numpy as np
    except ImportError:
        return filled

    for camera in camera_names:
        images = geometry_dir(root, clip, camera)
        masks = mask_dir(root, clip, camera)
        if not images.is_dir():
            continue
        blank = None
        for image in sorted(images.glob("*.*")):
            if image.name.startswith("."):
                continue
            target = masks / f"{image.stem}.png"
            if target.exists():
                continue
            if blank is None:
                sample = cv2.imread(str(image))
                if sample is None:
                    break
                blank = np.full(sample.shape[:2], 255, dtype=np.uint8)
            masks.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), blank)
            filled.append((camera, image.stem))
    return filled


def mirror_masks(root: str | Path, clip: str, camera_names) -> MirrorResult:
    """Rebuild `masks/` from the per-camera `.mask` folders, for COLMAP and Brush.

    Rebuilt rather than merged: a re-run with fewer frames must not leave the previous
    run's masks behind, silently masking images that no longer exist.
    """
    root = Path(root)
    result = MirrorResult()

    for camera in camera_names:
        source = mask_dir(root, clip, camera)
        subpath = relative_subpath(root, clip, camera)
        target = _reset(masks_dir(root, clip) / subpath)
        if not source.is_dir():
            continue
        result.cameras.append(camera)
        for mask in sorted(source.glob("*.png")):
            # Brush's name, then COLMAP's doubled-extension one, both linked to the same
            # bytes. Cheaper than choosing, and a mask neither can find is not applied.
            result.masks += _link(mask, target / mask.name)
            _link(mask, target / f"{mask.stem}.jpg.png")

    return result


def remove_frames(root: str | Path, clip: str, stems) -> RemovalResult:
    """Delete these frames from the dataset, the mirror, and the frame store.

    Removal is from `frames/` too, so regenerating cameras does not quietly bring them
    back -- the point of dropping a frame is that it never reaches the reconstruction.
    """
    root = Path(root)
    wanted = sorted({Path(str(stem)).stem for stem in stems if str(stem).strip()})
    removed = RemovalResult()

    for stem in wanted:
        for path in _glob(frames_dir(root, clip), f"{stem}.*"):
            removed.frames += _unlink(path)
        for camera in _subdirs(images_dir(root, clip)):
            name = camera.name
            for path in _glob(geometry_dir(root, clip, name), f"{stem}.*"):
                removed.images += _unlink(path)
            for path in _glob(mask_dir(root, clip, name), f"{stem}.*"):
                removed.masks += _unlink(path)
        _unlink(root / ".threesixty" / "masks" / "equirect_masks" / f"{stem}.png")

        for directory in _walk(masks_dir(root, clip)):
            for path in _glob(directory, f"{stem}.*"):
                removed.mirrored += _unlink(path)

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


def _has_images(directory: Path) -> bool:
    return any(_glob(directory, pattern) for pattern in ("*.jpg", "*.jpeg", "*.png"))


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


def _walk(directory: Path):
    """Every directory at or below this one."""
    if not directory.is_dir():
        return []
    return [directory] + [p for p in directory.rglob("*") if p.is_dir()]
