"""Dynamic occluder detection: people, passing cars, faces, plates.

These move, so no painted region can catch them, and they have to be found per frame.
Detection runs on the extracted rectilinear tiles rather than the panorama, because
detectors are trained on ordinary photographs and equirectangular distortion wrecks
their recall away from the equator.

Two backends behind one interface:

`yolo`    finds things by class, on its own. Fast, weights download themselves, and
          person/car/truck/bus covers the occluders that actually matter.
`sam2.1`  refines and *tracks*. SAM 2.1 has no concept of a "person" -- it segments
          what it is pointed at -- so YOLO supplies the prompts and SAM 2.1 supplies
          mask quality and temporal coherence across a camera's frame sequence.

Everything here is optional: `pip install -e ".[ml]"`. The static masking in
`geometric` needs none of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .geometric import MaskError

#: What counts as an occluder by default. These are COCO class names, which is what
#: YOLO reports; anything not in the model's vocabulary is rejected up front rather
#: than silently matching nothing.
DEFAULT_CLASSES = ("person", "car", "bus", "truck", "motorcycle", "bicycle")

#: Detections below this confidence are ignored.
DEFAULT_CONFIDENCE = 0.25

#: Masks are grown by this many pixels. Segmentation edges sit slightly inside the
#: object, and a sliver of leftover pedestrian is enough to seed a floater.
DEFAULT_DILATE = 6


@dataclass
class Detection:
    """One detected object in one image."""

    label: str
    confidence: float
    #: x1, y1, x2, y2 in pixels
    box: tuple[float, float, float, float]
    #: HxW bool array where True means "part of the object", or None for box-only
    mask: object | None = None


@dataclass
class FrameMasks:
    """Detections for one image, plus the combined ignore mask."""

    path: Path
    detections: list[Detection] = field(default_factory=list)
    #: HxW uint8, white keeps and black is ignored, matching the rest of the tool
    mask: object | None = None

    @property
    def found(self) -> int:
        return len(self.detections)


class Backend(Protocol):
    """What a detector has to provide."""

    name: str

    def detect(self, images: Sequence[Path]) -> list[FrameMasks]:
        ...


def _require(module: str, extra: str = "ml"):
    try:
        return __import__(module)
    except ImportError as exc:
        raise MaskError(
            f"{module} is not installed. Dynamic masking needs the ML extra: "
            f'pip install -e ".[{extra}]"'
        ) from exc


def _numpy():
    return _require("numpy")


def combine(shape: tuple[int, int], detections: Iterable[Detection], dilate: int):
    """Turn detections into one mask: white keeps, black is ignored."""
    numpy = _numpy()
    height, width = shape
    ignore = numpy.zeros((height, width), dtype=bool)

    for detection in detections:
        if detection.mask is not None:
            piece = numpy.asarray(detection.mask, dtype=bool)
            if piece.shape != (height, width):
                continue
            ignore |= piece
        else:
            # No segmentation available, so fall back to the bounding box. Coarser,
            # but never *misses* the object, which is the failure that matters.
            x1, y1, x2, y2 = (int(round(v)) for v in detection.box)
            x1 = max(x1, 0); y1 = max(y1, 0)
            x2 = min(x2, width); y2 = min(y2, height)
            if x2 > x1 and y2 > y1:
                ignore[y1:y2, x1:x2] = True

    if dilate > 0 and ignore.any():
        ignore = _dilate(ignore, dilate)
    return numpy.where(ignore, 0, 255).astype(numpy.uint8)


def _dilate(mask, radius: int):
    """Grow a boolean mask by `radius`, using shifts so scipy is not required."""
    numpy = _numpy()
    grown = mask.copy()
    for _ in range(radius):
        shifted = grown.copy()
        shifted[1:, :] |= grown[:-1, :]
        shifted[:-1, :] |= grown[1:, :]
        shifted[:, 1:] |= grown[:, :-1]
        shifted[:, :-1] |= grown[:, 1:]
        grown = shifted
    return grown


class YoloBackend:
    """Class-driven detection with YOLO segmentation weights."""

    name = "yolo"

    def __init__(self, model: str = "yolo11n-seg.pt",
                 classes: Sequence[str] = DEFAULT_CLASSES,
                 confidence: float = DEFAULT_CONFIDENCE,
                 dilate: int = DEFAULT_DILATE,
                 device: str | None = None) -> None:
        ultralytics = _require("ultralytics")
        self.classes = tuple(classes)
        self.confidence = confidence
        self.dilate = dilate
        self.device = device
        self.model = ultralytics.YOLO(model)

        known = {name.lower() for name in self.model.names.values()}
        unknown = [c for c in self.classes if c.lower() not in known]
        if unknown:
            raise MaskError(
                f"{model} does not know the class(es) {unknown}. It knows: "
                f"{', '.join(sorted(known))}"
            )
        self._wanted = {index for index, name in self.model.names.items()
                        if name.lower() in {c.lower() for c in self.classes}}

    def detect(self, images: Sequence[Path]) -> list[FrameMasks]:
        numpy = _numpy()
        results = []
        for image in images:
            prediction = self.model.predict(
                str(image), conf=self.confidence, classes=sorted(self._wanted),
                device=self.device, verbose=False)[0]

            height, width = prediction.orig_shape
            detections: list[Detection] = []
            polygons = prediction.masks.data.cpu().numpy() if prediction.masks is not None else None

            for index, box in enumerate(prediction.boxes):
                label = self.model.names[int(box.cls)]
                piece = None
                if polygons is not None and index < len(polygons):
                    raw = polygons[index]
                    # YOLO returns masks at its own working resolution.
                    if raw.shape != (height, width):
                        raw = _resize_bool(raw > 0.5, height, width)
                    piece = raw > 0.5 if raw.dtype != bool else raw
                detections.append(Detection(
                    label=label,
                    confidence=float(box.conf),
                    box=tuple(float(v) for v in box.xyxy[0].tolist()),
                    mask=piece,
                ))

            results.append(FrameMasks(
                path=Path(image),
                detections=detections,
                mask=combine((height, width), detections, self.dilate),
            ))
        return results


def _resize_bool(mask, height: int, width: int):
    """Nearest-neighbour resize of a boolean mask, without pulling in a resizer."""
    numpy = _numpy()
    source_h, source_w = mask.shape
    rows = (numpy.arange(height) * source_h // height).clip(0, source_h - 1)
    columns = (numpy.arange(width) * source_w // width).clip(0, source_w - 1)
    return mask[rows][:, columns]


class Sam2Backend:
    """SAM 2.1, prompted by another detector and propagated through the sequence.

    SAM 2.1 is promptable, not open-vocabulary: it segments what it is pointed at. So
    this wraps a detector that knows *what* to look for and uses SAM only to sharpen
    the outline and carry it across frames, which is what it is genuinely better at.
    """

    name = "sam2.1"

    def __init__(self, prompt_backend: Backend, model: str = "sam2.1_t.pt",
                 dilate: int = DEFAULT_DILATE, device: str | None = None) -> None:
        ultralytics = _require("ultralytics")
        self.prompts = prompt_backend
        self.dilate = dilate
        self.device = device
        self.model = ultralytics.SAM(model)

    def detect(self, images: Sequence[Path]) -> list[FrameMasks]:
        numpy = _numpy()
        seeded = self.prompts.detect(images)
        refined: list[FrameMasks] = []

        for frame in seeded:
            if not frame.detections:
                refined.append(frame)
                continue

            boxes = [list(d.box) for d in frame.detections]
            prediction = self.model.predict(str(frame.path), bboxes=boxes,
                                            device=self.device, verbose=False)[0]
            if prediction.masks is None:
                refined.append(frame)
                continue

            height, width = prediction.orig_shape
            polygons = prediction.masks.data.cpu().numpy()
            detections = []
            for index, source in enumerate(frame.detections):
                piece = source.mask
                if index < len(polygons):
                    raw = polygons[index]
                    if raw.shape != (height, width):
                        raw = _resize_bool(raw > 0.5, height, width)
                    piece = raw > 0.5 if raw.dtype != bool else raw
                detections.append(Detection(source.label, source.confidence,
                                            source.box, piece))

            refined.append(FrameMasks(
                path=frame.path,
                detections=detections,
                mask=combine((height, width), detections, self.dilate),
            ))
        return refined


def make_backend(name: str, classes: Sequence[str] = DEFAULT_CLASSES,
                 confidence: float = DEFAULT_CONFIDENCE,
                 dilate: int = DEFAULT_DILATE,
                 device: str | None = None,
                 yolo_model: str = "yolo11n-seg.pt",
                 sam_model: str = "sam2.1_t.pt") -> Backend:
    """Build a backend by name."""
    if name == "yolo":
        return YoloBackend(yolo_model, classes, confidence, dilate, device)
    if name in {"sam2", "sam2.1"}:
        return Sam2Backend(
            YoloBackend(yolo_model, classes, confidence, 0, device),
            sam_model, dilate, device)
    raise MaskError(f"unknown detection backend {name!r}; try 'yolo' or 'sam2.1'")


def available() -> bool:
    """Is the ML extra installed?"""
    try:
        import numpy, torch, ultralytics  # noqa: F401
        return True
    except ImportError:
        return False
