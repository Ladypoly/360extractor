"""Splitting a long capture into overlapping chunks.

Incremental SfM degrades on long linear captures: a street driven for ten minutes gives
thousands of frames in a nearly straight line, where drift accumulates and the whole
reconstruction can collapse late. Reconstructing overlapping chunks and merging them
keeps each problem small.

The overlap is the entire mechanism. `model_merger` aligns two models by their shared
registered images, so consecutive chunks must share enough frames to be alignable --
which is why overlap is expressed in frames rather than as a fraction.

**Honestly labelled:** the merge itself is not verified here, because COLMAP is not
installed on this machine and a synthetic model has no features to match. This generates
the plan and the commands; treat the merge as untested until it has been run against a
real capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CHUNK = 300
DEFAULT_OVERLAP = 40


@dataclass
class Batch:
    """One chunk of the capture."""

    index: int
    frames: list[int]

    @property
    def name(self) -> str:
        return f"chunk{self.index:03d}"

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]


@dataclass
class BatchPlan:
    batches: list[Batch] = field(default_factory=list)
    chunk: int = DEFAULT_CHUNK
    overlap: int = DEFAULT_OVERLAP

    def __len__(self) -> int:
        return len(self.batches)

    def summary(self) -> str:
        if not self.batches:
            return "no batches"
        sizes = [len(b.frames) for b in self.batches]
        return (f"{len(self.batches)} chunks of {min(sizes)}-{max(sizes)} frames, "
                f"{self.overlap} frames of overlap between neighbours")


def plan_batches(frames: list[int], chunk: int = DEFAULT_CHUNK,
                 overlap: int = DEFAULT_OVERLAP) -> BatchPlan:
    """Divide frame numbers into overlapping chunks."""
    ordered = sorted(set(frames))
    if not ordered:
        raise ValueError("no frames to batch")
    if chunk < 2:
        raise ValueError(f"chunk size must be at least 2, got {chunk}")
    if overlap >= chunk:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than the chunk size ({chunk}), "
            f"or the chunks never advance")
    if overlap < 1:
        raise ValueError(
            "overlap must be at least 1 frame; model_merger aligns chunks using the "
            "images they share")

    if len(ordered) <= chunk:
        return BatchPlan([Batch(0, ordered)], chunk, overlap)

    step = chunk - overlap
    batches: list[Batch] = []
    start = 0
    while start < len(ordered):
        window = ordered[start:start + chunk]
        if len(window) < overlap and batches:
            # A stub tail would not merge; fold it into the previous chunk instead.
            batches[-1].frames.extend(f for f in window if f not in batches[-1].frames)
            break
        batches.append(Batch(len(batches), window))
        if start + chunk >= len(ordered):
            break
        start += step

    return BatchPlan(batches, chunk, overlap)


def write_image_lists(plan: BatchPlan, root: Path, clip: str,
                      cameras: list[str], extension: str = "jpg",
                      digits: int = 5) -> list[Path]:
    """One `image_list.txt` per chunk, naming every camera's copy of each frame."""
    directory = root / "batches"
    directory.mkdir(parents=True, exist_ok=True)

    written = []
    for batch in plan.batches:
        lines = [f"{clip}/{camera}/{frame:0{digits}d}.{extension}"
                 for frame in batch.frames for camera in cameras]
        path = directory / f"{batch.name}_images.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def build_commands(plan: BatchPlan, root: Path) -> str:
    """Per-chunk reconstruction plus the merge chain."""
    root_text = str(root)
    lines = [
        "# Batched reconstruction for a long capture.",
        f"# {plan.summary()}",
        "#",
        "# The overlap is what lets model_merger align neighbouring chunks: it works",
        "# from the images they have in common.",
        "",
        "colmap feature_extractor \\",
        f"  --image_path {root_text}/images \\",
        f"  --database_path {root_text}/database.db \\",
        "  --ImageReader.single_camera_per_folder 1",
        "",
        "colmap rig_configurator \\",
        f"  --database_path {root_text}/database.db \\",
        f"  --rig_config_path {root_text}/rig_config.json",
        "",
    ]

    for batch in plan.batches:
        lines += [
            f"# {batch.name}: frames {batch.first}-{batch.last}",
            "colmap sequential_matcher \\",
            f"  --database_path {root_text}/database.db \\",
            f"  --SequentialMatching.image_list_path "
            f"{root_text}/batches/{batch.name}_images.txt",
            "colmap mapper \\",
            f"  --database_path {root_text}/database.db \\",
            f"  --image_path {root_text}/images \\",
            f"  --image_list_path {root_text}/batches/{batch.name}_images.txt \\",
            f"  --output_path {root_text}/batches/{batch.name} \\",
            "  --Mapper.ba_refine_sensor_from_rig 0",
            "",
        ]

    if len(plan) > 1:
        lines.append("# Merge the chunks in order, folding each into the running model.")
        merged = f"{root_text}/batches/{plan.batches[0].name}/0"
        for batch in plan.batches[1:]:
            target = f"{root_text}/batches/merged_{batch.index:03d}"
            lines += [
                "colmap model_merger \\",
                f"  --input_path1 {merged} \\",
                f"  --input_path2 {root_text}/batches/{batch.name}/0 \\",
                f"  --output_path {target}",
            ]
            merged = target
        lines += [
            "",
            "# A merged model benefits from one final bundle adjustment.",
            "colmap bundle_adjuster \\",
            f"  --input_path {merged} \\",
            f"  --output_path {root_text}/sparse/0",
        ]

    return "\n".join(lines) + "\n"
