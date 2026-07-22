"""Removing floaters from the volume the rig itself occupied.

Masking keeps the car and the operator out of the *images*. It cannot stop a trainer
putting gaussians **where the rig was**: that volume is seen by no camera from any
distance, so anything placed there explains away residual error and nothing contradicts
it. On a drive-through or walk-through the rig traces a path down the street, so the
floaters form a continuous trail of rubbish through the middle of the scene.

The fix is blunt and effective: delete every gaussian within a radius of any recorded
camera position. It works because Brush does not move the world -- its COLMAP loader
inverts world-to-cam and uses the translation as-is -- so camera centres and splat
coordinates are in the same frame with no alignment step.

**About the radius and the road.** A sphere centred on the rig also reaches below it. On
a car roof roughly 1.8 m above the ground, a 2.5 m radius eats road surface too. That
road is real data: the tarmac under the vehicle at time *t* genuinely is observed from
*t ± Δ* as the vehicle moves. `floor` spares anything more than N units below the rig
plane; it is off by default, and when it is off the report says how much of what was
removed lay underneath, because that is the number that tells you whether you just put
holes in the road.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..colmap.model import SparseModel
from .ply import Splats

#: Gaussians are tested against positions in blocks, to keep peak memory near
#: (block x positions) floats rather than (all gaussians x all positions).
BLOCK = 200_000


@dataclass
class Trajectory:
    """Where the rig was, one entry per frame."""

    positions: np.ndarray                      # (F, 3)
    frames: list[int] = field(default_factory=list)
    spread: float = 0.0                        # mean scatter within a frame

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def length(self) -> float:
        """Total path length, in whatever units the model uses."""
        if len(self) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    @property
    def median_spacing(self) -> float:
        """Typical distance between consecutive frames.

        Without geo-registration a COLMAP model has an arbitrary scale, so this is the
        only intuition available for choosing a radius: it is one capture interval.
        """
        if len(self) < 2:
            return 0.0
        steps = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        return float(np.median(steps))

    def estimate_up(self) -> tuple[np.ndarray | None, str]:
        """Guess which way is up from the shape of the path.

        A vehicle or a walker moves mostly horizontally, so the direction the
        trajectory varies in *least* is usually vertical. That inference collapses for
        a **straight** path -- driving down a street, which is the main use case here --
        because a line is symmetric about its own axis and every perpendicular
        direction is equally "least". Rather than pick one arbitrarily, this reports
        that it cannot tell, and the caller asks for `--up`.

        Returns the direction and a one-line reason, or `(None, reason)`.
        """
        if len(self) < 3:
            return None, "fewer than 3 frames"

        centred = self.positions - self.positions.mean(axis=0)
        _, singular, vectors = np.linalg.svd(centred, full_matrices=False)
        if singular[0] <= 1e-12:
            return None, "the trajectory does not move"

        # Second singular value tiny next to the first means a straight line, so the
        # two directions perpendicular to it are indistinguishable.
        if singular[1] / singular[0] < 0.02:
            return None, ("the trajectory is essentially straight, so its shape does "
                          "not reveal which way is up")

        direction = vectors[-1]
        return direction / (np.linalg.norm(direction) or 1.0), "least-varying axis"


def trajectory_from_model(model: SparseModel, frame_of=None) -> Trajectory:
    """Collapse a sparse model's image poses into one position per frame.

    Every camera in a frame shares an optical centre -- the rig has no baseline -- so
    their mean is the rig position, and how far they scatter is a free sanity check on
    the reconstruction. A large spread means COLMAP did not honour the rig.
    """
    if frame_of is None:
        from ..mask.dynamic import frame_number

        def frame_of(name):
            return frame_number(Path(name))

    grouped: dict[int, list[np.ndarray]] = {}
    for image in model.images.values():
        try:
            key = frame_of(image.name)
        except Exception:
            continue
        grouped.setdefault(key, []).append(image.center)

    if not grouped:
        raise ValueError(
            "no camera positions could be grouped into frames. Are the image names "
            "the ones 360extract wrote?")

    frames = sorted(grouped)
    positions = np.array([np.mean(grouped[f], axis=0) for f in frames])

    spreads = [float(np.linalg.norm(np.asarray(grouped[f]) - positions[i], axis=1).mean())
               for i, f in enumerate(frames) if len(grouped[f]) > 1]
    spread = float(np.mean(spreads)) if spreads else 0.0

    return Trajectory(positions=positions, frames=frames, spread=spread)


@dataclass
class CleanReport:
    """What the cleanup did, before anything is written."""

    total: int = 0
    removed: int = 0
    radius: float = 0.0
    below_floor_spared: int = 0
    removed_below_rig: int = 0
    trajectory_length: float = 0.0
    median_spacing: float = 0.0
    frame_spread: float = 0.0
    frames: int = 0
    up_known: bool = True

    @property
    def kept(self) -> int:
        return self.total - self.removed

    @property
    def removed_share(self) -> float:
        return self.removed / self.total if self.total else 0.0

    def lines(self) -> list[str]:
        out = [
            f"{self.total:,} gaussians, {self.removed:,} inside the cleanup volume "
            f"({self.removed_share * 100:.1f}%), {self.kept:,} kept",
            f"trajectory: {self.frames} frames, path length "
            f"{self.trajectory_length:.2f}, median frame spacing "
            f"{self.median_spacing:.3f}, radius {self.radius:.3f}",
        ]
        if self.frame_spread > 0.05 * max(self.median_spacing, 1e-9):
            out.append(
                f"warning: cameras within a frame scatter by {self.frame_spread:.3f}, "
                f"which is large next to the {self.median_spacing:.3f} frame spacing -- "
                f"COLMAP may not have honoured the rig")
        if self.below_floor_spared:
            out.append(f"{self.below_floor_spared:,} spared by the floor")
        elif not self.up_known:
            out.append(
                "note: which way is up could not be inferred from a straight "
                "trajectory, so --floor is unavailable until you pass --up")
        elif self.removed and self.removed_below_rig / max(self.removed, 1) > 0.3:
            out.append(
                f"note: {self.removed_below_rig / self.removed * 100:.0f}% of what was "
                f"removed sits below the rig. If that is road surface, --floor keeps it")
        return out


def cleanup_mask(positions: np.ndarray, trajectory: Trajectory, radius: float,
                 floor: float | None = None, up: np.ndarray | None = None
                 ) -> tuple[np.ndarray, dict]:
    """Which gaussians fall inside the cleanup volume.

    Returns a boolean array (true = remove) and a few counts for the report. `up` is
    only needed for `floor` and for the below-the-rig diagnostic; when it is unknown
    and unneeded, the sphere alone decides.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    positions = np.asarray(positions, dtype=np.float64)
    inside = np.zeros(positions.shape[0], dtype=bool)

    camera_positions = trajectory.positions
    radius_squared = radius * radius

    for start in range(0, positions.shape[0], BLOCK):
        block = positions[start:start + BLOCK]
        # (block, frames) squared distances, computed without forming the difference
        # tensor, which would be block x frames x 3 floats.
        squared = (
            (block * block).sum(axis=1)[:, None]
            - 2.0 * block @ camera_positions.T
            + (camera_positions * camera_positions).sum(axis=1)[None, :]
        )
        inside[start:start + BLOCK] = squared.min(axis=1) <= radius_squared

    if up is None:
        up, reason = trajectory.estimate_up()
        if up is None and floor is not None:
            raise ValueError(
                f"cannot apply --floor because {reason}. Give the up direction "
                f"explicitly, e.g. --up 0,1,0")
    else:
        up = np.asarray(up, dtype=np.float64)
        up = up / (np.linalg.norm(up) or 1.0)

    if up is None:
        # No floor requested and no reliable vertical: the sphere alone decides, and
        # the below-the-rig diagnostic is simply unavailable.
        return inside, {"removed_below_rig": 0, "below_floor_spared": 0,
                        "up_known": False}

    heights = positions @ up
    # The trajectory defines the reference plane: the rig sat at this height.
    rig_height = float(np.mean(camera_positions @ up))

    spared = 0
    if floor is not None:
        too_low = heights < (rig_height - floor)
        spared = int((inside & too_low).sum())
        inside = inside & ~too_low

    return inside, {
        "removed_below_rig": int((inside & (heights < rig_height)).sum()),
        "below_floor_spared": spared,
        "up_known": True,
    }


def clean(splats: Splats, trajectory: Trajectory, radius: float,
          floor: float | None = None, up: np.ndarray | None = None
          ) -> tuple[Splats, Splats, CleanReport]:
    """Split a splat file into what survives and what the cleanup removed.

    Both halves are returned so the removal can be inspected rather than trusted --
    loading `removed.ply` shows exactly what was taken out.
    """
    positions = splats.positions
    inside, counts = cleanup_mask(positions, trajectory, radius, floor, up)

    report = CleanReport(
        total=len(splats),
        removed=int(inside.sum()),
        radius=radius,
        trajectory_length=trajectory.length,
        median_spacing=trajectory.median_spacing,
        frame_spread=trajectory.spread,
        frames=len(trajectory),
        **counts,
    )
    return splats.select(~inside), splats.select(inside), report
