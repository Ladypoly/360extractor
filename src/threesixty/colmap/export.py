"""Writing a COLMAP project for an extracted dataset.

The point of this file: our cameras are *synthetic*, so their relative poses and their
intrinsics are **known exactly** rather than estimated. Handing COLMAP that knowledge
turns a hard problem into an easy one -- it no longer has to solve for where each camera
sits relative to the others, only for where the rig went. That is what stops panoramic
tile sets drifting.

Two conventions have to be bridged, and both are easy to get silently wrong:

* **Axes.** COLMAP cameras are OpenCV: +X right, +Y **down**, +Z forward. This tool is
  y-up (`mask/fuse.py:camera_basis`). So the rows of `cam_from_rig` are
  `[right, -up, forward]`. Flip the sign on that middle row and every camera is upside
  down while the reconstruction still looks superficially fine.
* **Filenames.** COLMAP groups images into frames by *matching filenames across camera
  folders*, so every camera's frame N must be called the same thing. That is why the
  brush layout names files `00001.jpg` inside `<clip>/<camera>/` rather than embedding
  the camera in the filename.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..mask.fuse import camera_basis
from ..rig import Camera, Rig, native_size
from .model import ColmapCamera, matrix_to_quaternion, write_cameras_text

#: COLMAP's mapper flag that keeps the rig we hand it, instead of re-solving it.
PIN_RIG_FLAG = "--Mapper.ba_refine_sensor_from_rig 0"


def focal_from_fov(pixels: int, fov_degrees: float) -> float:
    """Pinhole focal length in pixels for a given field of view.

    `f = (w / 2) / tan(fov / 2)`. Exact, because we chose both numbers ourselves when
    the tile was rendered.
    """
    if not 0 < fov_degrees < 180:
        raise ValueError(f"field of view must be in (0, 180), got {fov_degrees}")
    return (pixels / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)


def camera_axes_in_rig(camera: Camera) -> np.ndarray:
    """Maps rig coordinates into this camera's OpenCV axes.

    Rows, not columns: row 0 is the image's right, row 1 is image-down, row 2 is
    forward. That `-up` is the OpenCV convention (+Y points down the image).

    **This matrix has determinant −1, and that is expected.** Our equirect
    parameterisation, `dir(yaw, pitch) = (sin y cos p, sin p, cos y cos p)` with
    `right = dir(yaw + 90)`, is mirrored with respect to a right-handed world: yaw
    increases clockwise seen from above. Measured, not assumed — a marker at yaw +20
    lands on the right-hand side of a camera at yaw 0.

    It does not matter downstream, because COLMAP is only ever given *relative*
    rotations between cameras, and the product of two mirrors is a proper rotation.
    The reflection cancels exactly in `relative_rotation`, which is where it counts.
    """
    forward, right, up = camera_basis(camera)
    return np.array([right, -up, forward], dtype=np.float64)


def relative_rotation(camera: Camera, reference: Camera) -> np.ndarray:
    """Rotation from the reference camera's frame into this camera's frame.

    A proper rotation (determinant +1) even though both operands are reflections,
    which is why the mirrored world parameterisation never reaches COLMAP.
    """
    return camera_axes_in_rig(camera) @ camera_axes_in_rig(reference).T


def cam_from_rig_quaternion(camera: Camera, reference: Camera
                            ) -> tuple[float, float, float, float]:
    return matrix_to_quaternion(relative_rotation(camera, reference))


@dataclass
class ExportPaths:
    """Where the export put things."""

    rig_config: Path
    cameras: Path
    commands: Path
    geo_registration: Path | None = None


def build_rig_config(rig: Rig, clip: str, source_width: int,
                     include_intrinsics: bool = True) -> list[dict]:
    """The `rig_config.json` structure COLMAP's `rig_configurator` reads.

    The first enabled camera becomes the reference sensor with an identity pose; every
    other camera is posed relative to it. Translations are all zero because these
    cameras genuinely share one optical centre -- the panorama was sampled from a single
    point, so there is no baseline to model.
    """
    cameras = rig.normalized_cameras()
    if not cameras:
        raise ValueError("cannot export a rig with no enabled cameras")

    reference = cameras[0]

    entries = []
    for index, camera in enumerate(cameras):
        entry: dict = {"image_prefix": f"{clip}/{camera.name}/"}

        if index == 0:
            entry["ref_sensor"] = True
        else:
            # Relative to the reference sensor, not to the rig's nominal axes.
            entry["cam_from_rig_rotation"] = list(
                cam_from_rig_quaternion(camera, reference))
            entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]

        if include_intrinsics and source_width:
            width, height = native_size(source_width, camera.h_fov, camera.v_fov)
            entry["camera_model_name"] = "PINHOLE"
            entry["camera_params"] = [
                focal_from_fov(width, camera.h_fov),
                focal_from_fov(height, camera.v_fov),
                width / 2.0,
                height / 2.0,
            ]
        entries.append(entry)

    return [{"cameras": entries}]


def build_cameras(rig: Rig, source_width: int) -> dict[int, ColmapCamera]:
    """Intrinsics per camera, as a COLMAP `cameras.txt` would hold them."""
    cameras: dict[int, ColmapCamera] = {}
    for index, camera in enumerate(rig.normalized_cameras(), start=1):
        width, height = native_size(source_width, camera.h_fov, camera.v_fov)
        cameras[index] = ColmapCamera(
            id=index, model="PINHOLE", width=width, height=height,
            params=[focal_from_fov(width, camera.h_fov),
                    focal_from_fov(height, camera.v_fov),
                    width / 2.0, height / 2.0],
        )
    return cameras


def build_commands(root: Path, has_masks: bool, geo_registration: bool,
                   sequential: bool = True) -> str:
    """The exact command lines for this dataset, in order."""
    root_text = str(root)
    lines = [
        "# COLMAP reconstruction for a 360extract dataset.",
        "# Relative camera poses and intrinsics are known exactly, so COLMAP only has",
        "# to solve the rig trajectory.",
        "",
        "colmap feature_extractor \\",
        f"  --image_path {root_text}/images \\",
        f"  --database_path {root_text}/database.db \\",
        "  --ImageReader.single_camera_per_folder 1 \\",
    ]
    if has_masks:
        lines.append(f"  --ImageReader.mask_path {root_text}/masks \\")
    lines[-1] = lines[-1].rstrip(" \\")

    lines += [
        "",
        "# Configure the rig BEFORE matching: sequential matching pairs images by frame.",
        "colmap rig_configurator \\",
        f"  --database_path {root_text}/database.db \\",
        f"  --rig_config_path {root_text}/rig_config.json",
        "",
        f"colmap {'sequential' if sequential else 'exhaustive'}_matcher \\",
        f"  --database_path {root_text}/database.db",
        "",
        "colmap mapper \\",
        f"  --database_path {root_text}/database.db \\",
        f"  --image_path {root_text}/images \\",
        f"  --output_path {root_text}/sparse \\",
        f"  {PIN_RIG_FLAG}",
    ]

    if geo_registration:
        lines += [
            "",
            "# Geo-register: gives the model a real-world scale, which is what makes",
            "# the splat cleanup radius mean metres.",
            "colmap model_aligner \\",
            f"  --input_path {root_text}/sparse/0 \\",
            f"  --output_path {root_text}/sparse/aligned \\",
            f"  --ref_images_path {root_text}/geo_registration.txt \\",
            "  --ref_is_gps 1 \\",
            "  --alignment_type enu \\",
            "  --robust_alignment_max_error 3.0",
        ]

    lines += [
        "",
        "# Then train:",
        f"brush {root_text}",
    ]
    return "\n".join(lines) + "\n"


def export(root: str | Path, rig: Rig, clip: str, source_width: int,
           has_masks: bool = True, geo_registration: bool = False) -> ExportPaths:
    """Write rig_config.json, cameras.txt and the command list into `root`."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)

    rig_config = directory / "rig_config.json"
    rig_config.write_text(
        json.dumps(build_rig_config(rig, clip, source_width), indent=2) + "\n",
        encoding="utf-8")

    cameras_path = write_cameras_text(build_cameras(rig, source_width),
                                      directory / "colmap_cameras.txt")

    commands = directory / "run_colmap.sh"
    commands.write_text(build_commands(directory, has_masks, geo_registration),
                        encoding="utf-8")

    return ExportPaths(rig_config=rig_config, cameras=cameras_path, commands=commands,
                       geo_registration=(directory / "geo_registration.txt")
                       if geo_registration else None)
