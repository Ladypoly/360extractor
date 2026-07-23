"""Reading and writing COLMAP sparse models.

Only what this tool actually needs: camera intrinsics and image poses, in both the
binary and text encodings. `points3D` is deliberately not parsed -- it is the bulk of a
model and nothing here uses it.

The number that matters downstream is the **camera centre**. COLMAP stores the
world-to-camera transform, so the centre is `C = -R^T t`, and getting that inversion
wrong puts the cleanup spheres in a plausible-looking but completely wrong place.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: COLMAP camera model ids, and how many parameters each carries.
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_IDS = {name: index for index, (name, _) in CAMERA_MODELS.items()}


class ColmapError(RuntimeError):
    """A sparse model is missing, truncated, or in an unexpected shape."""


def quaternion_to_matrix(q) -> np.ndarray:
    """COLMAP quaternions are (w, x, y, z), not (x, y, z, w)."""
    w, x, y, z = (float(v) for v in q)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0:
        raise ColmapError("quaternion has zero length")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion(matrix) -> tuple[float, float, float, float]:
    """Rotation matrix to a COLMAP (w, x, y, z) quaternion.

    Uses the largest-diagonal branch, which stays numerically sound for rotations
    near 180 degrees where the naive trace formula loses all its precision.
    """
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    # A quaternion and its negation describe the same rotation; pick the one with a
    # non-negative real part so round-trips compare equal.
    if quaternion[0] < 0:
        quaternion = -quaternion
    return tuple(float(v) for v in quaternion)


@dataclass
class ColmapCamera:
    """Intrinsics for one camera."""

    id: int
    model: str
    width: int
    height: int
    params: list[float] = field(default_factory=list)

    @property
    def focal(self) -> tuple[float, float]:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            return self.params[0], self.params[0]
        return self.params[0], self.params[1]


@dataclass
class ColmapImage:
    """One registered image, with its world-to-camera pose."""

    id: int
    qvec: tuple[float, float, float, float]  # w, x, y, z
    tvec: tuple[float, float, float]
    camera_id: int
    name: str

    @property
    def rotation(self) -> np.ndarray:
        return quaternion_to_matrix(self.qvec)

    @property
    def center(self) -> np.ndarray:
        """Where the camera was, in world coordinates.

        COLMAP stores world-to-camera, so this is `-R^T t`. This is the single value
        the splat cleanup depends on.
        """
        return -self.rotation.T @ np.asarray(self.tvec, dtype=np.float64)


@dataclass
class SparseModel:
    cameras: dict[int, ColmapCamera] = field(default_factory=dict)
    images: dict[int, ColmapImage] = field(default_factory=dict)

    def centers(self) -> dict[str, np.ndarray]:
        """Camera centre for every image, keyed by the image name COLMAP stored."""
        return {image.name: image.center for image in self.images.values()}


# -- binary reading ---------------------------------------------------------


def _read(handle, fmt: str):
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) < size:
        raise ColmapError(f"unexpected end of file while reading {fmt!r}")
    return struct.unpack(fmt, data)


def read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with open(path, "rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read(handle, "<iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ColmapError(f"unknown camera model id {model_id}")
            name, param_count = CAMERA_MODELS[model_id]
            params = _read(handle, "<" + "d" * param_count)
            cameras[camera_id] = ColmapCamera(camera_id, name, width, height, list(params))
    return cameras


def read_images_binary(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with open(path, "rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(handle, "<idddddddi")

            letters = []
            while True:
                char = handle.read(1)
                if not char or char == b"\x00":
                    break
                letters.append(char)
            name = b"".join(letters).decode("utf-8", errors="replace")

            (num_points,) = _read(handle, "<Q")
            # The 2D observations are not needed here; skip them without allocating.
            handle.seek(num_points * struct.calcsize("<ddq"), 1)

            images[image_id] = ColmapImage(image_id, (qw, qx, qy, qz), (tx, ty, tz),
                                           camera_id, name)
    return images


# -- text reading -----------------------------------------------------------


def _text_rows(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras_text(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for row in _text_rows(path):
        parts = row.split()
        camera_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        cameras[camera_id] = ColmapCamera(camera_id, model, width, height,
                                          [float(v) for v in parts[4:]])
    return cameras


def _looks_like_pose(parts: list[str]) -> bool:
    """A pose row is `ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`.

    Recognised by shape rather than by position in the file: `images.txt` alternates
    pose rows with rows of 2D observations, but the observation row is *empty* when an
    image has none. Counting lines in pairs desynchronises the moment a blank row is
    skipped, and every subsequent image is then read from the wrong line.
    """
    if len(parts) < 10:
        return False
    try:
        int(parts[0])
        [float(v) for v in parts[1:8]]
        int(parts[8])
    except ValueError:
        return False
    return True


def read_images_text(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    for row in _text_rows(path):
        parts = row.split()
        if not _looks_like_pose(parts):
            continue  # a row of 2D observations
        image_id = int(parts[0])
        images[image_id] = ColmapImage(
            id=image_id,
            qvec=tuple(float(v) for v in parts[1:5]),
            tvec=tuple(float(v) for v in parts[5:8]),
            camera_id=int(parts[8]),
            name=" ".join(parts[9:]),
        )
    return images


def read_model(path: str | Path) -> SparseModel:
    """Read a sparse model directory, preferring binary over text."""
    directory = Path(path)
    if not directory.is_dir():
        raise ColmapError(f"{directory} is not a directory")

    if (directory / "images.bin").exists():
        return SparseModel(
            cameras=read_cameras_binary(directory / "cameras.bin"),
            images=read_images_binary(directory / "images.bin"),
        )
    if (directory / "images.txt").exists():
        return SparseModel(
            cameras=read_cameras_text(directory / "cameras.txt"),
            images=read_images_text(directory / "images.txt"),
        )
    raise ColmapError(
        f"no images.bin or images.txt in {directory}. Point at a sparse model "
        f"directory such as sparse/0/."
    )


# -- points (for the live reconstruction view) ------------------------------

#: One points3D.bin record's fixed part: id, xyz, rgb, error, track length. The
#: variable-length track (image_id, point2D_idx pairs) is skipped -- the view only wants
#: positions and colours.
_POINT_RECORD = struct.Struct("<QdddBBBdQ")


def read_points(model_dir: str | Path, limit: int | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Point positions (N,3 float32) and colours (N,3 uint8) from a sparse model.

    `limit` subsamples uniformly, so a huge model still returns a quick preview.
    """
    directory = Path(model_dir)
    binary = directory / "points3D.bin"
    text = directory / "points3D.txt"
    if binary.exists():
        positions, colors = _read_points_binary(binary)
    elif text.exists():
        positions, colors = _read_points_text(text)
    else:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    if limit and len(positions) > limit:
        step = len(positions) // limit
        positions, colors = positions[::step][:limit], colors[::step][:limit]
    return positions, colors


def _read_points_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    with open(path, "rb") as handle:
        (count,) = _read(handle, "<Q")
        size = _POINT_RECORD.size
        for _ in range(count):
            record = handle.read(size)
            if len(record) < size:
                break
            values = _POINT_RECORD.unpack(record)
            positions.append(values[1:4])
            colors.append(values[4:7])
            handle.read(values[8] * 8)   # skip the track (two int32 per element)
    return (np.array(positions, np.float32).reshape(-1, 3),
            np.array(colors, np.uint8).reshape(-1, 3))


def _read_points_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions, colors = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
        colors.append((int(parts[4]), int(parts[5]), int(parts[6])))
    return (np.array(positions, np.float32).reshape(-1, 3),
            np.array(colors, np.uint8).reshape(-1, 3))


# -- writing (used by tests and by the export step) -------------------------


def write_cameras_text(cameras: dict[int, ColmapCamera], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Camera list with one line of data per camera:",
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
             f"# Number of cameras: {len(cameras)}"]
    for camera in cameras.values():
        params = " ".join(f"{v:.10g}" for v in camera.params)
        lines.append(f"{camera.id} {camera.model} {camera.width} {camera.height} {params}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_images_text(images: dict[int, ColmapImage], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Image list with two lines of data per image:",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)",
             f"# Number of images: {len(images)}"]
    for image in images.values():
        q = " ".join(f"{v:.10g}" for v in image.qvec)
        t = " ".join(f"{v:.10g}" for v in image.tvec)
        lines.append(f"{image.id} {q} {t} {image.camera_id} {image.name}")
        lines.append("")  # no 2D observations
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_cameras_binary(cameras: dict[int, ColmapCamera], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in cameras.values():
            model_id = MODEL_IDS.get(camera.model)
            if model_id is None:
                raise ColmapError(f"cannot write unknown camera model {camera.model!r}")
            handle.write(struct.pack("<iiQQ", camera.id, model_id,
                                     camera.width, camera.height))
            handle.write(struct.pack("<" + "d" * len(camera.params), *camera.params))
    return path


def write_images_binary(images: dict[int, ColmapImage], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image in images.values():
            handle.write(struct.pack("<idddddddi", image.id, *image.qvec, *image.tvec,
                                     image.camera_id))
            handle.write(image.name.encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", 0))
    return path
