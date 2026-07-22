"""Reading, filtering and writing 3D Gaussian Splatting PLY files.

Deliberately generic. The header is parsed rather than assumed, the vertex block is
loaded into a numpy structured array, and writing reproduces the original header with
only the element count changed. That way a file survives a round trip whatever spherical
harmonic degree it carries and whatever extra properties a trainer decided to add --
nothing is dropped just because this tool did not recognise it.

Brush writes the ordinary INRIA layout (`x, y, z, scale_0..2, opacity, rot_0..3,
f_dc_0..2, f_rest_*`). It can also write a *packed* variant whose positions are
quantised into `u32` fields (`packed_position` and friends). Those cannot be filtered by
position without decoding, so they are refused by name rather than misread.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: PLY scalar types mapped to numpy, with the byte order applied later.
PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

#: Fields that mark a compressed/quantised splat file.
PACKED_FIELDS = {"packed_position", "packed_scale", "packed_rotation", "packed_color"}


class PlyError(RuntimeError):
    """A PLY file is malformed, or is not a splat file this tool can filter."""


@dataclass
class Splats:
    """The vertex block of a splat PLY, plus everything needed to write it back."""

    data: np.ndarray            # structured array, one row per gaussian
    header_lines: list[str]     # original header, verbatim
    byte_order: str             # '<' or '>'
    element_line_index: int     # which header line holds `element vertex N`

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def positions(self) -> np.ndarray:
        """An (N, 3) float64 view of the gaussian centres."""
        missing = [axis for axis in ("x", "y", "z") if axis not in self.data.dtype.names]
        if missing:
            raise PlyError(f"vertex data has no {', '.join(missing)} property")
        return np.stack([self.data["x"], self.data["y"], self.data["z"]],
                        axis=1).astype(np.float64)

    def select(self, keep: np.ndarray) -> "Splats":
        """A copy holding only the rows where `keep` is true."""
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (len(self),):
            raise PlyError(f"mask has shape {keep.shape}, expected ({len(self)},)")
        return Splats(self.data[keep], list(self.header_lines), self.byte_order,
                      self.element_line_index)


def _parse_header(handle) -> tuple[list[str], str, list[tuple[str, str]], int, int]:
    """Read the header. Returns lines, format, properties, vertex count, line index."""
    first = handle.readline()
    if first.strip() != b"ply":
        raise PlyError("not a PLY file (missing the 'ply' magic line)")

    lines = ["ply"]
    fmt = ""
    properties: list[tuple[str, str]] = []
    count = -1
    element_line_index = -1
    in_vertex = False

    while True:
        raw = handle.readline()
        if not raw:
            raise PlyError("header ended without 'end_header'")
        text = raw.decode("ascii", errors="replace").rstrip("\r\n")
        lines.append(text)
        parts = text.split()

        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
                element_line_index = len(lines) - 1
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise PlyError(
                    "vertex properties include a list; splat files do not use these "
                    "and this reader cannot filter them")
            properties.append((parts[1], parts[2]))
        elif parts[0] == "end_header":
            break

    if count < 0:
        raise PlyError("no 'element vertex' in the header")
    return lines, fmt, properties, count, element_line_index


def read(path: str | Path) -> Splats:
    """Load a splat PLY."""
    file_path = Path(path)
    if not file_path.exists():
        raise PlyError(f"no such file: {file_path}")

    with open(file_path, "rb") as handle:
        lines, fmt, properties, count, element_index = _parse_header(handle)

        names = [name for _, name in properties]
        packed = PACKED_FIELDS.intersection(names)
        if packed:
            raise PlyError(
                f"{file_path.name} is a compressed splat file (found "
                f"{', '.join(sorted(packed))}). Positions are quantised, so it cannot "
                f"be filtered by location -- export an uncompressed .ply instead."
            )

        if fmt == "ascii":
            raise PlyError(
                "ascii PLY is not supported; splat files are binary. Re-export as "
                "binary_little_endian.")
        if fmt == "binary_little_endian":
            order = "<"
        elif fmt == "binary_big_endian":
            order = ">"
        else:
            raise PlyError(f"unknown PLY format {fmt!r}")

        fields = []
        for type_name, name in properties:
            if type_name not in PLY_TYPES:
                raise PlyError(f"unknown property type {type_name!r} for {name!r}")
            fields.append((name, order + PLY_TYPES[type_name]))

        dtype = np.dtype(fields)
        raw = handle.read(dtype.itemsize * count)
        if len(raw) < dtype.itemsize * count:
            raise PlyError(
                f"{file_path.name} is truncated: expected {count} vertices "
                f"({dtype.itemsize * count} bytes), found {len(raw)}")
        data = np.frombuffer(raw, dtype=dtype, count=count)

    return Splats(data=data, header_lines=lines, byte_order=order,
                  element_line_index=element_index)


def write(splats: Splats, path: str | Path) -> Path:
    """Write a splat PLY, reproducing the original header with a new vertex count."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    lines = list(splats.header_lines)
    lines[splats.element_line_index] = f"element vertex {len(splats)}"

    with open(file_path, "wb") as handle:
        handle.write(("\n".join(lines) + "\n").encode("ascii"))
        handle.write(np.ascontiguousarray(splats.data).tobytes())
    return file_path


def describe(splats: Splats) -> str:
    """One line about a loaded file, for the CLI."""
    positions = splats.positions
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    return (f"{len(splats):,} gaussians, "
            f"bounds x[{low[0]:.2f}, {high[0]:.2f}] "
            f"y[{low[1]:.2f}, {high[1]:.2f}] "
            f"z[{low[2]:.2f}, {high[2]:.2f}]")
