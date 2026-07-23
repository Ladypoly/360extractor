"""Where the capture happened.

GPS earns its place three times over. It gives COLMAP position priors so a long street
capture does not wander; `model_aligner` uses it to geo-register the model, and that
similarity transform carries a **uniform scale** -- which is the only thing that makes a
cleanup radius mean metres rather than arbitrary reconstruction units.

Two sources, both deliberately vendor-neutral:

* **EXIF** for geotagged stills, parsed here rather than by a library, so the core stays
  dependency-free.
* **GPX sidecars** for video, interpolated to each frame's timestamp. Every camera and
  phone can produce a GPX, which beats reverse-engineering per-vendor telemetry.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ElementTree
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: EXIF tags we care about, from the GPS IFD.
GPS_TAGS = {
    1: "lat_ref", 2: "lat", 3: "lon_ref", 4: "lon", 5: "alt_ref", 6: "alt",
}


class GpsError(RuntimeError):
    """No usable position data."""


@dataclass(frozen=True)
class Fix:
    """One position in time."""

    latitude: float
    longitude: float
    altitude: float = 0.0
    time: float | None = None  # seconds since epoch


# -- distance ---------------------------------------------------------------
#
# GPX carries no traveled-distance, only positions, so the arc length along a track is
# ours to compute. Haversine (great-circle) is accurate to well under a metre at street
# scale and needs no projection or datum choice -- fine for deciding where to cut a
# 500 m segment.

EARTH_RADIUS_M = 6_371_000.0


def haversine(a: Fix, b: Fix) -> float:
    """Great-circle distance between two fixes, in metres. Ignores altitude."""
    from math import asin, cos, radians, sin, sqrt

    lat1, lat2 = radians(a.latitude), radians(b.latitude)
    dlat = lat2 - lat1
    dlon = radians(b.longitude - a.longitude)
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(min(1.0, sqrt(h)))


def cumulative_distance(fixes: list[Fix]) -> list[float]:
    """Distance travelled along the track up to each fix, in metres.

    `result[0] == 0`; `result[i]` is the summed haversine hops from the first fix to the
    i-th. One value per fix, so it lines up index-for-index with `fixes`.
    """
    totals = [0.0]
    for previous, current in zip(fixes, fixes[1:]):
        totals.append(totals[-1] + haversine(previous, current))
    return totals


# -- GPX --------------------------------------------------------------------


def read_gpx(path: str | Path) -> list[Fix]:
    """Every track point in a GPX file, in time order."""
    file_path = Path(path)
    if not file_path.exists():
        raise GpsError(f"no such GPX file: {file_path}")

    try:
        tree = ElementTree.parse(file_path)
    except ElementTree.ParseError as exc:
        raise GpsError(f"{file_path} is not valid XML: {exc}") from exc

    fixes: list[Fix] = []
    # GPX files are namespaced, and the namespace varies by version, so match on the
    # local tag name instead of hardcoding one.
    for element in tree.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in {"trkpt", "wpt", "rtept"}:
            continue
        try:
            latitude = float(element.attrib["lat"])
            longitude = float(element.attrib["lon"])
        except (KeyError, ValueError):
            continue

        altitude, moment = 0.0, None
        for child in element:
            child_tag = child.tag.rsplit("}", 1)[-1]
            if child_tag == "ele" and child.text:
                try:
                    altitude = float(child.text)
                except ValueError:
                    pass
            elif child_tag == "time" and child.text:
                moment = _parse_time(child.text)
        fixes.append(Fix(latitude, longitude, altitude, moment))

    if not fixes:
        raise GpsError(f"{file_path} contains no track points")

    timed = [f for f in fixes if f.time is not None]
    return sorted(timed, key=lambda f: f.time) if timed else fixes


def _parse_time(text: str) -> float | None:
    text = text.strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def interpolate(fixes: list[Fix], when: float) -> Fix:
    """Position at an arbitrary moment, linearly between the surrounding fixes."""
    timed = [f for f in fixes if f.time is not None]
    if not timed:
        raise GpsError("these fixes carry no timestamps, so they cannot be interpolated")
    if len(timed) == 1:
        return timed[0]

    times = [f.time for f in timed]
    if when <= times[0]:
        return timed[0]
    if when >= times[-1]:
        return timed[-1]

    index = bisect_left(times, when)
    before, after = timed[index - 1], timed[index]
    span = after.time - before.time
    ratio = 0.0 if span <= 0 else (when - before.time) / span
    return Fix(
        latitude=before.latitude + (after.latitude - before.latitude) * ratio,
        longitude=before.longitude + (after.longitude - before.longitude) * ratio,
        altitude=before.altitude + (after.altitude - before.altitude) * ratio,
        time=when,
    )


def fixes_for_frames(fixes: list[Fix], frame_times: dict[int, float]) -> dict[int, Fix]:
    """A position for each extracted frame, from its offset into the clip.

    Frame times are offsets in seconds; the GPX is absolute. They are tied together by
    treating the first track point as the start of the clip, which is right when the
    track was recorded alongside the video.
    """
    timed = [f for f in fixes if f.time is not None]
    if not timed:
        raise GpsError("GPX track has no timestamps")
    start = timed[0].time
    return {frame: interpolate(timed, start + offset)
            for frame, offset in frame_times.items()}


# -- EXIF -------------------------------------------------------------------


def read_exif_gps(path: str | Path) -> Fix | None:
    """GPS from a JPEG's EXIF, or None if it carries none.

    A deliberately small APP1/TIFF walk: enough to reach the GPS IFD and read the six
    tags that matter, and nothing else.
    """
    file_path = Path(path)
    data = file_path.read_bytes()
    if data[:2] != b"\xff\xd8":
        return None

    offset = 2
    while offset < len(data) - 4:
        if data[offset] != 0xFF:
            break
        marker = data[offset + 1]
        size = struct.unpack(">H", data[offset + 2:offset + 4])[0]
        if marker == 0xE1 and data[offset + 4:offset + 10] == b"Exif\x00\x00":
            return _gps_from_tiff(data[offset + 10:offset + 2 + size])
        offset += 2 + size
    return None


def _gps_from_tiff(tiff: bytes) -> Fix | None:
    if len(tiff) < 8:
        return None
    order = "<" if tiff[:2] == b"II" else ">" if tiff[:2] == b"MM" else None
    if order is None:
        return None

    def u16(at):
        return struct.unpack(order + "H", tiff[at:at + 2])[0]

    def u32(at):
        return struct.unpack(order + "I", tiff[at:at + 4])[0]

    ifd0 = u32(4)
    gps_offset = None
    count = u16(ifd0)
    for index in range(count):
        entry = ifd0 + 2 + index * 12
        if u16(entry) == 0x8825:  # GPSInfo IFD pointer
            gps_offset = u32(entry + 8)
            break
    if gps_offset is None or gps_offset + 2 > len(tiff):
        return None

    values: dict[str, object] = {}
    gps_count = u16(gps_offset)
    for index in range(gps_count):
        entry = gps_offset + 2 + index * 12
        if entry + 12 > len(tiff):
            break
        tag = u16(entry)
        if tag not in GPS_TAGS:
            continue
        type_id, components = u16(entry + 2), u32(entry + 4)

        if type_id == 2:  # ASCII
            at = u32(entry + 8) if components > 4 else entry + 8
            values[GPS_TAGS[tag]] = tiff[at:at + components].split(b"\x00")[0].decode(
                "ascii", errors="replace")
        elif type_id == 5:  # RATIONAL
            at = u32(entry + 8)
            numbers = []
            for part in range(components):
                base = at + part * 8
                if base + 8 > len(tiff):
                    break
                numerator, denominator = u32(base), u32(base + 4)
                numbers.append(numerator / denominator if denominator else 0.0)
            values[GPS_TAGS[tag]] = numbers
        elif type_id == 1:  # BYTE
            values[GPS_TAGS[tag]] = tiff[entry + 8]

    latitude = _degrees(values.get("lat"), values.get("lat_ref"), "S")
    longitude = _degrees(values.get("lon"), values.get("lon_ref"), "W")
    if latitude is None or longitude is None:
        return None

    altitude = 0.0
    raw_altitude = values.get("alt")
    if isinstance(raw_altitude, list) and raw_altitude:
        altitude = raw_altitude[0]
        if values.get("alt_ref") == 1:
            altitude = -altitude
    return Fix(latitude, longitude, altitude)


def _degrees(parts, reference, negative_ref) -> float | None:
    """Degrees/minutes/seconds triples into signed decimal degrees."""
    if not isinstance(parts, list) or len(parts) < 3:
        return None
    value = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if isinstance(reference, str) and reference.upper().startswith(negative_ref):
        value = -value
    return value


# -- COLMAP reference file --------------------------------------------------


def write_geo_registration(entries: dict[str, Fix], path: str | Path) -> Path:
    """The `image_name X Y Z` file `model_aligner --ref_is_gps 1` reads.

    Written as latitude, longitude, altitude; COLMAP converts to ENU or ECEF itself.
    At least three images are needed for it to estimate a transform.
    """
    file_path = Path(path)
    if len(entries) < 3:
        raise GpsError(
            f"model_aligner needs at least 3 positioned images, got {len(entries)}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name} {fix.latitude:.9f} {fix.longitude:.9f} {fix.altitude:.4f}"
             for name, fix in sorted(entries.items())]
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file_path
