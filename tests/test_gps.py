"""GPS: position priors, geo-registration, and therefore metric scale.

Without a real scale a cleanup radius is a number with no meaning. `model_aligner`
estimates a similarity transform -- including uniform scale -- from these coordinates,
which is what turns "2.5" into "2.5 metres".
"""

import pytest

from threesixty import gps


def write_gpx(path, points, namespace=True):
    """`points` are (lat, lon, ele, iso-time) tuples."""
    opening = ('<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
               if namespace else '<gpx version="1.1">')
    body = []
    for latitude, longitude, elevation, when in points:
        body.append(f'<trkpt lat="{latitude}" lon="{longitude}">'
                    f"<ele>{elevation}</ele>"
                    + (f"<time>{when}</time>" if when else "")
                    + "</trkpt>")
    path.write_text(f"{opening}<trk><trkseg>{''.join(body)}</trkseg></trk></gpx>",
                    encoding="utf-8")
    return path


class TestGpx:
    def test_reads_track_points(self, tmp_path):
        path = write_gpx(tmp_path / "t.gpx", [
            (52.5, 13.4, 34.0, "2026-07-22T10:00:00Z"),
            (52.6, 13.5, 36.0, "2026-07-22T10:00:10Z"),
        ])
        fixes = gps.read_gpx(path)
        assert len(fixes) == 2
        assert fixes[0].latitude == pytest.approx(52.5)
        assert fixes[0].altitude == pytest.approx(34.0)
        assert fixes[1].time > fixes[0].time

    def test_handles_an_unnamespaced_gpx(self, tmp_path):
        """Namespaces vary between GPX versions and exporters, so tags are matched by
        local name rather than a hardcoded namespace."""
        path = write_gpx(tmp_path / "t.gpx",
                         [(1.0, 2.0, 3.0, "2026-01-01T00:00:00Z")] * 2,
                         namespace=False)
        assert len(gps.read_gpx(path)) == 2

    def test_points_come_back_in_time_order(self, tmp_path):
        path = write_gpx(tmp_path / "t.gpx", [
            (3.0, 3.0, 0.0, "2026-07-22T10:00:20Z"),
            (1.0, 1.0, 0.0, "2026-07-22T10:00:00Z"),
            (2.0, 2.0, 0.0, "2026-07-22T10:00:10Z"),
        ])
        fixes = gps.read_gpx(path)
        assert [f.latitude for f in fixes] == [1.0, 2.0, 3.0]

    def test_points_without_timestamps_are_still_usable(self, tmp_path):
        path = write_gpx(tmp_path / "t.gpx", [(1.0, 2.0, 3.0, None)])
        assert len(gps.read_gpx(path)) == 1

    def test_empty_track_is_reported(self, tmp_path):
        path = tmp_path / "t.gpx"
        path.write_text("<gpx></gpx>", encoding="utf-8")
        with pytest.raises(gps.GpsError, match="no track points"):
            gps.read_gpx(path)

    def test_malformed_xml_is_reported(self, tmp_path):
        path = tmp_path / "t.gpx"
        path.write_text("<gpx><trkpt", encoding="utf-8")
        with pytest.raises(gps.GpsError, match="not valid XML"):
            gps.read_gpx(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(gps.GpsError, match="no such GPX"):
            gps.read_gpx(tmp_path / "nope.gpx")


class TestInterpolation:
    def _fixes(self, tmp_path):
        return gps.read_gpx(write_gpx(tmp_path / "t.gpx", [
            (0.0, 0.0, 0.0, "2026-07-22T10:00:00Z"),
            (10.0, 20.0, 100.0, "2026-07-22T10:00:10Z"),
        ]))

    def test_midpoint(self, tmp_path):
        fixes = self._fixes(tmp_path)
        middle = gps.interpolate(fixes, fixes[0].time + 5)
        assert middle.latitude == pytest.approx(5.0)
        assert middle.longitude == pytest.approx(10.0)
        assert middle.altitude == pytest.approx(50.0)

    def test_clamps_before_the_start_and_after_the_end(self, tmp_path):
        fixes = self._fixes(tmp_path)
        assert gps.interpolate(fixes, fixes[0].time - 100).latitude == pytest.approx(0.0)
        assert gps.interpolate(fixes, fixes[-1].time + 100).latitude == pytest.approx(10.0)

    def test_untimed_fixes_cannot_be_interpolated(self):
        with pytest.raises(gps.GpsError, match="cannot be interpolated"):
            gps.interpolate([gps.Fix(1.0, 2.0)], 0.0)

    def test_frames_map_to_positions_along_the_track(self, tmp_path):
        fixes = self._fixes(tmp_path)
        # Frame times are offsets into the clip; the track's first point is time zero.
        positions = gps.fixes_for_frames(fixes, {1: 0.0, 2: 5.0, 3: 10.0})
        assert positions[1].latitude == pytest.approx(0.0)
        assert positions[2].latitude == pytest.approx(5.0)
        assert positions[3].latitude == pytest.approx(10.0)


class TestGeoRegistration:
    def test_writes_the_format_model_aligner_reads(self, tmp_path):
        entries = {
            "clip/c00/00001.jpg": gps.Fix(52.5, 13.4, 34.0),
            "clip/c01/00001.jpg": gps.Fix(52.5, 13.4, 34.0),
            "clip/c00/00002.jpg": gps.Fix(52.6, 13.5, 35.0),
        }
        path = gps.write_geo_registration(entries, tmp_path / "geo.txt")
        lines = path.read_text(encoding="utf-8").strip().splitlines()

        assert len(lines) == 3
        for line in lines:
            parts = line.split()
            assert len(parts) == 4, "image_name X Y Z"
            float(parts[1]); float(parts[2]); float(parts[3])
        assert lines[0].startswith("clip/c00/00001.jpg")

    def test_too_few_images_is_refused(self, tmp_path):
        """model_aligner needs at least three to estimate a transform."""
        with pytest.raises(gps.GpsError, match="at least 3"):
            gps.write_geo_registration({"a.jpg": gps.Fix(1, 2, 3)}, tmp_path / "geo.txt")


class TestExif:
    def test_a_file_without_exif_returns_none(self, tmp_path):
        path = tmp_path / "a.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd9")
        assert gps.read_exif_gps(path) is None

    def test_a_non_jpeg_returns_none(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        assert gps.read_exif_gps(path) is None

    @pytest.mark.parametrize("parts,reference,expected", [
        ([52.0, 30.0, 0.0], "N", 52.5),
        ([52.0, 30.0, 0.0], "S", -52.5),
        ([13.0, 24.0, 36.0], "E", 13.41),
        ([0.0, 0.0, 0.0], "N", 0.0),
    ])
    def test_dms_to_decimal_degrees(self, parts, reference, expected):
        assert gps._degrees(parts, reference, "S" if reference in "NS" else "W") == \
               pytest.approx(expected, abs=1e-4)

    def test_incomplete_coordinates_give_none(self):
        assert gps._degrees([52.0], "N", "S") is None
        assert gps._degrees(None, "N", "S") is None
