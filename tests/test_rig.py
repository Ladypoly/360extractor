import math

import pytest

from threesixty.rig import (
    Camera,
    Orientation,
    Output,
    Rig,
    RigError,
    car_forward,
    cube,
    dome,
    handheld,
    ring,
    wrap180,
)


class TestWrap180:
    """v360 hard-errors outside [-180, 180], so this is load-bearing."""

    @pytest.mark.parametrize("raw,expected", [
        (0, 0), (90, 90), (180, 180), (-180, 180),
        (240, -120),   # the ring-of-3 case that made ffmpeg abort
        (270, -90), (360, 0), (450, 90), (-240, 120), (720, 0), (-450, -90),
    ])
    def test_wraps_into_range(self, raw, expected):
        assert wrap180(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [0, 45, 179.9, 180, 240, 359, 360, 1000, -1000, -240])
    def test_result_always_within_v360_limits(self, raw):
        assert -180.0 <= wrap180(raw) <= 180.0

    def test_preserves_direction(self):
        # 240 and -120 must point the same way, not merely both be in range.
        assert math.isclose(math.cos(math.radians(240)), math.cos(math.radians(wrap180(240))))
        assert math.isclose(math.sin(math.radians(240)), math.sin(math.radians(wrap180(240))))


class TestCamera:
    def test_rejects_empty_name(self):
        with pytest.raises(RigError, match="name must not be empty"):
            Camera(name=" ").validate()

    def test_rejects_path_separators_in_name(self):
        # Camera names become directory and file names.
        with pytest.raises(RigError, match="illegal in a filename"):
            Camera(name="fwd/left").validate()

    @pytest.mark.parametrize("fov", [0, -10, 361, float("nan"), float("inf")])
    def test_rejects_impossible_fov(self, fov):
        with pytest.raises(RigError, match="h_fov"):
            Camera(name="c", h_fov=fov).validate()

    def test_normalized_wraps_all_three_axes(self):
        camera = Camera(name="c", yaw=240, pitch=200, roll=-270).normalized()
        assert (camera.yaw, camera.pitch, camera.roll) == (-120.0, -160.0, 90.0)


class TestRigValidation:
    def test_rejects_empty_rig(self):
        with pytest.raises(RigError, match="no cameras"):
            Rig().validate()

    def test_rejects_duplicate_names(self):
        rig = Rig(cameras=[Camera(name="fwd"), Camera(name="FWD", yaw=90)])
        with pytest.raises(RigError, match="duplicate camera name"):
            rig.validate()

    def test_rejects_all_disabled(self):
        rig = Rig(cameras=[Camera(name="a", enabled=False)])
        with pytest.raises(RigError, match="every camera .* disabled"):
            rig.validate()

    def test_warns_on_fov_aspect_mismatch(self):
        # 90/90 fov into a fixed 4:3 frame stretches the image; ffmpeg will not
        # complain. Only reachable with automatic sizing off, since automatic sizing
        # derives the frame from the fov and so can never mismatch.
        rig = Rig(cameras=[Camera(name="c", h_fov=90, v_fov=90)],
                  output=Output(width=1920, height=1440, auto=False))
        assert any("stretched" in w for w in rig.warnings())

    def test_no_warning_when_fov_matches_aspect(self):
        assert ring(8).warnings() == []


class TestOrientation:
    def test_folds_into_every_camera(self):
        rig = Rig(cameras=[Camera(name="a", yaw=0), Camera(name="b", yaw=90)],
                  orientation=Orientation(pitch=-7))
        assert [c.pitch for c in rig.normalized_cameras()] == [-7.0, -7.0]

    def test_wraps_after_folding(self):
        # 170 + 20 = 190, which v360 would reject.
        rig = Rig(cameras=[Camera(name="a", yaw=170)], orientation=Orientation(yaw=20))
        assert rig.normalized_cameras()[0].yaw == pytest.approx(-170.0)

    def test_excludes_disabled_cameras(self):
        rig = Rig(cameras=[Camera(name="a"), Camera(name="b", yaw=90, enabled=False)])
        assert [c.name for c in rig.normalized_cameras()] == ["a"]


class TestPresets:
    def test_ring_spaces_cameras_evenly(self):
        yaws = sorted(c.yaw for c in ring(4).cameras)
        assert yaws == pytest.approx([-90.0, 0.0, 90.0, 180.0])

    def test_ring_of_three_stays_in_v360_range(self):
        # The preset that produces 0/120/240 -- the original failure.
        assert all(-180 <= c.yaw <= 180 for c in ring(3).cameras)

    @pytest.mark.parametrize("count", [1, 3, 6, 8, 12, 16, 36])
    def test_all_ring_sizes_valid(self, count):
        rig = ring(count)
        rig.validate()
        assert len(rig.cameras) == count

    def test_ring_rejects_zero_cameras(self):
        with pytest.raises(RigError, match="at least 1"):
            ring(0)

    def test_cube_covers_all_six_faces(self):
        rig = cube()
        assert len(rig.cameras) == 6
        assert {c.name for c in rig.cameras} == {"front", "back", "left", "right", "up", "down"}
        assert all(c.h_fov == 90 and c.v_fov == 90 for c in rig.cameras)

    def test_dome_never_looks_down(self):
        # The whole point: the operator and the stick live below the horizon.
        assert all(c.pitch >= 0 for c in dome().cameras)

    def test_car_forward_omits_rear_and_masks_nadir(self):
        rig = car_forward()
        assert all(abs(c.yaw) <= 90 for c in rig.cameras)
        assert rig.occluders == [{"type": "nadir_cone", "angle": 40}]

    def test_handheld_tilts_up_and_masks_nadir(self):
        rig = handheld()
        assert all(c.pitch > 0 for c in rig.cameras)
        assert rig.occluders[0]["type"] == "nadir_cone"


class TestSerialization:
    def test_roundtrip_preserves_everything(self):
        original = car_forward()
        original.orientation = Orientation(yaw=5, pitch=-7, roll=1)
        restored = Rig.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_roundtrip_via_file(self, tmp_path):
        original = dome(6)
        path = original.save(tmp_path / "nested" / "rig.json")
        assert Rig.load(path).to_dict() == original.to_dict()

    def test_occluders_survive_roundtrip(self):
        # Written by a later milestone, but must not be dropped by this one.
        occluders = [{"type": "ml", "backend": "sam3", "prompts": ["person"]}]
        rig = ring(4)
        rig.occluders = occluders
        assert Rig.from_dict(rig.to_dict()).occluders == occluders

    def test_rejects_future_schema_version(self):
        data = ring(4).to_dict()
        data["version"] = 99
        with pytest.raises(RigError, match="upgrade 360extract"):
            Rig.from_dict(data)

    def test_rejects_unknown_field(self):
        data = ring(4).to_dict()
        data["cameras"][0]["zoom"] = 2
        with pytest.raises(RigError, match="unrecognized field"):
            Rig.from_dict(data)

    def test_load_reports_bad_json_clearly(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(RigError, match="not valid JSON"):
            Rig.load(path)

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(RigError, match="no such rig file"):
            Rig.load(tmp_path / "absent.json")
