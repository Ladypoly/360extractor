"""COLMAP interop.

Two things here are worth more than the rest, because both fail *quietly*:

* `C = -R^T t`. Get the inversion wrong and the cleanup spheres land somewhere
  plausible but wrong, and nothing errors.
* The axis convention. COLMAP is OpenCV (+X right, +Y **down**, +Z forward); this tool
  is y-up. Flip the sign on the middle row of `cam_from_rig` and every camera is upside
  down while the reconstruction still looks fine.

COLMAP itself is not installed here, so the reader is verified by writing synthetic
models and reading them back.
"""

import json
import math

import numpy as np
import pytest

from threesixty.colmap import batches, export
from threesixty.colmap.model import (
    ColmapCamera,
    ColmapError,
    ColmapImage,
    matrix_to_quaternion,
    quaternion_to_matrix,
    read_model,
    write_cameras_binary,
    write_cameras_text,
    write_images_binary,
    write_images_text,
)
from threesixty.rig import Camera, Rig, ring


class TestQuaternions:
    def test_identity_round_trips(self):
        assert matrix_to_quaternion(np.eye(3)) == pytest.approx((1, 0, 0, 0))

    @pytest.mark.parametrize("axis,angle", [
        ((1, 0, 0), 30), ((0, 1, 0), 90), ((0, 0, 1), 45),
        ((1, 1, 0), 179),            # near 180, where the naive trace formula dies
        ((0.3, -0.5, 0.8), 120),
    ])
    def test_matrix_quaternion_round_trip(self, axis, angle):
        axis = np.array(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        radians = math.radians(angle)
        cross = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
        rotation = (np.eye(3) + math.sin(radians) * cross
                    + (1 - math.cos(radians)) * cross @ cross)

        recovered = quaternion_to_matrix(matrix_to_quaternion(rotation))
        assert np.allclose(recovered, rotation, atol=1e-9)

    def test_zero_quaternion_is_rejected(self):
        with pytest.raises(ColmapError, match="zero length"):
            quaternion_to_matrix((0, 0, 0, 0))


class TestCameraCentre:
    """`C = -R^T t` is the number the whole cleanup depends on."""

    def test_identity_pose_sits_at_the_origin(self):
        image = ColmapImage(1, (1, 0, 0, 0), (0, 0, 0), 1, "a.jpg")
        assert np.allclose(image.center, [0, 0, 0])

    def test_translation_only(self):
        # World-to-camera translation of t means the camera sits at -t.
        image = ColmapImage(1, (1, 0, 0, 0), (2, -3, 5), 1, "a.jpg")
        assert np.allclose(image.center, [-2, 3, -5])

    def test_rotated_pose(self):
        """A camera at a known place, encoded the way COLMAP would."""
        centre = np.array([4.0, 1.0, -2.0])
        rotation = quaternion_to_matrix((math.cos(0.3), 0.0, math.sin(0.3), 0.0))
        tvec = -rotation @ centre

        image = ColmapImage(1, matrix_to_quaternion(rotation), tuple(tvec), 1, "a.jpg")
        assert np.allclose(image.center, centre, atol=1e-9)


class TestModelIO:
    def _model(self):
        cameras = {1: ColmapCamera(1, "PINHOLE", 640, 480, [500.0, 500.0, 320.0, 240.0])}
        images = {
            1: ColmapImage(1, (1, 0, 0, 0), (0, 0, 0), 1, "clip/c00/00001.jpg"),
            2: ColmapImage(2, (0.7071067811865476, 0, 0.7071067811865476, 0),
                           (1.5, -2.0, 3.0), 1, "clip/c01/00001.jpg"),
        }
        return cameras, images

    def test_binary_round_trip(self, tmp_path):
        cameras, images = self._model()
        write_cameras_binary(cameras, tmp_path / "cameras.bin")
        write_images_binary(images, tmp_path / "images.bin")

        model = read_model(tmp_path)
        assert model.cameras[1].model == "PINHOLE"
        assert model.cameras[1].params == pytest.approx([500, 500, 320, 240])
        assert model.images[2].name == "clip/c01/00001.jpg"
        assert np.allclose(model.images[2].center, images[2].center)

    def test_text_round_trip(self, tmp_path):
        cameras, images = self._model()
        write_cameras_text(cameras, tmp_path / "cameras.txt")
        write_images_text(images, tmp_path / "images.txt")

        model = read_model(tmp_path)
        assert model.images[1].name == "clip/c00/00001.jpg"
        assert np.allclose(model.images[2].center, images[2].center, atol=1e-6)

    def test_binary_and_text_agree(self, tmp_path):
        cameras, images = self._model()
        binary = tmp_path / "bin"
        text = tmp_path / "txt"
        write_cameras_binary(cameras, binary / "cameras.bin")
        write_images_binary(images, binary / "images.bin")
        write_cameras_text(cameras, text / "cameras.txt")
        write_images_text(images, text / "images.txt")

        one, two = read_model(binary), read_model(text)
        for key in one.images:
            assert np.allclose(one.images[key].center, two.images[key].center, atol=1e-6)

    def test_binary_is_preferred_when_both_exist(self, tmp_path):
        cameras, images = self._model()
        write_cameras_binary(cameras, tmp_path / "cameras.bin")
        write_images_binary(images, tmp_path / "images.bin")
        write_cameras_text({}, tmp_path / "cameras.txt")
        write_images_text({}, tmp_path / "images.txt")
        assert len(read_model(tmp_path).images) == 2

    def test_missing_model_is_reported_clearly(self, tmp_path):
        with pytest.raises(ColmapError, match="sparse model directory"):
            read_model(tmp_path)

    def test_truncated_binary_is_caught(self, tmp_path):
        cameras, images = self._model()
        write_cameras_binary(cameras, tmp_path / "cameras.bin")
        write_images_binary(images, tmp_path / "images.bin")
        data = (tmp_path / "images.bin").read_bytes()
        (tmp_path / "images.bin").write_bytes(data[:len(data) // 2])
        with pytest.raises(ColmapError):
            read_model(tmp_path)


class TestAxisConvention:
    """COLMAP is OpenCV: +X right, +Y down, +Z forward. This tool is y-up."""

    def test_forward_maps_to_positive_z(self):
        camera = Camera(name="fwd", yaw=0, pitch=0, h_fov=90, v_fov=90)
        axes = export.camera_axes_in_rig(camera)
        # dir(yaw=0, pitch=0) is +Z in our frame; in the camera it must be straight ahead.
        assert np.allclose(axes @ np.array([0, 0, 1.0]), [0, 0, 1], atol=1e-9)

    def test_world_up_maps_to_negative_y(self):
        """The sign that flips every camera upside down if it is wrong."""
        camera = Camera(name="fwd", yaw=0, pitch=0, h_fov=90, v_fov=90)
        axes = export.camera_axes_in_rig(camera)
        assert np.allclose(axes @ np.array([0, 1.0, 0]), [0, -1, 0], atol=1e-9), (
            "up in the world must point up in the image, which is -Y in OpenCV")

    def test_right_maps_to_positive_x(self):
        """Measured against ffmpeg: a marker at yaw +20 lands on the right of a
        camera at yaw 0, so dir(yaw+90) really is the image's right."""
        camera = Camera(name="fwd", yaw=0, pitch=0, h_fov=90, v_fov=90)
        axes = export.camera_axes_in_rig(camera)
        assert np.allclose(axes @ np.array([1.0, 0, 0]), [1, 0, 0], atol=1e-9)

    @pytest.mark.parametrize("yaw,pitch", [(0, 0), (90, 0), (-45, -20), (170, 10)])
    def test_camera_axes_are_orthonormal(self, yaw, pitch):
        axes = export.camera_axes_in_rig(Camera(name="c", yaw=yaw, pitch=pitch))
        assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-9)

    @pytest.mark.parametrize("yaw,pitch", [(0, 0), (90, 0), (-45, -20), (170, 10)])
    def test_camera_axes_are_a_reflection_and_that_is_expected(self, yaw, pitch):
        """Our equirect world is mirrored: yaw increases clockwise seen from above.

        Documented rather than hidden, because it is the reason the *relative*
        rotations below are the only thing exported.
        """
        axes = export.camera_axes_in_rig(Camera(name="c", yaw=yaw, pitch=pitch))
        assert np.isclose(np.linalg.det(axes), -1.0, atol=1e-9)

    @pytest.mark.parametrize("yaw,pitch", [(0, 0), (90, 0), (-45, -20), (170, 10)])
    def test_relative_rotations_are_proper_rotations(self, yaw, pitch):
        """The mirror cancels: this is what COLMAP actually receives.

        A reflection reaching COLMAP would reconstruct a mirror-image world.
        """
        reference = Camera(name="ref", yaw=0, pitch=0)
        relative = export.relative_rotation(Camera(name="c", yaw=yaw, pitch=pitch),
                                            reference)
        assert np.allclose(relative @ relative.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(relative), 1.0, atol=1e-9), (
            "a reflection here would reconstruct a mirror world")

    def test_a_camera_looks_where_the_rig_says(self):
        """The point of the whole conversion: yaw 90 must see what is at yaw 90."""
        camera = Camera(name="east", yaw=90, pitch=0, h_fov=90, v_fov=90)
        in_camera = export.camera_axes_in_rig(camera) @ np.array([1.0, 0, 0])
        assert in_camera[2] > 0.99, "the thing it is aimed at must be straight ahead"

    def test_identity_when_a_camera_is_its_own_reference(self):
        camera = Camera(name="c", yaw=33, pitch=-12, roll=5)
        assert np.allclose(export.relative_rotation(camera, camera), np.eye(3), atol=1e-9)


class TestIntrinsics:
    def test_focal_from_fov(self):
        # 90 degrees across 1000 px: f = 500 / tan(45) = 500.
        assert export.focal_from_fov(1000, 90) == pytest.approx(500.0)

    def test_half_fov_reprojects_to_the_frame_edge(self):
        """A ray at exactly half the horizontal fov must land on the border."""
        width, fov = 1600, 90.0
        focal = export.focal_from_fov(width, fov)
        angle = math.radians(fov / 2)
        x_pixel = width / 2 + focal * math.tan(angle)
        assert x_pixel == pytest.approx(width, abs=1e-6)

    @pytest.mark.parametrize("fov", [0, -10, 180, 200])
    def test_impossible_fov_is_rejected(self, fov):
        with pytest.raises(ValueError, match="field of view"):
            export.focal_from_fov(1000, fov)

    def test_cameras_carry_exact_intrinsics(self):
        cameras = export.build_cameras(ring(4), source_width=4096)
        assert len(cameras) == 4
        first = cameras[1]
        assert first.model == "PINHOLE"
        # ring() is 90 x 67.5 fov; native size at 4096 is 1024 x 768.
        assert (first.width, first.height) == (1024, 768)
        assert first.params[0] == pytest.approx(export.focal_from_fov(1024, 90))
        assert first.params[2] == pytest.approx(512.0)


class TestRigConfig:
    def test_first_camera_is_the_reference_sensor(self):
        config = export.build_rig_config(ring(4), "clip", 4096)
        cameras = config[0]["cameras"]
        assert cameras[0]["ref_sensor"] is True
        assert "cam_from_rig_rotation" not in cameras[0]
        assert all("cam_from_rig_rotation" in c for c in cameras[1:])

    def test_image_prefixes_match_the_written_layout(self):
        config = export.build_rig_config(ring(3), "myclip", 4096)
        prefixes = [c["image_prefix"] for c in config[0]["cameras"]]
        assert prefixes == ["myclip/c00/", "myclip/c01/", "myclip/c02/"]

    def test_translations_are_zero(self):
        """The cameras genuinely share one optical centre: no baseline to model."""
        config = export.build_rig_config(ring(4), "clip", 4096)
        for camera in config[0]["cameras"][1:]:
            assert camera["cam_from_rig_translation"] == [0.0, 0.0, 0.0]

    def test_relative_rotation_recovers_the_yaw_difference(self):
        """A quarter turn between cameras must come back as a quarter turn."""
        rig = Rig(cameras=[Camera(name="a", yaw=0, h_fov=90, v_fov=90),
                           Camera(name="b", yaw=90, h_fov=90, v_fov=90)])
        config = export.build_rig_config(rig, "clip", 0, include_intrinsics=False)
        relative = quaternion_to_matrix(config[0]["cameras"][1]["cam_from_rig_rotation"])
        angle = math.degrees(math.acos((np.trace(relative) - 1) / 2))
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_intrinsics_are_included_when_the_source_size_is_known(self):
        config = export.build_rig_config(ring(2), "clip", 4096)
        assert config[0]["cameras"][0]["camera_model_name"] == "PINHOLE"
        assert len(config[0]["cameras"][0]["camera_params"]) == 4

    def test_an_empty_rig_is_rejected(self):
        rig = ring(2)
        for camera in rig.cameras:
            camera.enabled = False
        with pytest.raises(ValueError, match="no enabled cameras"):
            export.build_rig_config(rig, "clip", 4096)

    def test_written_file_is_valid_json_in_colmap_shape(self, tmp_path):
        paths = export.export(tmp_path, ring(4), "clip", 4096)
        data = json.loads(paths.rig_config.read_text(encoding="utf-8"))
        assert isinstance(data, list) and "cameras" in data[0]
        assert len(data[0]["cameras"]) == 4


class TestCommands:
    def test_rig_is_configured_before_matching(self):
        """COLMAP's docs are explicit: sequential matching pairs images by frame, so
        the rig has to exist first."""
        text = export.build_commands(__import__("pathlib").Path("/d"), True, False)
        assert text.index("rig_configurator") < text.index("sequential_matcher")

    def test_mapper_pins_the_rig(self):
        text = export.build_commands(__import__("pathlib").Path("/d"), True, False)
        assert export.PIN_RIG_FLAG in text

    def test_masks_are_wired_when_present(self):
        with_masks = export.build_commands(__import__("pathlib").Path("/d"), True, False)
        without = export.build_commands(__import__("pathlib").Path("/d"), False, False)
        assert "--ImageReader.mask_path" in with_masks
        assert "--ImageReader.mask_path" not in without

    def test_geo_registration_is_optional(self):
        text = export.build_commands(__import__("pathlib").Path("/d"), True, True)
        assert "model_aligner" in text and "--ref_is_gps 1" in text


class TestBatches:
    def test_short_capture_is_a_single_chunk(self):
        plan = batches.plan_batches(list(range(1, 50)), chunk=300, overlap=40)
        assert len(plan) == 1

    def test_chunks_overlap_by_the_requested_amount(self):
        plan = batches.plan_batches(list(range(1, 1001)), chunk=300, overlap=40)
        assert len(plan) > 1
        for first, second in zip(plan.batches, plan.batches[1:]):
            shared = set(first.frames) & set(second.frames)
            assert len(shared) == 40, (
                "model_merger aligns chunks using the images they share")

    def test_every_frame_appears_somewhere(self):
        frames = list(range(1, 1001))
        plan = batches.plan_batches(frames, chunk=250, overlap=30)
        covered = set()
        for batch in plan.batches:
            covered.update(batch.frames)
        assert covered == set(frames)

    def test_overlap_must_be_smaller_than_the_chunk(self):
        with pytest.raises(ValueError, match="never advance"):
            batches.plan_batches(list(range(500)), chunk=100, overlap=100)

    def test_zero_overlap_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1 frame"):
            batches.plan_batches(list(range(500)), chunk=100, overlap=0)

    def test_commands_merge_every_chunk(self, tmp_path):
        plan = batches.plan_batches(list(range(1, 1001)), chunk=300, overlap=40)
        text = batches.build_commands(plan, tmp_path)
        assert text.count("colmap model_merger") == len(plan) - 1
        assert "bundle_adjuster" in text

    def test_image_lists_name_every_camera(self, tmp_path):
        plan = batches.plan_batches(list(range(1, 100)), chunk=50, overlap=10)
        written = batches.write_image_lists(plan, tmp_path, "clip", ["c00", "c01"])
        lines = written[0].read_text(encoding="utf-8").strip().splitlines()
        assert any(line.startswith("clip/c00/") for line in lines)
        assert any(line.startswith("clip/c01/") for line in lines)
        assert lines[0].endswith(".jpg")
