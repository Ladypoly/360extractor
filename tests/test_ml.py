"""Dynamic occluder detection.

Split deliberately in two:

* Pipeline behaviour -- how detections become masks, how they are dilated, how the
  dataset is walked -- is tested against a stub detector, so it is deterministic and
  needs no weights or network.
* The real backends are exercised against ultralytics' own sample photographs, which
  genuinely contain people and a bus. Those tests skip when the ML extra or the model
  weights are unavailable, rather than silently passing.
"""

from pathlib import Path

import numpy as np
import pytest

from threesixty.mask import dynamic, fuse
from threesixty.mask.geometric import MaskError
from threesixty.rig import Camera, Output, Rig, ring

pytestmark = pytest.mark.ffmpeg

ml = pytest.importorskip("threesixty.mask.ml")


def ultralytics_asset(name):
    try:
        import ultralytics
    except ImportError:
        pytest.skip("ultralytics is not installed")
    path = Path(ultralytics.__file__).parent / "assets" / name
    if not path.exists():
        pytest.skip(f"ultralytics asset {name} is missing")
    return path


class StubBackend:
    """A detector with no model behind it, so the pipeline can be tested exactly."""

    name = "stub"

    def __init__(self, boxes_by_name=None, shape=(120, 160)):
        self.boxes_by_name = boxes_by_name or {}
        self.shape = shape
        self.seen = []

    def detect(self, images):
        results = []
        for image in images:
            self.seen.append(Path(image))
            boxes = self.boxes_by_name.get(Path(image).parent.name, [])
            detections = [ml.Detection("person", 0.9, box) for box in boxes]
            results.append(ml.FrameMasks(
                path=Path(image),
                detections=detections,
                mask=ml.combine(self.shape, detections, dilate=0),
            ))
        return results


class TestCombine:
    def test_no_detections_keeps_everything(self):
        mask = ml.combine((40, 60), [], dilate=0)
        assert mask.shape == (40, 60)
        assert (mask == 255).all()

    def test_a_box_becomes_a_black_rectangle(self):
        mask = ml.combine((40, 60), [ml.Detection("person", 0.9, (10, 5, 20, 15))], 0)
        assert (mask[5:15, 10:20] == 0).all()
        assert mask[0, 0] == 255

    def test_a_segmentation_mask_is_used_when_present(self):
        piece = np.zeros((40, 60), dtype=bool)
        piece[20:30, 30:40] = True
        mask = ml.combine((40, 60), [ml.Detection("person", 0.9, (0, 0, 60, 40), piece)], 0)
        # The box covers everything but the segmentation does not, so the segmentation
        # has to win -- otherwise every detection would black out the whole frame.
        assert (mask[20:30, 30:40] == 0).all()
        assert mask[0, 0] == 255

    def test_dilation_grows_the_masked_area(self):
        detection = ml.Detection("person", 0.9, (20, 20, 30, 30))
        tight = ml.combine((60, 60), [detection], dilate=0)
        grown = ml.combine((60, 60), [detection], dilate=4)
        assert (grown == 0).sum() > (tight == 0).sum()
        # A sliver of leftover pedestrian is enough to seed a floater, hence the grow.
        assert grown[19, 25] == 0

    def test_boxes_are_clipped_to_the_frame(self):
        mask = ml.combine((40, 60), [ml.Detection("person", 0.9, (-30, -30, 200, 200))], 0)
        assert (mask == 0).all()

    def test_a_mismatched_segmentation_is_ignored_not_crashed(self):
        piece = np.ones((5, 5), dtype=bool)
        mask = ml.combine((40, 60), [ml.Detection("person", 0.9, (0, 0, 1, 1), piece)], 0)
        assert mask.shape == (40, 60)


class TestDiscovery:
    def _dataset(self, tmp_path, cameras, frames=3):
        rig = ring(len(cameras), output=Output(width=160, height=120, auto=False))
        for index, camera in enumerate(rig.cameras):
            camera.name = cameras[index]
        for name in cameras:
            directory = tmp_path / "images" / "clip" / name
            directory.mkdir(parents=True)
            for number in range(1, frames + 1):
                (directory / f"clip_{name}_{number:05d}.jpg").write_bytes(b"x")
        return rig

    def test_finds_every_camera(self, tmp_path):
        rig = self._dataset(tmp_path, ["a", "b"])
        found = dynamic.discover(tmp_path, rig)
        assert {entry.camera.name for entry in found} == {"a", "b"}
        assert all(len(entry.frames) == 3 for entry in found)

    def test_mask_directories_mirror_image_directories(self, tmp_path):
        rig = self._dataset(tmp_path, ["a"])
        entry = dynamic.discover(tmp_path, rig)[0]
        assert entry.mask_directory == tmp_path / "masks" / "clip" / "a"

    def test_ignores_directories_not_in_the_rig(self, tmp_path):
        rig = self._dataset(tmp_path, ["a"])
        stray = tmp_path / "images" / "clip" / "not_a_camera"
        stray.mkdir(parents=True)
        (stray / "x_00001.jpg").write_bytes(b"x")
        assert len(dynamic.discover(tmp_path, rig)) == 1

    def test_missing_images_folder_says_so(self, tmp_path):
        with pytest.raises(MaskError, match="run `360extract extract` first"):
            dynamic.discover(tmp_path, ring(2))

    @pytest.mark.parametrize("name,expected", [
        ("clip_fwd_00042.jpg", 42), ("a_b_c_7.png", 7), ("x_000001.jpg", 1),
    ])
    def test_frame_numbers_come_from_the_filename(self, name, expected):
        assert dynamic.frame_number(Path(name)) == expected

    def test_unnumbered_filename_is_rejected(self):
        with pytest.raises(MaskError, match="frame number"):
            dynamic.frame_number(Path("nonsense.jpg"))


class TestPipeline:
    """End to end over a real extracted dataset, with a stub detector."""

    @pytest.fixture
    def dataset(self, ffmpeg, equirect_clip, tmp_path):
        from threesixty.extract import run_extraction
        from threesixty.ffmpeg import probe_media
        from threesixty.plan import FrameSelection, plan_extraction

        rig = Rig(
            cameras=[Camera(name="a", yaw=0, h_fov=90, v_fov=90),
                     Camera(name="b", yaw=45, h_fov=90, v_fov=90)],
            output=Output(width=160, height=160, format="jpg", auto=False),
        )
        media = probe_media(equirect_clip, ffmpeg)
        plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path,
                               ffmpeg=ffmpeg, mask_mode="none")
        run_extraction(plan, ffmpeg)
        return tmp_path, rig

    def test_writes_one_mask_per_image(self, ffmpeg, dataset):
        root, rig = dataset
        backend = StubBackend(shape=(160, 160))
        report = dynamic.run(ffmpeg, root, rig, backend, fuse=False)

        assert report.cameras == 2
        assert report.images > 0
        assert report.masks_written == report.images
        for entry in dynamic.discover(root, rig):
            images = sorted(entry.directory.glob("*.jpg"))
            masks = sorted(entry.mask_directory.glob("*.png"))
            assert {p.stem for p in images} == {p.stem for p in masks}

    def test_detections_reach_the_written_mask(self, ffmpeg, dataset):
        root, rig = dataset
        backend = StubBackend({"a": [(20, 20, 80, 80)]}, shape=(160, 160))
        dynamic.run(ffmpeg, root, rig, backend, fuse=False)

        entry = next(e for e in dynamic.discover(root, rig) if e.camera.name == "a")
        mask = fuse.read_gray(ffmpeg, sorted(entry.mask_directory.glob("*.png"))[0], 160, 160)
        assert mask[50, 50] < 128, "the stub's detection is missing from the mask"
        assert mask[5, 5] == 255

    def test_fusion_carries_a_detection_into_the_overlapping_camera(self, ffmpeg, dataset):
        """The reason fusion exists, over a real dataset.

        Only camera `a` detects anything. Camera `b` overlaps it, so after fusing, `b`
        must be masked too -- otherwise the trainer sees the object in `b` and bakes it in.
        """
        root, rig = dataset
        # Right-hand side of `a`, which is the side `b` overlaps.
        backend = StubBackend({"a": [(110, 40, 160, 120)]}, shape=(160, 160))
        dynamic.run(ffmpeg, root, rig, backend, fuse=True)

        entry_b = next(e for e in dynamic.discover(root, rig) if e.camera.name == "b")
        mask = fuse.read_gray(ffmpeg, sorted(entry_b.mask_directory.glob("*.png"))[0], 160, 160)
        assert (mask < 128).any(), (
            "camera b came back clean; the detection did not cross the overlap"
        )

    def test_without_fusion_the_neighbour_stays_clean(self, ffmpeg, dataset):
        """Confirms the previous test is actually measuring fusion."""
        root, rig = dataset
        backend = StubBackend({"a": [(110, 40, 160, 120)]}, shape=(160, 160))
        dynamic.run(ffmpeg, root, rig, backend, fuse=False)

        entry_b = next(e for e in dynamic.discover(root, rig) if e.camera.name == "b")
        mask = fuse.read_gray(ffmpeg, sorted(entry_b.mask_directory.glob("*.png"))[0], 160, 160)
        assert (mask < 128).sum() == 0

    def test_static_occluders_are_merged_in(self, ffmpeg, dataset):
        root, rig = dataset
        rig.occluders = [{"type": "nadir_cone", "angle": 20}]
        backend = StubBackend(shape=(160, 160))
        dynamic.run(ffmpeg, root, rig, backend, fuse=False, static=True)

        entry = next(e for e in dynamic.discover(root, rig) if e.camera.name == "a")
        mask = fuse.read_gray(ffmpeg, sorted(entry.mask_directory.glob("*.png"))[0], 160, 160)
        assert mask[155, 80] < 128, "the nadir cone did not survive into the final mask"
        assert mask[5, 80] == 255


@pytest.mark.slow
class TestRealBackends:
    """Against ultralytics' own sample photographs. Needs weights, so may download."""

    def _backend(self, name, **kwargs):
        try:
            return ml.make_backend(name, device="cpu", **kwargs)
        except Exception as exc:  # weights unavailable, offline, etc.
            pytest.skip(f"{name} backend unavailable: {exc}")

    def test_yolo_finds_the_people_and_the_bus(self):
        image = ultralytics_asset("bus.jpg")
        backend = self._backend("yolo", classes=("person", "bus"))
        frames = backend.detect([image])

        labels = {d.label for d in frames[0].detections}
        assert "person" in labels, f"no person found in bus.jpg, got {labels}"
        assert "bus" in labels, f"no bus found in bus.jpg, got {labels}"

    def test_yolo_masks_a_meaningful_share_of_the_frame(self):
        image = ultralytics_asset("bus.jpg")
        backend = self._backend("yolo", classes=("person", "bus"))
        mask = backend.detect([image])[0].mask
        ignored = (mask < 128).mean()
        assert 0.05 < ignored < 0.95, f"masked share looks wrong: {ignored:.3f}"

    def test_yolo_respects_the_class_filter(self):
        image = ultralytics_asset("bus.jpg")
        backend = self._backend("yolo", classes=("bus",))
        labels = {d.label for d in backend.detect([image])[0].detections}
        assert labels <= {"bus"}, f"class filter leaked: {labels}"

    def test_unknown_class_is_rejected_up_front(self):
        with pytest.raises(MaskError, match="does not know the class"):
            ml.make_backend("yolo", classes=("velociraptor",), device="cpu")

    def test_sam_refines_yolo_boxes(self):
        image = ultralytics_asset("bus.jpg")
        backend = self._backend("sam2.1", classes=("person", "bus"))
        frames = backend.detect([image])
        assert frames[0].found > 0
        assert any(d.mask is not None for d in frames[0].detections), (
            "SAM returned no segmentation at all"
        )


def test_unknown_backend_name_is_rejected():
    with pytest.raises(MaskError, match="unknown detection backend"):
        ml.make_backend("telepathy")
