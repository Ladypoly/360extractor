"""The dataset layout on disk: masks beside images, the trainers' mirror, pruning."""

from pathlib import Path

from threesixty import dataset


def build_working_set(root: Path, clip: str, cameras, frames, masks=True, legacy=False):
    """A minimal dataset, the shape camera generation leaves behind.

    `legacy` writes the older shape: images flat in `images/<camera>/`, masks in a
    separate `masks/<camera>/`, both under a repeated clip folder.
    """
    for camera in cameras:
        if legacy:
            image_dir = root / "images" / clip / camera
            mask_target = root / "masks" / clip / camera
        else:
            image_dir = root / "images" / camera / "images"
            mask_target = root / "images" / camera / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        for stem in frames:
            (image_dir / f"{stem}.jpg").write_bytes(b"jpg")
        if masks:
            mask_target.mkdir(parents=True, exist_ok=True)
            for stem in frames:
                (mask_target / f"{stem}.png").write_bytes(b"png")
    frames_dir = (root / "frames" / clip) if legacy else (root / "frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stem in frames:
        (frames_dir / f"{stem}.jpg").write_bytes(b"jpg")


class TestLayout:
    def test_a_camera_holds_its_images_and_its_masks(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00", "c01"], ["00001", "00002"])

        assert dataset.geometry_dir(tmp_path, "clip", "c00") == tmp_path / "images" / "c00" / "images"
        assert dataset.mask_dir(tmp_path, "clip", "c00") == tmp_path / "images" / "c00" / "masks"

    def test_the_subpath_the_trainers_see(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"])
        assert dataset.relative_subpath(tmp_path, "clip", "c00") == "c00/images"


class TestMirror:
    """COLMAP and Brush each insist on their own name for the same mask file."""

    def test_both_names_are_written(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00", "c01"], ["00001", "00002"])
        result = dataset.mirror_masks(tmp_path, "clip", ["c00", "c01"])

        mirror = tmp_path / "masks" / "c00" / "images"
        assert (mirror / "00001.png").exists()          # Brush
        assert (mirror / "00001.jpg.png").exists()      # COLMAP's doubled extension
        assert result.masks == 4                        # one per image, both cameras

    def test_the_mirror_shares_bytes_with_the_dataset(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"])
        dataset.mirror_masks(tmp_path, "clip", ["c00"])

        original = tmp_path / "images" / "c00" / "masks" / "00001.png"
        mirrored = tmp_path / "masks" / "c00" / "images" / "00001.png"
        assert mirrored.stat().st_ino == original.stat().st_ino

    def test_a_rerun_with_fewer_frames_leaves_nothing_stale(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001", "00002"])
        dataset.mirror_masks(tmp_path, "clip", ["c00"])

        (tmp_path / "images" / "c00" / "masks" / "00002.png").unlink()
        dataset.mirror_masks(tmp_path, "clip", ["c00"])

        mirror = tmp_path / "masks" / "c00" / "images"
        assert sorted(p.name for p in mirror.iterdir()) ==             ["00001.jpg.png", "00001.png"]


class TestEveryImageGetsAMask:
    """The reported complaint: exported images with no mask at all."""

    def test_a_missing_mask_is_filled_with_a_blank_one(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001", "00002"], masks=False)
        # A real image, so the blank can match its size.
        import numpy as np, cv2
        for stem in ("00001", "00002"):
            cv2.imwrite(str(tmp_path / "images" / "c00" / "images" / f"{stem}.jpg"),
                        np.zeros((8, 16, 3), np.uint8))

        filled = dataset.fill_missing_masks(tmp_path, "clip", ["c00"])

        assert sorted(stem for _, stem in filled) == ["00001", "00002"]
        mask = cv2.imread(
            str(tmp_path / "images" / "c00" / "masks" / "00001.png"),
            cv2.IMREAD_GRAYSCALE)
        assert mask.shape == (8, 16) and int(mask.min()) == 255   # keeps everything

    def test_masks_that_exist_are_left_alone(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"])
        assert dataset.fill_missing_masks(tmp_path, "clip", ["c00"]) == []
        assert (tmp_path / "images" / "c00" / "masks" / "00001.png").read_bytes() == b"png"


class TestRemoveFrames:
    def test_a_removed_frame_is_gone_from_every_tree(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00", "c01"], ["00001", "00002"])
        dataset.mirror_masks(tmp_path, "clip", ["c00", "c01"])
        equirect = tmp_path / ".threesixty" / "masks" / "equirect_masks"
        equirect.mkdir(parents=True)
        (equirect / "00002.png").write_bytes(b"png")

        removed = dataset.remove_frames(tmp_path, "clip", ["00002"])

        assert removed.frames == 1 and removed.images == 2 and removed.masks == 2
        assert removed.mirrored == 4          # both names, both cameras
        assert not (tmp_path / "frames" / "00002.jpg").exists()
        assert not (tmp_path / "images" / "c01" / "images" / "00002.jpg").exists()
        assert not (tmp_path / "images" / "c00" / "masks" / "00002.png").exists()
        assert not (equirect / "00002.png").exists()
        assert not (tmp_path / "masks" / "c00" / "images" / "00002.png").exists()

    def test_the_other_frames_survive(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001", "00002", "00003"])
        dataset.remove_frames(tmp_path, "clip", ["00002"])

        kept = sorted(p.stem for p in
                      (tmp_path / "images" / "c00" / "images").glob("*.jpg"))
        assert kept == ["00001", "00003"]

    def test_a_filename_is_accepted_as_well_as_a_stem(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"])
        removed = dataset.remove_frames(tmp_path, "clip", ["00001.jpg"])
        assert removed.frames == 1 and removed.images == 1

    def test_removing_something_absent_is_not_an_error(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"])
        removed = dataset.remove_frames(tmp_path, "clip", ["09999"])
        assert removed.frames == 0 and removed.images == 0


class TestOlderLayouts:
    """Datasets written before the clip level went, and before masks moved in."""

    def test_lookups_follow_the_layout_that_is_on_disk(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"], legacy=True)

        assert dataset.frames_dir(tmp_path, "clip") == tmp_path / "frames" / "clip"
        assert dataset.geometry_dir(tmp_path, "clip", "c00") == tmp_path / "images" / "clip" / "c00"
        assert dataset.mask_dir(tmp_path, "clip", "c00") == tmp_path / "masks" / "clip" / "c00"

    def test_a_fresh_project_gets_the_current_layout(self, tmp_path):
        assert dataset.frames_dir(tmp_path, "clip") == tmp_path / "frames"
        assert dataset.geometry_dir(tmp_path, "clip", "c00") == tmp_path / "images" / "c00" / "images"

    def test_an_older_dataset_still_prunes(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001", "00002"], legacy=True)
        removed = dataset.remove_frames(tmp_path, "clip", ["00002"])
        assert removed.frames == 1 and removed.images == 1 and removed.masks == 1


class TestImageList:
    """COLMAP scans image_path recursively and reads a camera's masks/ folder as more
    photographs -- measured against COLMAP 4.0.2, it doubled the camera count and fed
    all-white images into the reconstruction. The list is what stops it."""

    def test_lists_images_and_not_masks(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00", "c01"], ["00001", "00002"])
        names = dataset.image_list(tmp_path, "clip", ["c00", "c01"])

        assert names == ["c00/images/00001.jpg", "c00/images/00002.jpg",
                         "c01/images/00001.jpg", "c01/images/00002.jpg"]
        assert not any("masks" in name for name in names)

    def test_paths_are_relative_to_the_images_root(self, tmp_path):
        """COLMAP resolves them against --image_path, and records them verbatim."""
        build_working_set(tmp_path, "clip", ["c00"], ["00001"])
        name = dataset.image_list(tmp_path, "clip", ["c00"])[0]
        assert not Path(name).is_absolute()
        assert (tmp_path / "images" / name).exists()

    def test_an_older_flat_dataset_still_lists(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"], legacy=True)
        assert dataset.image_list(tmp_path, "clip", ["c00"]) == ["clip/c00/00001.jpg"]
