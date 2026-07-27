"""The exported RC_Dataset layout, and dropping frames out of a dataset."""

from pathlib import Path

from threesixty import dataset


def build_working_set(root: Path, clip: str, cameras, frames, masks=True):
    """A minimal images/ + masks/ tree, the shape camera generation leaves behind."""
    for camera in cameras:
        image_dir = root / "images" / clip / camera
        image_dir.mkdir(parents=True, exist_ok=True)
        for stem in frames:
            (image_dir / f"{stem}.jpg").write_bytes(b"jpg")
        if masks:
            mask_dir = root / "masks" / clip / camera
            mask_dir.mkdir(parents=True, exist_ok=True)
            for stem in frames:
                (mask_dir / f"{stem}.png").write_bytes(b"png")
    frames_dir = root / "frames" / clip
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stem in frames:
        (frames_dir / f"{stem}.jpg").write_bytes(b"jpg")


class TestExportLayout:
    def test_views_geometry_and_masks_are_named_by_frame_and_view(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00", "c01"], ["00001", "00002"])

        result = dataset.export_dataset(tmp_path, "clip", ["c00", "c01"])

        root = tmp_path / "RC_Dataset"
        assert sorted(p.name for p in root.iterdir()) == ["view_00", "view_01"]
        assert (root / "view_00" / ".geometry" / "frame_000001_v00.jpg").exists()
        assert (root / "view_00" / ".mask" / "frame_000001_v00.png").exists()
        assert (root / "view_01" / ".geometry" / "frame_000002_v01.jpg").exists()
        assert (root / "view_01" / ".mask" / "frame_000002_v01.png").exists()
        assert result.images == 4 and result.masks == 4

    def test_export_shares_bytes_with_the_working_set(self, tmp_path):
        """Hard links, not copies: the export must not double the dataset on disk."""
        build_working_set(tmp_path, "clip", ["c00"], ["00001"])
        dataset.export_dataset(tmp_path, "clip", ["c00"])

        original = tmp_path / "images" / "clip" / "c00" / "00001.jpg"
        exported = tmp_path / "RC_Dataset" / "view_00" / ".geometry" / "frame_000001_v00.jpg"
        assert exported.stat().st_ino == original.stat().st_ino

    def test_a_rerun_with_fewer_frames_leaves_nothing_stale(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001", "00002"])
        dataset.export_dataset(tmp_path, "clip", ["c00"])

        (tmp_path / "images" / "clip" / "c00" / "00002.jpg").unlink()
        dataset.export_dataset(tmp_path, "clip", ["c00"])

        geometry = tmp_path / "RC_Dataset" / "view_00" / ".geometry"
        assert sorted(p.name for p in geometry.iterdir()) == ["frame_000001_v00.jpg"]

    def test_images_without_masks_still_export(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"], masks=False)
        result = dataset.export_dataset(tmp_path, "clip", ["c00"])

        assert result.images == 1 and result.masks == 0
        assert (tmp_path / "RC_Dataset" / "view_00" / ".mask").is_dir()


class TestRemoveFrames:
    def test_a_removed_frame_is_gone_from_every_tree(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00", "c01"], ["00001", "00002"])
        dataset.export_dataset(tmp_path, "clip", ["c00", "c01"])
        equirect = tmp_path / ".threesixty" / "masks" / "equirect_masks"
        equirect.mkdir(parents=True)
        (equirect / "00002.png").write_bytes(b"png")

        removed = dataset.remove_frames(tmp_path, "clip", ["00002"])

        assert removed.frames == 1 and removed.images == 2 and removed.masks == 2
        assert removed.exported == 4          # a geometry and a mask file per view
        assert not (tmp_path / "frames" / "clip" / "00002.jpg").exists()
        assert not (tmp_path / "images" / "clip" / "c01" / "00002.jpg").exists()
        assert not (tmp_path / "masks" / "clip" / "c00" / "00002.png").exists()
        assert not (equirect / "00002.png").exists()
        assert not (tmp_path / "RC_Dataset" / "view_01" / ".geometry"
                    / "frame_000002_v01.jpg").exists()

    def test_the_other_frames_survive(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001", "00002", "00003"])
        dataset.export_dataset(tmp_path, "clip", ["c00"])

        dataset.remove_frames(tmp_path, "clip", ["00002"])

        kept = sorted(p.stem for p in (tmp_path / "images" / "clip" / "c00").glob("*.jpg"))
        assert kept == ["00001", "00003"]

    def test_a_filename_is_accepted_as_well_as_a_stem(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"])
        removed = dataset.remove_frames(tmp_path, "clip", ["00001.jpg"])
        assert removed.frames == 1 and removed.images == 1

    def test_removing_something_absent_is_not_an_error(self, tmp_path):
        build_working_set(tmp_path, "clip", ["c00"], ["00001"])
        removed = dataset.remove_frames(tmp_path, "clip", ["09999"])
        assert removed.frames == 0 and removed.images == 0
