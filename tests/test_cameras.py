"""Stage-B camera generation: equirect frames -> rectilinear camera tiles."""

import pytest

from threesixty.cameras import generate_cameras
from threesixty.ffmpeg import probe_media
from threesixty.frames import extract_frames, frames_dir
from threesixty.plan import FrameSelection
from threesixty.rig import ring

pytestmark = pytest.mark.ffmpeg


@pytest.fixture
def extracted(ffmpeg, equirect_clip, tmp_path):
    media = probe_media(equirect_clip, ffmpeg)
    extract_frames(ffmpeg, media, FrameSelection(mode="fps", value=5.0), tmp_path)
    return tmp_path, frames_dir(tmp_path, "clip")


def test_generates_a_tile_folder_per_camera(ffmpeg, extracted):
    root, frames = extracted
    rig = ring(4)
    result = generate_cameras(ffmpeg, frames, rig, root)

    assert result.images_written > 0
    assert not result.cancelled
    for camera in rig.normalized_cameras():
        camera_dir = root / "images" / camera.name / ".geometry"
        assert camera_dir.is_dir() and any(camera_dir.glob("*.jpg"))


def test_frame_numbers_match_across_cameras(ffmpeg, extracted):
    """COLMAP groups a frame's tiles by identical filenames across camera folders."""
    root, frames = extracted
    rig = ring(3)
    generate_cameras(ffmpeg, frames, rig, root)

    listings = [sorted(p.name for p in (root / "images" / c.name / ".geometry").glob("*.jpg"))
                for c in rig.normalized_cameras()]
    assert all(names == listings[0] for names in listings)
    assert len(listings[0]) >= 1


def test_sky_cone_writes_mask_sidecars(ffmpeg, extracted):
    """A sky cone projects to one mask per image, mirroring the images tree."""
    root, frames = extracted
    rig = ring(3)
    result = generate_cameras(ffmpeg, frames, rig, root, sky_cone_angle=30.0)

    assert result.masks_written > 0
    for camera in rig.normalized_cameras():
        image_dir = root / "images" / camera.name / ".geometry"
        mask_dir = root / "images" / camera.name / ".mask"
        images = sorted(p.stem for p in image_dir.glob("*.jpg"))
        masks = sorted(p.stem for p in mask_dir.glob("*.png"))
        assert masks == images and images       # a mask per image, same stems


def test_every_image_gets_a_mask_even_with_nothing_to_mask(ffmpeg, extracted):
    """The reported complaint: images shipped with no mask beside them at all."""
    root, frames = extracted
    result = generate_cameras(ffmpeg, frames, ring(2), root)   # no sky cone, no occluders

    assert result.blank_masks > 0
    for camera in ring(2).normalized_cameras():
        images = sorted(p.stem for p in
                        (root / "images" / camera.name / ".geometry").glob("*.jpg"))
        masks = sorted(p.stem for p in
                       (root / "images" / camera.name / ".mask").glob("*.png"))
        assert masks == images and images


def test_masks_are_mirrored_where_the_trainers_look(ffmpeg, extracted):
    """COLMAP and Brush read a mask root mirroring the image tree, under each one's
    own filename convention."""
    root, frames = extracted
    rig = ring(3)
    generate_cameras(ffmpeg, frames, rig, root, sky_cone_angle=30.0)

    for camera in rig.normalized_cameras():
        images = root / "images" / camera.name / ".geometry"
        mirror = root / "masks" / camera.name / ".geometry"
        for image in images.glob("*.jpg"):
            assert (mirror / f"{image.stem}.png").exists()          # Brush
            assert (mirror / f"{image.stem}.jpg.png").exists()      # COLMAP


def test_missing_frames_is_a_clear_error(ffmpeg, tmp_path):
    with pytest.raises(Exception, match="extract frames"):
        generate_cameras(ffmpeg, tmp_path / "frames" / "nope", ring(2), tmp_path)
