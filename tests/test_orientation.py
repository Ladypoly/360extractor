"""Does an extracted image actually point where the rig says it points?

Every other test would pass just as happily with the yaw sign inverted or the whole
sphere rotated 180 degrees -- the images would still be the right size, the right
count, and distinct from each other. These tests pin the convention down against
known-position markers, so a change in ffmpeg's v360 semantics is caught on upgrade
rather than discovered in a ruined dataset.

Equirect layout: x is yaw, x=0 is yaw -180 and x=width/2 is yaw 0.
y is pitch, y=0 is the zenith and y=height is the nadir.
"""

import math
import subprocess

import pytest

pytestmark = pytest.mark.ffmpeg

WIDTH, HEIGHT = 2048, 1024
STRIPE = 24


def _render(ffmpeg, args, output):
    subprocess.run([str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y", *args,
                    "-frames:v", "1", str(output)], check=True, capture_output=True)


def marker_at_yaw(ffmpeg, path, yaw):
    """Black equirect with a white vertical stripe centred on `yaw`."""
    x = int((yaw + 180.0) / 360.0 * WIDTH) - STRIPE // 2
    # Drawn twice so a stripe straddling the +/-180 seam is complete.
    boxes = ",".join(
        f"drawbox=x={offset}:y=0:w={STRIPE}:h={HEIGHT}:color=white:t=fill"
        for offset in (x % WIDTH, x % WIDTH - WIDTH)
    )
    _render(ffmpeg, ["-f", "lavfi", "-i", f"color=black:size={WIDTH}x{HEIGHT}", "-vf", boxes], path)
    return path


def marker_at(ffmpeg, path, yaw, pitch):
    """Black equirect with a white spot at a given bearing and elevation.

    A spot rather than a band: a point projects to a point under any projection,
    whereas a line of constant elevation becomes a curve once the camera tilts, which
    makes "which part of the frame is it in" meaningless at steep pitch.
    """
    # Equirect compresses longitude towards the poles, so a fixed-width box up there
    # subtends almost no angle and all but vanishes once projected. Widen it by
    # 1/cos(pitch) so the marker covers a constant solid angle at any elevation.
    width = min(int(STRIPE * 2 / max(math.cos(math.radians(pitch)), 0.02)), WIDTH // 2)
    x = int((yaw + 180.0) / 360.0 * WIDTH) - width // 2
    y = int((90.0 - pitch) / 180.0 * HEIGHT) - STRIPE
    spot = f"drawbox=x={x}:y={y}:w={width}:h={STRIPE * 2}:color=white:t=fill"
    _render(ffmpeg, ["-f", "lavfi", "-i", f"color=black:size={WIDTH}x{HEIGHT}", "-vf", spot], path)
    return path


#: Read the image at this resolution before aggregating. Scaling straight down to
#: 3x3 is unreliable twice over: the default scaler point-samples, so a small bright
#: marker vanishes entirely, and such tiny raw buffers do not read back dependably.
PROBE = 30


def _luma(ffmpeg, path, size):
    """The image as a `size` x `size` grid of luma values, area-averaged."""
    raw = subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vf", f"scale={size}:{size}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True).stdout
    assert len(raw) >= size * size, f"short read from ffmpeg: {len(raw)} of {size * size} bytes"
    return raw


def grid9(ffmpeg, path):
    """Mean luma of a 3x3 grid over the image; index 4 is the centre cell."""
    raw = _luma(ffmpeg, path, PROBE)
    step = PROBE // 3
    cells = []
    for row in range(3):
        for column in range(3):
            block = [
                raw[y * PROBE + x]
                for y in range(row * step, (row + 1) * step)
                for x in range(column * step, (column + 1) * step)
            ]
            cells.append(sum(block) // len(block))
    return tuple(cells)


def spot_is_centered(cells):
    """Bright in the centre cell, dark in all eight surrounding cells."""
    centre = cells[4]
    ring = [c for i, c in enumerate(cells) if i != 4]
    return centre > 5 and centre > 3 * max(max(ring), 1)


def extract_one(ffmpeg, source, output, yaw=0.0, pitch=0.0, fov=90.0):
    _render(ffmpeg, ["-i", str(source), "-vf",
                     f"v360=e:rectilinear:yaw={yaw:g}:pitch={pitch:g}:"
                     f"h_fov={fov:g}:v_fov={fov:g}:w=300:h=300"], output)
    return output


def thirds(ffmpeg, path):
    """Mean luma of the left, centre and right thirds of an image."""
    cells = grid9(ffmpeg, path)
    columns = [[cells[row * 3 + column] for row in range(3)] for column in range(3)]
    return tuple(sum(c) // 3 for c in columns)


def is_centered(bands):
    """Bright in the middle, dark either side."""
    return bands[1] > 5 and bands[1] > 3 * max(bands[0], bands[2], 1)


@pytest.mark.parametrize("yaw", [0.0, 45.0, 90.0, -90.0, 180.0, -135.0])
def test_camera_aimed_at_a_marker_centers_it(ffmpeg, tmp_path, yaw):
    source = marker_at_yaw(ffmpeg, tmp_path / "eq.png", yaw)
    image = extract_one(ffmpeg, source, tmp_path / "aimed.png", yaw=yaw)
    bands = thirds(ffmpeg, image)
    assert is_centered(bands), f"marker at yaw {yaw} did not land in the centre: {bands}"


@pytest.mark.parametrize("yaw", [0.0, 90.0, -90.0, 180.0])
def test_marker_absent_when_camera_turned_away(ffmpeg, tmp_path, yaw):
    """A 90 degree camera turned 90 degrees away must not see the marker at all.

    This is what distinguishes a correct yaw from an inverted one: both centre the
    marker for symmetric cases, only the correct one puts it out of frame here.
    """
    source = marker_at_yaw(ffmpeg, tmp_path / "eq.png", yaw)
    away = (yaw + 90.0 + 180.0) % 360.0 - 180.0
    image = extract_one(ffmpeg, source, tmp_path / "away.png", yaw=away)
    assert max(thirds(ffmpeg, image)) < 3


def test_yaw_sign_is_not_mirrored(ffmpeg, tmp_path):
    """A marker to one side must not appear on the other.

    Catches a sign inversion, which symmetric rigs like `ring` would otherwise hide
    completely -- the set of images would be identical, just labelled wrongly.
    """
    source = marker_at_yaw(ffmpeg, tmp_path / "eq.png", 45.0)
    correct = thirds(ffmpeg, extract_one(ffmpeg, source, tmp_path / "a.png", yaw=45.0))
    mirrored = thirds(ffmpeg, extract_one(ffmpeg, source, tmp_path / "b.png", yaw=-45.0))
    assert is_centered(correct)
    assert max(mirrored) < 3, "a marker at +45 was visible from -45; yaw sign is mirrored"


@pytest.mark.parametrize("pitch", [0.0, 30.0, -30.0, 60.0, -60.0, 85.0, -85.0])
def test_camera_pitched_at_a_marker_centers_it(ffmpeg, tmp_path, pitch):
    source = marker_at(ffmpeg, tmp_path / "eq.png", 0.0, pitch)
    image = extract_one(ffmpeg, source, tmp_path / "aimed.png", pitch=pitch)
    cells = grid9(ffmpeg, image)
    assert spot_is_centered(cells), f"spot at pitch {pitch} did not land in the centre: {cells}"


@pytest.mark.parametrize("yaw,pitch", [(45.0, 30.0), (-90.0, -30.0), (135.0, -45.0)])
def test_yaw_and_pitch_compose(ffmpeg, tmp_path, yaw, pitch):
    """Both axes at once -- catches an axis swap that neither alone would reveal."""
    source = marker_at(ffmpeg, tmp_path / "eq.png", yaw, pitch)
    image = extract_one(ffmpeg, source, tmp_path / "aimed.png", yaw=yaw, pitch=pitch)
    assert spot_is_centered(grid9(ffmpeg, image))


def test_rig_angles_reach_the_image_through_the_real_pipeline(ffmpeg, tmp_path):
    """The end-to-end claim: a camera named in a rig points where the rig says.

    Every other test in this file drives ffmpeg directly, which pins down ffmpeg's
    convention but proves nothing about our mapping onto it. This one goes through
    plan_extraction and run_extraction, so a bug in rig normalisation, orientation
    folding or filter-graph construction shows up here.
    """
    from threesixty.extract import run_extraction
    from threesixty.ffmpeg import probe_media
    from threesixty.plan import FrameSelection, plan_extraction
    from threesixty.rig import Camera, Orientation, Output, Rig

    bearings = {"north": 0.0, "east": 90.0, "west": -90.0, "south": 180.0}

    for name, bearing in bearings.items():
        source = marker_at(ffmpeg, tmp_path / f"eq_{name}.png", bearing, 0.0)
        rig = Rig(
            cameras=[Camera(name=n, yaw=y, h_fov=90, v_fov=90) for n, y in bearings.items()],
            output=Output(width=300, height=300, format="png"),
        )
        media = probe_media(source, ffmpeg)
        plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path / name)
        run_extraction(plan, ffmpeg)

        for job in plan.passes[0].jobs:
            image = next(iter(sorted(job.directory.glob("*.png"))))
            cells = grid9(ffmpeg, image)
            if job.camera.name == name:
                assert spot_is_centered(cells), (
                    f"marker at bearing {bearing} was not centred in the camera "
                    f"named {name!r}: {cells}"
                )
            else:
                assert max(cells) < 3, (
                    f"marker at bearing {bearing} leaked into camera "
                    f"{job.camera.name!r}, which points elsewhere: {cells}"
                )


def test_rig_orientation_offset_actually_rotates_the_view(ffmpeg, tmp_path):
    """Rig-level orientation is how a tilted mount gets levelled; it must apply.

    A marker at bearing 90 must be centred by a camera at yaw 0 once the rig is
    rotated 90 degrees. If orientation were dropped, the camera would still be
    looking at bearing 0 and see nothing.
    """
    from threesixty.extract import run_extraction
    from threesixty.ffmpeg import probe_media
    from threesixty.plan import FrameSelection, plan_extraction
    from threesixty.rig import Camera, Orientation, Output, Rig

    source = marker_at(ffmpeg, tmp_path / "eq.png", 90.0, 0.0)
    rig = Rig(
        cameras=[Camera(name="fwd", yaw=0.0, h_fov=90, v_fov=90)],
        output=Output(width=300, height=300, format="png"),
        orientation=Orientation(yaw=90.0),
    )
    media = probe_media(source, ffmpeg)
    plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path / "out")
    run_extraction(plan, ffmpeg)

    job = plan.passes[0].jobs[0]
    image = next(iter(sorted(job.directory.glob("*.png"))))
    assert spot_is_centered(grid9(ffmpeg, image)), "rig orientation was not applied"


def test_negative_pitch_looks_down(ffmpeg, tmp_path):
    """The `car-forward` and `dome` presets depend on this sign.

    If negative pitch looked *up*, `car-forward` would aim at the sky and `dome`
    would point straight at the operator -- the exact failure the tool exists to
    prevent, and one no other test would notice.
    """
    ground = marker_at(ffmpeg, tmp_path / "ground.png", 0.0, -60.0)
    looking_down = grid9(ffmpeg, extract_one(ffmpeg, ground, tmp_path / "d.png", pitch=-60.0))
    looking_up = grid9(ffmpeg, extract_one(ffmpeg, ground, tmp_path / "u.png", pitch=60.0))
    assert spot_is_centered(looking_down), "a spot below the horizon was missed by a downward camera"
    assert max(looking_up) < 3, "negative pitch looks UP; every preset's pitch sign is inverted"
