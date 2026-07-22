"""Does the drawn footprint match what ffmpeg actually extracts?

The overlay is what the user reads to answer "is the car inside this camera?".
If it disagrees with the real extraction, the UI lies about the one thing it
exists for -- and nothing else in the suite would notice, because the extraction
would still be perfectly correct.

These tests run the shipped `static/geometry.js` under node (not a copy of it, so
it cannot drift), ask where it claims a camera's edges land, then place a marker
just inside and just outside each edge and check with ffmpeg whether the extracted
image really contains it.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from threesixty.web.server import STATIC

pytestmark = pytest.mark.ffmpeg

GEOMETRY = STATIC / "js" / "geometry.js"
WIDTH, HEIGHT = 4096, 2048
#: Small on purpose: a fat marker straddles the frustum edge and reads as visible
#: well past it, which looks like an overlay bug but is only the probe's own size.
MARK = 5

CAMERAS = [
    ("level fwd", {"yaw": 0, "pitch": 0, "h_fov": 90, "v_fov": 67.5}),
    ("off-axis", {"yaw": 30, "pitch": -10, "h_fov": 90, "v_fov": 67.5}),
    ("car tilt", {"yaw": -45, "pitch": -20, "h_fov": 90, "v_fov": 67.5}),
    ("across the seam", {"yaw": 170, "pitch": 0, "h_fov": 90, "v_fov": 67.5}),
    ("narrow", {"yaw": 60, "pitch": 5, "h_fov": 45, "v_fov": 34}),
    ("rolled", {"yaw": 0, "pitch": 0, "h_fov": 90, "v_fov": 45, "roll": 90}),
]


@pytest.fixture(scope="session")
def node():
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not installed; overlay geometry cannot be checked")
    return found


def edges_from_js(node, camera, orientation=None):
    """Ask the shipped geometry module where this camera's edges land."""
    script = (
        f"import {{ probeEdges }} from {json.dumps(GEOMETRY.as_uri())};\n"
        f"console.log(JSON.stringify(probeEdges("
        f"{json.dumps(camera)}, {json.dumps(orientation or {})})));\n"
    )
    result = subprocess.run([node, "--input-type=module", "-e", script],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def write_marker(ffmpeg, path, yaw, pitch):
    """A small white spot at a bearing and elevation, widened near the poles."""
    width = min(int(MARK * 2 / max(math.cos(math.radians(pitch)), 0.02)), WIDTH // 2)
    x = int((yaw + 180.0) / 360.0 * WIDTH) - width // 2
    y = int((90.0 - pitch) / 180.0 * HEIGHT) - MARK
    subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=black:size={WIDTH}x{HEIGHT}",
         "-vf", f"drawbox=x={x}:y={y}:w={width}:h={MARK * 2}:color=white:t=fill",
         "-frames:v", "1", str(path)], check=True, capture_output=True)
    return path


def peak_luma(ffmpeg, source, camera, output):
    """Brightest region of what this camera extracts. ~1 when the marker is absent."""
    subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-vf", (f"v360=e:rectilinear:yaw={camera['yaw']:g}:pitch={camera['pitch']:g}:"
                 f"roll={camera.get('roll', 0):g}:"
                 f"h_fov={camera['h_fov']:g}:v_fov={camera['v_fov']:g}:w=400:h=300"),
         "-frames:v", "1", str(output)], check=True, capture_output=True)
    raw = subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-i", str(output),
         "-vf", "scale=30:30:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True).stdout
    return max(raw[:900])


#: A present marker reads 16-66 through a 30x30 probe, an absent one reads 1.
VISIBLE = 8


@pytest.mark.parametrize("label,camera", CAMERAS, ids=[c[0] for c in CAMERAS])
def test_overlay_centre_is_the_camera_axis(node, label, camera):
    edges = edges_from_js(node, camera)
    assert edges["center"]["yaw"] == pytest.approx(camera["yaw"], abs=0.01)
    assert edges["center"]["pitch"] == pytest.approx(camera["pitch"], abs=0.01)


@pytest.mark.parametrize("label,camera", [c for c in CAMERAS if "roll" not in c[1]],
                         ids=[c[0] for c in CAMERAS if "roll" not in c[1]])
def test_overlay_is_not_mirrored(node, label, camera):
    """An inverted up-vector flips the footprint vertically.

    Symmetric fields of view hide this completely -- the outline is identical --
    so it has to be checked against the axis directly.
    """
    edges = edges_from_js(node, camera)
    assert edges["top"]["pitch"] > camera["pitch"], "'top' edge is below the axis"
    assert edges["bottom"]["pitch"] < camera["pitch"], "'bottom' edge is above the axis"
    delta = ((edges["right"]["yaw"] - camera["yaw"] + 540) % 360) - 180
    assert delta > 0, "'right' edge is at decreasing yaw"


@pytest.mark.parametrize("label,camera", CAMERAS, ids=[c[0] for c in CAMERAS])
@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_marker_inside_the_drawn_edge_is_extracted(node, ffmpeg, tmp_path, label, camera, edge):
    edges = edges_from_js(node, camera)
    yaw, pitch = _towards(camera, edges[edge], 0.85)
    source = write_marker(ffmpeg, tmp_path / "eq.png", yaw, pitch)
    peak = peak_luma(ffmpeg, source, camera, tmp_path / "out.png")
    assert peak > VISIBLE, (
        f"the overlay draws {edge} edge as covered, but extraction did not see a "
        f"marker just inside it (peak {peak})"
    )


@pytest.mark.parametrize("label,camera", CAMERAS, ids=[c[0] for c in CAMERAS])
@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_marker_outside_the_drawn_edge_is_not_extracted(node, ffmpeg, tmp_path, label, camera, edge):
    """The failure that matters: the overlay claiming coverage it does not have.

    That is what would let an occluder the user believes is excluded end up in
    every extracted frame.
    """
    edges = edges_from_js(node, camera)
    yaw, pitch = _towards(camera, edges[edge], 1.4)
    source = write_marker(ffmpeg, tmp_path / "eq.png", yaw, pitch)
    peak = peak_luma(ffmpeg, source, camera, tmp_path / "out.png")
    assert peak <= VISIBLE, (
        f"a marker well outside the drawn {edge} edge was still extracted "
        f"(peak {peak}); the overlay understates this camera's coverage"
    )


def _towards(camera, point, factor):
    """Walk from the camera axis towards (and past) a claimed edge."""
    delta_yaw = ((point["yaw"] - camera["yaw"] + 540) % 360) - 180
    delta_pitch = point["pitch"] - camera["pitch"]
    yaw = ((camera["yaw"] + delta_yaw * factor + 180) % 360) - 180
    pitch = max(-89.0, min(89.0, camera["pitch"] + delta_pitch * factor))
    return yaw, pitch


def test_occlusion_fraction_matches_geometry(node):
    """The % column that tells the user how much of a camera the car eats."""
    script = (
        f"import {{ occlusionFraction }} from {json.dumps(GEOMETRY.as_uri())};\n"
        "const cam = {yaw:0, pitch:0, h_fov:90, v_fov:90};\n"
        "console.log(JSON.stringify({"
        "  off: occlusionFraction(cam, {}, 0),"
        "  half: occlusionFraction({yaw:0,pitch:-45,h_fov:90,v_fov:90}, {}, 45),"
        "  none: occlusionFraction({yaw:0,pitch:60,h_fov:40,v_fov:40}, {}, 45),"
        "  all: occlusionFraction({yaw:0,pitch:-80,h_fov:20,v_fov:20}, {}, 45)"
        "}));\n"
    )
    result = subprocess.run([node, "--input-type=module", "-e", script],
                            check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)

    assert data["off"] == 0, "no occluder should mean no occlusion"
    assert data["none"] == 0, "a camera aimed at the sky is not occluded by the ground"
    assert data["all"] == 1, "a camera buried in the occluder should read fully occluded"
    # Aimed exactly at the cone edge: approximately half the view falls inside.
    assert 0.3 < data["half"] < 0.7, f"expected roughly half, got {data['half']}"
