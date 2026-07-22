"""Choosing a grade automatically.

The measurements are deterministic, so most of this runs on synthetic pixel arrays with
no ffmpeg involved. The property that matters is that auto is *conservative*: it must
rescue bad footage, leave good footage alone, and never produce something wild.
"""

import numpy as np
import pytest

from threesixty import autograde
from threesixty.rig import Grade


def flat(value, count=4096):
    return np.full((count, 3), value, dtype=np.float32)


def spread(low, high, count=4096, chroma=0.0):
    """A ramp between two luma levels, optionally with some colour in it."""
    ramp = np.linspace(low, high, count, dtype=np.float32)
    pixels = np.repeat(ramp[:, None], 3, axis=1)
    if chroma:
        pixels[:, 0] = np.clip(pixels[:, 0] + chroma / 2, 0, 1)
        pixels[:, 2] = np.clip(pixels[:, 2] - chroma / 2, 0, 1)
    return pixels


def well_exposed():
    """An image measuring exactly on target, so auto has nothing to correct.

    A linear ramp's 10th-90th percentile band covers only 80% of its full range, so the
    ramp has to be widened by that factor to land the *measured* span on target.
    """
    full = autograde.TARGET_SPAN / 0.8
    return spread(autograde.TARGET_MEDIAN - full / 2,
                  autograde.TARGET_MEDIAN + full / 2,
                  chroma=autograde.TARGET_CHROMA)


class TestAnalysis:
    def test_measures_a_known_ramp(self):
        analysis = autograde.analyse(spread(0.2, 0.8))
        assert analysis.median == pytest.approx(0.5, abs=0.02)
        assert analysis.low == pytest.approx(0.2, abs=0.02)
        assert analysis.high == pytest.approx(0.8, abs=0.02)

    def test_detects_clipping(self):
        pixels = np.vstack([flat(1.0, 1000), spread(0.1, 0.6, 1000)])
        analysis = autograde.analyse(pixels)
        assert analysis.clipped_high > 0.4

    def test_measures_chroma(self):
        grey = autograde.analyse(spread(0.2, 0.8, chroma=0.0))
        colourful = autograde.analyse(spread(0.2, 0.8, chroma=0.4))
        assert grey.chroma < 0.01
        assert colourful.chroma > 0.3


class TestGradeChoice:
    def test_dark_footage_is_brightened(self):
        grade = autograde.grade_for(autograde.analyse(spread(0.02, 0.25)))
        assert grade.exposure > 0.5

    def test_bright_footage_is_pulled_down(self):
        grade = autograde.grade_for(autograde.analyse(spread(0.7, 0.95)))
        assert grade.exposure < 0

    def test_well_exposed_footage_is_left_alone(self):
        """Auto has to be safe to press on footage that is already fine."""
        analysis = autograde.analyse(well_exposed())
        assert autograde.grade_for(analysis).is_identity

    def test_flat_footage_gains_contrast(self):
        grade = autograde.grade_for(autograde.analyse(spread(0.42, 0.5)))
        assert grade.contrast > 1.2

    def test_already_contrasty_footage_is_not_pushed_further(self):
        grade = autograde.grade_for(autograde.analyse(spread(0.0, 1.0)))
        assert grade.contrast <= 1.0

    def test_washed_out_colour_is_boosted(self):
        grade = autograde.grade_for(autograde.analyse(spread(0.2, 0.8, chroma=0.05)))
        assert grade.saturation > 1.0

    def test_monochrome_footage_is_not_given_colour(self):
        """Dividing by a near-zero chroma would produce something lurid."""
        grade = autograde.grade_for(autograde.analyse(spread(0.2, 0.8, chroma=0.0)))
        assert grade.saturation == 1.0

    def test_clipped_highlights_hold_the_exposure_back(self):
        """Brightening pixels that are already at full scale only destroys more."""
        dark = spread(0.05, 0.2, 3000)
        clipped = np.vstack([dark, flat(1.0, 600)])
        free = autograde.grade_for(autograde.analyse(dark))
        held = autograde.grade_for(autograde.analyse(clipped))
        assert held.exposure < free.exposure

    @pytest.mark.parametrize("low,high", [
        (0.0, 0.02), (0.98, 1.0), (0.0, 1.0), (0.5, 0.5001),
    ])
    def test_never_produces_anything_out_of_range(self, low, high):
        grade = autograde.grade_for(autograde.analyse(spread(low, high)))
        grade.validate()   # raises if any control is outside what ffmpeg accepts
        assert abs(grade.exposure) <= autograde.EXPOSURE_LIMIT
        assert autograde.CONTRAST_RANGE[0] <= grade.contrast <= autograde.CONTRAST_RANGE[1]

    def test_pure_black_does_not_divide_by_zero(self):
        grade = autograde.grade_for(autograde.analyse(flat(0.0)))
        grade.validate()

    def test_the_result_is_a_usable_filter(self):
        grade = autograde.grade_for(autograde.analyse(spread(0.02, 0.25)))
        assert "exposure=" in grade.filter_chain()


class TestDescribe:
    def test_explains_what_it_did(self):
        analysis = autograde.analyse(spread(0.02, 0.25))
        lines = autograde.describe(analysis, autograde.grade_for(analysis))
        assert any("median" in line for line in lines)
        assert any("stops" in line for line in lines)

    def test_says_so_when_nothing_needs_changing(self):
        analysis = autograde.analyse(well_exposed())
        lines = autograde.describe(analysis, autograde.grade_for(analysis))
        assert any("already well exposed" in line for line in lines)

    def test_mentions_held_back_exposure(self):
        pixels = np.vstack([spread(0.02, 0.2, 3000), flat(1.0, 600)])
        analysis = autograde.analyse(pixels)
        lines = autograde.describe(analysis, autograde.grade_for(analysis))
        assert any("full brightness" in line for line in lines)


@pytest.mark.ffmpeg
class TestAgainstRealImages:
    def _write(self, ffmpeg, path, colour, size="256x128"):
        import subprocess
        subprocess.run(
            [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"color={colour}:size={size}",
             "-frames:v", "1", str(path)], check=True, capture_output=True)
        return path

    def test_samples_a_real_file(self, ffmpeg, tmp_path):
        image = self._write(ffmpeg, tmp_path / "grey.png", "gray")
        pixels = autograde.sample(ffmpeg, image)
        assert pixels.shape[1] == 3
        assert 0.4 < float(pixels.mean()) < 0.6

    def test_a_dark_image_gets_a_positive_exposure(self, ffmpeg, tmp_path):
        image = self._write(ffmpeg, tmp_path / "dark.png", "0x101010")
        grade, analysis = autograde.auto_grade(ffmpeg, image)
        assert analysis.median < 0.1
        assert grade.exposure > 0

    def test_reading_a_missing_file_is_reported(self, ffmpeg, tmp_path):
        from threesixty.ffmpeg import FFmpegError
        with pytest.raises(FFmpegError, match="could not read"):
            autograde.sample(ffmpeg, tmp_path / "nope.png")

    def test_a_real_frame_round_trips_into_a_valid_filter(self, ffmpeg, equirect_clip,
                                                          tmp_path):
        import subprocess

        frame = tmp_path / "frame.png"
        subprocess.run([str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(equirect_clip), "-frames:v", "1", str(frame)],
                       check=True, capture_output=True)
        grade, _ = autograde.auto_grade(ffmpeg, frame)

        chain = grade.filter_chain()
        out = tmp_path / "graded.png"
        argv = [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(frame)]
        if chain:
            argv += ["-vf", chain]
        argv += ["-frames:v", "1", str(out)]
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert out.exists()
