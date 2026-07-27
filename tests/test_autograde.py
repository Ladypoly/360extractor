"""Choosing a grade automatically.

The measurements are deterministic, so most of this runs on synthetic pixel arrays with
no ffmpeg involved. Two properties matter, and they are what the algorithm is *for*:
auto must rescue bad footage without destroying anything, and it must never invent
colour. A splat trainer fits what it is given -- a clipped highlight has no gradient and
a boosted chroma is baked-in speckle, and neither is recoverable afterwards.
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


def panorama(sky, ground, height=180, width=360):
    """An equirect frame: bright sky on top, the scene below it."""
    frame = np.full((height, width, 3), ground, dtype=np.float32)
    frame[:int(height * 0.25)] = sky
    return frame


def well_exposed():
    """An image measuring exactly on target, so auto has nothing to correct.

    A linear ramp's 10th-90th percentile band covers only 80% of its full range, so the
    ramp has to be widened by that factor to land the *measured* span on target.
    """
    full = autograde.TARGET_SPAN / 0.8
    return spread(autograde.TARGET_MEDIAN - full / 2,
                  autograde.TARGET_MEDIAN + full / 2)


def graded(pixels, grade):
    """The luma this grade would actually produce."""
    values = autograde.luma(autograde.trained_band(pixels))
    return autograde.apply_contrast(
        autograde.apply_exposure(values, grade.exposure, grade.black), grade.contrast)


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


class TestTrainedBand:
    """A third of a panorama is sky, and the sky is masked out of the splat."""

    def test_the_sky_does_not_vote_on_how_the_street_is_exposed(self):
        frame = panorama(sky=0.95, ground=0.20)
        analysis = autograde.analyse(frame)
        assert analysis.median == pytest.approx(0.20, abs=0.02)
        assert analysis.banded

    def test_measuring_the_whole_frame_would_get_it_wrong(self):
        """The same frame, flattened: the sky drags the median well off the ground."""
        frame = panorama(sky=0.95, ground=0.20)
        whole = autograde.analyse(frame.reshape(-1, 3))
        assert whole.median > 0.20
        assert not whole.banded

    def test_the_rig_underneath_is_excluded_too(self):
        frame = panorama(sky=0.5, ground=0.5)
        frame[int(180 * 0.9):] = 0.02          # the car roof, straight down
        assert autograde.analyse(frame).median == pytest.approx(0.5, abs=0.02)

    def test_a_flat_array_is_measured_as_given(self):
        pixels = spread(0.2, 0.8)
        assert autograde.trained_band(pixels).shape == pixels.shape


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
        pixels = spread(0.42, 0.5)
        grade = autograde.grade_for(autograde.analyse(pixels),
                                    autograde.luma(pixels))
        assert grade.contrast > 1.0

    def test_already_contrasty_footage_is_not_pushed_further(self):
        grade = autograde.grade_for(autograde.analyse(spread(0.0, 1.0)))
        assert grade.contrast <= 1.0

    def test_a_lifted_black_is_pulled_down(self):
        """Haze and lens flare lift the bottom of the histogram; that is recoverable."""
        pixels = spread(0.14, 0.5)            # lifted and flat: hazy
        grade = autograde.grade_for(autograde.analyse(pixels), autograde.luma(pixels))
        assert 0 < grade.black <= autograde.BLACK_LIMIT

    def test_black_is_never_pulled_out_of_a_picture_that_reaches_zero(self):
        pixels = spread(0.0, 0.6)
        assert autograde.grade_for(autograde.analyse(pixels)).black == 0.0

    def test_a_full_range_picture_with_no_deep_shadows_is_not_crushed(self):
        """Lifted *and* flat is haze; lifted alone is just a scene without black in it,
        and pulling its black away throws shadow detail out for cosmetic contrast."""
        pixels = spread(0.13, 0.93)
        assert autograde.grade_for(autograde.analyse(pixels)).black == 0.0


class TestSaturationIsNeverRaised:
    """The reported complaint, and the property that answers it."""

    @pytest.mark.parametrize("chroma", [0.0, 0.02, 0.05, 0.1, 0.2, 0.29])
    def test_colour_is_never_invented(self, chroma):
        pixels = spread(0.2, 0.8, chroma=chroma)
        assert autograde.grade_for(autograde.analyse(pixels)).saturation == 1.0

    def test_a_lurid_camera_profile_is_brought_down(self):
        pixels = spread(0.3, 0.7, chroma=0.6)
        grade = autograde.grade_for(autograde.analyse(pixels))
        assert grade.saturation < 1.0
        assert grade.saturation >= autograde.SATURATION_RANGE[0]

    def test_monochrome_footage_is_not_given_colour(self):
        grade = autograde.grade_for(autograde.analyse(spread(0.2, 0.8, chroma=0.0)))
        assert grade.saturation == 1.0


class TestClippingGuard:
    """Nothing auto proposes may destroy more than the budget it is allowed."""

    @pytest.mark.parametrize("low,high", [
        (0.02, 0.25), (0.05, 0.9), (0.3, 0.99), (0.14, 0.6), (0.42, 0.5),
    ])
    def test_the_result_never_clips_more_than_the_budget(self, low, high):
        pixels = spread(low, high)
        values = autograde.luma(pixels)
        analysis = autograde.analyse(pixels)
        grade = autograde.grade_for(analysis, values)

        after = autograde.apply_contrast(
            autograde.apply_exposure(values, grade.exposure, grade.black), grade.contrast)
        was_high = float((values > 0.995).mean())
        added = float((after > 0.995).mean()) - was_high
        assert added <= autograde.CLIP_BUDGET / 100 + 0.01

    def test_a_dark_street_under_a_blown_sky_is_not_lifted_into_the_highlights(self):
        """Real footage: black tarmac and a near-white facade in the same band."""
        pixels = np.vstack([spread(0.03, 0.12, 3000), spread(0.90, 0.96, 400)])
        values = autograde.luma(pixels)
        grade = autograde.grade_for(autograde.analyse(pixels), values)

        after = autograde.apply_exposure(values, grade.exposure, grade.black)
        assert float((after > 0.995).mean()) < 0.02

    def test_clipped_highlights_hold_the_exposure_back(self):
        """Brightening pixels that are already at full scale only destroys more."""
        dark = spread(0.05, 0.2, 3000)
        clipped = np.vstack([dark, flat(1.0, 600)])
        free = autograde.grade_for(autograde.analyse(dark), autograde.luma(dark))
        held = autograde.grade_for(autograde.analyse(clipped), autograde.luma(clipped))
        assert held.exposure < free.exposure


class TestSafety:
    @pytest.mark.parametrize("low,high", [
        (0.0, 0.02), (0.98, 1.0), (0.0, 1.0), (0.5, 0.5001),
    ])
    def test_never_produces_anything_out_of_range(self, low, high):
        grade = autograde.grade_for(autograde.analyse(spread(low, high)))
        grade.validate()   # raises if any control is outside what ffmpeg accepts
        assert abs(grade.exposure) <= autograde.EXPOSURE_LIMIT
        assert autograde.CONTRAST_RANGE[0] <= grade.contrast <= autograde.CONTRAST_RANGE[1]
        assert grade.saturation <= 1.0

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

    def test_says_where_it_measured(self):
        analysis = autograde.analyse(panorama(sky=0.95, ground=0.2))
        lines = autograde.describe(analysis, autograde.grade_for(analysis))
        assert any("masked out of the splat" in line for line in lines)

    def test_says_when_it_declined_to_reach_the_target(self):
        """Auto doing less than expected must not look like auto doing nothing."""
        pixels = np.vstack([spread(0.03, 0.12, 3000), spread(0.90, 0.96, 400)])
        analysis = autograde.analyse(pixels)
        grade = autograde.grade_for(analysis, autograde.luma(pixels))
        lines = autograde.describe(analysis, grade)
        assert any("would clip" in line for line in lines)

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
        assert pixels.ndim == 3 and pixels.shape[2] == 3
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
