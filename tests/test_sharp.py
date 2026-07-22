"""Sharp frame selection.

The claim being tested is not "it runs" but "it actually picks sharper frames than
uniform sampling would". That needs a clip with known blur in known places.
"""

import subprocess

import pytest

from threesixty import sharp
from threesixty.ffmpeg import probe_media
from threesixty.plan import FrameSelection, plan_extraction
from threesixty.rig import Output, ring

pytestmark = pytest.mark.ffmpeg


#: One sharp frame in every five. At 10 fps and a 2 fps target the selection window
#: is exactly five frames, so each window holds one good frame and four bad ones.
#: The blur has to vary *inside* the window: if whole windows were uniformly blurred,
#: picking the best of them would still return a blurred frame and prove nothing.
SHARP_EVERY = 5
CLIP_FRAMES = 30


@pytest.fixture(scope="session")
def mixed_clip(ffmpeg, tmp_path_factory):
    """10 fps, 3s, where exactly one frame in five is sharp and the rest are blurred.

    Built by concatenating single-frame segments: expressing a per-frame blur
    schedule inside one filter needs an eval expression that would itself need
    testing, and this way the ground truth is unarguable.
    """
    directory = tmp_path_factory.mktemp("mixed")
    segments = []
    for index in range(CLIP_FRAMES):
        segment = directory / f"seg{index:03d}.mp4"
        blur = "null" if index % SHARP_EVERY == 0 else "gblur=sigma=12"
        subprocess.run(
            [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=640x320:rate=10:duration=0.1",
             "-vf", blur, "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p",
             str(segment)], check=True, capture_output=True)
        segments.append(segment)

    listing = directory / "list.txt"
    listing.write_text("".join(f"file '{s.as_posix()}'\n" for s in segments), encoding="utf-8")
    output = directory / "mixed.mp4"
    subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(output)],
        check=True, capture_output=True)
    return output


def is_sharp_frame(index):
    return index % SHARP_EVERY == 0


class TestAnalysis:
    def test_scores_every_frame(self, ffmpeg, mixed_clip):
        media = probe_media(mixed_clip, ffmpeg)
        scores = sharp.analyze(ffmpeg, media)
        assert len(scores) == pytest.approx(media.frame_count, abs=2)
        assert all(value >= 0 for value in scores.scores)

    def test_blurred_frames_score_worse(self, ffmpeg, mixed_clip):
        """The metric must actually track blur, or everything downstream is noise."""
        media = probe_media(mixed_clip, ffmpeg)
        scores = sharp.analyze(ffmpeg, media).scores

        sharp_frames = [v for i, v in enumerate(scores) if is_sharp_frame(i)]
        blurred_frames = [v for i, v in enumerate(scores) if not is_sharp_frame(i)]

        mean_sharp = sum(sharp_frames) / len(sharp_frames)
        mean_blurred = sum(blurred_frames) / len(blurred_frames)
        assert mean_blurred > mean_sharp * 1.5, (
            f"blurdetect did not separate blurred from sharp frames "
            f"({mean_blurred:.2f} vs {mean_sharp:.2f})"
        )

    def test_analysis_respects_the_time_range(self, ffmpeg, mixed_clip):
        media = probe_media(mixed_clip, ffmpeg)
        whole = sharp.analyze(ffmpeg, media)
        part = sharp.analyze(ffmpeg, media, start=0.5, end=1.5)
        assert len(part) < len(whole)


class TestChoosing:
    def test_picks_one_frame_per_window(self):
        scores = sharp.Sharpness([float(i % 5) for i in range(50)])
        chosen = sharp.choose(scores, source_fps=10, target_fps=2)
        assert len(chosen) == 10

    def test_picks_the_sharpest_in_each_window(self):
        # Lower is sharper; frame 3 of each block of 5 is the best.
        scores = sharp.Sharpness([9, 9, 9, 1, 9] * 4)
        chosen = sharp.choose(scores, source_fps=10, target_fps=2)
        assert chosen == [3, 8, 13, 18]

    def test_target_above_source_rate_keeps_every_frame(self):
        scores = sharp.Sharpness([1.0] * 10)
        assert sharp.choose(scores, source_fps=10, target_fps=30) == list(range(10))

    def test_handles_a_ragged_final_window(self):
        scores = sharp.Sharpness([5, 4, 3, 2])  # 4 frames, blocks of 3
        chosen = sharp.choose(scores, source_fps=9, target_fps=3)
        assert chosen == [2, 3]

    def test_rejects_nonpositive_target(self):
        with pytest.raises(ValueError, match="must be positive"):
            sharp.choose(sharp.Sharpness([1.0]), source_fps=10, target_fps=0)

    def test_empty_analysis_yields_nothing(self):
        assert sharp.choose(sharp.Sharpness([]), source_fps=10, target_fps=2) == []


class TestSelectExpression:
    def test_escapes_commas(self):
        expression = sharp.select_expression([1, 5])
        assert expression == r"select='eq(n\,1)+eq(n\,5)'"
        assert r"\," in expression

    def test_empty_selection_selects_nothing(self):
        assert sharp.select_expression([]) == "select=0"


class TestEndToEnd:
    def test_selection_lands_on_the_sharp_frames(self, ffmpeg, mixed_clip, tmp_path):
        """The whole point: pick the good frame out of each window."""
        media = probe_media(mixed_clip, ffmpeg)
        rig = ring(1, output=Output(width=320, height=240, format="png", auto=False))

        plan = plan_extraction(media, rig, FrameSelection("sharp", 2), tmp_path,
                               ffmpeg=ffmpeg)
        chosen = plan.selection.frames
        assert chosen, "sharp selection produced no frames"

        hits = sum(1 for frame in chosen if is_sharp_frame(frame))
        assert hits / len(chosen) > 0.8, (
            f"only {hits}/{len(chosen)} selected frames were the sharp one in their "
            f"window; chose {list(chosen)}"
        )

    def test_beats_uniform_sampling_on_the_same_clip(self, ffmpeg, mixed_clip, tmp_path):
        """Sharp mode has to be better than --fps, not merely different.

        Uniform sampling takes whatever frame lands on the tick, which on this clip
        is blurred four times out of five.
        """
        media = probe_media(mixed_clip, ffmpeg)
        rig = ring(1, output=Output(width=160, height=120, format="png", auto=False))
        scores = sharp.analyze(ffmpeg, media).scores

        chosen = plan_extraction(media, rig, FrameSelection("sharp", 2), tmp_path,
                                 ffmpeg=ffmpeg).selection.frames
        # What plain --fps 2 would take: every 5th frame from a 10 fps source.
        uniform = list(range(0, len(scores), 5))

        mean_sharp = sum(scores[f] for f in chosen) / len(chosen)
        mean_uniform = sum(scores[f] for f in uniform) / len(uniform)
        assert mean_sharp <= mean_uniform, (
            f"sharp selection ({mean_sharp:.2f}) was not sharper than uniform "
            f"sampling ({mean_uniform:.2f})"
        )

    def test_sharp_mode_needs_ffmpeg(self, ffmpeg, mixed_clip, tmp_path):
        media = probe_media(mixed_clip, ffmpeg)
        with pytest.raises(ValueError, match="needs ffmpeg"):
            plan_extraction(media, ring(1), FrameSelection("sharp", 2), tmp_path)

    def test_sharp_extraction_writes_the_expected_count(self, ffmpeg, mixed_clip, tmp_path):
        from threesixty.extract import run_extraction

        media = probe_media(mixed_clip, ffmpeg)
        rig = ring(2, output=Output(width=160, height=120, format="png", auto=False))
        plan = plan_extraction(media, rig, FrameSelection("sharp", 2), tmp_path,
                               ffmpeg=ffmpeg)
        result = run_extraction(plan, ffmpeg)
        assert result.images_written == len(plan.selection.frames) * 2

    def test_stills_bypass_analysis(self, ffmpeg, equirect_clip, tmp_path):
        still = tmp_path / "still.png"
        subprocess.run([str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(equirect_clip), "-frames:v", "1", str(still)],
                       check=True, capture_output=True)
        media = probe_media(still, ffmpeg)
        # No ffmpeg passed: a still must not need an analysis pass at all.
        plan = plan_extraction(media, ring(1), FrameSelection("sharp", 2), tmp_path / "o")
        assert plan.estimated_frames == 1
