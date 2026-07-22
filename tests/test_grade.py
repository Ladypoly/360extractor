"""Tonal correction of the panorama.

Two properties carry the weight:

* **Identity is free.** The default grade must emit no filter and leave output
  byte-identical, so nobody pays for a feature they are not using and existing datasets
  do not silently change.
* **It happens before the split.** Grading each tile separately would let overlapping
  cameras disagree about exposure, which shows up as a seam in the trained splat and
  weakens feature matching between views of the same wall.
"""

import subprocess

import pytest

from threesixty.extract import run_extraction
from threesixty.ffmpeg import probe_media
from threesixty.plan import FrameSelection, build_filter_graph, plan_extraction
from threesixty.rig import Camera, Grade, Output, Rig, RigError, ring


class TestGradeFilter:
    def test_default_is_the_identity(self):
        assert Grade().is_identity
        assert Grade().filter_chain() == ""

    @pytest.mark.parametrize("field,value", [
        ("exposure", 0.5), ("black", 0.1), ("brightness", 0.2),
        ("contrast", 1.3), ("saturation", 0.8), ("gamma", 1.2),
    ])
    def test_any_change_leaves_the_identity(self, field, value):
        grade = Grade(**{field: value})
        assert not grade.is_identity
        assert grade.filter_chain() != ""

    def test_exposure_and_black_use_the_exposure_filter(self):
        chain = Grade(exposure=1.0, black=0.05).filter_chain()
        assert chain.startswith("exposure=")
        assert "eq=" not in chain

    def test_tone_controls_use_eq(self):
        chain = Grade(contrast=1.2).filter_chain()
        assert chain.startswith("eq=")
        assert "exposure=" not in chain

    def test_exposure_is_applied_before_tone(self):
        """Light first, then the curve -- the order a photographer expects."""
        chain = Grade(exposure=1.0, contrast=1.2).filter_chain()
        assert chain.index("exposure=") < chain.index("eq=")

    @pytest.mark.parametrize("field,value", [
        ("exposure", 5.0), ("exposure", -5.0), ("black", 2.0),
        ("brightness", 3.0), ("saturation", -1.0), ("gamma", 0.0),
        ("gamma", 50.0), ("contrast", float("nan")),
    ])
    def test_out_of_range_is_refused(self, field, value):
        with pytest.raises(RigError, match=f"grade.{field}"):
            Grade(**{field: value}).validate()

    def test_a_bad_grade_fails_rig_validation(self):
        rig = ring(2)
        rig.grade = Grade(exposure=9.0)
        with pytest.raises(RigError, match="grade.exposure"):
            rig.validate()


class TestSerialization:
    def test_grade_survives_a_round_trip(self):
        rig = ring(4)
        rig.grade = Grade(exposure=0.75, contrast=1.2, saturation=0.9, gamma=1.05)
        restored = Rig.from_dict(rig.to_dict())
        assert restored.grade == rig.grade

    def test_a_rig_without_a_grade_still_loads(self):
        """Rigs written before grading existed must keep working."""
        data = ring(4).to_dict()
        del data["grade"]
        assert Rig.from_dict(data).grade.is_identity

    def test_unknown_grade_keys_are_ignored(self):
        data = ring(4).to_dict()
        data["grade"]["vibrance"] = 2.0
        assert Rig.from_dict(data).grade.is_identity


class TestFilterGraph:
    def _graph(self, grade):
        rig = ring(3)
        rig.grade = grade
        return build_filter_graph(rig.normalized_cameras(), rig, "fps=2")[0]

    def test_identity_adds_nothing_to_the_graph(self):
        assert self._graph(Grade()) == self._graph(Grade())
        assert "eq=" not in self._graph(Grade())
        assert "exposure=" not in self._graph(Grade())

    def test_grade_appears_once_no_matter_how_many_cameras(self):
        """Before the split, so it is paid for once and every camera agrees."""
        graph = self._graph(Grade(exposure=0.5, contrast=1.2))
        assert graph.count("exposure=exposure") == 1
        assert graph.count("eq=") == 1
        assert graph.count("v360=") == 3

    def test_grade_precedes_the_split(self):
        graph = self._graph(Grade(contrast=1.2))
        assert graph.index("eq=") < graph.index("split=")

    def test_grade_precedes_the_burn(self):
        """Grading after the burn would lift the blacked-out occluder back out of black."""
        rig = ring(2)
        rig.grade = Grade(brightness=0.3)
        graph = build_filter_graph(rig.normalized_cameras(), rig, "fps=2",
                                   burn=True, source_size=(64, 32))[0]
        assert graph.index("eq=") < graph.index("blend=all_mode=multiply")


@pytest.mark.ffmpeg
class TestAgainstFfmpeg:
    def _extract(self, ffmpeg, clip, root, grade):
        rig = ring(2, output=Output(width=160, height=120, format="png", auto=False))
        rig.grade = grade
        media = probe_media(clip, ffmpeg)
        plan = plan_extraction(media, rig, FrameSelection("fps", 1), root,
                               ffmpeg=ffmpeg, mask_mode="none")
        run_extraction(plan, ffmpeg)
        return sorted((root / "images").rglob("*.png"))

    def test_identity_grade_is_byte_identical(self, ffmpeg, equirect_clip, tmp_path):
        """Nobody should pay for a feature they are not using."""
        plain = self._extract(ffmpeg, equirect_clip, tmp_path / "plain", Grade())
        graded = self._extract(ffmpeg, equirect_clip, tmp_path / "identity", Grade())
        assert [p.read_bytes() for p in plain] == [p.read_bytes() for p in graded]

    def test_a_real_grade_changes_the_pixels(self, ffmpeg, equirect_clip, tmp_path):
        plain = self._extract(ffmpeg, equirect_clip, tmp_path / "plain", Grade())
        graded = self._extract(ffmpeg, equirect_clip, tmp_path / "bright",
                               Grade(exposure=1.0, contrast=1.2))
        assert len(plain) == len(graded)
        assert [p.read_bytes() for p in plain] != [p.read_bytes() for p in graded]

    def test_brightening_actually_brightens(self, ffmpeg, equirect_clip, tmp_path):
        def mean_luma(path):
            raw = subprocess.run(
                [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-i", str(path),
                 "-vf", "scale=1:1:flags=area,format=gray", "-f", "rawvideo",
                 "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout
            return raw[0]

        plain = self._extract(ffmpeg, equirect_clip, tmp_path / "plain", Grade())
        bright = self._extract(ffmpeg, equirect_clip, tmp_path / "bright",
                               Grade(brightness=0.25))
        assert mean_luma(bright[0]) > mean_luma(plain[0])

    def test_every_camera_gets_the_same_grade(self, ffmpeg, equirect_clip, tmp_path):
        """Overlapping cameras disagreeing about exposure leaves a seam in the splat."""
        rig = Rig(
            cameras=[Camera(name="a", yaw=0, h_fov=90, v_fov=90),
                     Camera(name="b", yaw=0, h_fov=90, v_fov=90)],   # same direction
            output=Output(width=120, height=120, format="png", auto=False),
        )
        rig.grade = Grade(exposure=0.8, saturation=1.3)
        media = probe_media(equirect_clip, ffmpeg)
        plan = plan_extraction(media, rig, FrameSelection("fps", 1), tmp_path,
                               ffmpeg=ffmpeg, mask_mode="none")
        run_extraction(plan, ffmpeg)

        jobs = {job.camera.name: job for p in plan.passes for job in p.jobs}
        first = sorted(jobs["a"].directory.glob("*.png"))[0].read_bytes()
        second = sorted(jobs["b"].directory.glob("*.png"))[0].read_bytes()
        assert first == second, "two cameras pointing the same way graded differently"
