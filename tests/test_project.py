"""Projects: settings that survive closing the window, and honest stage tracking.

The interesting logic here is staleness. "Files exist" is not the same as "up to date":
a project that was extracted and then had its rig edited must say so, or the user
re-runs masking over images that no longer match the rig they are looking at.
"""

import json

import pytest

from threesixty.project import (
    PROJECT_FILENAME,
    STAGES,
    DetectSettings,
    FrameSettings,
    Project,
    ProjectError,
    find,
)
from threesixty.rig import Camera, ring


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / "job", sources=[], rig=ring(4), name="job")


class TestCreation:
    def test_writes_a_project_file(self, tmp_path):
        made = Project.create(tmp_path / "p")
        assert made.file == tmp_path / "p" / PROJECT_FILENAME
        assert made.file.exists()

    def test_refuses_to_clobber_an_existing_project(self, tmp_path):
        Project.create(tmp_path / "p")
        with pytest.raises(ProjectError, match="already exists"):
            Project.create(tmp_path / "p")

    def test_force_replaces_it(self, tmp_path):
        Project.create(tmp_path / "p", name="first")
        again = Project.create(tmp_path / "p", name="second", overwrite=True)
        assert again.name == "second"

    def test_defaults_the_name_to_the_folder(self, tmp_path):
        assert Project.create(tmp_path / "hauptstrasse").name == "hauptstrasse"


class TestRoundTrip:
    def test_settings_survive_save_and_load(self, project):
        project.frames = FrameSettings(mode="every", value=10, start=1.5, end=9.0)
        project.detect = DetectSettings(backend="yolo", classes=["person"], dilate=12)
        project.output.layout = "flat"
        project.rig.cameras[0].name = "renamed"
        project.save()

        reopened = Project.load(project.root)
        assert reopened.frames.mode == "every"
        assert reopened.frames.start == 1.5
        assert reopened.detect.classes == ["person"]
        assert reopened.detect.dilate == 12
        assert reopened.output.layout == "flat"
        assert reopened.rig.cameras[0].name == "renamed"

    def test_can_be_loaded_from_the_folder_or_the_file(self, project):
        assert Project.load(project.root).name == project.name
        assert Project.load(project.file).name == project.name

    def test_occluders_survive(self, project):
        project.rig.occluders = [{"type": "nadir_cone", "angle": 35}]
        project.save()
        assert Project.load(project.root).rig.occluders[0]["angle"] == 35

    def test_missing_project_is_reported_clearly(self, tmp_path):
        with pytest.raises(ProjectError, match="project new"):
            Project.load(tmp_path)

    def test_broken_json_is_reported_clearly(self, tmp_path):
        (tmp_path / PROJECT_FILENAME).write_text("{oh dear", encoding="utf-8")
        with pytest.raises(ProjectError, match="not valid JSON"):
            Project.load(tmp_path)

    def test_future_schema_is_refused(self, project):
        data = json.loads(project.file.read_text(encoding="utf-8"))
        data["version"] = 99
        project.file.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ProjectError, match="upgrade 360extract"):
            Project.load(project.root)

    def test_an_invalid_rig_is_reported_as_such(self, project):
        data = json.loads(project.file.read_text(encoding="utf-8"))
        data["rig"]["cameras"][0]["h_fov"] = 0
        project.file.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ProjectError, match="invalid rig"):
            Project.load(project.root)

    def test_opens_a_file_saved_with_a_byte_order_mark(self, project):
        """Notepad and PowerShell's Set-Content both write UTF-8 with a BOM.

        A hand-edited project has to keep working, so the reader tolerates one.
        """
        data = project.file.read_text(encoding="utf-8")
        project.file.write_text(data, encoding="utf-8-sig")
        assert Project.load(project.root).name == project.name

    def test_saving_never_leaves_a_truncated_file(self, project):
        """Written via a temp file and renamed, so a crash cannot lose every setting."""
        project.save()
        assert not list(project.root.glob("*.tmp"))
        json.loads(project.file.read_text(encoding="utf-8"))


class TestSources:
    def test_paths_inside_the_project_are_stored_relative(self, tmp_path):
        root = tmp_path / "p"
        root.mkdir()
        clip = root / "clip.mp4"
        clip.write_bytes(b"x")
        made = Project.create(root, sources=[str(clip)])
        assert made.sources == ["clip.mp4"], "an internal path should be portable"
        assert made.resolved_sources() == [clip]

    def test_paths_outside_the_project_stay_absolute(self, tmp_path):
        outside = tmp_path / "footage" / "clip.mp4"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"x")
        made = Project.create(tmp_path / "p", sources=[str(outside)])
        assert made.resolved_sources() == [outside]

    def test_missing_sources_are_reported_not_raised(self, tmp_path):
        """A project whose drive is unplugged must still open."""
        made = Project.create(tmp_path / "p", sources=[str(tmp_path / "gone.mp4")])
        reopened = Project.load(made.root)
        assert len(reopened.missing_sources()) == 1


class TestStages:
    def test_a_fresh_project_has_nothing_done(self, project):
        assert all(project.status(stage) == "pending" for stage in STAGES)

    def test_marking_done_records_details(self, project):
        project.mark_done("extract", images=160, cameras=8)
        assert project.status("extract") == "done"
        assert project.stages["extract"].details["images"] == 160
        assert project.stages["extract"].done_at

    def test_editing_the_rig_makes_extraction_stale(self, project):
        project.mark_done("extract", images=160)
        assert project.status("extract") == "done"

        project.rig.cameras[0].yaw += 15
        assert project.status("extract") == "stale", (
            "changing where a camera points must invalidate the extracted images"
        )

    def test_changing_frame_rate_makes_extraction_stale(self, project):
        project.mark_done("extract")
        project.frames.value = 5
        assert project.is_stale("extract")

    def test_detection_settings_do_not_disturb_extraction(self, project):
        """Changing the detector must not force a re-extract; that is the point of
        keeping separate fingerprints per stage."""
        project.mark_done("extract")
        project.detect.backend = "yolo"
        project.detect.confidence = 0.5
        assert project.status("extract") == "done"

    def test_detection_settings_do_make_masking_stale(self, project):
        project.mark_done("extract")
        project.mark_done("mask", masks=160)
        project.detect.confidence = 0.5
        assert project.status("mask") == "stale"

    def test_redoing_a_stage_invalidates_later_ones(self, project):
        project.mark_done("extract")
        project.mark_done("mask", masks=160)
        assert project.status("mask") == "done"

        project.mark_done("extract", images=200)
        assert project.status("mask") == "pending", (
            "masks describe images that were just replaced, so they cannot be current"
        )

    def test_status_survives_save_and_load(self, project):
        project.mark_done("extract", images=160)
        project.save()
        assert Project.load(project.root).status("extract") == "done"

    def test_staleness_survives_save_and_load(self, project):
        project.mark_done("extract", images=160)
        project.rig.cameras[0].pitch -= 10
        project.save()
        assert Project.load(project.root).status("extract") == "stale"

    def test_unknown_stage_is_rejected(self, project):
        with pytest.raises(ProjectError, match="unknown stage"):
            project.mark_done("teleport")
        with pytest.raises(ProjectError, match="unknown stage"):
            project.fingerprint("teleport")

    def test_fingerprints_are_stable_across_identical_projects(self, tmp_path):
        one = Project.create(tmp_path / "a", rig=ring(4))
        two = Project.create(tmp_path / "b", rig=ring(4))
        assert one.fingerprint("extract") == two.fingerprint("extract")


class TestSnapshots:
    def test_saves_and_lists(self, project):
        project.snapshot("before-retiming")
        assert project.snapshots() == ["before-retiming"]

    def test_restores_earlier_settings(self, project):
        project.frames.value = 2
        project.snapshot("two-fps")

        project.frames.value = 12
        project.rig.cameras.append(Camera(name="extra", yaw=123))
        restored = project.restore("two-fps")

        assert restored.frames.value == 2
        assert "extra" not in [c.name for c in restored.rig.cameras]

    def test_restoring_an_unknown_snapshot_lists_what_exists(self, project):
        project.snapshot("good")
        with pytest.raises(ProjectError, match="Available: good"):
            project.restore("nope")

    def test_names_are_made_safe_for_the_filesystem(self, project):
        path = project.snapshot("before / after: v2")
        assert path.exists()
        assert "/" not in path.name and ":" not in path.name

    def test_an_empty_name_is_rejected(self, project):
        with pytest.raises(ProjectError, match="needs a name"):
            project.snapshot("///")


class TestFind:
    def test_finds_a_project_in_a_parent_folder(self, project):
        nested = project.root / "images" / "clip" / "c00"
        nested.mkdir(parents=True)
        assert find(nested) == project.file

    def test_returns_none_when_there_is_none(self, tmp_path):
        assert find(tmp_path) is None


class TestLaterStages:
    """Training and cleanup are stages too. `mark_done("train")` used to raise -- at the
    end of a successful run, after the splat had already been written."""

    def test_training_can_be_marked_done(self, tmp_path):
        project = Project.create(tmp_path / "p")
        project.mark_done("train", steps=30000, splat="splat_30000.ply")
        assert project.status("train") == "done"
        assert project.stages["train"].details["steps"] == 30000

    def test_cleanup_can_be_marked_done(self, tmp_path):
        project = Project.create(tmp_path / "p")
        project.mark_done("clean", removed=1234)
        assert project.status("clean") == "done"

    def test_a_new_reconstruction_makes_the_splat_stale(self, tmp_path):
        """The splat describes the model under it; redo the model and it no longer does."""
        project = Project.create(tmp_path / "p")
        project.mark_done("export")
        project.mark_done("train", steps=1000)
        assert project.status("train") == "done"

        project.rig.cameras[0].yaw += 10       # changes extract -> mask -> export
        project.mark_done("extract")
        project.mark_done("mask")
        project.mark_done("export")
        assert project.status("train") == "pending"   # redoing export cleared it

    def test_stages_round_trip_through_the_file(self, tmp_path):
        project = Project.create(tmp_path / "p")
        project.mark_done("train", steps=30000)
        project.save()
        assert Project.load(tmp_path / "p").status("train") == "done"


class TestSourcePaths:
    def test_a_relative_source_is_stored_absolute(self, tmp_path, monkeypatch):
        """`--source drive.mp4` run beside the video used to be read back as
        <project>/drive.mp4 and reported missing."""
        video = tmp_path / "drive.mp4"
        video.write_bytes(b"x")
        monkeypatch.chdir(tmp_path)

        project = Project.create(tmp_path / "dataset", sources=["drive.mp4"])
        assert project.resolved_sources() == [video]
        assert project.missing_sources() == []

    def test_a_source_inside_the_project_stays_relative(self, tmp_path):
        """A dataset that carries its own footage should still be movable."""
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "clip.mp4").write_bytes(b"x")

        project = Project.create(root, sources=[str(root / "clip.mp4")])
        assert project.sources == ["clip.mp4"]
        assert project.resolved_sources() == [root / "clip.mp4"]


class TestToolPathStore:
    """Where the external tools are is a property of the machine, so it is stored per
    user rather than per project -- and the environment still wins over it."""

    def test_round_trips(self):
        from threesixty import toolpaths

        toolpaths.save("colmap", "/opt/colmap/bin/colmap")
        assert toolpaths.get("colmap") == "/opt/colmap/bin/colmap"
        assert toolpaths.stored() == {"colmap": "/opt/colmap/bin/colmap"}

    def test_an_empty_value_clears_it(self):
        from threesixty import toolpaths

        toolpaths.save("brush", "/usr/bin/brush")
        toolpaths.save("brush", "")
        assert toolpaths.get("brush") == ""

    def test_an_unknown_tool_is_refused(self):
        from threesixty import toolpaths

        with pytest.raises(ValueError, match="unknown tool"):
            toolpaths.save("nonsense", "/x")

    def test_a_configured_path_is_actually_used(self, tmp_path):
        """The real chain, not a re-implementation of it."""
        from threesixty import tools, toolpaths

        binary = tmp_path / "brush.exe"
        binary.write_bytes(b"")
        toolpaths.save("brush", str(binary))

        found = tools.find_brush()
        assert found.path == binary
        assert found.source == "configured"

    def test_the_environment_wins_over_the_stored_path(self, tmp_path, monkeypatch):
        """Exporting a variable for one run is a deliberate override of the settled
        answer, so it has to come first."""
        from threesixty import tools, toolpaths

        stored = tmp_path / "stored.exe"
        stored.write_bytes(b"")
        chosen = tmp_path / "env.exe"
        chosen.write_bytes(b"")
        toolpaths.save("brush", str(stored))
        monkeypatch.setenv("THREESIXTY_BRUSH", str(chosen))

        found = tools.find_brush()
        assert found.path == chosen
        assert found.source == "THREESIXTY_BRUSH"

    def test_an_explicit_argument_wins_over_both(self, tmp_path, monkeypatch):
        from threesixty import tools, toolpaths

        for name in ("stored", "env", "explicit"):
            (tmp_path / f"{name}.exe").write_bytes(b"")
        toolpaths.save("brush", str(tmp_path / "stored.exe"))
        monkeypatch.setenv("THREESIXTY_BRUSH", str(tmp_path / "env.exe"))

        found = tools.find_brush(tmp_path / "explicit.exe")
        assert found.path == tmp_path / "explicit.exe"

    def test_common_locations_are_not_one_machine_s_layout(self):
        """A personal convention belongs at the end of the search, not the front."""
        from threesixty import toolpaths

        def personal(text):
            return text.startswith("C:") and "Tools" in text

        for locations in (toolpaths.colmap_locations(), toolpaths.brush_locations(),
                          toolpaths.supersplat_locations()):
            texts = [str(p) for p in locations]
            mine = [i for i, text in enumerate(texts) if personal(text)]
            rest = [i for i, text in enumerate(texts) if not personal(text)]
            assert mine and rest
            assert min(mine) > max(rest), texts
