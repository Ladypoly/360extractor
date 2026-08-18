"""The source projection: what the footage arrives as, and the maths that follows.

These are the numbers every later stage inherits -- tile sizes, the working set's
resolution -- so they are pinned here rather than left to be discovered downstream.
"""

import pytest

from threesixty.source import (
    DEFAULT_LENS_FOV,
    PROJECTIONS,
    SourceError,
    SourceFormat,
)


class TestDefaults:
    def test_equirect_is_the_default_and_costs_nothing(self):
        fmt = SourceFormat()
        assert fmt.is_equirect
        assert fmt.to_equirect(4096, 2048) == ""
        assert fmt.input_spec() == ("e", [])
        assert fmt.equirect_size(4096, 2048) == (4096, 2048)

    def test_equirect_height_is_derived_when_it_is_not_given(self):
        assert SourceFormat().equirect_size(4096) == (4096, 2048)

    def test_every_projection_names_a_v360_input(self):
        for name in PROJECTIONS:
            SourceFormat(projection=name).validate()


class TestValidation:
    def test_an_unknown_projection_is_refused(self):
        with pytest.raises(SourceError, match="projection"):
            SourceFormat(projection="cubemap").validate()

    def test_an_impossible_lens_is_refused(self):
        with pytest.raises(SourceError, match="field of view"):
            SourceFormat("dfisheye", lens_fov=400).validate()

    def test_the_lens_is_irrelevant_to_an_equirect_source(self):
        # Nothing reads it, so a nonsense value must not block a stitched file.
        SourceFormat("equirect", lens_fov=0).validate()


class TestPixelDensity:
    def test_two_lenses_share_the_width(self):
        # 5760 across two 190-degree circles is 2880 per lens: 15.16 px per degree,
        # which is 5456 across a full turn.
        fmt = SourceFormat("dfisheye", 190)
        assert fmt.equirect_size(5760, 2880) == (5456, 2728)

    def test_one_lens_spends_the_whole_width(self):
        assert SourceFormat("fisheye", 180).equirect_size(2000, 2000) == (4000, 2000)

    def test_the_equirect_width_is_what_sizes_the_tiles(self):
        fmt = SourceFormat("dfisheye", 190)
        assert fmt.source_width_as_equirect(5760, 2880) == 5456

    def test_both_dimensions_come_out_even(self):
        # Encoders reject odd dimensions, and 2:1 means the width decides the height
        # as well -- so the width has to be a multiple of four, not merely even.
        for width in (1001, 2003, 3999):
            equirect_width, equirect_height = SourceFormat("dfisheye").equirect_size(width)
            assert equirect_width % 4 == 0
            assert equirect_height % 2 == 0
            assert equirect_height * 2 == equirect_width

    def test_a_wider_lens_packs_more_sphere_into_the_same_pixels(self):
        narrow = SourceFormat("dfisheye", 180).equirect_size(4096)
        wide = SourceFormat("dfisheye", 220).equirect_size(4096)
        assert wide[0] < narrow[0]


class TestFilters:
    def test_the_conversion_names_both_lens_fields(self):
        chain = SourceFormat("dfisheye", 190).to_equirect(5760, 2880)
        assert chain.startswith("v360=dfisheye:e:")
        assert "ih_fov=190" in chain and "iv_fov=190" in chain
        assert "w=5456:h=2728" in chain

    def test_an_explicit_size_wins(self):
        chain = SourceFormat("dfisheye").to_equirect(5760, 2880, size=(1280, 640))
        assert "w=1280:h=640" in chain

    def test_the_input_spec_projects_straight_to_a_camera(self):
        name, options = SourceFormat("dfisheye", 200).input_spec()
        assert name == "dfisheye"
        assert options == ["ih_fov=200", "iv_fov=200"]

    def test_the_interpolation_is_carried_through(self):
        assert "interp=lanczos" in SourceFormat("fisheye").to_equirect(2000, 2000,
                                                                      interp="lanczos")


class TestSerialization:
    def test_round_trip(self):
        fmt = SourceFormat("dfisheye", 187.5)
        assert SourceFormat.from_dict(fmt.to_dict()) == fmt

    def test_missing_data_reads_as_equirect(self):
        assert SourceFormat.from_dict(None) == SourceFormat()
        assert SourceFormat.from_dict({}) == SourceFormat()

    def test_a_partial_record_keeps_the_default_lens(self):
        assert SourceFormat.from_dict({"projection": "fisheye"}).lens_fov == DEFAULT_LENS_FOV

    def test_the_label_says_what_it_is(self):
        assert SourceFormat().label == "equirectangular"
        assert SourceFormat("dfisheye", 190).label == "dual fisheye, 190° lenses"


class TestLensLayout:
    """Where the lenses are, and which way up -- the two things a file cannot tell you."""

    def test_a_dual_fisheye_source_defaults_to_side_by_side(self):
        assert SourceFormat("dfisheye").layout == "sbs"

    def test_layout_is_meaningless_without_two_lenses(self):
        assert SourceFormat("equirect", layout="streams").layout == "single"
        assert SourceFormat("fisheye", layout="sbs").layout == "single"

    def test_a_lens_per_stream_doubles_the_effective_width(self):
        fmt = SourceFormat("dfisheye", 190, layout="streams")
        # 3840 per lens is a 7680-wide pair, whatever the file's own dimensions say.
        assert fmt.frame_width(3840, streams=2) == 7680
        assert fmt.equirect_size(3840, 3840, streams=2)[0] > fmt.equirect_size(3840)[0]

    def test_one_stream_is_read_as_one_frame_even_in_streams_layout(self):
        # A file that lost its second track must not be scaled up as if it had one.
        fmt = SourceFormat("dfisheye", 190, layout="streams")
        assert fmt.frame_width(3840, streams=1) == 3840

    def test_separate_streams_are_stacked_and_turned_upright(self):
        fmt = SourceFormat("dfisheye", 195, layout="streams", rotate=(90, -90))
        chains = fmt.ingest_chains((2048, 1024))
        assert chains[0] == "[0:v:0]transpose=1[lens0]"
        assert chains[1] == "[0:v:1]transpose=2[lens1]"
        assert "hstack=inputs=2" in chains[2]
        assert chains[-1].startswith("[lenses]v360=dfisheye:e:ih_fov=195")

    def test_thinning_happens_per_lens_before_the_stack(self):
        # Frames that are about to be dropped must not cost a stack or a resample.
        fmt = SourceFormat("dfisheye", 190, layout="streams", rotate=(90, -90))
        chains = fmt.ingest_chains((1024, 512), thin="fps=2")
        assert chains[0] == "[0:v:0]fps=2,transpose=1[lens0]"
        assert chains[1] == "[0:v:1]fps=2,transpose=2[lens1]"
        assert "fps=2" not in chains[-1]

    def test_a_plain_source_needs_no_graph_at_all(self):
        assert not SourceFormat().needs_graph
        assert not SourceFormat("dfisheye", 190).needs_graph
        assert SourceFormat("dfisheye", 190, layout="streams").needs_graph
        assert SourceFormat("dfisheye", 190, rotate=(90, 90)).needs_graph

    def test_rotation_is_stated_per_lens_and_padded(self):
        fmt = SourceFormat("dfisheye", 190, rotate=(90,))
        assert fmt.rotation(0) == 90
        assert fmt.rotation(1) == 0

    def test_half_turns_and_negatives_become_transposes(self):
        from threesixty.source import _rotation_filter
        assert _rotation_filter(0) == ""
        assert _rotation_filter(90) == "transpose=1"
        assert _rotation_filter(-90) == "transpose=2"
        assert _rotation_filter(270) == "transpose=2"
        assert _rotation_filter(180) == "transpose=1,transpose=1"


class TestLensTrim:
    """The circular trim: cutting the soft, badly stitched rim off each lens."""

    def test_off_by_default(self):
        assert SourceFormat("dfisheye").seam_band == 0.0

    def test_the_band_is_stated_in_degrees(self):
        assert SourceFormat("dfisheye", trim=8).seam_band == 8.0

    def test_an_equirect_source_has_no_seam_to_trim(self):
        assert SourceFormat("equirect", trim=8).seam_band == 0.0

    def test_a_nonsense_trim_is_refused(self):
        with pytest.raises(SourceError, match="trim"):
            SourceFormat("dfisheye", trim=120).validate()


class TestDetection:
    """What can honestly be read off a file, and what must be left to the user."""

    class Fake:
        def __init__(self, width, height, streams):
            self.width, self.height, self.video_streams = width, height, streams

        @property
        def aspect(self):
            return self.width / self.height

        @property
        def looks_circular(self):
            return 0.95 <= self.aspect <= 1.05

    def test_two_square_streams_is_a_lens_each(self):
        found = SourceFormat.detect(self.Fake(3840, 3840, 2))
        assert found == SourceFormat("dfisheye", 190, layout="streams",
                                     rotate=(90.0, -90.0))

    def test_one_square_stream_is_a_single_fisheye(self):
        assert SourceFormat.detect(self.Fake(2000, 2000, 1)).projection == "fisheye"

    def test_a_2_to_1_file_is_left_alone(self):
        # A raw side-by-side file looks exactly like a stitched panorama, so guessing
        # here would silently reproject footage that was fine.
        assert SourceFormat.detect(self.Fake(5760, 2880, 1)) is None
