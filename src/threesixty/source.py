"""What projection the footage arrives in, and how to get it to equirectangular.

Every stage after ingest -- the rig, the occluder masks, detection, the canvas -- is
written against an equirectangular panorama, where a pixel's position *is* its direction.
That assumption is worth keeping: it is what makes a mask paintable in one image and a
camera a pair of angles.

So the projection of the *source* is handled in exactly one place, at the moment the
pixels are first read. A dual-fisheye camera (two circular lenses, the raw output of a
360 rig before its desktop app stitches anything) is turned into equirectangular by the
same ``v360`` filter that later cuts the camera tiles, and everything downstream carries
on unchanged.

Three things vary between cameras and all three have to be stated, because none of them
can be read off the pixels:

* **Where the lenses are.** Side by side in one frame (`sbs`), or -- the QooCam 8K, and
  the reason this module has a layout at all -- one video *stream* per lens (`streams`),
  where the file's dimensions say nothing about there being a second lens at all.
* **Which way up each lens is.** Sensors are mounted however the body needed them. The
  QooCam's two are rotated in opposite directions, so its second lens arrives upside
  down relative to the first and the panorama comes out with its back half inverted.
* **How much each lens sees.** `lens_fov` past 180 is the overlap a stitcher blends;
  v360 hands over at exactly 90 degrees from each axis, so the overlap is what pays for
  `trim` -- see `seam_band`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Input projections, mapped to the name ``v360`` knows them by.
PROJECTIONS = {
    "equirect": "e",
    "dfisheye": "dfisheye",
    "fisheye": "fisheye",
}

#: How the lenses are packed into the file.
#:
#: `single` is one image that is already the whole source (equirect, or one fisheye).
#: `sbs` is two circles side by side in one frame. `streams` is one video stream per
#: lens, which has to be stacked back together before v360 can read it as a pair.
LAYOUTS = ("single", "sbs", "streams")

#: What each projection needs to describe itself, and what a sensible default is.
#: `equirect` covers the sphere by definition, so its lens fov is meaningless.
DEFAULT_LENS_FOV = 190.0

#: ffmpeg refuses dimensions past this, and no real source is anywhere near it.
MAX_DIMENSION = 16384


class SourceError(ValueError):
    """The source projection is not one this build understands."""


def _rotation_filter(degrees: float) -> str:
    """The ffmpeg chain that rotates a lens upright, clockwise, in whole quarter turns."""
    quarters = int(round((degrees % 360) / 90)) % 4
    return {0: "", 1: "transpose=1", 2: "transpose=1,transpose=1", 3: "transpose=2"}[quarters]


@dataclass(frozen=True)
class SourceFormat:
    """How to read the source's pixels as directions.

    The default is equirectangular, which is what a stitched 360 file is, and it costs
    nothing: `is_equirect` short-circuits every conversion below.
    """

    projection: str = "equirect"
    #: Field of view of a single lens, in degrees. Ignored for `equirect`.
    lens_fov: float = DEFAULT_LENS_FOV
    #: Where the lenses are: one frame each (`streams`), side by side (`sbs`), or a
    #: single image that is the whole source (`single`).
    layout: str = "single"
    #: Clockwise rotation to apply to each lens before projecting, in degrees. One entry
    #: per lens; a short list is padded with zeroes.
    rotate: tuple[float, ...] = ()
    #: Degrees trimmed off each lens's usable edge. v360 hands one lens over to the other
    #: at 90 degrees from its axis, and that is exactly where a fisheye is at its worst:
    #: softest, most vignetted, and where the stitch's parallax error lives. Trimming
    #: ignores a band of this width either side of the seam. See `seam_band`.
    trim: float = 0.0

    def __post_init__(self) -> None:
        # A frozen dataclass, so the normalisation goes through object.__setattr__.
        if isinstance(self.rotate, list):
            object.__setattr__(self, "rotate", tuple(float(r) for r in self.rotate))
        if self.projection != "dfisheye" and self.layout in ("sbs", "streams"):
            object.__setattr__(self, "layout", "single")
        if self.projection == "dfisheye" and self.layout == "single":
            object.__setattr__(self, "layout", "sbs")

    def validate(self) -> None:
        if self.projection not in PROJECTIONS:
            raise SourceError(
                f"source projection must be one of {sorted(PROJECTIONS)}, "
                f"got {self.projection!r}")
        if self.layout not in LAYOUTS:
            raise SourceError(f"lens layout must be one of {list(LAYOUTS)}, "
                              f"got {self.layout!r}")
        if not self.is_equirect and not 1.0 <= float(self.lens_fov) <= 360.0:
            raise SourceError(
                f"lens field of view must be in [1, 360] degrees, got {self.lens_fov}")
        if not 0.0 <= float(self.trim) < 90.0:
            raise SourceError(f"lens trim must be in [0, 90) degrees, got {self.trim}")

    @property
    def is_equirect(self) -> bool:
        return self.projection == "equirect"

    @property
    def lens_count(self) -> int:
        return 2 if self.projection == "dfisheye" else 1

    @property
    def needs_merge(self) -> bool:
        """True when the lenses arrive as separate streams and have to be stacked."""
        return self.layout == "streams"

    @property
    def needs_graph(self) -> bool:
        """True when reading this source takes more than a plain ``-vf`` chain."""
        return self.needs_merge or any(self.rotate)

    @property
    def label(self) -> str:
        if self.is_equirect:
            return "equirectangular"
        lens = "dual fisheye" if self.projection == "dfisheye" else "fisheye"
        where = " (one stream per lens)" if self.needs_merge else ""
        return f"{lens}, {self.lens_fov:g}° lenses{where}"

    def rotation(self, lens: int) -> float:
        return float(self.rotate[lens]) if lens < len(self.rotate) else 0.0

    # -- geometry -----------------------------------------------------------

    def frame_width(self, width: int, streams: int = 1) -> int:
        """The width of the picture v360 will read.

        With a lens per stream the file's own width is one lens; the pair is twice that
        once stacked. Getting this wrong halves every size derived from it.
        """
        return int(width) * 2 if self.needs_merge and streams > 1 else int(width)

    def degrees_per_pixel(self, width: int, streams: int = 1) -> float:
        """How many degrees one horizontal pixel of the source covers.

        This is the number that decides output sizes. An equirect frame spreads 360
        degrees over its width; a dual fisheye spreads `lens_fov` over *half* of it,
        because the two circles sit side by side.
        """
        merged = max(self.frame_width(width, streams), 1)
        if self.is_equirect:
            return 360.0 / merged
        span = float(self.lens_fov)
        if self.projection == "dfisheye":
            return span / max(merged / 2.0, 1.0)
        return span / merged

    def equirect_size(self, width: int, height: int = 0,
                      streams: int = 1) -> tuple[int, int]:
        """The equirect size that keeps the source's own pixel density.

        Bigger than this invents detail; smaller throws away detail the capture paid
        for. The width comes out a multiple of four, so that half of it -- the height --
        is still even; encoders reject odd dimensions.
        """
        if self.is_equirect:
            return int(width), int(height or (int(width) // 2))
        per_degree = 1.0 / self.degrees_per_pixel(width, streams)
        equirect_width = int(round(360.0 * per_degree / 4)) * 4
        equirect_width = max(min(equirect_width, MAX_DIMENSION), 4)
        return equirect_width, equirect_width // 2

    def equirect_size_for(self, media) -> tuple[int, int]:
        """`equirect_size` for a probed source, streams and all."""
        return self.equirect_size(media.width, media.height,
                                  getattr(media, "video_streams", 1))

    def source_width_as_equirect(self, width: int, height: int = 0,
                                 streams: int = 1) -> int:
        """The width an equirect frame of this pixel density would have.

        Camera tiles are sized from it (`rig.native_size`), so a dual-fisheye source
        yields the same tile sizes as the stitched equirect of the same footage would.
        """
        return self.equirect_size(width, height, streams)[0]

    @property
    def seam_band(self) -> float:
        """Half-width, in degrees, of the ignored band around the stitch line.

        Zero unless the lenses are trimmed. v360 switches lenses at 90 degrees from each
        axis, so trimming `trim` degrees off each lens leaves exactly this much of the
        sphere covered by neither -- a band around the great circle where the two images
        meet, which is where the stitch is worst.
        """
        return float(self.trim) if self.projection == "dfisheye" else 0.0

    # -- filters ------------------------------------------------------------

    def input_spec(self) -> tuple[str, list[str]]:
        """The ``v360`` input name and the options describing it.

        Used to project straight from the source to a camera, skipping the intermediate
        equirect entirely -- one resample instead of two.
        """
        name = PROJECTIONS[self.projection]
        if self.is_equirect:
            return name, []
        return name, [f"ih_fov={self.lens_fov:g}", f"iv_fov={self.lens_fov:g}"]

    def to_equirect(self, width: int, height: int = 0, interp: str = "line",
                    size: tuple[int, int] | None = None) -> str:
        """A filter chain that normalizes a single-image source to equirectangular.

        Empty for an equirect source, so callers can splice the result into a filter
        chain without a special case. Lenses arriving as separate streams cannot be read
        by a chain at all -- use `ingest_chains`.
        """
        if self.is_equirect:
            return ""
        self.validate()
        out_width, out_height = size or self.equirect_size(width, height)
        name, options = self.input_spec()
        params = ":".join([name, "e", *options,
                           f"w={out_width}", f"h={out_height}", f"interp={interp}"])
        return f"v360={params}"

    def lens_chains(self, label: str = "lenses", scale: int = 0,
                    thin: str = "") -> list[str]:
        """Filter chains that put every lens into one frame, side by side, upright.

        This is the picture a person should be shown for a raw source: two circles, each
        the right way up, nothing warped. It is also the input v360 reads as `dfisheye`.
        Returns an empty list when the source is already one frame and needs no rotation.

        `thin` is a frame-selection filter applied to each lens *before* they are stacked:
        the streams carry the same timestamps, so the same expression keeps them in step,
        and the frames that are about to be thrown away never cost a stack or a resample.
        """
        if not self.needs_graph:
            return []
        chains: list[str] = []
        parts = []
        size = f"scale={scale}:{scale}" if scale else ""
        if self.needs_merge:
            for lens in range(self.lens_count):
                rotate = _rotation_filter(self.rotation(lens))
                steps = ",".join(filter(None, [thin, rotate, size])) or "null"
                chains.append(f"[0:v:{lens}]{steps}[lens{lens}]")
                parts.append(f"[lens{lens}]")
            # shortest=1: the two tracks are the same length, but a file whose tracks
            # disagree by a frame should end rather than hang.
            chains.append(f"{''.join(parts)}hstack=inputs={len(parts)}:shortest=1[{label}]")
            return chains
        # One frame holding both lenses: rotate the halves in place.
        rotate = _rotation_filter(self.rotation(0))
        if self.layout == "sbs" and self.rotation(1) != self.rotation(0):
            chains.append(f"[0:v]{thin + ',' if thin else ''}split=2[whole0][whole1]")
            for lens in range(2):
                crop = f"crop=iw/2:ih:{'0' if lens == 0 else 'iw/2'}:0"
                steps = ",".join(filter(None, [crop, _rotation_filter(self.rotation(lens))]))
                chains.append(f"[whole{lens}]{steps}[lens{lens}]")
            chains.append(f"[lens0][lens1]hstack=inputs=2:shortest=1[{label}]")
            return chains
        chains.append(f"[0:v]{','.join(filter(None, [thin, rotate])) or 'null'}[{label}]")
        return chains

    def ingest_chains(self, size: tuple[int, int], interp: str = "line",
                      label: str = "src", thin: str = "") -> list[str]:
        """Filter chains that turn the raw source into one equirect stream `[label]`.

        Everything the pipeline does downstream reads that label. Empty for an equirect
        source that needs no rearranging, in which case the caller keeps using `[0:v]`.
        """
        if self.is_equirect and not self.needs_graph and not thin:
            return []
        self.validate()
        chains = self.lens_chains(label="lenses", thin=thin)
        head = "[lenses]" if chains else f"[0:v]{thin + ',' if thin else ''}"
        convert = self.to_equirect(0, 0, interp=interp, size=size)
        if not convert:
            # Already equirect, only rotated: hand the rearranged stream straight on.
            return [chain.replace("[lenses]", f"[{label}]") for chain in chains]
        chains.append(f"{head}{convert}[{label}]")
        return chains

    # -- detection ----------------------------------------------------------

    @classmethod
    def detect(cls, media) -> "SourceFormat | None":
        """A guess at what a just-opened file is, or None to leave the choice alone.

        Only claims what the container makes plain: two video streams is a lens each,
        and nothing else writes that. A square single stream is a fisheye of some kind,
        which is worth proposing but not worth being sure about. A 2:1 file is left as
        equirect even though a raw side-by-side file looks identical -- guessing wrong
        there would silently reproject footage that was fine.
        """
        streams = getattr(media, "video_streams", 1)
        if streams >= 2 and getattr(media, "looks_circular", False):
            # Both known cameras of this shape mount their sensors opposed.
            return cls("dfisheye", DEFAULT_LENS_FOV, layout="streams", rotate=(90.0, -90.0))
        if streams == 1 and getattr(media, "looks_circular", False):
            return cls("fisheye", DEFAULT_LENS_FOV)
        return None

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {"projection": self.projection, "lens_fov": float(self.lens_fov),
                "layout": self.layout, "rotate": list(self.rotate),
                "trim": float(self.trim)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "SourceFormat":
        if not data:
            return cls()
        return cls(projection=str(data.get("projection", "equirect")),
                   lens_fov=float(data.get("lens_fov", DEFAULT_LENS_FOV)),
                   layout=str(data.get("layout", "single")),
                   rotate=tuple(float(r) for r in (data.get("rotate") or ())),
                   trim=float(data.get("trim", 0.0)))


# -- fitting the lens field of view -----------------------------------------
#
# `lens_fov` is the one number a camera's spec sheet often rounds ("about 200 degrees")
# and getting it wrong is visible: the two lenses meet at a great circle, and a wrong
# figure makes the picture jump across it. That jump is measurable, so it can be fitted
# rather than guessed -- render the panorama at several candidates and keep the one whose
# seam agrees with itself best.

#: Where the sweep looks. Consumer 360 lenses live in this range; below 180 the sphere is
#: not covered at all and above 220 nothing is made.
FIT_RANGE = (180.0, 220.0)


def _seam_mismatch(image) -> float:
    """Mean absolute difference across both stitch lines of an equirect frame.

    The seams sit a quarter and three quarters of the way across, where one lens hands
    over to the other. Sky and the vehicle directly below agree whatever the field of
    view is, so the top and bottom sixths are left out of the measurement.
    """
    import numpy as np

    frame = np.asarray(image, dtype=float)
    height, width = frame.shape[:2]
    band = slice(height // 6, 5 * height // 6)
    scores = []
    for seam in (width // 4, 3 * width // 4):
        scores.append(float(np.abs(frame[band, seam - 2] - frame[band, seam + 2]).mean()))
    return float(sum(scores) / len(scores))


#: Rotation pairs worth trying when fitting. Only the *relative* rotation is
#: measurable from the seam -- two lenses turned the same way still meet correctly, they
#: just tilt the whole panorama -- so this is a short list of the ways a body can mount
#: two sensors, not an exhaustive sweep.
ROTATION_CANDIDATES = ((0.0, 0.0), (90.0, -90.0), (-90.0, 90.0), (90.0, 90.0),
                       (180.0, 180.0), (0.0, 180.0))


def _lens_pair(ffmpeg, path, fmt: "SourceFormat", seek: float, width: int, target):
    """Decode both lenses side by side, *unrotated*, so candidates are cheap.

    Every candidate rotation and field of view is then projected from this image rather
    than from the video, which is the difference between one decode and a hundred.
    """
    import subprocess

    argv = [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y"]
    if seek:
        argv += ["-ss", f"{seek:g}"]
    argv += ["-i", str(path)]
    if fmt.needs_merge:
        argv += ["-filter_complex",
                 f"[0:v:0]scale={width}:{width}[a];[0:v:1]scale={width}:{width}[b];"
                 f"[a][b]hstack=inputs=2:shortest=1[out]", "-map", "[out]"]
    else:
        argv += ["-vf", f"scale={width * 2}:{width}"]
    argv += ["-frames:v", "1", str(target)]
    result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    return target if result.returncode == 0 and target.exists() else None


def _project_candidate(ffmpeg, lenses, rotate, lens_fov: float, width: int, target):
    """Project one (rotation, field of view) candidate from a decoded lens pair."""
    import subprocess

    left, right = _rotation_filter(rotate[0]), _rotation_filter(rotate[1])
    chains = [
        "[0:v]split=2[x][y]",
        ",".join(filter(None, ["[x]crop=iw/2:ih:0:0", left])) + "[l]",
        ",".join(filter(None, ["[y]crop=iw/2:ih:iw/2:0", right])) + "[r]",
        f"[l][r]hstack=inputs=2[pair]",
        f"[pair]v360=dfisheye:e:ih_fov={lens_fov:g}:iv_fov={lens_fov:g}"
        f":w={width}:h={width // 2}[out]",
    ]
    result = subprocess.run(
        [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(lenses), "-filter_complex", ";".join(chains), "-map", "[out]",
         "-frames:v", "1", str(target)],
        capture_output=True, text=True, errors="replace")
    return target if result.returncode == 0 and target.exists() else None


def _sky_above_ground(image) -> float:
    """How much brighter the top of a panorama is than the bottom.

    The tie-breaker the seam cannot provide. Turning *both* lenses by 180 degrees leaves
    the stitch exactly as good and the world upside down, so the two candidates are
    indistinguishable by mismatch alone. Outdoors -- which is what a 360 rig on a vehicle
    is doing -- the sky is the brighter half, and that is enough to tell them apart.

    Wrong at night, or under a ceiling brighter than the floor. It only ever chooses
    between two candidates the measurement already called equally good, and the rotation
    control still overrides it.
    """
    import numpy as np

    frame = np.asarray(image, dtype=float)
    height = frame.shape[0]
    return float(frame[:height // 6].mean() - frame[5 * height // 6:].mean())


def fit_lens_fov(ffmpeg, path, fmt: "SourceFormat", seeks=(0.0,),
                 workdir=None, width: int = 1024, coarse_step: float = 5.0,
                 fit_rotation: bool = True) -> dict:
    """Measure how the two lenses have to be read for them to line up.

    Returns `{"lens_fov": deg, "rotate": [deg, deg], "mismatch": score, "scores": [...]}`.

    Two things are fitted, in that order: how each lens is mounted, then how much it
    sees. Both are measured the same way -- project the panorama and look at how much
    the picture jumps where one lens hands over to the other. Only the *relative*
    rotation is visible to that measurement: two sensors turned the same way still meet
    correctly, they just tilt the whole panorama, which is the rig's orientation to fix
    rather than the source's.

    Every candidate is projected from an already-decoded pair of lenses, so the cost is
    one video decode per `seeks` entry rather than one per candidate.
    """
    import tempfile
    from pathlib import Path as _Path

    if fmt.projection != "dfisheye":
        raise SourceError("fitting the lens field of view needs a dual-fisheye source")

    workdir = _Path(workdir or tempfile.mkdtemp(prefix="threesixty-fit"))
    workdir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for index, seek in enumerate(seeks or (0.0,)):
        decoded = _lens_pair(ffmpeg, path, fmt, seek, width, workdir / f"pair{index}.png")
        if decoded is not None:
            pairs.append(decoded)
    if not pairs:
        from .ffmpeg import FFmpegError
        raise FFmpegError(f"could not read the lenses of {path}")

    from PIL import Image as PILImage

    def measure(rotate, lens_fov: float, samples) -> tuple[float, float]:
        """(seam mismatch, how much brighter the top is) for one candidate."""
        mismatch, upright = [], []
        for index, lenses in enumerate(samples):
            name = f"fit_{int(rotate[0])}_{int(rotate[1])}_{lens_fov:g}_{index}.png"
            out = _project_candidate(ffmpeg, lenses, rotate, lens_fov, width,
                                     workdir / name)
            if out is None:
                return float("inf"), 0.0
            with PILImage.open(out) as image:
                grey = image.convert("L")
                mismatch.append(_seam_mismatch(grey))
                upright.append(_sky_above_ground(grey))
        return sum(mismatch) / len(mismatch), sum(upright) / len(upright)

    def score(rotate, lens_fov: float, samples) -> float:
        return measure(rotate, lens_fov, samples)[0]

    low, high = FIT_RANGE
    rotate = tuple(fmt.rotate[:2]) if len(fmt.rotate) >= 2 else (0.0, 0.0)
    rotations: list[list] = []
    if fit_rotation:
        # One sample and a coarse sweep is plenty to tell a mounting apart: the wrong
        # one puts half the panorama upside down, which no field of view can rescue.
        coarse = [low + i * (high - low) / 4 for i in range(5)]
        upright_by_candidate = {}
        for candidate in ROTATION_CANDIDATES:
            results = [measure(candidate, fov, pairs[:1]) for fov in coarse]
            best_here = min(results, key=lambda entry: entry[0])
            rotations.append([list(candidate), best_here[0]])
            upright_by_candidate[candidate] = best_here[1]
        floor = min(entry[1] for entry in rotations)
        # Every candidate the seam cannot separate -- notably a mounting and the same
        # mounting turned over, which stitch identically -- is settled by which one puts
        # the sky at the top.
        tied = [tuple(entry[0]) for entry in rotations if entry[1] <= floor * 1.15]
        rotate = max(tied, key=lambda candidate: upright_by_candidate[candidate])

    scores: dict[float, float] = {}
    candidate = low
    while candidate <= high + 1e-6:
        scores[candidate] = score(rotate, candidate, pairs)
        candidate += coarse_step
    best = min(scores, key=scores.get)
    for fine in range(int(best - coarse_step + 1), int(best + coarse_step)):
        value = float(fine)
        if low <= value <= high and value not in scores:
            scores[value] = score(rotate, value, pairs)
    best = min(scores, key=scores.get)
    return {"lens_fov": float(best), "rotate": [float(r) for r in rotate],
            "mismatch": scores[best], "measured": True,
            "rotations": rotations,
            "scores": sorted([float(k), float(v)] for k, v in scores.items())}
