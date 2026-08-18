"""A local web UI for designing rigs and running extractions.

Deliberately built on the standard library: the whole point of the rig editor is
seeing what each camera covers, and that should not require installing a web
framework. Binds to localhost only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import subprocess
import tempfile
import threading
import traceback
import webbrowser
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import picker, stages
from .jobs import AlreadyRunning, JobRegistry
from ..mask import geometric
from ..tools import survey as tool_survey
from ..extract import run_extraction
from ..ffmpeg import FFmpegError, MediaInfo, probe_media, resolve_ffmpeg
from ..plan import (
    FrameSelection,
    camera_size,
    frames_per_second as plan_frames_per_second,
    plan_extraction,
    safe_stem,
)
from ..project import (
    PROJECT_FILENAME,
    STAGES,
    DetectSettings,
    FrameSettings,
    OutputSettings,
    Project,
    ProjectError,
)
from .. import (
    cameras,
    dataset,
    frames,
    motion,
    recent,
    segment,
    toolpaths,
    userpresets,
)
from ..rig import PRESETS, Camera, Grade, Orientation, Output, Rig, RigError
from ..source import PROJECTIONS, SourceFormat

#: Occluder kinds a global rig preset may carry. `nadir_cone`/`zenith_cone` are angles
#: and travel between projects; `equirect_mask` (a painted file) and `ml` are tied to one
#: project's assets, so they are stripped before a preset is saved.
PORTABLE_OCCLUDERS = {"nadir_cone", "zenith_cone"}

STATIC = Path(__file__).parent / "static"
PREVIEW_WIDTH = 1600

#: Extensions the UI and the embedded viewer need, with the types browsers insist on.
#: A module served as text/plain is refused outright, which is a confusing way to find
#: out you forgot one.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".wasm": "application/wasm",
    ".ply": "application/octet-stream",
    ".txt": "text/plain; charset=utf-8",
}


def _static_type(path: str) -> str:
    return CONTENT_TYPES.get(Path(path).suffix.lower(), "")


class Session:
    """Shared server state: the resolved ffmpeg, preview cache, current job."""

    def __init__(self, ffmpeg, project=None) -> None:
        self.ffmpeg = ffmpeg
        self.cache = Path(tempfile.mkdtemp(prefix="360extract-ui-"))
        self.counter = 0
        self.lock = threading.Lock()
        #: The open project, if any. Owns the painted occluder and remembers settings.
        self.project = project
        #: The most recently decoded panorama frame, *ungraded*. Seeking an 8K source
        #: costs around 600 ms; regrading this cached frame costs around 50 ms, which
        #: is what makes the grade sliders usable live.
        self.preview_source: Path | None = None
        self.preview_key: tuple | None = None
        #: (path, time, SourceFormat) of whatever produced `preview_source`, so a
        #: measurement can go back to the panorama even when the lenses are on screen.
        self.preview_origin: tuple | None = None
        #: Detection backends, kept alive between requests. Constructing one loads the
        #: model weights, which is several seconds -- doing that per preview is what made
        #: scrubbing with the mask overlay unusable.
        self.backends: dict[tuple, object] = {}
        #: One job per pipeline stage. Replaces the single session-wide job, which
        #: could not tell extraction from detection and blocked both.
        self.jobs = JobRegistry()

    def next_name(self, suffix: str) -> Path:
        with self.lock:
            self.counter += 1
            return self.cache / f"p{self.counter:05d}{suffix}"


def rig_from_payload(data: dict) -> Rig:
    """Build a Rig from the browser's JSON, validating it properly."""
    return Rig.from_dict({
        "version": data.get("version", 1),
        "name": data.get("name", "rig"),
        "cameras": data.get("cameras", []),
        "output": data.get("output", {}),
        "orientation": data.get("orientation", {}),
        "grade": data.get("grade", {}),
        "occluders": data.get("occluders", []),
    })


def media_payload(info: MediaInfo) -> dict:
    payload = asdict(info)
    payload["path"] = str(info.path)
    payload["aspect"] = info.aspect
    payload["looks_equirectangular"] = info.looks_equirectangular
    payload["looks_circular"] = info.looks_circular
    # What the file looks like it is, when the container makes that plain -- two video
    # streams is a lens each, and no stitched 360 file is ever shaped that way. Offered,
    # never applied: the user confirms it in Start.
    suggestion = SourceFormat.detect(info)
    payload["suggested_source"] = suggestion.to_dict() if suggestion else None
    return payload


class Handler(BaseHTTPRequestHandler):
    session: Session  # injected by serve()

    server_version = "360extract"

    def log_message(self, fmt, *args):  # noqa: A003 - quieten the default access log
        return

    # -- plumbing -----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _safe_join(self, root: Path, relative: str) -> Path | None:
        """Resolve `relative` under `root`, refusing anything that escapes it."""
        candidate = (root / relative.lstrip("/")).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return None
        return candidate

    def _serve_static(self, route_path: str) -> None:
        target = self._safe_join(STATIC, route_path)
        if target is None:
            self._json({"error": "not found"}, 404)
            return
        self._file(target, _static_type(route_path))

    def _serve_viewer(self, relative: str) -> None:
        """Serve the local SuperSplat build."""
        from ..tools import find_supersplat

        viewer = find_supersplat()
        if not viewer.found:
            self._json({"error": "no SuperSplat build found"}, 404)
            return
        target = self._safe_join(viewer.path, relative or "index.html")
        if target is None or not target.exists():
            self._json({"error": f"not found: {relative}"}, 404)
            return
        self._file(target, _static_type(target.name) or "application/octet-stream")

    def _serve_project_frame(self, relative: str, width: int = 0,
                             view: str = "panorama") -> None:
        """Serve an extracted frame so the Capture canvas can show it.

        `?w=` returns a cached copy at that width. Frames come off an 8K source, and
        shipping (and decoding) all of those pixels for a canvas a fraction of the size
        is most of what made stepping through them feel slow.

        `?view=lenses` sends the frame back through the projection it arrived in, so a
        capture shot on two fisheye lenses can be rigged in the picture it was shot in.
        The frames on disk stay equirect either way -- that is what the rest of the
        pipeline reads, and what the tiles are cut from.

        Only the filename is used, so `/frames/00001.jpg` and the older
        `/frames/<clip>/00001.jpg` both land on whichever layout the project has.
        """
        project = self.session.project
        if project is None:
            self._json({"error": "no project is open"}, 404)
            return
        sources = project.resolved_sources()
        clip = safe_stem(sources[0].stem) if sources else None
        target = self._safe_join(dataset.frames_dir(project.root, clip),
                                 Path(relative).name)
        if target is None or not target.exists():
            self._json({"error": f"not found: {relative}"}, 404)
            return
        fmt = project.source_format
        lenses = view == "lenses" and not fmt.is_equirect
        if width or lenses:
            width = max(min(width or 1600, 8192), 64)
            proxy = self._derived(
                f"frame|{target}|{target.stat().st_mtime_ns}|{width}|{view}", ".jpg")
            try:
                target = (self._as_lenses(target, fmt, width, proxy) if lenses
                          else self._scaled(target, width, width // 2, proxy))
            except FFmpegError:
                pass                        # fall back to the original, whole
        self._file(target, "image/jpeg")

    def _as_lenses(self, equirect: Path, fmt: SourceFormat, width: int,
                   target: Path, mask: bool = False) -> Path:
        """Project an equirect image back onto the lenses it was shot with.

        Same call as `_mask_as_lenses`, for pictures rather than masks: the frames on
        disk are panoramas, and this is what the Capture canvas draws when the rig is
        being placed on the raw view.
        """
        if target.exists():
            return target
        height = max(width // 2, 2)
        params = ":".join(["e", "dfisheye" if fmt.lens_count == 2 else "fisheye",
                           f"h_fov={fmt.lens_fov:g}", f"v_fov={fmt.lens_fov:g}",
                           f"w={width}", f"h={height}",
                           f"interp={'near' if mask else 'line'}"])
        # Outside each circle there is no picture -- a real lens frame is black there.
        # v360 fills those corners by smearing the rim instead, which reads as detail
        # that does not exist, so they are cut back to black.
        inside = ("lte(pow((X-if(lt(X,W/2),W/4,3*W/4))/(W/4),2)"
                  "+pow((Y-H/2)/(H/2),2),1)")
        # A mask keeps its surround *white*: outside the circles there is no picture to
        # exclude, and tinting it red would drown the frame in a warning about nothing.
        circles = (f"format=gray,geq=lum='if({inside},lum(X,Y),255)'" if mask else
                   f"format=gbrp,geq=r='if({inside},r(X,Y),0)'"
                   f":g='if({inside},g(X,Y),0)':b='if({inside},b(X,Y),0)'")
        result = subprocess.run(
            [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(equirect), "-vf", f"v360={params},{circles}",
             "-frames:v", "1", "-q:v", "3", str(target)],
            capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not target.exists():
            raise FFmpegError(f"lens projection failed: {result.stderr.strip()}")
        return target

    def _serve_project_file(self, relative: str) -> None:
        """Serve a file from the open project, so the viewer can fetch a .ply."""
        project = self.session.project
        if project is None:
            self._json({"error": "no project is open"}, 404)
            return
        target = self._safe_join(project.root, relative)
        if target is None or not target.exists():
            self._json({"error": f"not found: {relative}"}, 404)
            return
        self._send(200, target.read_bytes(), "application/octet-stream")

    def _file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._json({"error": f"not found: {path.name}"}, 404)
            return
        self._send(200, path.read_bytes(), content_type)

    # -- routing ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - required name
        route = urlparse(self.path)
        query = parse_qs(route.query)
        try:
            if route.path in ("/", "/index.html"):
                self._file(STATIC / "index.html", "text/html; charset=utf-8")
            elif route.path.startswith("/viewer/"):
                # SuperSplat's own build, served from wherever it was found so the
                # viewer can be embedded rather than launched separately.
                self._serve_viewer(route.path[len("/viewer/"):])
            elif route.path.startswith("/splat/"):
                self._serve_project_file(route.path[len("/splat/"):])
            elif route.path.startswith("/frames/"):
                self._serve_project_frame(route.path[len("/frames/"):],
                                          int(query.get("w", ["0"])[0] or 0),
                                          query.get("view", ["panorama"])[0])
            elif route.path.startswith("/preview/"):
                # Before the generic static handler: these are generated files in the
                # session cache, and `.jpg` would otherwise be looked for in static/.
                name = Path(route.path).name
                self._file(self.session.cache / name,
                           _static_type(name) or "image/jpeg")
            elif _static_type(route.path):
                self._serve_static(route.path)
            elif route.path == "/api/presets":
                self._json(self._presets_payload())
            elif route.path == "/api/progress":
                # Kept for the CLI-era clients and tests: report whichever stage is
                # running, or capture's last state when nothing is.
                running = self.session.jobs.any_running()
                self._json((running or self.session.jobs["capture"]).snapshot(0))
            elif route.path == "/api/detect/status":
                from ..mask import ml
                self._json({"available": ml.available()})
            elif route.path == "/api/jobs":
                # Every stage at once, so the pipeline navigation can show a stage as
                # running even while the user is looking at a different one.
                self._json({
                    "jobs": self.session.jobs.snapshot(
                        log_limit=int(query.get("log", ["0"])[0])),
                    "stages": stages.readiness(self.session.project),
                })
            elif route.path == "/api/system":
                self._json({"tools": tool_survey(),
                            "configured": toolpaths.stored()})
            elif route.path == "/api/project":
                project = self.session.project
                self._json({"project": self._project_payload(project)
                            if project else None})
            elif route.path == "/api/recent":
                self._json({"recent": recent.entries()})
            elif route.path == "/api/frames/list":
                self._json(self.api_frames_list())
            elif route.path == "/api/reconstruct/points":
                self._serve_reconstruct_points(query)
            elif route.path == "/api/train/latest":
                self._json(self.api_train_latest())
            elif route.path.startswith("/preview/"):
                name = Path(route.path).name
                self._file(self.session.cache / name, "image/jpeg")
            else:
                self._json({"error": "no such endpoint"}, 404)
        except Exception as exc:  # surface errors in the UI rather than the console
            traceback.print_exc()
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802 - required name
        route = urlparse(self.path)
        try:
            payload = self._read_json()
            if route.path == "/api/probe":
                self._json(self.api_probe(payload))
            elif route.path == "/api/preview":
                self._json(self.api_preview(payload))
            elif route.path == "/api/source/fit":
                self._json(self.api_source_fit(payload))
            elif route.path == "/api/camera-preview":
                self._json(self.api_camera_preview(payload))
            elif route.path == "/api/rig/validate":
                self._json(self.api_validate(payload))
            elif route.path == "/api/rig/save":
                self._json(self.api_rig_save(payload))
            elif route.path == "/api/rig/load":
                self._json(self.api_rig_load(payload))
            elif route.path == "/api/preset/save":
                self._json(self.api_preset_save(payload))
            elif route.path == "/api/preset/delete":
                self._json(self.api_preset_delete(payload))
            elif route.path == "/api/extract":
                self._json(self.api_extract(payload))
            elif route.path == "/api/frames/extract":
                self._json(self.api_frames_extract(payload))
            elif route.path == "/api/frames/remove":
                self._json(self.api_frames_remove(payload))
            elif route.path == "/api/cameras/generate":
                self._json(self.api_cameras_generate(payload))
            elif route.path == "/api/job/cancel":
                job = self.session.jobs[payload["stage"]]
                job.cancel.set()
                self._json({"cancelled": job.stage})
            elif route.path == "/api/job/status":
                self._json(self.session.jobs[payload["stage"]].snapshot(
                    log_limit=int(payload.get("log", 400))))
            elif route.path == "/api/reconstruct/run":
                self._json(self._start("reconstruct", stages.run_reconstruction, payload))
            elif route.path == "/api/train/run":
                self._json(self._start("train", stages.run_training, payload))
            elif route.path == "/api/inspect/clean":
                self._json(self._start("inspect", stages.run_cleanup, payload))
            elif route.path == "/api/detect/frames":
                self._json(self.api_detect_frames(payload))
            elif route.path == "/api/detect/preview":
                self._json(self.api_detect_preview(payload))
            elif route.path == "/api/detect/run":
                self._json(self.api_detect_run(payload))
            elif route.path == "/api/export/colmap":
                self._json(self.api_export_colmap(payload))
            elif route.path == "/api/splat/clean":
                self._json(self.api_splat_clean(payload))
            elif route.path == "/api/project/open":
                self._json(self.api_project_open(payload))
            elif route.path == "/api/project/save":
                self._json(self.api_project_save(payload))
            elif route.path == "/api/project/new":
                self._json(self.api_project_new(payload))
            elif route.path == "/api/project/for-source":
                self._json(self.api_project_for_source(payload))
            elif route.path == "/api/segment":
                self._json(self.api_segment(payload))
            elif route.path == "/api/recent/remove":
                recent.remove(payload["root"])
                self._json({"recent": recent.entries()})
            elif route.path == "/api/preview/grade":
                self._json(self.api_preview_grade(payload))
            elif route.path == "/api/grade/auto":
                self._json(self.api_grade_auto(payload))
            elif route.path == "/api/mask/paint":
                self._json(self.api_mask_paint(payload))
            elif route.path == "/api/mask/coverage":
                self._json(self.api_mask_coverage(payload))
            elif route.path == "/api/mask/preview":
                self._json(self.api_mask_preview(payload))
            elif route.path == "/api/mask/frame":
                self._json(self.api_mask_frame(payload))
            elif route.path == "/api/system/tools":
                self._json(self.api_system_tools(payload))
            elif route.path == "/api/pick":
                self._json(self.api_pick(payload))
            elif route.path == "/api/cancel":
                self.session.jobs.cancel_all()
                self._json({"ok": True})
            else:
                self._json({"error": "no such endpoint"}, 404)
        except AlreadyRunning as exc:
            # Name what is running and where, so the UI can offer to go there rather
            # than saying "something is already running" and leaving the user stuck.
            self._json({"error": str(exc), "running_stage": exc.stage}, 409)
        except (FFmpegError, RigError, ProjectError, stages.StageError, ValueError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": str(exc)}, 500)

    # -- endpoints ----------------------------------------------------------

    def _project_payload(self, project: Project) -> dict:
        """Everything the UI needs to restore itself from a project."""
        return {
            "root": str(project.root),
            "file": str(project.file),
            "name": project.name,
            "sources": [str(p) for p in project.resolved_sources()],
            "missing": [str(p) for p in project.missing_sources()],
            "rig": project.rig.to_dict(),
            "frames": asdict(project.frames),
            "source_format": project.source_format.to_dict(),
            "output": asdict(project.output),
            "detect": asdict(project.detect),
            "stages": {name: project.status(name) for name in STAGES},
            # How much of the import is already on disk, so Start can say "done" on a
            # project reopened weeks later rather than looking untouched.
            "imported": self._frame_count(project),
            # Survives a reload, so the "nothing was detected" warning does not vanish
            # just because the page was refreshed after masking.
            "undetected": list(
                project.stages["mask"].details.get("undetected", []))
            if "mask" in project.stages else [],
        }

    def _frame_count(self, project: Project) -> int:
        sources = project.resolved_sources()
        clip = safe_stem(sources[0].stem) if sources else None
        directory = dataset.frames_dir(project.root, clip)
        return len(list(directory.glob("*.jpg"))) if directory.is_dir() else 0

    def _start(self, stage: str, work, payload: dict) -> dict:
        """Kick off a stage's job, with the project and settings bound in."""
        project = self._open_project()
        job = self.session.jobs[stage]
        job.start(lambda j: work(j, project, payload), name="starting")
        return {"started": True, "stage": stage}

    def _open_project(self) -> Project:
        if self.session.project is None:
            raise ValueError("no project is open; open or save one on the Capture tab")
        return self.session.project

    def api_detect_frames(self, payload: dict) -> dict:
        """What has been extracted, and how much of it already has masks."""
        from ..mask.dynamic import discover

        project = self._open_project()
        found = discover(project.root, project.rig)
        # Masks live in each camera's own folder now, which `discover` already resolved.
        masked = sum(len(list(entry.mask_directory.glob("*.png")))
                     for entry in found if entry.mask_directory.is_dir())

        return {
            "cameras": [{"name": entry.camera.name,
                         "frames": sorted(entry.frames)} for entry in found],
            "masked": masked,
        }

    def api_detect_preview(self, payload: dict) -> dict:
        """An extracted frame with its mask tinted over it.

        Composited here rather than in the browser so the preview uses the mask file
        that will actually be handed to the trainer, not an approximation of it.
        """
        from ..mask.dynamic import discover

        project = self._open_project()
        found = discover(project.root, project.rig)
        entry = next((e for e in found if e.camera.name == payload["camera"]), None)
        if entry is None:
            raise ValueError(f"no camera named {payload['camera']!r}")

        frame = int(payload["frame"])
        image = entry.frames.get(frame)
        if image is None:
            raise ValueError(f"camera {entry.camera.name} has no frame {frame}")

        target = self.session.next_name(".jpg")
        mask = entry.mask_directory / f"{image.stem}.png"
        opacity = float(payload.get("opacity", 0.55))

        if mask.exists():
            # Masked area shown in red: invert the mask so the ignored region is what
            # gets tinted, then blend it over the picture.
            graph = (
                "[1:v]format=gray,negate[m];"
                "color=red:size=16x16,format=rgba[c];"
                "[c][0:v]scale2ref[cr][img];"
                f"[cr][m]alphamerge,colorchannelmixer=aa={opacity:g}[tint];"
                "[img][tint]overlay,scale=520:-2[out]"
            )
            argv = [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error",
                    "-y", "-i", str(image), "-i", str(mask),
                    "-filter_complex", graph, "-map", "[out]",
                    "-frames:v", "1", "-q:v", "4", str(target)]
        else:
            argv = [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error",
                    "-y", "-i", str(image), "-vf", "scale=520:-2",
                    "-frames:v", "1", "-q:v", "4", str(target)]

        result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not target.exists():
            raise FFmpegError(f"preview failed: {result.stderr.strip()}")
        return {"url": f"/preview/{target.name}", "has_mask": mask.exists()}

    def api_detect_run(self, payload: dict) -> dict:
        """Run dynamic masking over the extracted dataset, in the background."""
        from ..mask import dynamic, ml

        project = self._open_project()
        job = self.session.jobs["refine"]
        if not ml.available():
            raise ValueError('dynamic masking needs the ML extra: pip install -e ".[ml]"')

        settings = DetectSettings(
            backend=payload.get("backend", "sam2.1"),
            classes=list(payload.get("classes") or DetectSettings().classes),
            confidence=float(payload.get("confidence", 0.25)),
            dilate=int(payload.get("dilate", 6)),
            fuse=bool(payload.get("fuse", True)),
        )
        project.detect = settings
        project.save()
        session = self.session

        def work(running_job) -> dict:
            running_job.update(message="loading the model")
            backend = ml.make_backend(
                settings.backend, classes=settings.classes,
                confidence=settings.confidence, dilate=settings.dilate,
                device=settings.device)

            report = dynamic.run(
                session.ffmpeg, project.root, project.rig, backend,
                fuse=settings.fuse,
                on_progress=lambda note: running_job.log(note),
                on_fraction=lambda done, total, message:
                    running_job.progress(done / max(total, 1e-9), message),
                should_cancel=running_job.cancel.is_set,
            )
            project.mark_done("mask", masks=report.masks_written,
                              detections=report.detections)
            project.save()
            return {"masks": report.masks_written, "detections": report.detections,
                    "summary": report.summary()}

        job.start(work, name="detecting")
        return {"started": True, "stage": "refine"}

    def api_export_colmap(self, payload: dict) -> dict:
        """Write rig_config.json, intrinsics and the command list for the open project."""
        from ..colmap import export as colmap_export
        from ..plan import safe_stem

        project = self.session.project
        if project is None:
            raise ValueError("no project is open")
        sources = project.resolved_sources()
        if not sources:
            raise ValueError("this project has no sources, so there is nothing to describe")

        clip = safe_stem(sources[0].stem)
        width = probe_media(sources[0], self.session.ffmpeg).width

        geo_path = None
        gpx = (payload.get("gpx") or "").strip()
        if gpx:
            geo_path = self._write_geo(project, clip, gpx)

        paths = colmap_export.export(
            project.root, project.rig, clip, width,
            has_masks=(project.root / "masks").exists(),
            geo_registration=geo_path is not None,
        )
        written = [paths.rig_config.name, paths.cameras.name, paths.commands.name]
        if geo_path:
            written.append(geo_path.name)

        project.mark_done("export", rig_config=str(paths.rig_config))
        project.save()
        return {"written": written}

    def _write_geo(self, project, clip: str, gpx: str) -> Path:
        from .. import gps
        from ..mask.dynamic import frame_number

        fixes = gps.read_gpx(gpx)
        images_root = dataset.images_dir(project.root, clip)
        if not images_root.exists():
            raise ValueError(f"{images_root} does not exist; extract first")

        per_second = plan_frames_per_second(project.frames.mode, project.frames.value)
        start = project.frames.start or 0.0

        entries = {}
        # Keyed by the path COLMAP records, which is the image's path below images/.
        for camera in sorted(p for p in images_root.iterdir() if p.is_dir()):
            directory = dataset.geometry_dir(project.root, clip, camera.name)
            for image in sorted(directory.iterdir()):
                if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                offset = start + (frame_number(image) - 1) / max(per_second, 1e-6)
                key = image.relative_to(project.root / "images").as_posix()
                entries[key] = gps.interpolate(fixes, fixes[0].time + offset)
        return gps.write_geo_registration(entries, project.root / "geo_registration.txt")

    def api_splat_clean(self, payload: dict) -> dict:
        """Preview or perform the floater removal."""
        from ..colmap.model import read_model
        from ..splat import clean as splat_clean
        from ..splat import ply

        splat_path = Path(payload["splat"])
        model = read_model(payload["sparse"])
        trajectory = splat_clean.trajectory_from_model(model)
        splats = ply.read(splat_path)

        named = {"enu": [0.0, 0.0, 1.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
        up = named.get((payload.get("up") or "").lower())

        kept, removed, report = splat_clean.clean(
            splats, trajectory, float(payload["radius"]),
            payload.get("floor"), up)

        lines = [ply.describe(splats)] + report.lines()
        if payload.get("dry_run"):
            return {"report": lines + ["preview only, nothing written"]}

        cleaned_path = splat_path.with_name(splat_path.stem + "_cleaned.ply")
        removed_path = splat_path.with_name(splat_path.stem + "_removed.ply")
        ply.write(kept, cleaned_path)
        ply.write(removed, removed_path)
        return {"report": lines + [f"wrote {cleaned_path.name} and {removed_path.name}"]}

    def api_project_new(self, payload: dict) -> dict:
        project = Project.create(
            payload["root"],
            sources=payload.get("sources", []),
            name=payload.get("name"),
            overwrite=bool(payload.get("force")),
        )
        self.session.project = project
        recent.record(project.root, project.name)
        return {"project": self._project_payload(project)}

    def api_project_open(self, payload: dict) -> dict:
        project = Project.load(payload["path"])
        self.session.project = project
        recent.record(project.root, project.name)
        return {"project": self._project_payload(project)}

    def api_project_for_source(self, payload: dict) -> dict:
        """Ensure a project for a just-opened source, in a folder beside it.

        Opening a video *is* opening a project now: the dataset lands in a subfolder
        named after the clip (`<video folder>/<clip>/`), so several clips in one folder
        never collide. Reopening a clip whose folder already has a `project.json`
        resumes it -- what is on disk wins over whatever the UI is currently showing,
        so a settings payload is honoured only when the project is created fresh.
        """
        source = Path(payload["path"])
        if not source.exists():
            raise ValueError(f"{source} does not exist")

        root = source.parent / safe_stem(source.stem)
        if (root / PROJECT_FILENAME).exists():
            project = Project.load(root)
            resolved = str(source.resolve())
            if resolved not in {str(project.absolute(s).resolve())
                                for s in project.sources}:
                project.sources.append(project.relative(str(source)))
                project.save()
        else:
            project = Project.create(root, sources=[str(source)])
            if "rig" in payload:
                project.rig = rig_from_payload(payload["rig"])
            for key, target in (("frames", FrameSettings), ("output", OutputSettings)):
                if key in payload:
                    current = asdict(getattr(project, key))
                    current.update(payload[key])
                    setattr(project, key, target(**current))
            if "source_format" in payload:
                project.source_format = SourceFormat.from_dict(payload["source_format"])
                project.source_format.validate()
            else:
                # Nobody said what the footage is -- so if the container makes it plain
                # (two video streams of the same square size is a lens each), record
                # that rather than defaulting to a panorama. A tab other than Start can
                # open a source, and reading one lens as a whole sphere is not a
                # recoverable mistake: it is silently wrong all the way to the splat.
                try:
                    probed = probe_media(source, self.session.ffmpeg)
                except FFmpegError:
                    probed = None       # unreadable here is the extraction's problem
                detected = SourceFormat.detect(probed) if probed else None
                if detected is not None:
                    project.source_format = detected
            project.save()

        self.session.project = project
        recent.record(project.root, project.name)
        return {"project": self._project_payload(project)}

    def api_segment(self, payload: dict) -> dict:
        """Split one source into a project per segment, beside the video.

        Each segment is a `<stem>_segNN/` project carrying its own start/end window; the
        user then extracts each independently (a long drive reconstructs far better as
        several short datasets than as one). Modes: `duration` (seconds), `motion`
        (forward travel estimated from the video), `gpx` (metres along a sidecar track).
        """
        from .. import gps

        source = Path(payload["path"])
        if not source.exists():
            raise ValueError(f"{source} does not exist")
        info = probe_media(source, self.session.ffmpeg)
        mode = payload.get("mode", "duration")
        source_format = self._source_format(payload)

        if mode == "duration":
            segments = segment.segment_by_duration(info.duration, float(payload["seconds"]))
        elif mode == "gpx":
            track = source.with_suffix(".gpx")
            if not track.exists():
                raise ValueError(f"no GPX sidecar next to the video (expected {track.name})")
            segments = segment.segment_by_gpx(gps.read_gpx(track), float(payload["meters"]))
        elif mode == "motion":
            if not motion.available():
                raise ValueError('motion segmentation needs OpenCV: pip install -e ".[ml]"')
            samples = motion.forward_motion(self.session.ffmpeg, source,
                                            source_format=source_format)
            if payload.get("speed_kph"):
                segments = segment.segment_by_motion(
                    samples, meters=float(payload["meters"]),
                    speed_kph=float(payload["speed_kph"]))
            else:
                segments = segment.segment_by_motion(samples, count=int(payload["count"]))
        else:
            raise ValueError(f"unknown segment mode {mode!r}")

        pad = max(2, len(str(len(segments))))
        stem = safe_stem(source.stem)
        created = []
        for seg in segments:
            root = source.parent / f"{stem}_seg{seg.index + 1:0{pad}d}"
            project = Project.create(root, sources=[str(source)],
                                     name=root.name, overwrite=True)
            project.frames.start = round(seg.start, 3)
            project.frames.end = round(seg.end, 3)
            # Every segment reads the same file, so they all read it the same way.
            project.source_format = source_format
            project.save()
            recent.record(project.root, project.name)
            created.append({
                "index": seg.index, "start": seg.start, "end": seg.end,
                "distance": seg.distance, "approximate": seg.approximate,
                "root": str(project.root), "name": project.name,
            })
        return {"segments": created, "mode": mode}

    def api_project_save(self, payload: dict) -> dict:
        """Write the UI's current state into the project.

        The project is the source of truth on disk, so the browser hands over
        everything it holds rather than the server guessing what changed.
        """
        project = self.session.project
        if project is None:
            root = payload.get("root")
            if not root:
                raise ValueError("no project is open; choose a folder first")
            project = Project(root=Path(root), name=payload.get("name") or Path(root).name)
            self.session.project = project

        if "rig" in payload:
            project.rig = rig_from_payload(payload["rig"])
        if "sources" in payload:
            project.sources = [project.relative(s) for s in payload["sources"]]
        if "source_format" in payload:
            project.source_format = SourceFormat.from_dict(payload["source_format"])
            project.source_format.validate()
        for key, target in (("frames", FrameSettings), ("output", OutputSettings),
                            ("detect", DetectSettings)):
            if key in payload:
                current = asdict(getattr(project, key))
                current.update(payload[key])
                setattr(project, key, target(**current))

        if payload.get("snapshot"):
            project.snapshot(payload["snapshot"])
        project.save()
        return {"project": self._project_payload(project)}

    def api_mask_paint(self, payload: dict) -> dict:
        """Store the painted occluder as an equirect mask file.

        An entirely white image means nothing is painted, so the occluder is dropped
        rather than written -- otherwise clearing the brush would leave a no-op mask
        behind that still forces every camera into masked handling.
        """
        data = payload.get("image", "")
        _, _, encoded = data.partition(",")
        if not encoded:
            raise ValueError("no image data received")

        raw = base64.b64decode(encoded)
        # Into the project when there is one. The temp cache is wiped on reboot, which
        # would leave the rig pointing at an occluder that no longer exists.
        if self.session.project is not None:
            self.session.project.assets_dir.mkdir(parents=True, exist_ok=True)
            target = self.session.project.assets_dir / "painted_occluder.png"
        else:
            target = self.session.cache / "painted_occluder.png"
        target.write_bytes(raw)

        if geometric.ignored_fraction(self.session.ffmpeg, target) <= 0.0005:
            target.unlink(missing_ok=True)
            return {"path": None}
        return {"path": str(target)}

    def api_mask_preview(self, payload: dict) -> dict:
        """The mask tinted red over the panorama at one frame, so masking can be checked.

        The sky cone and static occluders are deterministic (same on every frame). With
        ``objects: true`` it also runs detection on this frame -- more expensive, so it
        is on-demand behind the Preview button rather than every scrub.
        """
        width = int(payload.get("width", 1280))
        height = width // 2
        ffmpeg = self.session.ffmpeg

        # Source: an extracted frame from the project (Capture) or a source path/time.
        project = self.session.project
        frame_name = payload.get("frame")
        seek = None
        extracted = bool(frame_name and project and project.resolved_sources())
        if extracted:
            clip = safe_stem(project.resolved_sources()[0].stem)
            source_path = frames.frames_dir(project.root, clip) / frame_name
            if not source_path.exists():
                raise ValueError(f"no such frame: {frame_name}")
        else:
            source_path = Path(payload["path"])
            if probe_media(source_path, ffmpeg).is_video:
                seek = float(payload.get("time", 0.0)) or None

        # Extracted frames are equirect by construction; a raw source is whatever the
        # user says it is, and is previewed in whichever view they are looking at.
        fmt = SourceFormat() if extracted else self._source_format(payload)
        view = "panorama" if extracted else self._preview_view(payload, fmt)
        frame = self._decode_source_frame(source_path, seek, width, fmt, view,
                                          self.session.next_name(".jpg"), quality=4)

        # Masking is decided on the panorama whatever is on screen. That is where the
        # pipeline itself detects, and it is the only place the two agree: a detector
        # run on the lens view finds different things (a fisheye circle is not a
        # photograph of anything it was trained on), and the occluders are equirect by
        # construction, so mixing the two produced a mask that matched neither.
        measured = frame if view == "panorama" else self._decode_source_frame(
            source_path, seek, width, fmt, "panorama",
            self.session.next_name(".jpg"), quality=4)

        mask = self._preview_mask(measured, width, height, payload, source_format=fmt)
        if mask is None:
            return {"url": f"/preview/{frame.name}", "empty": True}
        if view == "lenses":
            # The mask is equirect; the picture under it is not. Send the mask back out
            # through the same projection, so it lands on the lenses -- and the trimmed
            # edge shows up as what it is, a ring cut off the rim of each circle.
            mask = self._mask_as_lenses(mask, fmt, width, height)

        target = self.session.next_name(".jpg")
        opacity = float(payload.get("opacity", 0.5))
        # Invert the mask so the ignored (black) region is what gets tinted red.
        graph = (
            "[1:v]format=gray,negate[m];"
            "color=red:size=16x16,format=rgba[c];"
            "[c][0:v]scale2ref[cr][img];"
            f"[cr][m]alphamerge,colorchannelmixer=aa={opacity:g}[tint];"
            "[img][tint]overlay[out]"
        )
        composite = [str(ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
                     "-i", str(frame), "-i", str(mask),
                     "-filter_complex", graph, "-map", "[out]",
                     "-frames:v", "1", "-q:v", "4", str(target)]
        result = subprocess.run(composite, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not target.exists():
            raise FFmpegError(f"mask preview failed: {result.stderr.strip()}")
        return {"url": f"/preview/{target.name}", "objects": bool(payload.get("objects"))}

    # -- preview caching --------------------------------------------------
    #
    # Everything the canvas asks for repeatedly -- a frame at canvas size, the mask over
    # it -- is derived from files that do not change. So derive it once, under a name
    # that is a hash of what produced it, and the second request is a file that is
    # already there. This is what makes scrubbing feel like scrubbing.

    def _derived(self, key: str, suffix: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        return self.session.cache / f"d{digest}{suffix}"

    def _scaled(self, source: Path, width: int, height: int, target: Path,
                sharp: bool = False) -> Path:
        """A resized copy of `source`, written once. `sharp` keeps a mask binary."""
        if target.exists():
            return target
        argv = [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y"]
        if sharp:
            argv += ["-sws_flags", "neighbor"]
        argv += ["-i", str(source), "-vf", f"scale={width}:{height}",
                 "-frames:v", "1", str(target)]
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not target.exists():
            raise FFmpegError(f"resize failed: {result.stderr.strip()}")
        return target

    def _mask_backend(self, detect: dict):
        """A detection backend for these settings, built once per session."""
        from ..mask import ml
        if not ml.available():
            raise ValueError('object detection needs the ML extra: pip install -e ".[ml]"')
        defaults = DetectSettings()
        key = (detect.get("backend") or defaults.backend,
               tuple(detect.get("classes") or defaults.classes),
               float(detect.get("confidence", defaults.confidence)),
               int(detect.get("dilate", defaults.dilate)),
               detect.get("device"))
        with self.session.lock:
            backend = self.session.backends.get(key)
        if backend is None:
            backend = ml.make_backend(key[0], classes=list(key[1]), confidence=key[2],
                                      dilate=key[3], device=key[4])
            with self.session.lock:
                self.session.backends[key] = backend
        return backend

    def api_mask_frame(self, payload: dict) -> dict:
        """The ignore-mask for one extracted frame, as a plain image the canvas tints.

        Three ways to answer, cheapest first: the mask camera generation already wrote
        (when the settings that produced it still hold), one this session has derived
        before, or a fresh detection run. Only the last is slow, and only once per frame.
        """
        project = self._open_project()
        sources = project.resolved_sources()
        if not sources:
            raise ValueError("this project has no source")
        name = str(payload.get("frame") or "")
        frame_path = frames.frames_dir(project.root, safe_stem(sources[0].stem)) / name
        if not name or not frame_path.exists():
            raise ValueError(f"no such frame: {name}")
        width = max(int(payload.get("width", 1280)), 64)
        height = width // 2
        # The mask is measured on the panorama, as always, and only then follows the
        # picture onto the lenses -- so the two always agree about what is excluded.
        fmt = project.source_format
        lenses = str(payload.get("view") or "") == "lenses" and not fmt.is_equirect

        def answer(mask: Path, source: str) -> dict:
            shown = mask
            if lenses:
                shown = self._as_lenses(
                    mask, fmt, width,
                    self._derived(f"lens|{mask}|{width}|{fmt.lens_fov:g}", ".png"),
                    mask=True)
            return {"url": f"/preview/{shown.name}", "source": source}

        generated = (project.root / ".threesixty" / "masks" / "equirect_masks"
                     / f"{Path(name).stem}.png")
        if generated.exists() and project.status("mask") == "done":
            target = self._derived(
                f"gen|{generated}|{generated.stat().st_mtime_ns}|{width}", ".png")
            self._scaled(generated, width, height, target, sharp=True)
            return answer(target, "generated")

        detect = payload.get("detect") or asdict(project.detect)
        signature = json.dumps({"detect": detect, "occluders": payload.get("occluders")},
                               sort_keys=True, default=str)
        target = self._derived(f"live|{frame_path}|{width}|{signature}", ".png")
        if target.exists():
            return answer(target, "cached")

        scaled = self._derived(f"frame|{frame_path}|{width}", ".jpg")
        self._scaled(frame_path, width, height, scaled)
        mask = self._preview_mask(scaled, width, height,
                                  {"objects": True, "detect": detect,
                                   "occluders": payload.get("occluders") or []},
                                  out=target)
        if mask is None:
            return {"url": None, "empty": True}
        return answer(target, "detected")

    def _mask_as_lenses(self, equirect_mask: Path, fmt: SourceFormat,
                        width: int, height: int) -> Path:
        """Project an equirect mask back onto the lens view, for the preview."""
        return self._as_lenses(equirect_mask, fmt, width,
                               self.session.next_name(".png"), mask=True)

    def _preview_mask(self, frame: Path, width: int, height: int,
                      payload: dict, out: Path | None = None,
                      source_format: SourceFormat | None = None) -> Path | None:
        """Build the combined equirect mask (cone/occluders + optional detection) as PNG.

        Returns None when nothing would be masked, so the caller shows the plain frame.
        """
        raw = list(payload.get("occluders") or [])
        angle = payload.get("sky_cone_angle")
        if angle:
            raw.append({"type": "zenith_cone", "angle": float(angle)})
        # Trimming the lenses is part of the picture the user is checking, so it belongs
        # in the preview even though it comes from the source rather than the rig.
        fmt = source_format or self._source_format(payload)
        if fmt.seam_band > 0:
            raw.append({"type": "seam_band", "angle": fmt.seam_band})
        occluders = [o for o in (geometric.Occluder.from_dict(d) for d in raw)
                     if o.kind != "ml"]

        cone_path = None
        if occluders:
            cone_path = geometric.build_equirect_mask(
                self.session.ffmpeg, occluders, width, height,
                self.session.cache / "mask_preview_eq.png")

        want_objects = bool(payload.get("objects"))
        if not want_objects:
            return cone_path

        import numpy as np

        backend = self._mask_backend(payload.get("detect") or {})
        objects = backend.detect([frame])[0].mask     # HxW uint8, white keeps
        combined = np.asarray(objects)
        if cone_path is not None:
            import cv2
            cone = cv2.imread(str(cone_path), cv2.IMREAD_GRAYSCALE)
            if cone.shape != combined.shape:
                cone = cv2.resize(cone, (combined.shape[1], combined.shape[0]))
            combined = np.minimum(combined, cone)     # stricter of the two wins

        import cv2
        # Named by the caller when the result is being cached; otherwise one scratch file.
        out = out or (self.session.cache / "mask_preview_combined.png")
        cv2.imwrite(str(out), combined)
        return out

    def api_mask_coverage(self, payload: dict) -> dict:
        """Measure what each camera actually loses to the occluders.

        Rendered rather than estimated: a painted occluder is an arbitrary shape with
        no closed form, and this uses the same projection the extraction will.
        """
        rig = rig_from_payload(payload["rig"])
        width = int(payload.get("source_width") or 4096)
        height = int(payload.get("source_height") or width // 2)
        # The mask is equirect whatever the file is. Measuring it on a raw source's own
        # 1:1 frame would put every angle in the wrong place.
        fmt = self._source_format(payload)
        if not fmt.is_equirect:
            width, height = fmt.equirect_size(
                width, height, int(payload.get("source_streams") or 1))

        occluders = geometric.occluders_of(rig)
        if not occluders:
            return {"coverage": {}}

        equirect = geometric.build_equirect_mask(
            self.session.ffmpeg, occluders, width, height,
            self.session.cache / "coverage_equirect.png")

        coverage = {}
        for camera in rig.normalized_cameras():
            rendered = geometric.render_camera_mask(
                self.session.ffmpeg, equirect, camera, 160, 120,
                self.session.cache / f"coverage_{camera.name}.png")
            coverage[camera.name] = geometric.ignored_fraction(self.session.ffmpeg, rendered)
        return {"coverage": coverage}

    def api_system_tools(self, payload: dict) -> dict:
        """Remember where a tool is, then report what that resolved to.

        Answering with a fresh survey is the point: a path that does not hold the binary
        should say so immediately, in the dialog, rather than at the start of a
        reconstruction twenty minutes later.
        """
        configured = toolpaths.save_many(payload.get("tools") or {})
        return {"tools": tool_survey(), "configured": configured}

    def api_pick(self, payload: dict) -> dict:
        """Raise a native file dialog. The browser cannot supply real paths itself."""
        if not picker.available():
            raise ValueError(
                "no file dialog available (tkinter is missing from this Python). "
                "Type the path into the field instead."
            )
        paths = picker.ask(
            mode=payload.get("mode", "open"),
            title=payload.get("title", "Select"),
            kind=payload.get("kind", "media"),
            initial=payload.get("initial", ""),
        )
        return {"paths": paths}

    def _source_format(self, payload: dict | None = None) -> SourceFormat:
        """What projection the source pixels are in.

        The UI's current choice wins while a source is being set up (the project may not
        exist yet, or may not have been saved since the dropdown moved); otherwise the
        open project answers. Extracted frames are equirect by construction, so callers
        working on those skip this entirely.
        """
        if payload and payload.get("source_format"):
            chosen = SourceFormat.from_dict(payload["source_format"])
            chosen.validate()
            return chosen
        project = self.session.project
        return project.source_format if project else SourceFormat()

    def _decode_source_frame(self, path: Path, seek: float | None, width: int,
                             fmt: SourceFormat, view: str, target: Path,
                             quality: int = 3) -> Path:
        """One frame out of a source file, as either view, at `width`.

        `view` is `panorama` (assembled and projected to equirectangular, what the
        pipeline will actually work on) or `lenses` (each lens upright, side by side,
        nothing warped -- what a raw file honestly looks like). Both come out 2:1, so
        the canvas does not have to care which it is showing.
        """
        height = max(width // 2, 2)
        argv = [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y"]
        if seek:
            argv += ["-ss", f"{seek:g}"]
        argv += ["-i", str(path)]

        if view == "lenses" and not fmt.is_equirect:
            chains = fmt.lens_chains(label="out", scale=width // 2)
            if chains:
                argv += ["-filter_complex", ";".join(chains), "-map", "[out]"]
            else:
                # Both lenses already sit in one frame the right way up.
                argv += ["-vf", f"scale={width}:{height}"]
        elif fmt.needs_graph:
            argv += ["-filter_complex",
                     ";".join(fmt.ingest_chains((width, height), label="out")),
                     "-map", "[out]"]
        else:
            argv += ["-vf", fmt.to_equirect(0, 0, size=(width, height))
                     or f"scale={width}:-2"]

        argv += ["-frames:v", "1", "-q:v", str(quality), str(target)]
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not target.exists():
            raise FFmpegError(f"preview failed: {result.stderr.strip()}")
        return target

    def _preview_view(self, payload: dict, fmt: SourceFormat) -> str:
        """Which of the two views to show.

        The panorama unless the caller explicitly asks for the lenses. That default is
        load-bearing: the panorama is the picture the whole pipeline works in, and every
        other tab draws over it in equirect coordinates -- so a caller that says nothing
        must not be handed two fisheye circles. Start asks for the lenses by name.
        """
        return "lenses" if str(payload.get("view") or "") == "lenses" else "panorama"

    def api_source_fit(self, payload: dict) -> dict:
        """Measure the lens field of view from how well the two lenses line up.

        The spec sheet rounds it and the difference is visible, so this fits it instead:
        the seam is a great circle, and a wrong figure makes the picture jump across it.
        """
        from .. import source as source_module

        path = Path(payload["path"])
        fmt = self._source_format(payload)
        info = probe_media(path, self.session.ffmpeg)
        # Spread the samples through the clip: one frame with a wall close on one side
        # scores its own parallax as if it were a bad field of view.
        seeks = [info.duration * fraction for fraction in (0.2, 0.5, 0.8)]             if info.is_video and info.duration else [0.0]
        return source_module.fit_lens_fov(
            self.session.ffmpeg, path, fmt, seeks=seeks,
            workdir=self.session.cache / "fit",
            # Fitting the mounting as well is the default: it is the setting nothing in
            # the file reveals, and the one a user has no way to look up.
            fit_rotation=bool(payload.get("fit_rotation", True)))

    def api_probe(self, payload: dict) -> dict:
        info = probe_media(payload["path"], self.session.ffmpeg)
        return {"media": media_payload(info)}

    def api_preview(self, payload: dict) -> dict:
        """One equirect frame, downscaled, for the rig editor canvas.

        Graded exactly as the extraction will grade it, so the canvas is not a
        flattering or unflattering lie about what comes out.
        """
        info = probe_media(payload["path"], self.session.ffmpeg)
        time = float(payload.get("time", 0.0))
        target = self.session.next_name(".jpg")

        # Decode once and keep it ungraded; grading happens from the cache, so moving a
        # slider never re-seeks the video.
        fmt = self._source_format(payload)
        view = self._preview_view(payload, fmt)
        source = self._decode_source_frame(
            info.path, time if info.is_video and time > 0 else None,
            PREVIEW_WIDTH, fmt, view, self.session.next_name(".jpg"))

        self.session.preview_source = source
        self.session.preview_key = (str(info.path), time)
        # Grading has to measure the panorama, not two circles in a black frame, so the
        # auto-grade path re-decodes from these rather than from what is on screen.
        self.session.preview_origin = (info.path, time if info.is_video else 0.0, fmt)

        graded = self._regrade(source, payload.get("grade"), PREVIEW_WIDTH, target)
        return {"url": f"/preview/{graded.name}", "media": media_payload(info),
                "view": view}

    def _regrade(self, source: Path, grade_data, width: int, target: Path) -> Path:
        """Apply a grade to an already-decoded frame."""
        grade = ""
        if grade_data:
            grade = Grade(**{k: float(v) for k, v in grade_data.items()
                             if k in Grade.LIMITS}).filter_chain()

        chain = ",".join(filter(None, [grade, f"scale={width}:-2"]))
        result = subprocess.run(
            [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(source), "-vf", chain, "-frames:v", "1", "-q:v", "4", str(target)],
            capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not target.exists():
            raise FFmpegError(f"regrade failed: {result.stderr.strip()}")
        return target

    def _grade_source(self, payload: dict) -> Path:
        """The image the grade controls are acting on.

        In the two-stage flow Capture has no decoded video preview -- it works on an
        extracted frame -- so `frame` names one and it is graded directly. Falls back to
        the decoded panorama for the single-pass path.
        """
        name = str(payload.get("frame") or "")
        project = self.session.project
        if name and project and project.resolved_sources():
            clip = safe_stem(project.resolved_sources()[0].stem)
            path = frames.frames_dir(project.root, clip) / name
            if path.exists():
                proxy = self._derived(
                    f"frame|{path}|{path.stat().st_mtime_ns}|{PREVIEW_WIDTH}", ".jpg")
                return self._scaled(path, PREVIEW_WIDTH, -2, proxy)

        source = self.session.preview_source
        if source is None or not source.exists():
            raise ValueError("load a source first; grading works on what is on screen")
        return source

    def api_preview_grade(self, payload: dict) -> dict:
        """Re-grade the frame already on screen, without touching the video.

        `width` lets the browser ask for a small proxy while a slider is moving and the
        full-size frame when it is released. Grading an extracted frame answers to a
        stable name, so the browser can cache it and going back to a frame is free.
        """
        source = self._grade_source(payload)
        width = max(64, min(int(payload.get("width", PREVIEW_WIDTH)), PREVIEW_WIDTH))
        grade_data = payload.get("grade")

        if payload.get("frame"):
            key = json.dumps(grade_data, sort_keys=True, default=str)
            target = self._derived(f"grade|{source}|{width}|{key}", ".jpg")
            if target.exists():
                return {"url": f"/preview/{target.name}", "width": width}
        else:
            target = self.session.next_name(".jpg")

        graded = self._regrade(source, grade_data, width, target)
        # The grade is applied to the panorama and then follows it onto the lenses, so
        # what the canvas shows is graded the same way whichever view it is in.
        project = self.session.project
        fmt = project.source_format if project else SourceFormat()
        if str(payload.get("view") or "") == "lenses" and not fmt.is_equirect:
            graded = self._as_lenses(
                graded, fmt, width,
                self._derived(f"lensgrade|{graded}|{width}|{fmt.lens_fov:g}", ".jpg"))
        return {"url": f"/preview/{graded.name}", "width": width}

    def api_grade_auto(self, payload: dict) -> dict:
        """Measure the frame on screen and propose a grade for it.

        With a raw source on screen the measurement is taken from the panorama instead:
        two fisheye circles come with black corners, and averaging those in would push
        every exposure decision the wrong way.
        """
        from .. import autograde

        source = self._grade_source(payload)
        origin = self.session.preview_origin
        if not payload.get("frame") and origin and not origin[2].is_equirect:
            path, when, fmt = origin
            source = self._decode_source_frame(
                path, when or None, PREVIEW_WIDTH, fmt, "panorama",
                self.session.next_name(".jpg"))
        grade, analysis = autograde.auto_grade(self.session.ffmpeg, source)
        return {
            "grade": asdict(grade),
            "notes": autograde.describe(analysis, grade),
        }

    def api_camera_preview(self, payload: dict) -> dict:
        """What a single camera sees. In the two-stage flow this projects an extracted
        frame (`frame`) rather than the source video, and can overlay that camera's mask.
        """
        rig = rig_from_payload(payload["rig"])
        name = payload["camera"]
        matches = [c for c in rig.normalized_cameras() if c.name == name]
        if not matches:
            raise RigError(f"no enabled camera named {name!r}")
        camera = matches[0]

        # Resolve the source: an extracted frame from the project, or a source path/time.
        project = self.session.project
        frame_name = payload.get("frame")
        seek = None
        extracted = bool(frame_name and project and project.resolved_sources())
        if extracted:
            clip = safe_stem(project.resolved_sources()[0].stem)
            source_path = frames.frames_dir(project.root, clip) / frame_name
            if not source_path.exists():
                raise ValueError(f"no such frame: {frame_name}")
        else:
            source_path = Path(payload["path"])
            if probe_media(source_path, self.session.ffmpeg).is_video:
                seek = float(payload.get("time", 0.0)) or None

        width = int(payload.get("width", 480))
        aspect = (camera.h_fov / camera.v_fov) if rig.output.auto else rig.output.aspect
        height = max(int(width / aspect), 1)
        tile = self.session.next_name(".jpg")

        argv = [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y"]
        if seek:
            argv += ["-ss", f"{seek:g}"]
        grade = rig.grade.filter_chain()
        # An extracted frame is equirect; a raw source is whatever the project says it
        # is, and v360 cuts the camera straight out of it -- the same one-step
        # projection the extraction uses, so the preview is not a different picture.
        # Lenses that arrive separately are the exception: they have to be assembled
        # first, at roughly the panorama size this tile is a slice of.
        fmt = SourceFormat() if extracted else self._source_format(payload)
        input_name, input_options = fmt.input_spec()
        projection = ":".join([input_name, "rectilinear", *input_options])
        camera_filter = (f"v360={projection}:yaw={camera.yaw:g}:pitch={camera.pitch:g}:"
                         f"roll={camera.roll:g}:h_fov={camera.h_fov:g}:"
                         f"v_fov={camera.v_fov:g}:w={width}:h={height}:"
                         f"interp={rig.output.interp}")
        argv += ["-i", str(source_path)]
        if fmt.needs_graph:
            assembled = max(width * 4, 512)
            chains = fmt.ingest_chains((assembled, assembled // 2), label="src")
            camera_filter = camera_filter.replace(f"v360={projection}",
                                                  "v360=e:rectilinear")
            chains.append(f"[src]{grade + ',' if grade else ''}{camera_filter}[tile]")
            argv += ["-filter_complex", ";".join(chains), "-map", "[tile]"]
        else:
            argv += ["-vf", (f"{grade}," if grade else "") + camera_filter]
        argv += ["-frames:v", "1", "-q:v", "4", str(tile)]
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not tile.exists():
            raise FFmpegError(f"camera preview failed: {result.stderr.strip()}")

        # Overlay the generated mask for this camera+frame, tinted red, when asked.
        mask = None
        if payload.get("overlay") and frame_name and project:
            candidate = (dataset.mask_dir(project.root, clip, name)
                         / f"{Path(frame_name).stem}.png")
            if candidate.exists():
                mask = candidate
        if mask is None:
            return {"url": f"/preview/{tile.name}", "masked": False}

        tinted = self.session.next_name(".jpg")
        # The mask sidecar is at the camera's output size, the tile at preview size, so
        # scale the mask (and the red) to the tile before compositing -- alphamerge needs
        # matching dimensions.
        graph = (
            f"[1:v]format=gray,negate,scale={width}:{height}[m];"
            f"color=red:size={width}x{height},format=rgba[c];"
            "[c][m]alphamerge,colorchannelmixer=aa=0.5[tint];"
            "[0:v][tint]overlay[out]"
        )
        composite = [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error",
                     "-y", "-i", str(tile), "-i", str(mask),
                     "-filter_complex", graph, "-map", "[out]",
                     "-frames:v", "1", "-q:v", "4", str(tinted)]
        result = subprocess.run(composite, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not tinted.exists():
            return {"url": f"/preview/{tile.name}", "masked": False}
        return {"url": f"/preview/{tinted.name}", "masked": True}

    def api_validate(self, payload: dict) -> dict:
        rig = rig_from_payload(payload["rig"])
        source_width = int(payload.get("source_width") or 0)

        # Report the size each camera will really be written at, so the UI can show
        # it rather than making the user infer it from the auto setting.
        sizes = {}
        if source_width:
            fmt = self._source_format(payload)
            fake = MediaInfo(path=Path("."), width=source_width, height=source_width // 2,
                             fps=0.0, duration=0.0, frame_count=1, codec="", is_video=False,
                             video_streams=int(payload.get("source_streams") or 1))
            for camera in rig.normalized_cameras():
                width, height = camera_size(camera, rig, fake, fmt)
                sizes[camera.name] = [width, height]

        return {"ok": True, "warnings": rig.warnings(),
                "enabled": len(rig.enabled_cameras), "sizes": sizes}

    def api_rig_save(self, payload: dict) -> dict:
        rig = rig_from_payload(payload["rig"])
        path = rig.save(payload["path"])
        return {"path": str(path)}

    def api_rig_load(self, payload: dict) -> dict:
        return {"rig": Rig.load(payload["path"]).to_dict()}

    def _presets_payload(self) -> dict:
        """Built-in presets merged with the user's saved ones, for the rig dropdown.

        User names cannot collide with built-ins (save refuses that), so a plain merge
        is unambiguous; `user` tells the UI which ones it may delete.
        """
        presets = {name: factory().to_dict() for name, factory in PRESETS.items()}
        stored = userpresets.stored()
        presets.update(stored)
        return {"presets": presets, "user": sorted(stored)}

    def api_preset_save(self, payload: dict) -> dict:
        name = (payload.get("name") or "").strip()
        if not name:
            raise ValueError("name the preset before saving it")
        if name in PRESETS:
            raise ValueError(f"{name!r} is a built-in preset; choose another name")

        # Validate by building a real rig, then store the normalized form minus the
        # occluders that only make sense inside one project.
        rig = rig_from_payload(payload["rig"]).to_dict()
        rig["occluders"] = [o for o in rig.get("occluders", [])
                            if o.get("type") in PORTABLE_OCCLUDERS]
        userpresets.save(name, rig)
        return self._presets_payload()

    def api_preset_delete(self, payload: dict) -> dict:
        name = (payload.get("name") or "").strip()
        if name in PRESETS:
            raise ValueError(f"{name!r} is a built-in preset and cannot be deleted")
        userpresets.delete(name)
        return self._presets_payload()

    def _model_mtime(self, directory: Path) -> float:
        best = 0.0
        for name in ("points3D.bin", "points3D.txt"):
            candidate = directory / name
            if candidate.exists():
                best = max(best, candidate.stat().st_mtime)
        return best

    def _latest_sparse(self, project: Project) -> Path | None:
        """The most recently written model with points -- a snapshot mid-run, or the
        final sparse/0 -- so the view follows the reconstruction as it builds."""
        sparse = project.root / "sparse"
        candidates = [sparse / "0", sparse / "aligned"]
        snapshots = sparse / "snapshots"
        if snapshots.exists():
            candidates += [d for d in snapshots.iterdir() if d.is_dir()]
        withpoints = [d for d in candidates if self._model_mtime(d) > 0]
        return max(withpoints, key=self._model_mtime) if withpoints else None

    def _serve_reconstruct_points(self, query: dict) -> None:
        """The current sparse cloud as a compact binary: mtime, count, xyz, rgb.

        `?since=<mtime>` gets a 204 when the model has not changed, so polling during a
        run is cheap.
        """
        project = self.session.project
        model_dir = self._latest_sparse(project) if project else None
        mtime = self._model_mtime(model_dir) if model_dir else 0.0
        since = float(query.get("since", ["0"])[0] or 0)
        if model_dir is None or (since and since >= mtime):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        from ..colmap.model import read_points

        try:
            positions, colors = read_points(model_dir, limit=200_000)
        except Exception:
            # The model may be mid-write; skip this poll rather than 500.
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        blob = (struct.pack("<dI", mtime, len(positions))
                + positions.astype("<f4").tobytes()
                + colors.astype("<u1").tobytes())
        self._send(200, blob, "application/octet-stream")

    def api_train_latest(self) -> dict:
        """The newest exported splat, for the Train viewer to load.

        Brush writes one every `--export-every` steps, so this is how the splat can be
        watched taking shape -- and, since a piped Brush says nothing at all, it is most
        of what tells the user anything is happening.
        """
        project = self.session.project
        found = stages.trained_splats(project) if project else []
        if not found:
            return {"splat": None}
        newest = found[0]
        return {
            "splat": project.relative(newest),
            "step": stages.exported_step(newest),
            "mtime": newest.stat().st_mtime,
            "bytes": newest.stat().st_size,
            # Every export, so the cleanup can be pointed at an earlier one without
            # having to have watched the run that produced it.
            "splats": [{"path": str(p), "relative": project.relative(p),
                        "name": p.name, "step": stages.exported_step(p)}
                       for p in found],
        }

    def api_frames_list(self) -> dict:
        """The extracted equirect frames for the open project, for the canvas viewer."""
        project = self.session.project
        if project is None:
            return {"clip": None, "frames": []}
        sources = project.resolved_sources()
        if not sources:
            return {"clip": None, "frames": []}
        clip = safe_stem(sources[0].stem)
        directory = frames.frames_dir(project.root, clip)
        names = sorted(p.name for p in directory.glob("*.jpg")) if directory.exists() else []
        return {"clip": clip, "frames": names}

    def api_frames_remove(self, payload: dict) -> dict:
        """Drop frames from the working set, the export, and the frame store.

        Used by the "nothing was detected here" warning: an all-white mask means the
        frame goes into training unmasked, so the honest options are to re-run detection
        or to throw the frame away. This is the second.
        """
        project = self._open_project()
        sources = project.resolved_sources()
        if not sources:
            raise ValueError("this project has no source")
        clip = safe_stem(sources[0].stem)
        wanted = payload.get("frames") or []
        if not wanted:
            raise ValueError("no frames were named for removal")

        removed = dataset.remove_frames(project.root, clip, wanted)
        # Whatever the mask stage recorded about these frames is now history.
        stage = project.stages.get("mask")
        if stage is not None:
            dropped = {Path(str(name)).stem for name in wanted}
            stage.details["undetected"] = [
                name for name in stage.details.get("undetected", [])
                if name not in dropped]
            project.save()

        remaining = frames.frames_dir(project.root, clip)
        return {
            "frames": removed.frames, "images": removed.images,
            "masks": removed.masks, "mirrored": removed.mirrored,
            "remaining": len(list(remaining.glob("*.jpg"))) if remaining.exists() else 0,
        }

    def api_frames_extract(self, payload: dict) -> dict:
        """Stage A: pull the chosen equirect frames into the project working set.

        This is the first half of the two-stage capture -- decode + frame thinning, no
        rig. The rig and masking are applied later by camera generation, so the frames
        can be re-rigged without decoding the video again.

        It runs on the `start` job, not `capture`: importing a video is Start's work, and
        reporting it as Capture's made that stage look like it had run and finished.
        """
        project = self._open_project()
        sources = project.resolved_sources()
        if not sources:
            raise ValueError("this project has no source to extract frames from")
        selection = FrameSelection(
            mode=payload.get("mode", "sharp"), value=float(payload.get("value", 2.0)),
            start=payload.get("start"), end=payload.get("end"))
        selection.validate()
        # The projection is a property of the footage, so it is stored on the project:
        # every later stage (previews, camera generation) reads it from there.
        if "source_format" in payload:
            project.source_format = SourceFormat.from_dict(payload["source_format"])
            project.source_format.validate()
        source_format = project.source_format
        session = self.session
        job = self.session.jobs["start"]

        def work(running_job) -> dict:
            info = probe_media(sources[0], session.ffmpeg)
            if not source_format.is_equirect:
                running_job.log(f"source is {source_format.label}; "
                                "projecting to equirectangular on the way in")
            result = frames.extract_frames(
                session.ffmpeg, info, selection, project.root,
                on_progress=lambda frac, n, _t: running_job.progress(frac, f"frame {n}"),
                on_analysis=lambda note: running_job.log(note),
                should_cancel=running_job.cancel.is_set,
                source_format=source_format)
            project.frames = FrameSettings(
                mode=selection.mode, value=selection.value,
                start=selection.start, end=selection.end)
            # Frame extraction invalidates any cameras already generated from older frames.
            project.stages.pop("extract", None)
            project.save()
            return {"frames": result.count, "clip": result.clip,
                    "summary": f"{result.count} frames extracted"}

        job.start(work, name="extracting frames")
        return {"started": True, "stage": "start"}

    def api_cameras_generate(self, payload: dict) -> dict:
        """Stage B: project the extracted frames through the rig into camera tiles."""
        project = self._open_project()
        sources = project.resolved_sources()
        if not sources:
            raise ValueError("this project has no source")
        if "rig" in payload:
            project.rig = rig_from_payload(payload["rig"])
        clip = safe_stem(sources[0].stem)
        frames_directory = frames.frames_dir(project.root, clip)
        if not frames_directory.exists():
            raise ValueError("extract frames before generating cameras")
        session = self.session
        rig = project.rig
        detect = project.detect
        # Sky exclusion via the cone until the semantic model lands ("auto" falls back to
        # the cone when there is no model); "off"/"model" pass None for the cone here.
        sky_cone = (detect.sky_cone_angle
                    if detect.exclude_sky and detect.sky_method in ("auto", "cone")
                    else None)
        # Trimming the lens edges is a property of the footage, so it comes from the
        # source format rather than the rig or the detector.
        seam_band = project.source_format.seam_band
        job = self.session.jobs["capture"]

        def work(running_job) -> dict:
            result = cameras.generate_cameras(
                session.ffmpeg, frames_directory, rig, project.root, clip=clip,
                sky_cone_angle=sky_cone, detect=detect, seam_band=seam_band,
                on_progress=lambda frac, n, _t:
                    running_job.progress(0.4 * frac, f"projecting frame {n}"),
                on_mask_progress=lambda frac, n, _t:
                    running_job.progress(0.4 + 0.6 * frac, f"masking frame {n}"),
                should_cancel=running_job.cancel.is_set)
            project.mark_done("extract", images=result.images_written)
            if result.masks_written:
                # mark_done("extract") cascades a reset of mask/export, so record the
                # masks produced here after it, not before.
                project.mark_done("mask", masks=result.masks_written,
                                  undetected=result.undetected)
            project.save()
            if result.undetected:
                running_job.log(
                    f"{len(result.undetected)} frames had no detections at all "
                    f"(their masks are entirely white)", "warn")
            return {"images": result.images_written, "masks": result.masks_written,
                    "undetected": result.undetected,
                    "blank_masks": result.blank_masks,
                    "summary": f"{result.images_written} camera images"}

        job.start(work, name="generating cameras")
        return {"started": True, "stage": "capture"}

    def api_extract(self, payload: dict) -> dict:
        job = self.session.jobs["capture"]
        rig = rig_from_payload(payload["rig"])
        sources = payload["sources"]
        if not sources:
            raise ValueError("no source files selected")
        selection = FrameSelection(
            mode=payload.get("mode", "fps"),
            value=float(payload.get("value", 2.0)),
            start=payload.get("start"),
            end=payload.get("end"),
        )
        selection.validate()
        # With a project open its folder is the destination, so the dataset always
        # lands beside the project.json that describes it. The output_dir field is only
        # a fallback for the project-less path the tests still exercise.
        project = self.session.project
        output_dir = str(project.root) if project else (payload.get("output_dir") or "dataset")
        source_format = self._source_format(payload)
        session = self.session

        def work(running_job) -> dict:
            total = 0
            for index, source in enumerate(sources):
                running_job.raise_if_cancelled()
                info = probe_media(source, session.ffmpeg)
                if selection.mode == "sharp" and info.is_video:
                    running_job.update(message=f"{info.path.name}: analysing sharpness…")

                plan = plan_extraction(
                    info, rig, selection, output_dir,
                    resume=bool(payload.get("resume", True)),
                    ffmpeg=session.ffmpeg,
                    on_analysis=lambda note: running_job.log(note),
                    mask_mode=payload.get("mask_mode", "sidecar"),
                    source_format=source_format,
                )
                if not plan.passes:
                    running_job.log(f"{info.path.name}: already extracted")
                    continue

                def report(progress, index=index, info=info):
                    running_job.progress(
                        (index + progress.fraction) / len(sources),
                        f"pass {progress.pass_index + 1} / {progress.pass_count}"
                        f"  ·  frame {progress.frame}",
                        detail=info.path.name)

                result = run_extraction(plan, session.ffmpeg, on_progress=report)
                total += result.images_written
                running_job.log(f"{info.path.name}: {result.images_written} images")

            # Record the step in the project so the pipeline knows extraction is done
            # -- and with which rig/frames, so it can tell "done" from "stale" later.
            if project is not None:
                project.rig = rig
                project.source_format = source_format
                project.frames = FrameSettings(
                    mode=selection.mode, value=selection.value,
                    start=selection.start, end=selection.end)
                project.output.mask_mode = payload.get(
                    "mask_mode", project.output.mask_mode)
                project.mark_done("extract", images=total)
                project.save()

            return {"images": total,
                    "summary": f"{total} images written to {output_dir}"}

        job.start(work, name="extracting")
        return {"started": True, "stage": "capture"}


def serve(host: str = "127.0.0.1", port: int = 8360, open_browser: bool = True,
          ffmpeg_path: str | None = None, project_path: str | None = None) -> None:
    """Run the UI until interrupted."""
    ffmpeg = resolve_ffmpeg(ffmpeg_path)

    project = None
    if project_path:
        project = Project.load(project_path)
        print(f"project: {project.file}")
    session = Session(ffmpeg, project)

    handler = type("BoundHandler", (Handler,), {"session": session})
    server = ThreadingHTTPServer((host, port), handler)

    url = f"http://{host}:{port}/"
    print(f"360extract UI on {url}")
    print(f"ffmpeg: {ffmpeg.path} ({ffmpeg.version})")
    print("press Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
