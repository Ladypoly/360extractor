"""A local web UI for designing rigs and running extractions.

Deliberately built on the standard library: the whole point of the rig editor is
seeing what each camera covers, and that should not require installing a web
framework. Binds to localhost only.
"""

from __future__ import annotations

import base64
import json
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
from ..plan import FrameSelection, camera_size, plan_extraction
from ..project import (
    STAGES,
    DetectSettings,
    FrameSettings,
    OutputSettings,
    Project,
    ProjectError,
)
from ..rig import PRESETS, Camera, Grade, Orientation, Output, Rig, RigError

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
            elif route.path.startswith("/preview/"):
                # Before the generic static handler: these are generated files in the
                # session cache, and `.jpg` would otherwise be looked for in static/.
                name = Path(route.path).name
                self._file(self.session.cache / name, "image/jpeg")
            elif _static_type(route.path):
                self._serve_static(route.path)
            elif route.path == "/api/presets":
                self._json({"presets": {name: factory().to_dict()
                                        for name, factory in PRESETS.items()}})
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
                self._json({"tools": tool_survey()})
            elif route.path == "/api/project":
                project = self.session.project
                self._json({"project": self._project_payload(project)
                            if project else None})
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
            elif route.path == "/api/camera-preview":
                self._json(self.api_camera_preview(payload))
            elif route.path == "/api/rig/validate":
                self._json(self.api_validate(payload))
            elif route.path == "/api/rig/save":
                self._json(self.api_rig_save(payload))
            elif route.path == "/api/rig/load":
                self._json(self.api_rig_load(payload))
            elif route.path == "/api/extract":
                self._json(self.api_extract(payload))
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
            elif route.path == "/api/preview/grade":
                self._json(self.api_preview_grade(payload))
            elif route.path == "/api/grade/auto":
                self._json(self.api_grade_auto(payload))
            elif route.path == "/api/mask/paint":
                self._json(self.api_mask_paint(payload))
            elif route.path == "/api/mask/coverage":
                self._json(self.api_mask_coverage(payload))
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
            "output": asdict(project.output),
            "detect": asdict(project.detect),
            "stages": {name: project.status(name) for name in STAGES},
        }

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
        masked = sum(1 for _ in (project.root / "masks").rglob("*.png")) \
            if (project.root / "masks").exists() else 0

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
        images_root = project.root / "images" / clip
        if not images_root.exists():
            raise ValueError(f"{images_root} does not exist; extract first")

        per_second = project.frames.value if project.frames.mode in {"fps", "sharp"} else 1.0
        start = project.frames.start or 0.0

        entries = {}
        for camera_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
            for image in sorted(camera_dir.iterdir()):
                if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                offset = start + (frame_number(image) - 1) / max(per_second, 1e-6)
                entries[f"{clip}/{camera_dir.name}/{image.name}"] = gps.interpolate(
                    fixes, fixes[0].time + offset)
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
        return {"project": self._project_payload(project)}

    def api_project_open(self, payload: dict) -> dict:
        project = Project.load(payload["path"])
        self.session.project = project
        return {"project": self._project_payload(project)}

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

    def api_mask_coverage(self, payload: dict) -> dict:
        """Measure what each camera actually loses to the occluders.

        Rendered rather than estimated: a painted occluder is an arbitrary shape with
        no closed form, and this uses the same projection the extraction will.
        """
        rig = rig_from_payload(payload["rig"])
        width = int(payload.get("source_width") or 4096)
        height = int(payload.get("source_height") or width // 2)

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

        # Decode the panorama once and keep it ungraded; grading happens from the
        # cache, so moving a slider never re-seeks the video.
        source = self.session.next_name(".jpg")
        argv = [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y"]
        if info.is_video and time > 0:
            argv += ["-ss", f"{time:g}"]
        argv += ["-i", str(info.path), "-vf", f"scale={PREVIEW_WIDTH}:-1",
                 "-frames:v", "1", "-q:v", "3", str(source)]
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not source.exists():
            raise FFmpegError(f"preview failed: {result.stderr.strip()}")

        self.session.preview_source = source
        self.session.preview_key = (str(info.path), time)

        graded = self._regrade(source, payload.get("grade"), PREVIEW_WIDTH, target)
        return {"url": f"/preview/{graded.name}", "media": media_payload(info)}

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

    def api_preview_grade(self, payload: dict) -> dict:
        """Re-grade the frame already on screen, without touching the video.

        `width` lets the browser ask for a small proxy while a slider is moving and the
        full-size frame when it is released.
        """
        source = self.session.preview_source
        if source is None or not source.exists():
            raise ValueError("no preview loaded yet")
        width = max(64, min(int(payload.get("width", PREVIEW_WIDTH)), PREVIEW_WIDTH))
        target = self.session.next_name(".jpg")
        graded = self._regrade(source, payload.get("grade"), width, target)
        return {"url": f"/preview/{graded.name}", "width": width}

    def api_grade_auto(self, payload: dict) -> dict:
        """Measure the frame on screen and propose a grade for it."""
        from .. import autograde

        source = self.session.preview_source
        if source is None or not source.exists():
            raise ValueError("load a source first; auto grades what is on screen")

        grade, analysis = autograde.auto_grade(self.session.ffmpeg, source)
        return {
            "grade": asdict(grade),
            "notes": autograde.describe(analysis, grade),
        }

    def api_camera_preview(self, payload: dict) -> dict:
        """What a single camera actually sees -- the ground truth for the overlay."""
        info = probe_media(payload["path"], self.session.ffmpeg)
        rig = rig_from_payload(payload["rig"])
        name = payload["camera"]

        matches = [c for c in rig.normalized_cameras() if c.name == name]
        if not matches:
            raise RigError(f"no enabled camera named {name!r}")
        camera = matches[0]

        time = float(payload.get("time", 0.0))
        width = int(payload.get("width", 480))
        # Match the aspect the camera will actually be written at, or the preview
        # misrepresents the framing it exists to show.
        aspect = (camera.h_fov / camera.v_fov) if rig.output.auto else rig.output.aspect
        height = max(int(width / aspect), 1)
        target = self.session.next_name(".jpg")

        argv = [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y"]
        if info.is_video and time > 0:
            argv += ["-ss", f"{time:g}"]
        # Grade first, exactly as the extraction does.
        grade = rig.grade.filter_chain()
        argv += [
            "-i", str(info.path),
            "-vf", (f"{grade}," if grade else "")
                   + (f"v360=e:rectilinear:yaw={camera.yaw:g}:pitch={camera.pitch:g}:"
                      f"roll={camera.roll:g}:h_fov={camera.h_fov:g}:v_fov={camera.v_fov:g}:"
                      f"w={width}:h={height}:interp={rig.output.interp}"),
            "-frames:v", "1", "-q:v", "4", str(target),
        ]
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not target.exists():
            raise FFmpegError(f"camera preview failed: {result.stderr.strip()}")

        return {"url": f"/preview/{target.name}"}

    def api_validate(self, payload: dict) -> dict:
        rig = rig_from_payload(payload["rig"])
        source_width = int(payload.get("source_width") or 0)

        # Report the size each camera will really be written at, so the UI can show
        # it rather than making the user infer it from the auto setting.
        sizes = {}
        if source_width:
            fake = MediaInfo(path=Path("."), width=source_width, height=source_width // 2,
                             fps=0.0, duration=0.0, frame_count=1, codec="", is_video=False)
            for camera in rig.normalized_cameras():
                width, height = camera_size(camera, rig, fake)
                sizes[camera.name] = [width, height]

        return {"ok": True, "warnings": rig.warnings(),
                "enabled": len(rig.enabled_cameras), "sizes": sizes}

    def api_rig_save(self, payload: dict) -> dict:
        rig = rig_from_payload(payload["rig"])
        path = rig.save(payload["path"])
        return {"path": str(path)}

    def api_rig_load(self, payload: dict) -> dict:
        return {"rig": Rig.load(payload["path"]).to_dict()}

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
        output_dir = payload.get("output_dir") or "dataset"
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
