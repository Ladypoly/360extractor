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
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import picker
from ..mask import geometric
from ..extract import run_extraction
from ..ffmpeg import FFmpegError, MediaInfo, probe_media, resolve_ffmpeg
from ..plan import FrameSelection, camera_size, plan_extraction
from ..rig import PRESETS, Camera, Orientation, Output, Rig, RigError

STATIC = Path(__file__).parent / "static"
PREVIEW_WIDTH = 1600


class Job:
    """A running extraction, polled by the browser."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = "idle"  # idle | running | done | error | cancelled
        self.message = ""
        self.fraction = 0.0
        self.images = 0
        self.thread: threading.Thread | None = None
        self.cancel = threading.Event()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "message": self.message,
                "fraction": self.fraction,
                "images": self.images,
            }

    def update(self, **fields) -> None:
        with self.lock:
            for key, value in fields.items():
                setattr(self, key, value)


class Session:
    """Shared server state: the resolved ffmpeg, preview cache, current job."""

    def __init__(self, ffmpeg) -> None:
        self.ffmpeg = ffmpeg
        self.cache = Path(tempfile.mkdtemp(prefix="360extract-ui-"))
        self.job = Job()
        self.counter = 0
        self.lock = threading.Lock()

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
            elif route.path.endswith(".js"):
                # Served as a module by index.html; the wrong MIME type makes the
                # browser refuse the import outright.
                name = Path(route.path).name
                self._file(STATIC / name, "text/javascript; charset=utf-8")
            elif route.path == "/api/presets":
                self._json({"presets": {name: factory().to_dict()
                                        for name, factory in PRESETS.items()}})
            elif route.path == "/api/progress":
                self._json(self.session.job.snapshot())
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
            elif route.path == "/api/mask/paint":
                self._json(self.api_mask_paint(payload))
            elif route.path == "/api/mask/coverage":
                self._json(self.api_mask_coverage(payload))
            elif route.path == "/api/pick":
                self._json(self.api_pick(payload))
            elif route.path == "/api/cancel":
                self.session.job.cancel.set()
                self._json({"ok": True})
            else:
                self._json({"error": "no such endpoint"}, 404)
        except (FFmpegError, RigError, ValueError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": str(exc)}, 500)

    # -- endpoints ----------------------------------------------------------

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
        """One equirect frame, downscaled, for the rig editor canvas."""
        info = probe_media(payload["path"], self.session.ffmpeg)
        time = float(payload.get("time", 0.0))
        target = self.session.next_name(".jpg")

        argv = [str(self.session.ffmpeg.path), "-hide_banner", "-loglevel", "error", "-y"]
        if info.is_video and time > 0:
            argv += ["-ss", f"{time:g}"]
        argv += ["-i", str(info.path), "-vf", f"scale={PREVIEW_WIDTH}:-1",
                 "-frames:v", "1", "-q:v", "4", str(target)]
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not target.exists():
            raise FFmpegError(f"preview failed: {result.stderr.strip()}")

        return {"url": f"/preview/{target.name}", "media": media_payload(info)}

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
        argv += [
            "-i", str(info.path),
            "-vf", (f"v360=e:rectilinear:yaw={camera.yaw:g}:pitch={camera.pitch:g}:"
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
        job = self.session.job
        if job.state == "running":
            raise ValueError("an extraction is already running")

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

        def work() -> None:
            job.cancel.clear()
            job.update(state="running", message="starting", fraction=0.0, images=0)
            total = 0
            try:
                for index, source in enumerate(sources):
                    if job.cancel.is_set():
                        job.update(state="cancelled", message="cancelled")
                        return
                    info = probe_media(source, session.ffmpeg)
                    if selection.mode == "sharp" and info.is_video:
                        job.update(message=f"{info.path.name}: analysing sharpness…")
                    plan = plan_extraction(
                        info, rig, selection, output_dir,
                        resume=bool(payload.get("resume", True)),
                        ffmpeg=session.ffmpeg,
                        on_analysis=lambda note: job.update(message=note),
                        mask_mode=payload.get("mask_mode", "sidecar"),
                    )
                    if not plan.passes:
                        job.update(message=f"{info.path.name}: already extracted")
                        continue

                    def report(progress, index=index, info=info):
                        if job.cancel.is_set():
                            raise KeyboardInterrupt
                        overall = (index + progress.fraction) / len(sources)
                        job.update(
                            fraction=overall,
                            message=f"{info.path.name}: pass "
                                    f"{progress.pass_index + 1}/{progress.pass_count}, "
                                    f"frame {progress.frame}",
                        )

                    result = run_extraction(plan, session.ffmpeg, on_progress=report)
                    if result.cancelled:
                        job.update(state="cancelled", message="cancelled")
                        return
                    total += result.images_written
                    job.update(images=total)

                job.update(state="done", fraction=1.0,
                           message=f"{total} images written to {output_dir}")
            except KeyboardInterrupt:
                job.update(state="cancelled", message="cancelled")
            except Exception as exc:
                traceback.print_exc()
                job.update(state="error", message=str(exc))

        job.thread = threading.Thread(target=work, daemon=True)
        job.thread.start()
        return {"started": True}


def serve(host: str = "127.0.0.1", port: int = 8360, open_browser: bool = True,
          ffmpeg_path: str | None = None) -> None:
    """Run the UI until interrupted."""
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    session = Session(ffmpeg)

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
