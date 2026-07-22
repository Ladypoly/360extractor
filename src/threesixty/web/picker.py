"""Native open/save dialogs for the UI.

A browser can never hand the server a real filesystem path -- `<input type=file>`
deliberately hides it -- so a local tool that wants a normal "Browse..." button has
to raise the dialog itself.

Tk is run in a *subprocess* rather than in the server's thread: Tk demands to own the
main thread, and creating windows from a request-handler thread deadlocks or crashes
depending on the platform. One short-lived process per dialog sidesteps all of it.
"""

from __future__ import annotations

import json
import subprocess
import sys

MEDIA_TYPES = [
    ("360 media", "*.mp4 *.mov *.mkv *.insv *.jpg *.jpeg *.png *.tif *.tiff"),
    ("Video", "*.mp4 *.mov *.mkv *.insv *.webm *.avi"),
    ("Images", "*.jpg *.jpeg *.png *.tif *.tiff"),
    ("All files", "*.*"),
]
RIG_TYPES = [("Rig files", "*.json"), ("All files", "*.*")]


def _child(mode: str, title: str, filetypes, initial: str) -> None:
    """Runs inside the subprocess: show one dialog, print the result as JSON."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    # Without this the dialog can open behind the browser and look like a hang.
    root.attributes("-topmost", True)

    common = {"title": title, "initialdir": initial or None}
    if mode == "directory":
        result = filedialog.askdirectory(**common, mustexist=False)
    elif mode == "save":
        result = filedialog.asksaveasfilename(
            **common, filetypes=filetypes, defaultextension=".json")
    else:
        result = filedialog.askopenfilenames(**common, filetypes=filetypes)

    root.destroy()
    if isinstance(result, str):
        paths = [result] if result else []
    else:
        paths = list(result)
    print(json.dumps({"paths": paths}))


def ask(mode: str = "open", title: str = "Select", kind: str = "media",
        initial: str = "", timeout: int = 300) -> list[str]:
    """Show a dialog and return the chosen paths, or [] if cancelled."""
    filetypes = RIG_TYPES if kind == "rig" else MEDIA_TYPES
    argv = [sys.executable, "-c",
            "import sys, json;"
            "sys.path.insert(0, sys.argv[1]);"
            "from threesixty.web.picker import _child;"
            "_child(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]), sys.argv[5])",
            str(_package_root()), mode, title, json.dumps(filetypes), initial]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        raise RuntimeError(
            "could not open a file dialog"
            + (f": {proc.stderr.strip()}" if proc.stderr.strip() else "")
            + ". Type the path into the field instead."
        )
    try:
        return json.loads(proc.stdout.strip() or '{"paths": []}')["paths"]
    except (json.JSONDecodeError, KeyError):
        return []


def _package_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def available() -> bool:
    """Is Tk importable? Some slim Python builds ship without it."""
    try:
        import tkinter  # noqa: F401
        return True
    except Exception:
        return False
