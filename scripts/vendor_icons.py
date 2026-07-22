"""Vendor the Lucide icons the UI uses into one JS module.

Only the icons actually referenced are copied, and only their inner markup, so the
application carries a couple of kilobytes rather than a dependency and a build step.

Run from this directory after `npm install lucide-static`:

    npm install lucide-static
    python scripts/vendor_icons.py

Lucide is ISC licensed: https://lucide.dev
"""
import re
from pathlib import Path

SOURCE = Path(__file__).parent / "node_modules" / "lucide-static" / "icons"
TARGET = Path(__file__).parents[1] / "src" / "threesixty" / "web" / "static" / "js" / "icons.js"

WANTED = {
    # pipeline stages
    "camera": "video",
    "refine": "scan-eye",
    "reconstruct": "box",
    "train": "sparkles",
    "inspect": "eye",
    # stage states
    "pending": "circle",
    "ready": "circle-dashed",
    "running": "loader-circle",
    "done": "circle-check",
    "stale": "triangle-alert",
    "error": "circle-x",
    "disabled": "ban",
    # actions and chrome
    "play": "play",
    "stop": "square",
    "cancel": "x",
    "folder": "folder-open",
    "save": "save",
    "settings": "settings",
    "system": "activity",
    "chevron": "chevron-down",
    "copy": "copy",
    "pause": "pause",
    "image": "image",
    "layers": "layers",
    "wand": "wand-sparkles",
    "brush": "paintbrush",
    "eraser": "eraser",
    "trash": "trash-2",
    "download": "download",
    "refresh": "refresh-cw",
    "info": "info",
    "film": "film",
    "gauge": "gauge",
    "sun": "sun",
    "crosshair": "crosshair",
    "arrow-right": "arrow-right",
}

INNER = re.compile(r"<svg[^>]*>(.*)</svg>", re.S)

entries = []
missing = []
for name, filename in sorted(WANTED.items()):
    path = SOURCE / f"{filename}.svg"
    if not path.exists():
        missing.append(f"{name} ({filename})")
        continue
    match = INNER.search(path.read_text(encoding="utf-8"))
    if not match:
        missing.append(f"{name} (unparsable)")
        continue
    body = re.sub(r"\s+", " ", match.group(1)).strip()
    entries.append(f'  "{name}": \'{body}\',')

if missing:
    print("MISSING:", ", ".join(missing))

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(
    '// Lucide icons, vendored.\n'
    '//\n'
    '// Only the icons this interface uses, and only their inner markup, so there is no\n'
    '// runtime dependency, no build step and nothing to fetch. Regenerate with\n'
    '// scripts/vendor_icons.py if a new icon is needed.\n'
    '//\n'
    '// Lucide is ISC licensed: https://lucide.dev\n\n'
    'const PATHS = {\n' + "\n".join(entries) + '\n};\n\n'
    'const ATTRS = \'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" \'\n'
    '  + \'stroke="currentColor" stroke-width="2" stroke-linecap="round" \'\n'
    '  + \'stroke-linejoin="round"\';\n\n'
    '/** Inline SVG markup for one icon, or an empty string if it is unknown. */\n'
    'export function icon(name, { size = 16, className = "" } = {}) {\n'
    '  const body = PATHS[name];\n'
    '  if (!body) return "";\n'
    '  const cls = className ? ` class="${className}"` : "";\n'
    '  return `<svg ${ATTRS} width="${size}" height="${size}"${cls}'
    ' aria-hidden="true">${body}</svg>`;\n'
    '}\n\n'
    'export function hasIcon(name) {\n'
    '  return Boolean(PATHS[name]);\n'
    '}\n\n'
    'export const ICON_NAMES = Object.keys(PATHS);\n',
    encoding="utf-8", newline="\n")

print(f"vendored {len(entries)} icons -> {TARGET.name} ({TARGET.stat().st_size} bytes)")
