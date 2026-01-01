#!/usr/bin/env python3
"""
Bootstrap the panstack repository structure.

Usage:
  python bootstrap_panstack.py
  python bootstrap_panstack.py --force
"""

import argparse
from pathlib import Path

REPO_NAME = "panstack"

DIRS = [
    "src/panstack",
    "scripts",
    "data/bursts",
    "data/outputs",
    "notebooks",
    "tests",
]

FILES = {
    "README.md": (
        "# panstack\n\n"
        "Compute a Pixel-style 'action pan' composite from Sony A6700 burst RAWs (ARW):\n"
        "- choose best base frame (face sharpness)\n"
        "- align frames on background (mask out face)\n"
        "- stack frames for controllable motion blur\n"
        "- freeze the face from the base frame\n\n"
        "## Setup\n\n"
        "```bash\n"
        "python -m venv .venv\n"
        "# Windows:\n"
        ".\\.venv\\Scripts\\activate\n"
        "# macOS/Linux:\n"
        "# source .venv/bin/activate\n\n"
        "pip install -U pip\n"
        "pip install -e .\n"
        "```\n\n"
        "## Data layout\n\n"
        "Put bursts here (not tracked by git):\n"
        "- data/bursts/<session_name>/  (ARW files)\n\n"
        "Outputs go here (not tracked by git):\n"
        "- data/outputs/\n\n"
        "## Run (once CLI is implemented)\n\n"
        "```bash\n"
        "panstack --in data/bursts/<session_name> --out data/outputs/<session_name>.png\n"
        "```\n"
    ),
    ".gitignore": (
        "# Python\n"
        "__pycache__/\n"
        "*.py[cod]\n"
        "*.pyd\n"
        "*.so\n"
        ".venv/\n"
        "venv/\n\n"
        "# OS\n"
        ".DS_Store\n"
        "Thumbs.db\n\n"
        "# IDE\n"
        ".vscode/\n"
        ".idea/\n\n"
        "# Local data\n"
        "data/bursts/\n"
        "data/outputs/\n\n"
        "# Notebooks\n"
        ".ipynb_checkpoints/\n"
    ),
    "pyproject.toml": (
        "[project]\n"
        "name = \"panstack\"\n"
        "version = \"0.0.1\"\n"
        "description = \"Compute controllable action-pan composites from burst RAWs.\"\n"
        "readme = \"README.md\"\n"
        "requires-python = \">=3.10\"\n"
        "dependencies = [\n"
        "  \"rawpy\",\n"
        "  \"opencv-python\",\n"
        "  \"numpy\",\n"
        "  \"mediapipe\",\n"
        "]\n\n"
        "[project.scripts]\n"
        "panstack = \"panstack.cli:main\"\n\n"
        "[build-system]\n"
        "requires = [\"setuptools>=68\"]\n"
        "build-backend = \"setuptools.build_meta\"\n"
    ),
    "src/panstack/__init__.py": (
        "__all__ = [\"__version__\"]\n"
        "__version__ = \"0.0.1\"\n"
    ),
    "src/panstack/cli.py": (
        "import argparse\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(prog=\"panstack\")\n"
        "    parser.add_argument(\"--in\", dest=\"inp\", required=True,\n"
        "                        help=\"Input burst folder containing ARW files\")\n"
        "    parser.add_argument(\"--out\", dest=\"out\", required=True,\n"
        "                        help=\"Output image path (png/jpg)\")\n"
        "    parser.add_argument(\"--k\", type=int, default=8,\n"
        "                        help=\"Blur strength: number of frames around base\")\n"
        "    parser.add_argument(\"--mode\", choices=[\"symmetric\", \"trailing\", \"leading\"],\n"
        "                        default=\"symmetric\", help=\"Frame selection mode\")\n"
        "    parser.add_argument(\"--scale\", type=float, default=0.5,\n"
        "                        help=\"Decode scale for speed (1.0 = full res)\")\n"
        "    args = parser.parse_args()\n\n"
        "    # TODO: implement pipeline.make_panstack(...) and call it here.\n"
        "    raise SystemExit(\"CLI stub. Next: implement src/panstack/pipeline.py and wire it here.\")\n\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n"
    ),
    "src/panstack/pipeline.py": (
        "\"\"\"Pipeline orchestrator.\n\n"
        "Goal:\n"
        "- load burst RAWs\n"
        "- pick base frame (best face sharpness)\n"
        "- align frames on background\n"
        "- stack frames (weighted) to create motion blur background\n"
        "- freeze face from base frame\n"
        "\"\"\"\n\n"
        "def make_panstack(*args, **kwargs):\n"
        "    raise NotImplementedError(\"Implement make_panstack in pipeline.py\")\n"
    ),
    "src/panstack/io_raw.py": (
        "\"\"\"RAW loading utilities (rawpy).\"\"\"\n"
    ),
    "src/panstack/face.py": (
        "\"\"\"Face detection + sharpness scoring + face mask creation.\"\"\"\n"
    ),
    "src/panstack/align.py": (
        "\"\"\"Background alignment utilities.\"\"\"\n"
    ),
    "src/panstack/stack.py": (
        "\"\"\"Frame selection + weighted stacking.\"\"\"\n"
    ),
    "src/panstack/composite.py": (
        "\"\"\"Compositing utilities.\"\"\"\n"
    ),
    "scripts/run_one.ps1": (
        "param(\n"
        "  [Parameter(Mandatory=$true)][string]$In,\n"
        "  [Parameter(Mandatory=$true)][string]$Out\n"
        ")\n"
        "panstack --in $In --out $Out\n"
    ),
    "scripts/run_one.bat": (
        "@echo off\n"
        "REM Example:\n"
        "REM   scripts\\run_one.bat data\\bursts\\session1 data\\outputs\\session1.png\n"
        "panstack --in %1 --out %2\n"
    ),
}

GITKEEP_DIRS = [
    "data/bursts",
    "data/outputs",
]

def write_text(path, content, force):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    path.write_text(content, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    root = Path(".").resolve()

    # Create directories
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)

    # Add .gitkeep for data dirs
    for d in GITKEEP_DIRS:
        write_text(root / d / ".gitkeep", "", args.force)

    # Create files
    for rel, content in FILES.items():
        write_text(root / rel, content, args.force)

    print("OK: panstack repo structure created.")

if __name__ == "__main__":
    main()
