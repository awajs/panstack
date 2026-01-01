# panstack

Compute a Pixel-style 'action pan' composite from Sony A6700 burst RAWs (ARW):
- choose best base frame (face sharpness)
- align frames on background (mask out face)
- stack frames for controllable motion blur
- freeze the face from the base frame

## Setup

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -U pip
pip install -e .
```

## Data layout

Put bursts here (not tracked by git):
- data/bursts/<session_name>/  (ARW files)

Outputs go here (not tracked by git):
- data/outputs/

## Run (once CLI is implemented)

```bash
panstack --in data/bursts/<session_name> --out data/outputs/<session_name>.png
```
