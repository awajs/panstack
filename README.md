# panstack

**panstack** is a computational photography tool for creating Pixel-style *action pan* images from Sony A6700 (or similar) burst RAWs.

It combines:
- burst capture
- background stacking
- optional background alignment
- selective face freezing
- directional motion blur

to produce images where **faces remain sharp** while **time and motion accumulate around them**.

This is not a one-click effect.  
It is a controllable pipeline designed for photographers who want repeatability and intent.

---

## What panstack does

Given a burst of RAW images:

1. Selects a **base frame** automatically (sharpest detected face).
2. Detects **all faces in the base frame**.
3. Lets you choose which faces stay sharp:
   - a specific face (`--freeze-faces 1`)
   - multiple faces (`--freeze-faces 1,2`)
   - all detected faces (`--freeze-faces all`)
4. Builds a **motion-blurred background** by stacking frames:
   - optionally aligned (Pixel-like)
   - or unaligned (raw, chaotic motion)
5. Optionally applies **extra directional motion blur**.
6. Outputs:
   - PNG / TIFF
   - auto-named files
   - sidecar JSON metadata
   - embedded metadata in TIFF files

---

## What panstack deliberately does *not* do

- It does **not** track faces across frames.
- It does **not** do per-pixel optical-flow blur.
- It does **not** try to look “perfect”.

The aesthetic goal is:
> *one frozen moment embedded in flowing time*  
not synthetic realism.

---

## Requirements

- Python **3.10+** (3.11 recommended, **not** RC builds)
- Windows / macOS / Linux
- Camera bursts (Sony ARW tested)
- CPU only (GPU not required)

### Python dependencies

Installed automatically:
- `rawpy`
- `opencv-python`
- `numpy`
- `tifffile`

---

## Installation

```bash
git clone https://github.com/<your-username>/panstack.git
cd panstack

python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# or: source .venv/bin/activate # macOS/Linux

pip install -U pip
pip install -e .
```

---

## Directory structure

```
panstack/
├── src/panstack/
│   ├── pipeline.py
│   ├── cli.py
│   └── models/
│       ├── deploy.prototxt
│       └── res10_300x300_ssd_iter_140000.caffemodel
├── data/
│   ├── bursts/      # input bursts (ignored by git)
│   └── outputs/     # generated images + metadata
└── README.md
```

Place each burst into its own folder:

```
data/bursts/session1/
  DSC08801.ARW
  DSC08802.ARW
  ...
```

---

## Quick start

### 1. Preview detected faces

```bash
panstack --in data/bursts/session1 --out data/outputs --preview-faces
```

Writes:

```
data/outputs/preview_faces.png
```

Faces are numbered starting at **1**.

---

### 2. Generate a Pixel-style pan image

```bash
panstack --in data/bursts/session1 --out data/outputs   --bps 16   --freeze-faces 1   --align-bg on   --k 12 --mode trailing   --extra-blur 16 --blur-angle auto
```

---

### 3. Freeze *all* faces

```bash
panstack --in data/bursts/session1 --out data/outputs   --freeze-faces all   --align-bg on   --k 12 --mode trailing   --extra-blur 14
```

---

## Output behavior

### Auto-naming

If `--out` is a **directory**, panstack generates a unique filename encoding the settings.

### Metadata

For every output:
- A sidecar JSON is written (`output.tif.json`)
- If output is TIFF, the same JSON is embedded in the TIFF **ImageDescription** tag

---

## Key parameters (conceptual)

- `--k`: temporal window (how much time is accumulated)
- `--align-bg`: align background or not
- `--freeze-faces`: which faces stay sharp
- `--extra-blur`: synthetic directional blur
- `--detect-gain`, `--detect-gamma`: detection-only brightening

---

## Philosophy

panstack treats the city as a **force** and people as **anchors**.

It is less about perfect reconstruction and more about:
- attention
- speed
- pressure
- isolation

---

## License

MIT (or your preferred license)
