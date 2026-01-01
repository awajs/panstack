from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rawpy

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------
Box = Tuple[int, int, int, int]  # x, y, w, h


# -----------------------------------------------------------------------------
# Robust color helpers
# -----------------------------------------------------------------------------
def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel BGR."""
    if img is None:
        raise ValueError("ensure_bgr() got None")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unexpected image shape in ensure_bgr: {img.shape}")


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale uint8. Works for uint8/uint16/float.
    ORB requires uint8.
    """
    if img is None:
        raise ValueError("to_gray_u8() got None")

    # Convert to gray (keeping dtype)
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] == 1:
        g = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unexpected image shape in to_gray_u8: {img.shape}")

    # Convert to uint8 safely
    if g.dtype == np.uint8:
        return g
    if g.dtype == np.uint16:
        return (g >> 8).astype(np.uint8)
    # float or other integer types
    g = np.clip(g, 0, 255).astype(np.uint8) if np.issubdtype(g.dtype, np.floating) else g
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return g


def bgr_to_u8(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to uint8 BGR (for face detection & ORB), preserving shape."""
    bgr = ensure_bgr(bgr)
    if bgr.dtype == np.uint8:
        return bgr
    if bgr.dtype == np.uint16:
        return (bgr >> 8).astype(np.uint8)
    if np.issubdtype(bgr.dtype, np.floating):
        return np.clip(bgr, 0, 255).astype(np.uint8)
    return cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# -----------------------------------------------------------------------------
# Face detection (OpenCV DNN)
# -----------------------------------------------------------------------------
_FACE_PROTO = (Path(__file__).parent / "models" / "deploy.prototxt").resolve()
_FACE_MODEL = (Path(__file__).parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel").resolve()

if not _FACE_PROTO.exists():
    raise FileNotFoundError(f"Missing face prototxt: {_FACE_PROTO}")
if not _FACE_MODEL.exists():
    raise FileNotFoundError(f"Missing face caffemodel: {_FACE_MODEL}")

_face_net = cv2.dnn.readNetFromCaffe(str(_FACE_PROTO), str(_FACE_MODEL))


def detect_face_box(bgr: np.ndarray, conf_thresh: float = 0.5) -> Optional[Box]:
    """
    Return best face bounding box (x,y,w,h) or None.
    Input should be uint8 BGR for best results.
    """
    bgr = bgr_to_u8(bgr)
    h, w = bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    _face_net.setInput(blob)
    det = _face_net.forward()

    best_box: Optional[Box] = None
    best_conf = conf_thresh

    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf <= best_conf:
            continue

        box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 > x1 and y2 > y1:
            best_box = (x1, y1, x2 - x1, y2 - y1)
            best_conf = conf

    return best_box


# -----------------------------------------------------------------------------
# RAW loading
# -----------------------------------------------------------------------------
def load_arw_bgr(path: str, scale: float = 0.5, bps: int = 8) -> np.ndarray:
    """
    Load Sony ARW using rawpy and return BGR image (uint8 if bps=8, uint16 if bps=16).
    scale < 1.0 downscales for speed.
    """
    if bps not in (8, 16):
        raise ValueError("bps must be 8 or 16")

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=bps,
            gamma=(1, 1),
            user_flip=0,
        )

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = ensure_bgr(bgr)

    if scale != 1.0:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return bgr


# -----------------------------------------------------------------------------
# Face sharpness + face mask
# -----------------------------------------------------------------------------
def face_sharpness(bgr_u8: np.ndarray, box: Optional[Box]) -> float:
    """Laplacian variance in face ROI on uint8 proxy."""
    if box is None:
        return 0.0
    bgr_u8 = bgr_to_u8(bgr_u8)
    x, y, w, h = box
    roi = bgr_u8[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    g = to_gray_u8(roi)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def face_mask(shape: Tuple[int, int, int], box: Optional[Box], feather: float = 35.0, expand: float = 0.40) -> np.ndarray:
    """Feathered expanded face-box mask. Returns (H,W,1) float in [0,1]."""
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.float32)
    if box is None:
        return m[..., None]

    x, y, bw, bh = box
    pad = int(expand * max(bw, bh))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)

    m[y0:y1, x0:x1] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), feather)
    return np.clip(m, 0.0, 1.0)[..., None]


# -----------------------------------------------------------------------------
# Alignment on background (ORB on uint8 proxies)
# -----------------------------------------------------------------------------
def _mask_out_face(shape: Tuple[int, int, int], box: Optional[Box]) -> np.ndarray:
    """Mask for ORB feature detection: 255 usable, 0 face region."""
    h, w = shape[:2]
    m = np.ones((h, w), dtype=np.uint8) * 255
    if box is None:
        return m
    x, y, bw, bh = box
    pad = int(0.25 * max(bw, bh))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    m[y0:y1, x0:x1] = 0
    return m


def estimate_affine_orb(base_u8: np.ndarray, img_u8: np.ndarray, base_face: Optional[Box], img_face: Optional[Box]) -> Optional[np.ndarray]:
    """
    Estimate affine mapping img->base using ORB+RANSAC on uint8 grayscale.
    """
    base_u8 = bgr_to_u8(base_u8)
    img_u8 = bgr_to_u8(img_u8)

    g1 = to_gray_u8(base_u8)
    g2 = to_gray_u8(img_u8)

    m1 = _mask_out_face(base_u8.shape, base_face)
    m2 = _mask_out_face(img_u8.shape, img_face)

    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(g1, m1)
    k2, d2 = orb.detectAndCompute(g2, m2)
    if d1 is None or d2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:400]
    if len(matches) < 30:
        return None

    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])

    M, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return M


def warp_affine(img: np.ndarray, M: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Warp supports uint8/uint16."""
    h, w = shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# -----------------------------------------------------------------------------
# Frame selection + weighting
# -----------------------------------------------------------------------------
def select_indices(n: int, base_idx: int, k: int, mode: str) -> List[int]:
    if mode == "symmetric":
        lo, hi = max(0, base_idx - k), min(n, base_idx + k + 1)
        return list(range(lo, hi))
    if mode == "trailing":
        lo, hi = max(0, base_idx - 2 * k), base_idx + 1
        return list(range(lo, hi))
    if mode == "leading":
        lo, hi = base_idx, min(n, base_idx + 2 * k + 1)
        return list(range(lo, hi))
    raise ValueError(f"Unknown mode: {mode}")


def gaussian_weight(i: int, base_idx: int, k: int, sigma: float = 0.7) -> float:
    d = abs(i - base_idx)
    s = sigma * k + 1e-6
    return math.exp(-(d * d) / (2 * s * s))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def make_panstack(
    in_dir: str,
    out_path: str,
    k: int = 8,
    mode: str = "symmetric",
    scale: float = 0.5,
    weight_sigma: float = 0.7,
    face_conf: float = 0.5,
    bps: int = 8,
) -> Dict[str, object]:
    """
    Create a panstack composite.

    - If bps=8: everything runs on uint8.
    - If bps=16: we still do face detection & ORB alignment on uint8 proxies,
      but we warp/stack/composite the uint16 images for final output.

    Returns info dict.
    """
    in_dir_p = Path(in_dir)
    paths = sorted(glob.glob(str(in_dir_p / "*.ARW"))) + sorted(glob.glob(str(in_dir_p / "*.arw")))
    if not paths:
        raise FileNotFoundError(f"No ARW files found in: {in_dir}")

    # Load render frames (uint8 or uint16)
    frames: List[np.ndarray] = [load_arw_bgr(p, scale=scale, bps=bps) for p in paths]

    # Build uint8 proxies for detection & alignment (always)
    frames_u8: List[np.ndarray] = [bgr_to_u8(f) for f in frames]

    # Face boxes and sharpness computed on proxies
    face_boxes: List[Optional[Box]] = [detect_face_box(f, conf_thresh=face_conf) for f in frames_u8]
    sharp_scores: List[float] = [face_sharpness(f, b) for f, b in zip(frames_u8, face_boxes)]

    base_idx = int(np.argmax(sharp_scores))
    base = frames[base_idx]          # render base (uint8 or uint16)
    base_u8 = frames_u8[base_idx]    # proxy base
    base_face = face_boxes[base_idx]

    idxs = select_indices(len(frames), base_idx, k, mode)

    acc = np.zeros_like(base, dtype=np.float32)
    wsum = 0.0
    used = 0

    for i in idxs:
        M = estimate_affine_orb(base_u8, frames_u8[i], base_face, face_boxes[i])
        if M is None:
            continue

        fw = warp_affine(frames[i], M, base.shape)  # warp render frame
        w = gaussian_weight(i, base_idx, k, sigma=weight_sigma)
        acc += fw.astype(np.float32) * w
        wsum += w
        used += 1

    if used == 0:
        raise RuntimeError("Alignment failed for all frames. Try different burst or adjust k/mode/face_conf.")

    bg = (acc / max(wsum, 1e-6)).astype(base.dtype)

    m = face_mask(base.shape, base_face, feather=35.0, expand=0.40)
    out = (base.astype(np.float32) * m + bg.astype(np.float32) * (1.0 - m)).astype(base.dtype)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(out_p), out)
    if not ok:
        raise RuntimeError(f"Failed to write output: {out_p}")

    return {
        "base_idx": base_idx,
        "base_file": Path(paths[base_idx]).name,
        "frames_total": len(frames),
        "frames_used": used,
        "mode": mode,
        "k": k,
        "scale": scale,
        "bps": bps,
        "out_path": str(out_p),
    }
