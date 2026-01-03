from __future__ import annotations

import datetime
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rawpy

Box = Tuple[int, int, int, int]
_ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# -----------------------------------------------------------------------------
# Debug overlay helpers
# -----------------------------------------------------------------------------
def _draw_box(img, box: Box, label: str, color=(0, 255, 255), thickness=2):
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(
            img,
            label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_arrow(img, origin_xy, dx, dy, label="", color=(0, 255, 0), thickness=2, scale=6.0):
    ox, oy = int(origin_xy[0]), int(origin_xy[1])
    ex = int(ox + dx * scale)
    ey = int(oy + dy * scale)
    cv2.arrowedLine(img, (ox, oy), (ex, ey), color, thickness, tipLength=0.25)
    if label:
        cv2.putText(img, label, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _default_debug_path(out_path: str, base_file: str) -> Path:
    p = Path(out_path)
    if p.suffix.lower() in _ALLOWED_EXTS:
        return p.with_suffix(".debug.png")
    return Path(out_path) / f"{Path(base_file).stem}.debug.png"


# -----------------------------------------------------------------------------
# JSON safety
# -----------------------------------------------------------------------------
def to_builtin(x):
    """Recursively convert numpy/scalars/arrays into JSON-safe Python builtins."""
    import numpy as _np

    if isinstance(x, _np.generic):
        return x.item()
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    return x


# -----------------------------------------------------------------------------
# Color / dtype helpers
# -----------------------------------------------------------------------------
def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def bgr_to_u8(bgr: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr(bgr)
    if bgr.dtype == np.uint8:
        return bgr
    if bgr.dtype == np.uint16:
        return (bgr >> 8).astype(np.uint8)
    if np.issubdtype(bgr.dtype, np.floating):
        return (
            np.clip(bgr * 255.0, 0, 255).astype(np.uint8)
            if bgr.max() <= 1.0
            else np.clip(bgr, 0, 255).astype(np.uint8)
        )
    return cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif img.ndim == 3 and img.shape[2] == 1:
        g = img[:, :, 0]
    else:
        raise ValueError(f"Unexpected gray conversion shape: {img.shape}")

    if g.dtype == np.uint8:
        return g
    if g.dtype == np.uint16:
        return (g >> 8).astype(np.uint8)
    if np.issubdtype(g.dtype, np.floating):
        return (
            np.clip(g * 255.0, 0, 255).astype(np.uint8)
            if g.max() <= 1.0
            else np.clip(g, 0, 255).astype(np.uint8)
        )
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def brighten_for_detect(bgr_u8: np.ndarray, gain: float = 1.6, gamma: float = 0.6, clahe: bool = False) -> np.ndarray:
    bgr_u8 = bgr_to_u8(bgr_u8)
    x = bgr_u8.astype(np.float32) / 255.0
    x = np.clip(x * max(gain, 0.0), 0.0, 1.0)
    x = np.power(x, max(gamma, 1e-6))
    y = (x * 255.0).astype(np.uint8)

    if clahe:
        lab = cv2.cvtColor(y, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = c.apply(l)
        y = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    return y


def linear_to_srgb(linear01: np.ndarray) -> np.ndarray:
    x = np.clip(linear01, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(x, 1 / 2.4) - a)


# -----------------------------------------------------------------------------
# Segmentation (MediaPipe Tasks ImageSegmenter)
# -----------------------------------------------------------------------------
_MP_TASKS_AVAILABLE = False
_MP_TASKS_IMPORT_ERROR = ""


def _soften_mask(mask01: np.ndarray, feather_px: float = 25.0) -> np.ndarray:
    m = np.clip(mask01.astype(np.float32), 0.0, 1.0)
    if feather_px and feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), float(feather_px))
    return np.clip(m, 0.0, 1.0)


def _mp_tasks_import():
    global _MP_TASKS_AVAILABLE, _MP_TASKS_IMPORT_ERROR
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore

        _MP_TASKS_AVAILABLE = True
        _MP_TASKS_IMPORT_ERROR = ""
        return mp, mp_python, mp_vision
    except Exception as e:
        _MP_TASKS_AVAILABLE = False
        _MP_TASKS_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        return None, None, None


def person_segmentation_mask_u8(
    bgr_u8: np.ndarray,
    thresh: float = 0.5,
    erode_px: int = 0,
    dilate_px: int = 0,
    feather_px: float = 25.0,
    model_path: Optional[Path] = None,
) -> Tuple[Optional[np.ndarray], str]:
    mp, mp_python, mp_vision = _mp_tasks_import()
    if mp is None:
        return None, f"MediaPipe Tasks import failed: {_MP_TASKS_IMPORT_ERROR}"

    if model_path is None:
        model_path = (Path(__file__).parent / "models" / "selfie_multiclass_256x256.tflite").resolve()
    if not model_path.exists():
        return None, f"Segmentation model not found: {model_path}"

    img = bgr_to_u8(bgr_u8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        with mp_vision.ImageSegmenter.create_from_options(options) as segmenter:
            result = segmenter.segment(mp_image)
            if result is None or result.category_mask is None:
                return None, "segmenter.segment returned no category_mask"
            cat = result.category_mask.numpy_view()
    except Exception as e:
        return None, f"segmenter failed: {type(e).__name__}: {e}"

    m_bin = (cat != 0).astype(np.uint8) * 255

    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m_bin = cv2.erode(m_bin, k, iterations=1)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m_bin = cv2.dilate(m_bin, k, iterations=1)

    m01 = (m_bin.astype(np.float32) / 255.0)
    m01 = _soften_mask(m01, feather_px=feather_px)
    m01 = (m01 >= float(thresh)).astype(np.float32)
    m01 = _soften_mask(m01, feather_px=max(1.0, float(feather_px) * 0.5))
    return m01[..., None], ""


# -----------------------------------------------------------------------------
# Face detection (OpenCV DNN)
# -----------------------------------------------------------------------------
_FACE_PROTO = (Path(__file__).parent / "models" / "deploy.prototxt").resolve()
_FACE_MODEL = (Path(__file__).parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel").resolve()

_face_net = None
_face_load_error = ""
if _FACE_PROTO.exists() and _FACE_MODEL.exists():
    try:
        _face_net = cv2.dnn.readNetFromCaffe(str(_FACE_PROTO), str(_FACE_MODEL))
    except cv2.error as e:
        _face_net = None
        _face_load_error = f"{type(e).__name__}: {e}"
else:
    _face_net = None
    _face_load_error = f"Missing face models: {_FACE_PROTO.name}, {_FACE_MODEL.name}"


def detect_face_boxes(bgr_u8: np.ndarray, conf_thresh: float = 0.5) -> List[Tuple[Box, float]]:
    if _face_net is None:
        return []
    bgr_u8 = bgr_to_u8(bgr_u8)
    h, w = bgr_u8.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr_u8, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    _face_net.setInput(blob)
    det = _face_net.forward()

    out: List[Tuple[Box, float]] = []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            out.append(((x1, y1, x2 - x1, y2 - y1), conf))
    return out


def sort_boxes_for_ids(boxes: List[Tuple[Box, float]]) -> List[Box]:
    return [b for (b, c) in sorted(boxes, key=lambda bc: (bc[0][2] * bc[0][3], bc[1]), reverse=True)]


# -----------------------------------------------------------------------------
# RAW loading
# -----------------------------------------------------------------------------
def load_arw_bgr_u16(path: str, scale: float = 0.5) -> np.ndarray:
    with rawpy.imread(path) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            user_flip=0,
        )
    bgr16 = cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
    bgr16 = ensure_bgr(bgr16)
    if scale != 1.0:
        h, w = bgr16.shape[:2]
        bgr16 = cv2.resize(bgr16, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr16


def u16_to_linear01(bgr_u16: np.ndarray) -> np.ndarray:
    out = ensure_bgr(bgr_u16).astype(np.float32)
    out *= (1.0 / 65535.0)
    return out


# -----------------------------------------------------------------------------
# Mask helpers
# -----------------------------------------------------------------------------
def region_box_from_face(face_box: Box, shape_hw3: Tuple[int, int, int], region: str) -> Box:
    h_img, w_img = shape_hw3[:2]
    x, y, w, h = face_box
    if region == "face":
        x0, y0, x1, y1 = x, y, x + w, y + h
    elif region == "upper":
        x0 = x - int(0.30 * w)
        y0 = y - int(0.20 * h)
        x1 = x + w + int(0.30 * w)
        y1 = y + int(4.50 * h)
    elif region == "full":
        x0 = x - int(0.60 * w)
        y0 = y - int(0.20 * h)
        x1 = x + w + int(0.60 * w)
        y1 = y + int(7.50 * h)
    else:
        raise ValueError(region)

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w_img, x1)
    y1 = min(h_img, y1)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def boxes_union_mask(shape_hw3: Tuple[int, int, int], boxes: List[Box], feather: float = 35.0) -> np.ndarray:
    h, w = shape_hw3[:2]
    m = np.zeros((h, w), dtype=np.float32)
    for (x, y, bw, bh) in boxes:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + bw)
        y1 = min(h, y + bh)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), feather)
    return np.clip(m, 0.0, 1.0)[..., None]


def select_face_boxes(face_boxes_sorted: List[Box], freeze_faces: Union[str, List[int]]) -> List[Box]:
    if not face_boxes_sorted:
        return []
    if freeze_faces == "all":
        return face_boxes_sorted
    chosen: List[Box] = []
    for fid in freeze_faces:
        j = fid - 1
        if 0 <= j < len(face_boxes_sorted):
            chosen.append(face_boxes_sorted[j])
    if not chosen:
        chosen = [face_boxes_sorted[0]]
    return chosen


def union_box(boxes: List[Box], shape_hw3: Tuple[int, int, int]) -> Optional[Box]:
    if not boxes:
        return None
    h, w = shape_hw3[:2]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def arms_band_from_upper(upper_box: Box, shape_hw3: Tuple[int, int, int]) -> Box:
    H, W = shape_hw3[:2]
    x, y, w, h = upper_box
    y0 = y + int(0.35 * h)
    y1 = y + int(0.90 * h)
    x0 = x - int(0.15 * w)
    x1 = x + int(1.15 * w)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def face_sharpness(bgr_u8: np.ndarray, box: Optional[Box]) -> float:
    if box is None:
        return 0.0
    x, y, w, h = box
    roi = bgr_u8[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    g = to_gray_u8(roi)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def ellipse_mask_from_box(shape_hw3: Tuple[int, int, int], box: Box, feather_px: float = 10.0) -> np.ndarray:
    H, W = shape_hw3[:2]
    x, y, w, h = [int(v) for v in box]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)

    m = np.zeros((H, W), np.float32)
    if x1 <= x0 or y1 <= y0:
        return m[..., None]

    cx = int((x0 + x1) / 2)
    cy = int((y0 + y1) / 2)
    ax = max(1, int((x1 - x0) / 2))
    ay = max(1, int((y1 - y0) / 2))

    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 1.0, thickness=-1)
    if feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), float(feather_px))
    return np.clip(m, 0.0, 1.0)[..., None]


# -----------------------------------------------------------------------------
# Alignment helpers
# -----------------------------------------------------------------------------
def build_alignment_mask(base_shape: Tuple[int, int, int], base_face: Optional[Box], seg_mask01: Optional[np.ndarray], bg_only: bool) -> np.ndarray:
    h, w = base_shape[:2]
    m = np.ones((h, w), dtype=np.uint8) * 255

    if seg_mask01 is not None and bg_only:
        bg = (seg_mask01 < 0.35).astype(np.uint8) * 255
        m = cv2.bitwise_and(m, bg)

    if base_face is not None:
        x, y, bw, bh = base_face
        pad = int(0.25 * max(bw, bh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        m[y0:y1, x0:x1] = 0

    return m


def estimate_transform_orb(
    base_u8: np.ndarray,
    img_u8: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    model: str,
    nfeatures: int,
    max_matches: int,
    ransac_thresh: float,
) -> Tuple[Optional[np.ndarray], float, float, int]:
    g1 = to_gray_u8(base_u8)
    g2 = to_gray_u8(img_u8)

    orb = cv2.ORB_create(nfeatures)
    k1, d1 = orb.detectAndCompute(g1, mask1)
    k2, d2 = orb.detectAndCompute(g2, mask2)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None, 0.0, float("inf"), 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if not matches:
        return None, 0.0, float("inf"), 0

    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    if len(matches) < 40:
        return None, 0.0, float("inf"), len(matches)

    pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if model == "affine":
        M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if M is None or inliers is None:
            return None, 0.0, float("inf"), len(matches)
        inliers = inliers.ravel().astype(bool)
        inlr = float(inliers.mean()) if inliers.size else 0.0

        pts2_in = pts2[inliers].reshape(-1, 2)
        pts1_in = pts1[inliers].reshape(-1, 2)
        if pts2_in.shape[0] < 10:
            return None, inlr, float("inf"), len(matches)

        pts2_h = np.hstack([pts2_in, np.ones((pts2_in.shape[0], 1), np.float32)])
        proj = (M @ pts2_h.T).T
        err = np.linalg.norm(proj - pts1_in, axis=1)
        return M, inlr, float(np.mean(err)), len(matches)

    H, inliers = cv2.findHomography(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if H is None or inliers is None:
        return None, 0.0, float("inf"), len(matches)
    inliers = inliers.ravel().astype(bool)
    inlr = float(inliers.mean()) if inliers.size else 0.0

    pts2_in = pts2[inliers].reshape(-1, 2)
    pts1_in = pts1[inliers].reshape(-1, 2)
    if pts2_in.shape[0] < 10:
        return None, inlr, float("inf"), len(matches)

    pts2_h = np.hstack([pts2_in, np.ones((pts2_in.shape[0], 1), np.float32)])
    proj_h = (H @ pts2_h.T).T
    proj = proj_h[:, :2] / np.maximum(proj_h[:, 2:3], 1e-6)
    err = np.linalg.norm(proj - pts1_in, axis=1)
    return H, inlr, float(np.mean(err)), len(matches)


def warp_image_prealloc(src: np.ndarray, T: np.ndarray, dst: np.ndarray, model: str, border_value=0) -> np.ndarray:
    h, w = dst.shape[:2]
    if model == "homography":
        cv2.warpPerspective(
            src, T, (w, h),
            dst=dst,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
    else:
        cv2.warpAffine(
            src, T, (w, h),
            dst=dst,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
    return dst


def warp_valid_mask(T: np.ndarray, shape_hw3: Tuple[int, int, int], model: str) -> np.ndarray:
    h, w = shape_hw3[:2]
    src = np.ones((h, w), dtype=np.uint8) * 255
    if model == "homography":
        m = cv2.warpPerspective(src, T, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        m = cv2.warpAffine(src, T, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (m.astype(np.float32) / 255.0)


def transform_displacement(T: np.ndarray, model: str) -> Tuple[float, float]:
    if model == "homography":
        return float(T[0, 2]), float(T[1, 2])
    return float(T[0, 2]), float(T[1, 2])


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
# Kernel helpers
# -----------------------------------------------------------------------------
def _cap_size(size: int, cap: int) -> int:
    if cap is None or cap < 0:
        return size
    if cap == 0:
        return 1
    if cap % 2 == 0:
        cap += 1
    return min(size, cap)


def build_trajectory_kernel(
    displacements: List[Tuple[float, float]],
    weights: List[float],
    min_kernel: int = 9,
    softness: float = 0.0,
    max_kernel: int = -1,
) -> np.ndarray:
    if not displacements:
        k = np.zeros((min_kernel, min_kernel), dtype=np.float32)
        k[min_kernel // 2, min_kernel // 2] = 1.0
        return k

    max_dx = max(abs(dx) for dx, _ in displacements)
    max_dy = max(abs(dy) for _, dy in displacements)
    radius = int(math.ceil(max(max_dx, max_dy))) + 2
    size = max(min_kernel, 2 * radius + 1)
    if size % 2 == 0:
        size += 1

    size = _cap_size(size, max_kernel)
    size = max(min_kernel, size)
    if size % 2 == 0:
        size += 1

    k = np.zeros((size, size), dtype=np.float32)
    cx = cy = size // 2
    max_r = (size // 2) - 1

    for (dx, dy), w in zip(displacements, weights):
        dx = float(np.clip(dx, -max_r, max_r))
        dy = float(np.clip(dy, -max_r, max_r))
        x = int(round(cx + dx))
        y = int(round(cy + dy))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] += float(w)

    k[cy, cx] += 1e-6
    if softness and softness > 0:
        k = cv2.GaussianBlur(k, (0, 0), softness)

    s = float(k.sum())
    if s > 0:
        k /= s
    return k


def apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, kernel)


# -----------------------------------------------------------------------------
# Background plate stacking (prealloc, uint16 warp)
# -----------------------------------------------------------------------------
def stack_background_plate(
    idxs: List[int],
    base_idx: int,
    frames_u16: List[np.ndarray],
    frames_u8: List[np.ndarray],
    best_boxes: List[Optional[Box]],
    weight_sigma: float,
    k: int,
    seg_mask01: np.ndarray,  # HxW in base coords
    align_model: str,
    align_quality_thresh: float,
    align_ransac_thresh: float,
    align_nfeatures: int,
    max_matches: int = 600,
) -> Tuple[np.ndarray, int, int]:
    H, W = frames_u16[base_idx].shape[:2]

    bg_keep = np.clip(1.0 - seg_mask01, 0.0, 1.0).astype(np.float32)
    bg_keep3 = bg_keep[..., None]  # HxWx1

    acc = np.zeros((H, W, 3), dtype=np.float32)
    wacc = np.zeros((H, W, 1), dtype=np.float32)

    warp_u16 = np.empty((H, W, 3), dtype=np.uint16)
    warp_f32 = np.empty((H, W, 3), dtype=np.float32)
    tmp = np.empty((H, W, 3), dtype=np.float32)
    wmap = np.empty((H, W, 1), dtype=np.float32)

    used = 0
    skipped = 0

    base_mask = build_alignment_mask(frames_u8[base_idx].shape, best_boxes[base_idx], seg_mask01, bg_only=True)

    for i in idxs:
        w = gaussian_weight(i, base_idx, k, sigma=weight_sigma)

        img_mask = build_alignment_mask(frames_u8[i].shape, best_boxes[i], None, bg_only=False)

        T, inlr, merr, _ = estimate_transform_orb(
            frames_u8[base_idx],
            frames_u8[i],
            base_mask,
            img_mask,
            model=align_model,
            nfeatures=align_nfeatures,
            max_matches=max_matches,
            ransac_thresh=align_ransac_thresh,
        )

        quality = inlr * math.exp(-merr / 6.0)
        if T is None or quality < align_quality_thresh:
            skipped += 1
            continue

        warp_image_prealloc(frames_u16[i], T, warp_u16, align_model, border_value=0)
        warp_f32[:] = warp_u16
        warp_f32 *= (1.0 / 65535.0)

        valid1 = warp_valid_mask(T, frames_u16[base_idx].shape, align_model)  # HxW
        wmap[:] = bg_keep3
        wmap *= (w * valid1[..., None]).astype(np.float32)

        tmp[:] = warp_f32
        tmp *= wmap

        acc += tmp
        wacc += wmap
        used += 1

    bg_plate = acc / np.maximum(wacc, 1e-6)
    return bg_plate, used, skipped


# -----------------------------------------------------------------------------
# Output naming + metadata
# -----------------------------------------------------------------------------
def is_image_file_path(p: Path) -> bool:
    return p.suffix.lower() in _ALLOWED_EXTS


def make_unique_outpath(out_path: str, base_file: str, meta: dict, default_ext: str = ".tif") -> Path:
    outp = Path(out_path)
    if is_image_file_path(outp):
        outp.parent.mkdir(parents=True, exist_ok=True)
        return outp

    out_dir = outp
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(base_file).stem

    meta2 = to_builtin(meta)
    h = hashlib.sha1(json.dumps(meta2, sort_keys=True).encode("utf-8")).hexdigest()[:8]

    name = f"{stem}__B{meta2.get('base_idx','X')}__{ts}__{h}{default_ext}"
    return out_dir / name


def write_sidecar_json(image_path: Path, meta: dict) -> None:
    meta2 = to_builtin(meta)
    sidecar = image_path.with_name(image_path.name + ".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(meta2, indent=2, sort_keys=True), encoding="utf-8")


def srgb_icc_bytes() -> bytes:
    from PIL import ImageCms
    prof = ImageCms.createProfile("sRGB")
    return ImageCms.ImageCmsProfile(prof).tobytes()


def save_image_with_metadata(path: Path, img_srgb01_bgr: np.ndarray, out_bps: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_sidecar_json(path, meta)

    ext = path.suffix.lower()
    rgb01 = img_srgb01_bgr[..., ::-1].copy()

    if out_bps == 16:
        arr = np.clip(rgb01 * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    else:
        arr = np.clip(rgb01 * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if ext in [".tif", ".tiff"]:
        import tifffile
        desc = json.dumps(to_builtin(meta), sort_keys=True)
        icc = srgb_icc_bytes()
        tifffile.imwrite(str(path), arr, photometric="rgb", description=desc, metadata=None, iccprofile=icc)
        return

    ok = cv2.imwrite(str(path), arr[..., ::-1])
    if not ok:
        raise RuntimeError(f"Failed to write output: {path}")


# -----------------------------------------------------------------------------
# Auto exposure in linear space
# -----------------------------------------------------------------------------
def auto_expose_gain(
    linear01_bgr: np.ndarray,
    percentile: float = 99.5,
    target: float = 0.98,
    min_gain: float = 0.25,
    max_gain: float = 8.0,
) -> float:
    rgb = linear01_bgr[..., ::-1]
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    p = float(np.percentile(lum, percentile))
    if p <= 1e-6:
        return 1.0
    g = target / p
    return float(np.clip(g, min_gain, max_gain))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def make_panstack(
    in_dir: str,
    out_path: str,
    k: int = 10,
    mode: str = "trailing",
    scale: float = 0.5,
    weight_sigma: float = 0.7,
    face_conf: float = 0.5,
    align_bg: bool = True,
    bg_mode: str = "blur",
    bg_stack: str = "mean",
    blur_scope: str = "all",
    bg_max_kernel: int = -1,
    fg_motion: str = "residual",
    kernel_scale: float = 1.0,
    motion_roi: str = "arms",
    motion_feature: str = "edges",
    freeze_faces: Union[str, List[int]] = "all",
    freeze_region: str = "full",
    preview_faces: bool = False,
    detect_gain: float = 1.6,
    detect_gamma: float = 0.6,
    detect_clahe: bool = False,
    out_bps: int = 16,
    auto_expose: bool = True,
    exposure_gain: float = 1.0,
    expose_percentile: float = 99.5,
    expose_target: float = 0.98,
    min_gain: float = 0.25,
    max_gain: float = 8.0,
    blur_model: str = "trajectory",
    bg_blur: float = 1.0,
    body_blur: float = 0.8,
    kernel_softness: float = 0.0,
    min_kernel: int = 9,
    debug_overlay: bool = False,
    debug_out: str = "",
    segmentation: str = "auto",
    seg_thresh: float = 0.5,
    seg_erode_px: int = 0,
    seg_dilate_px: int = 0,
    seg_feather_px: float = 25.0,
    face_opacity: float = 1.0,
    body_opacity: float = 1.0,
    align_model: str = "homography",
    align_quality_thresh: float = 0.22,
    align_ransac_thresh: float = 3.0,
    align_nfeatures: int = 8000,
) -> Dict[str, object]:

    disp_debug: List[Tuple[float, float]] = []

    in_dir_p = Path(in_dir)
    paths = sorted(glob.glob(str(in_dir_p / "*.ARW"))) + sorted(glob.glob(str(in_dir_p / "*.arw")))
    if not paths:
        raise FileNotFoundError(f"No ARW files found in: {in_dir}")

    frames_u16 = [load_arw_bgr_u16(p, scale=scale) for p in paths]
    frames_lin = [u16_to_linear01(f) for f in frames_u16]  # base + final
    frames_u8 = [brighten_for_detect(bgr_to_u8(f), gain=detect_gain, gamma=detect_gamma, clahe=detect_clahe) for f in frames_u16]

    # base selection via sharpest face
    best_boxes: List[Optional[Box]] = []
    scores: List[float] = []
    for f8 in frames_u8:
        dets = detect_face_boxes(f8, conf_thresh=face_conf)
        sboxes = sort_boxes_for_ids(dets)
        box = sboxes[0] if sboxes else None
        best_boxes.append(box)
        scores.append(face_sharpness(f8, box))

    base_idx = int(np.argmax(scores)) if any(s > 0 for s in scores) else 0
    base_lin = frames_lin[base_idx]
    base_u8 = frames_u8[base_idx]
    base_face_boxes_sorted = sort_boxes_for_ids(detect_face_boxes(base_u8, conf_thresh=face_conf))

    if preview_faces:
        preview = base_u8.copy()
        for i, b in enumerate(base_face_boxes_sorted, start=1):
            _draw_box(preview, b, f"{i}", color=(0, 255, 255))
        outp = Path(out_path)
        prev_path = outp if is_image_file_path(outp) else (outp / "preview_faces.png")
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(prev_path), preview)
        return {"preview_faces": True, "base_idx": base_idx, "base_file": Path(paths[base_idx]).name, "preview_path": str(prev_path)}

    if blur_scope == "foreground":
        bg_mode = "stabilize"

    chosen_faces = select_face_boxes(base_face_boxes_sorted, freeze_faces)
    sharp_boxes = [region_box_from_face(fb, base_lin.shape, freeze_region) for fb in chosen_faces]
    body_boxes_upper = [region_box_from_face(fb, base_lin.shape, "upper") for fb in chosen_faces]

    motion_boxes = body_boxes_upper
    if motion_roi == "arms":
        motion_boxes = [arms_band_from_upper(b, base_u8.shape) for b in body_boxes_upper]
    motion_roi_box = union_box(motion_boxes, base_u8.shape)
    upper_roi_box = union_box(body_boxes_upper, base_u8.shape)

    # segmentation
    seg_mode = segmentation.strip().lower()
    seg_mask: Optional[np.ndarray] = None
    seg_error = ""

    if seg_mode in ("auto", "on"):
        base_for_seg = bgr_to_u8(frames_u16[base_idx])
        seg_mask, seg_error = person_segmentation_mask_u8(
            base_for_seg, thresh=seg_thresh, erode_px=seg_erode_px, dilate_px=seg_dilate_px, feather_px=seg_feather_px
        )
        if seg_mask is None and seg_mode == "on":
            raise RuntimeError(f"Segmentation failed: {seg_error}")

    using_seg = seg_mask is not None

    idxs = select_indices(len(frames_lin), base_idx, k, mode)

    # sharp mask (ellipse union)
    sharp_mask = np.zeros((base_lin.shape[0], base_lin.shape[1], 1), dtype=np.float32)
    for b in sharp_boxes:
        sharp_mask = np.maximum(sharp_mask, ellipse_mask_from_box(base_lin.shape, b, feather_px=10.0))
    sharp = np.clip(sharp_mask[..., 0], 0.0, 1.0)

    if using_seg:
        seg = np.clip(seg_mask[..., 0].astype(np.float32), 0.0, 1.0)
        body = np.clip(seg - sharp, 0.0, 1.0)
        bg = np.clip(1.0 - seg, 0.0, 1.0)
        body_mask = body[..., None].astype(np.float32)
        bg_mask = bg[..., None].astype(np.float32)
        sharp_mask = sharp[..., None].astype(np.float32)
    else:
        seg = None
        sharp_mask = boxes_union_mask(base_lin.shape, sharp_boxes, feather=35.0).astype(np.float32)
        body_mask_boxes = boxes_union_mask(base_lin.shape, body_boxes_upper, feather=45.0).astype(np.float32)
        body_mask = np.clip(body_mask_boxes - sharp_mask, 0.0, 1.0).astype(np.float32)
        bg_mask = (1.0 - np.clip(sharp_mask + body_mask, 0.0, 1.0)).astype(np.float32)

    # background plate (no ghosts)
    bg_plate = None
    bg_plate_used = 0
    bg_plate_skipped = 0
    if using_seg and seg is not None:
        bg_plate, bg_plate_used, bg_plate_skipped = stack_background_plate(
            idxs=idxs,
            base_idx=base_idx,
            frames_u16=frames_u16,
            frames_u8=frames_u8,
            best_boxes=best_boxes,
            weight_sigma=weight_sigma,
            k=k,
            seg_mask01=seg,
            align_model=align_model,
            align_quality_thresh=align_quality_thresh,
            align_ransac_thresh=align_ransac_thresh,
            align_nfeatures=align_nfeatures,
        )

    # Full-image stack (prealloc warp, uint16)
    H, W = base_lin.shape[:2]
    warp_u16 = np.empty((H, W, 3), dtype=np.uint16)
    warp_f32 = np.empty((H, W, 3), dtype=np.float32)

    acc = np.zeros((H, W, 3), dtype=np.float32)
    wacc = np.zeros((H, W, 1), dtype=np.float32)

    disps: List[Tuple[float, float]] = []
    disp_ws: List[float] = []
    used = 0
    skipped = 0

    base_mask = build_alignment_mask(frames_u8[base_idx].shape, best_boxes[base_idx], seg if using_seg else None, bg_only=using_seg)

    for i in idxs:
        w = gaussian_weight(i, base_idx, k, sigma=weight_sigma)

        img_mask = build_alignment_mask(frames_u8[i].shape, best_boxes[i], None, bg_only=False)

        T, inlr, merr, _ = estimate_transform_orb(
            frames_u8[base_idx],
            frames_u8[i],
            base_mask,
            img_mask,
            model=align_model,
            nfeatures=align_nfeatures,
            max_matches=600,
            ransac_thresh=align_ransac_thresh,
        )

        quality = inlr * math.exp(-merr / 6.0)
        if T is None or quality < align_quality_thresh:
            skipped += 1
            continue

        warp_image_prealloc(frames_u16[i], T, warp_u16, align_model, border_value=0)
        warp_f32[:] = warp_u16
        warp_f32 *= (1.0 / 65535.0)

        valid = warp_valid_mask(T, base_lin.shape, align_model).astype(np.float32)
        wmap = (w * valid)[..., None].astype(np.float32)

        acc += warp_f32 * wmap
        wacc += wmap
        used += 1

        dx, dy = transform_displacement(T, align_model)
        disps.append((kernel_scale * dx, kernel_scale * dy))
        disp_ws.append(w)

    stacked_lin = acc / np.maximum(wacc, 1e-6)

    # background source and blur behavior
    bg_kernel_disabled = (bg_mode == "stabilize") or (bg_max_kernel == 0) or (bg_blur <= 0.0)
    bg_for_blur = bg_plate if (bg_plate is not None) else stacked_lin

    if blur_model == "none" or len(disps) == 0:
        bg_blurred = bg_for_blur
        body_blurred = stacked_lin
    else:
        ktraj = build_trajectory_kernel(
            disps,
            disp_ws if disp_ws else [1.0] * len(disps),
            min_kernel=min_kernel,
            softness=kernel_softness,
            max_kernel=bg_max_kernel,
        )

        ident = np.zeros_like(ktraj, dtype=np.float32)
        ident[ktraj.shape[0] // 2, ktraj.shape[1] // 2] = 1.0

        if bg_kernel_disabled:
            bg_blurred = bg_for_blur
        else:
            bg_strength = float(np.clip(bg_blur, 0.0, 1.0))
            k_bg = (1.0 - bg_strength) * ident + bg_strength * ktraj
            k_bg /= max(float(k_bg.sum()), 1e-6)
            bg_blurred = apply_kernel(bg_for_blur, k_bg)

        body_strength = float(np.clip(body_blur, 0.0, 1.0))
        k_body = (1.0 - body_strength) * ident + body_strength * ktraj
        k_body /= max(float(k_body.sum()), 1e-6)
        body_blurred = apply_kernel(stacked_lin, k_body)

    # Memory-safe opacity compositing (reuse one tmp buffer)
    fo = float(np.clip(face_opacity, 0.0, 1.0))
    bo = float(np.clip(body_opacity, 0.0, 1.0))

    out_lin = np.zeros_like(base_lin, dtype=np.float32)
    tmp = np.empty_like(out_lin, dtype=np.float32)

    # bg
    np.copyto(tmp, bg_blurred)
    tmp *= bg_mask
    out_lin += tmp

    # body: base*(1-bo) + blurred*(bo)
    w_base_body = body_mask * (1.0 - bo)
    w_blur_body = body_mask * bo

    np.copyto(tmp, base_lin)
    tmp *= w_base_body
    out_lin += tmp

    np.copyto(tmp, body_blurred)
    tmp *= w_blur_body
    out_lin += tmp

    # face: base*(fo) + stacked*(1-fo) in sharp region
    w_base_face = sharp_mask * fo
    w_stack_face = sharp_mask * (1.0 - fo)

    np.copyto(tmp, base_lin)
    tmp *= w_base_face
    out_lin += tmp

    np.copyto(tmp, stacked_lin)
    tmp *= w_stack_face
    out_lin += tmp

    auto_g = 1.0
    if auto_expose:
        auto_g = auto_expose_gain(out_lin, percentile=expose_percentile, target=expose_target, min_gain=min_gain, max_gain=max_gain)
        out_lin = np.clip(out_lin * auto_g, 0.0, None)

    out_lin = np.clip(out_lin * float(exposure_gain), 0.0, None)
    out_srgb = linear_to_srgb(out_lin)

    meta = {
        "tool": "panstack",
        "version": "0.7.0-prealloc-warp-memorysafe",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "in_dir": str(in_dir_p),
        "base_idx": base_idx,
        "base_file": Path(paths[base_idx]).name,
        "frames_total": len(frames_lin),
        "frames_used": used,
        "frames_skipped_alignment": skipped,
        "k": k,
        "mode": mode,
        "scale": scale,
        "align_model": align_model,
        "align_quality_thresh": align_quality_thresh,
        "align_ransac_thresh": align_ransac_thresh,
        "align_nfeatures": align_nfeatures,
        "bg_mode": bg_mode,
        "bg_stack": bg_stack,
        "blur_scope": blur_scope,
        "bg_max_kernel": bg_max_kernel,
        "fg_motion": fg_motion,
        "kernel_scale": kernel_scale,
        "motion_roi": motion_roi,
        "motion_feature": motion_feature,
        "freeze_faces": freeze_faces,
        "freeze_region": freeze_region,
        "faces_found_in_base": len(base_face_boxes_sorted),
        "chosen_faces_count": len(chosen_faces),
        "blur_model": blur_model,
        "bg_blur": bg_blur,
        "body_blur": body_blur,
        "kernel_softness": kernel_softness,
        "min_kernel": min_kernel,
        "weight_sigma": weight_sigma,
        "face_conf": face_conf,
        "detect_gain": detect_gain,
        "detect_gamma": detect_gamma,
        "detect_clahe": detect_clahe,
        "out_bps": out_bps,
        "auto_expose": auto_expose,
        "auto_gain_used": auto_g,
        "exposure_gain": exposure_gain,
        "expose_percentile": expose_percentile,
        "expose_target": expose_target,
        "min_gain": min_gain,
        "max_gain": max_gain,
        "displacements_count": len(disps),
        "motion_roi_box": motion_roi_box,
        "upper_roi_box": upper_roi_box,
        "segmentation": seg_mode,
        "segmentation_used": using_seg,
        "seg_thresh": seg_thresh,
        "seg_erode_px": seg_erode_px,
        "seg_dilate_px": seg_dilate_px,
        "seg_feather_px": seg_feather_px,
        "mediapipe_tasks_available": _MP_TASKS_AVAILABLE,
        "mediapipe_tasks_import_error": _MP_TASKS_IMPORT_ERROR,
        "segmentation_error": seg_error,
        "bg_plate_used_frames": bg_plate_used,
        "bg_plate_skipped_alignment": bg_plate_skipped,
        "face_opacity": fo,
        "body_opacity": bo,
        "face_model_loaded": _face_net is not None,
        "face_model_error": _face_load_error,
    }

    if debug_overlay:
        overlay = base_u8.copy()

        for idx, fb in enumerate(base_face_boxes_sorted, start=1):
            _draw_box(overlay, fb, f"face {idx}", color=(0, 255, 255))
        for rb in sharp_boxes:
            _draw_box(overlay, rb, f"sharp:{freeze_region}", color=(0, 200, 255))
        if motion_roi_box is not None:
            _draw_box(overlay, motion_roi_box, f"motion_roi:{motion_roi}", color=(0, 128, 255))

        if using_seg and seg_mask is not None:
            seg_vis = (np.clip(seg_mask[..., 0], 0, 1) * 255).astype(np.uint8)
            edges = cv2.Canny(seg_vis, 60, 180)
            overlay[edges > 0] = (0, 255, 0)

        dbg_path = Path(debug_out.strip()) if debug_out.strip() else _default_debug_path(out_path, meta["base_file"])
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg_path), overlay)
        meta["debug_overlay_path"] = str(dbg_path)

    final_path = make_unique_outpath(out_path, meta["base_file"], meta, default_ext=".tif")
    save_image_with_metadata(final_path, out_srgb, out_bps, meta)
    meta["out_path"] = str(final_path)
    return meta
