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

Box = Tuple[int, int, int, int]  # x, y, w, h
_ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# -----------------------------------------------------------------------------
# Color / dtype helpers
# -----------------------------------------------------------------------------
def ensure_bgr(img: np.ndarray) -> np.ndarray:
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


def bgr_to_u8(bgr: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr(bgr)
    if bgr.dtype == np.uint8:
        return bgr
    if bgr.dtype == np.uint16:
        return (bgr >> 8).astype(np.uint8)
    if np.issubdtype(bgr.dtype, np.floating):
        return np.clip(bgr * 255.0, 0, 255).astype(np.uint8) if bgr.max() <= 1.0 else np.clip(bgr, 0, 255).astype(np.uint8)
    return cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def to_gray_u8(img: np.ndarray) -> np.ndarray:
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

    if g.dtype == np.uint8:
        return g
    if g.dtype == np.uint16:
        return (g >> 8).astype(np.uint8)
    if np.issubdtype(g.dtype, np.floating):
        return np.clip(g * 255.0, 0, 255).astype(np.uint8) if g.max() <= 1.0 else np.clip(g, 0, 255).astype(np.uint8)
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
    out = np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(x, 1 / 2.4) - a)
    return out


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


def detect_face_boxes(bgr_u8: np.ndarray, conf_thresh: float = 0.5) -> List[Tuple[Box, float]]:
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
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            out.append(((x1, y1, x2 - x1, y2 - y1), conf))
    return out


def sort_boxes_for_ids(boxes: List[Tuple[Box, float]]) -> List[Box]:
    return [b for (b, c) in sorted(boxes, key=lambda bc: (bc[0][2] * bc[0][3], bc[1]), reverse=True)]


# -----------------------------------------------------------------------------
# RAW loading (linear-ish)
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
    bgr_u16 = ensure_bgr(bgr_u16)
    return (bgr_u16.astype(np.float32) / 65535.0)


# -----------------------------------------------------------------------------
# Masks (face / upper / full)
# -----------------------------------------------------------------------------
def region_box_from_face(face_box: Box, shape_hw3: Tuple[int, int, int], region: str) -> Box:
    """
    Expand a face box into a larger region box.
    region:
      - face: just the face
      - upper: head+torso heuristic
      - full: larger heuristic
    """
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
        raise ValueError(f"Unknown freeze_region: {region}")

    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w_img, x1); y1 = min(h_img, y1)

    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def boxes_union_mask(shape_hw3: Tuple[int, int, int], boxes: List[Box], feather: float = 35.0) -> np.ndarray:
    h, w = shape_hw3[:2]
    m = np.zeros((h, w), dtype=np.float32)
    for (x, y, bw, bh) in boxes:
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(w, x + bw); y1 = min(h, y + bh)
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


def face_sharpness(bgr_u8: np.ndarray, box: Optional[Box]) -> float:
    if box is None:
        return 0.0
    x, y, w, h = box
    roi = bgr_u8[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    g = to_gray_u8(roi)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# -----------------------------------------------------------------------------
# Alignment on background (ORB on uint8 proxies)
# -----------------------------------------------------------------------------
def _mask_out_face(shape_hw3: Tuple[int, int, int], box: Optional[Box]) -> np.ndarray:
    h, w = shape_hw3[:2]
    m = np.ones((h, w), dtype=np.uint8) * 255
    if box is None:
        return m
    x, y, bw, bh = box
    pad = int(0.25 * max(bw, bh))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
    m[y0:y1, x0:x1] = 0
    return m


def estimate_affine_orb(base_u8: np.ndarray, img_u8: np.ndarray, base_face: Optional[Box], img_face: Optional[Box]) -> Optional[np.ndarray]:
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


def warp_affine(img: np.ndarray, M: np.ndarray, shape_hw3: Tuple[int, int, int]) -> np.ndarray:
    h, w = shape_hw3[:2]
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
# Extra synthetic motion blur (background only)
# -----------------------------------------------------------------------------
def motion_blur(img: np.ndarray, length: float, angle_deg: float) -> np.ndarray:
    if length <= 0.0:
        return img
    length_i = max(3, int(round(length)))
    k = np.zeros((length_i, length_i), dtype=np.float32)
    k[length_i // 2, :] = 1.0
    k /= max(k.sum(), 1e-6)

    center = (length_i / 2.0, length_i / 2.0)
    R = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    k_rot = cv2.warpAffine(k, R, (length_i, length_i), flags=cv2.INTER_LINEAR)
    s = k_rot.sum()
    if s > 0:
        k_rot /= s
    return cv2.filter2D(img, -1, k_rot)


def estimate_motion_angle_deg(translations: List[Tuple[float, float]]) -> Optional[float]:
    if not translations:
        return None
    dx = float(np.mean([t[0] for t in translations]))
    dy = float(np.mean([t[1] for t in translations]))
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


# -----------------------------------------------------------------------------
# Output naming + metadata writing (TIFF ICC)
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
    ext = default_ext

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    settings_blob = json.dumps(meta, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(settings_blob).hexdigest()[:8]
    stem = Path(base_file).stem

    name = (
        f"{stem}__B{meta.get('base_idx')}"
        f"__k{meta.get('k')}_{meta.get('mode')}"
        f"__align{1 if meta.get('align_bg') else 0}"
        f"__faces{meta.get('freeze_faces')}"
        f"__reg{meta.get('freeze_region')}"
        f"__eb{meta.get('extra_blur')}_{meta.get('blur_angle')}"
        f"__{ts}__{h}{ext}"
    )
    return out_dir / name


def write_sidecar_json(image_path: Path, meta: dict) -> None:
    sidecar = image_path.with_suffix(image_path.suffix + ".json")
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


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
        desc = json.dumps(meta, sort_keys=True)
        icc = srgb_icc_bytes()
        tifffile.imwrite(str(path), arr, photometric="rgb", description=desc, metadata=None, iccprofile=icc)
        return

    ok = cv2.imwrite(str(path), arr[..., ::-1])
    if not ok:
        raise RuntimeError(f"Failed to write output: {path}")


# -----------------------------------------------------------------------------
# Auto exposure in linear space
# -----------------------------------------------------------------------------
def auto_expose_gain(linear01_bgr: np.ndarray, percentile: float = 99.5, target: float = 0.98,
                     min_gain: float = 0.25, max_gain: float = 8.0) -> float:
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
    freeze_faces: Union[str, List[int]] = "1",
    freeze_region: str = "face",
    preview_faces: bool = False,
    extra_blur: float = 0.0,
    blur_angle: Union[str, float] = "auto",
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
) -> Dict[str, object]:

    in_dir_p = Path(in_dir)
    paths = sorted(glob.glob(str(in_dir_p / "*.ARW"))) + sorted(glob.glob(str(in_dir_p / "*.arw")))
    if not paths:
        raise FileNotFoundError(f"No ARW files found in: {in_dir}")

    # Decode RAWs to uint16 BGR then to float linear 0..1 (BGR)
    frames_u16 = [load_arw_bgr_u16(p, scale=scale) for p in paths]
    frames_lin = [u16_to_linear01(f) for f in frames_u16]

    # Detection/alignment proxies (uint8, brightened)
    frames_u8 = [brighten_for_detect(bgr_to_u8(f), gain=detect_gain, gamma=detect_gamma, clahe=detect_clahe) for f in frames_u16]

    # Determine base frame by best face sharpness (best face per frame)
    best_boxes: List[Optional[Box]] = []
    scores: List[float] = []
    for f8 in frames_u8:
        dets = detect_face_boxes(f8, conf_thresh=face_conf)
        boxes_sorted = sort_boxes_for_ids(dets)
        box = boxes_sorted[0] if boxes_sorted else None
        best_boxes.append(box)
        scores.append(face_sharpness(f8, box))

    base_idx = int(np.argmax(scores))
    base_lin = frames_lin[base_idx]
    base_u8 = frames_u8[base_idx]

    # Detect all faces in base for selection
    base_dets = detect_face_boxes(base_u8, conf_thresh=face_conf)
    base_face_boxes_sorted = sort_boxes_for_ids(base_dets)

    # Preview faces
    if preview_faces:
        preview = base_u8.copy()
        for i, b in enumerate(base_face_boxes_sorted, start=1):
            x, y, w, h = b
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(preview, str(i), (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        outp = Path(out_path)
        prev_path = outp if is_image_file_path(outp) else (outp / "preview_faces.png")
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(prev_path), preview)
        return {
            "preview_faces": True,
            "base_idx": base_idx,
            "base_file": Path(paths[base_idx]).name,
            "faces_found": len(base_face_boxes_sorted),
            "preview_path": str(prev_path),
        }

    # Choose faces and convert to region boxes
    chosen_faces = select_face_boxes(base_face_boxes_sorted, freeze_faces)
    region_boxes = [region_box_from_face(fb, base_lin.shape, freeze_region) for fb in chosen_faces]
    mask = boxes_union_mask(base_lin.shape, region_boxes, feather=35.0)

    # Stack
    idxs = select_indices(len(frames_lin), base_idx, k, mode)

    acc = np.zeros_like(base_lin, dtype=np.float32)
    wsum = 0.0
    used = 0
    translations: List[Tuple[float, float]] = []

    for i in idxs:
        w = gaussian_weight(i, base_idx, k, sigma=weight_sigma)

        M = None
        if align_bg or (blur_angle == "auto"):
            M = estimate_affine_orb(frames_u8[base_idx], frames_u8[i], best_boxes[base_idx], best_boxes[i])

        if M is not None:
            translations.append((float(M[0, 2]), float(M[1, 2])))
            fw = warp_affine(frames_lin[i], M, base_lin.shape) if align_bg else frames_lin[i]
        else:
            fw = frames_lin[i]  # fallback (prevents frames_used=1)

        acc += fw * w
        wsum += w
        used += 1

    bg_lin = acc / max(wsum, 1e-6)

    # extra blur (linear)
    angle_used = None
    if extra_blur > 0:
        if blur_angle == "auto":
            angle_used = estimate_motion_angle_deg(translations) or 0.0
        else:
            angle_used = float(blur_angle)
        bg_lin = motion_blur(bg_lin, length=extra_blur, angle_deg=angle_used)

    # composite (linear)
    out_lin = (base_lin * mask + bg_lin * (1.0 - mask)).astype(np.float32)

    # exposure (linear)
    auto_g = 1.0
    if auto_expose:
        auto_g = auto_expose_gain(out_lin, percentile=expose_percentile, target=expose_target, min_gain=min_gain, max_gain=max_gain)
        out_lin = np.clip(out_lin * auto_g, 0.0, None)

    out_lin = np.clip(out_lin * float(exposure_gain), 0.0, None)

    # encode to sRGB
    out_srgb = linear_to_srgb(out_lin)

    # metadata
    meta = {
        "tool": "panstack",
        "version": "0.0.3",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "in_dir": str(in_dir_p),
        "base_idx": base_idx,
        "base_file": Path(paths[base_idx]).name,
        "frames_total": len(frames_lin),
        "frames_used": used,
        "k": k,
        "mode": mode,
        "scale": scale,
        "align_bg": align_bg,
        "freeze_faces": freeze_faces,
        "freeze_region": freeze_region,
        "faces_found_in_base": len(base_face_boxes_sorted),
        "chosen_faces_count": len(chosen_faces),
        "extra_blur": extra_blur,
        "blur_angle": ("auto" if blur_angle == "auto" else float(blur_angle)),
        "estimated_angle": angle_used,
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
    }

    final_path = make_unique_outpath(out_path, meta["base_file"], meta, default_ext=".tif")
    save_image_with_metadata(final_path, out_srgb, out_bps, meta)
    meta["out_path"] = str(final_path)
    return meta
