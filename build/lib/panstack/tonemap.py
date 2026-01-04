# tonemap.py
from __future__ import annotations
import cv2
import numpy as np

from panstack.sky_mask import heuristic_sky_mask


def _to_float01(img: np.ndarray) -> np.ndarray:
    """Convert uint8/uint16/float image to float32 in [0,1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    img = img.astype(np.float32)
    # If user passed floats in 0..255, try to guess
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _from_float01(img01: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    """Convert float [0,1] back to requested dtype."""
    img01 = np.clip(img01, 0.0, 1.0)
    if out_dtype == np.uint8:
        return (img01 * 255.0 + 0.5).astype(np.uint8)
    if out_dtype == np.uint16:
        return (img01 * 65535.0 + 0.5).astype(np.uint16)
    return img01.astype(out_dtype)


def _luma_bt709(rgb01: np.ndarray) -> np.ndarray:
    """BT.709 luma for RGB in [0,1]."""
    # assume last dim is channels
    return 0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]


def _apply_to_luminance(rgb01: np.ndarray, f) -> np.ndarray:
    """
    Apply tone curve to luminance and rescale RGB to preserve chroma.
    This avoids channel-wise hue shifts.
    """
    eps = 1e-6
    L = _luma_bt709(rgb01)
    L2 = f(L)
    scale = (L2 + eps) / (L + eps)
    out = rgb01 * scale[..., None]
    return np.clip(out, 0.0, 1.0)


def log_tonemap(x01: np.ndarray, alpha: float = 8.0) -> np.ndarray:
    """
    Simple log tonemap:
      y = log(1 + alpha*x) / log(1 + alpha)
    """
    x01 = np.clip(x01, 0.0, 1.0)
    denom = np.log1p(alpha)
    return np.log1p(alpha * x01) / denom


def log_knee_tonemap(x01: np.ndarray, t: float = 0.7, alpha: float = 8.0) -> np.ndarray:
    """
    Soft-knee log:
      y = x                         for x <= t
      y = t + (log(1 + a(x-t))/log(1 + a(1-t))) * (1-t)   for x > t
    """
    x01 = np.clip(x01, 0.0, 1.0)
    t = float(np.clip(t, 0.0, 1.0))
    out = x01.copy()

    mask = x01 > t
    if np.any(mask):
        denom = np.log1p(alpha * (1.0 - t))
        out[mask] = t + (np.log1p(alpha * (x01[mask] - t)) / denom) * (1.0 - t)
    return np.clip(out, 0.0, 1.0)


def reinhard_log_tonemap(x01: np.ndarray, alpha: float = 6.0) -> np.ndarray:
    """
    Reinhard-log hybrid:
      L = log(1 + alpha*x)
      y = L / (1 + L)
    """
    x01 = np.clip(x01, 0.0, 1.0)
    L = np.log1p(alpha * x01)
    return L / (1.0 + L)


def apply_tonemap(
    img: np.ndarray,
    mode: str = "log_knee",
    alpha: float = 8.0,
    threshold: float = 0.7,
    use_luminance: bool = True,
) -> np.ndarray:
    """
    Apply tone mapping to an image.

    Args:
      img: HxWxC or HxW, uint8/uint16/float
      mode: 'log', 'log_knee', 'reinhard_log'
      alpha: strength of compression
      threshold: knee threshold (only for log_knee)
      use_luminance: if True and img is color, apply curve to luminance to reduce hue shifts

    Returns:
      Same dtype as input.
    """
    mode = mode.lower().strip()
    out_dtype = img.dtype
    x01 = _to_float01(img)

    def curve(v):
        if mode == "log":
            return log_tonemap(v, alpha=alpha)
        if mode in ("log_knee", "knee", "logknee"):
            return log_knee_tonemap(v, t=threshold, alpha=alpha)
        if mode in ("reinhard_log", "reinhard", "rlog"):
            return reinhard_log_tonemap(v, alpha=alpha)
        raise ValueError(f"Unknown mode: {mode}")

    if x01.ndim == 3 and x01.shape[2] >= 3 and use_luminance:
        y01 = _apply_to_luminance(x01[..., :3], curve)
        out01 = x01.copy()
        out01[..., :3] = y01
    else:
        out01 = curve(x01)

    return _from_float01(out01, out_dtype)

def _apply_sky_only_tonemap(
    bgr: np.ndarray,
    mode: str,
    alpha: float,
    threshold: float,
    use_luma: bool,
    feather: int,
    sky_mask: np.ndarray | None,
    allow_heuristic: bool,
) -> np.ndarray:
    h, w = bgr.shape[:2]

    if sky_mask is None:
        if not allow_heuristic:
            return bgr
        sky_mask = heuristic_sky_mask(bgr)

    # Feather the skyline edge
    if feather > 0:
        m = (sky_mask.astype(np.uint8) * 255)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather, sigmaY=feather)
        a = m.astype(np.float32) / 255.0
    else:
        a = sky_mask.astype(np.float32)

    tm = apply_tonemap(bgr, mode=mode, alpha=alpha, threshold=threshold, use_luminance=use_luma)

    orig01 = _to_float01(bgr)
    tm01 = _to_float01(tm)
    out01 = tm01 * a[..., None] + orig01 * (1.0 - a[..., None])
    return _from_float01(out01, bgr.dtype)

def _smoothstep(edge0: float, edge1: float, x):
    import numpy as np
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def apply_highlight_tonemap(
    bgr,
    mode: str = "log_knee",
    alpha: float = 20.0,
    threshold: float = 0.6,
    use_luma: bool = True,
    hi_start: float = 0.70,   # start compressing highlights here
    hi_end: float = 0.98,     # fully compress by here
    top_bias: float = 0.6,    # 0=no top bias, 1=strong top bias
    feather: int = 8,
):
    import numpy as np
    import cv2

    # bgr is float in [0,1] in your pipeline at this stage (out_srgb)
    img = np.clip(bgr.astype(np.float32), 0.0, 1.0)

    # luminance (BT.709) in sRGB space (fine for masking)
    rgb = img[..., ::-1]
    L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    # highlight weight
    w = _smoothstep(hi_start, hi_end, L).astype(np.float32)

    # optional top bias (push effect to sky region without hard segmentation)
    if top_bias and top_bias > 0:
        h = img.shape[0]
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]  # 0 top -> 1 bottom
        top = 1.0 - y
        w = w * ((1.0 - top_bias) + top_bias * top)

    # feather to prevent banding edges in clouds
    if feather and feather > 0:
        w = cv2.GaussianBlur(w, (0, 0), sigmaX=float(feather), sigmaY=float(feather))

    # tonemap whole image
    tm = apply_tonemap(img, mode=mode, alpha=float(alpha), threshold=float(threshold), use_luminance=bool(use_luma))

    # blend
    out = tm * w[..., None] + img * (1.0 - w[..., None])
    return np.clip(out, 0.0, 1.0)
