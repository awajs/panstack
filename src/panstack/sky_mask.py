# sky_mask.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2


def load_mask(mask_path: str | Path, shape_hw: tuple[int, int]) -> np.ndarray:
    """
    Load a mask image (any format). Non-zero is treated as True.
    Resizes to match shape_hw if needed.
    """
    mask_path = Path(mask_path)
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if (m.shape[0], m.shape[1]) != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0)


def heuristic_sky_mask(
    bgr: np.ndarray,
    top_frac: float = 0.55,
    min_v: float = 0.62,
    max_s: float = 0.55,
    blue_boost: float = 0.03,
    morph: int = 5,
) -> np.ndarray:
    """
    A decent heuristic sky detector:
    - Focus on the top portion of the image
    - Prefer bright-ish, low/medium saturation regions
    - Prefer blue channel slightly stronger than red
    """
    h, w = bgr.shape[:2]
    top_h = int(h * float(np.clip(top_frac, 0.1, 1.0)))
    roi = bgr[:top_h, :, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = hsv[..., 0] / 179.0
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0

    b = (roi[..., 0].astype(np.float32) / 255.0)
    r = (roi[..., 2].astype(np.float32) / 255.0)

    # Bright + not too saturated + blue-ish preference (helps against bright walls)
    m = (V >= min_v) & (S <= max_s) & ((b - r) >= blue_boost)

    # Clean up
    m_u8 = (m.astype(np.uint8) * 255)
    if morph > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, k, iterations=1)
        m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, k, iterations=1)

    mask = np.zeros((h, w), dtype=bool)
    mask[:top_h, :] = (m_u8 > 0)
    return mask
