# apply_tonemap.py
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from tonemap import apply_tonemap, _to_float01, _from_float01
from sky_mask import load_mask, heuristic_sky_mask


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def iter_images(in_path: Path):
    if in_path.is_file():
        yield in_path
        return
    for p in sorted(in_path.rglob("*")):
        if p.suffix.lower() in ALLOWED_EXTS and p.is_file():
            yield p


def apply_sky_only(
    bgr: np.ndarray,
    mode: str,
    alpha: float,
    threshold: float,
    use_luminance: bool,
    mask_path: str | None,
    use_heuristic: bool,
    feather: int,
) -> np.ndarray:
    h, w = bgr.shape[:2]

    if mask_path:
        sky = load_mask(mask_path, (h, w))
    elif use_heuristic:
        sky = heuristic_sky_mask(bgr)
    else:
        raise ValueError("sky-only requested but no mask provided and --sky-heuristic not set")

    # Feather to avoid a hard skyline edge
    if feather > 0:
        sky_u8 = (sky.astype(np.uint8) * 255)
        sky_u8 = cv2.GaussianBlur(sky_u8, (0, 0), sigmaX=feather, sigmaY=feather)
        sky_alpha = sky_u8.astype(np.float32) / 255.0
    else:
        sky_alpha = sky.astype(np.float32)

    # tonemap full image (or you can tonemap only sky region, but full is simpler and stable)
    tm = apply_tonemap(bgr, mode=mode, alpha=alpha, threshold=threshold, use_luminance=use_luminance)

    # Blend: out = sky*tm + (1-sky)*orig
    orig01 = _to_float01(bgr)
    tm01 = _to_float01(tm)
    out01 = tm01 * sky_alpha[..., None] + orig01 * (1.0 - sky_alpha[..., None])

    return _from_float01(out01, bgr.dtype)


def main():
    ap = argparse.ArgumentParser(description="Apply log/soft-knee tone mapping, optionally sky-only.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input image file or folder")
    ap.add_argument("--out", dest="out_path", required=True, help="Output file or folder")
    ap.add_argument("--mode", default="log_knee", choices=["log", "log_knee", "reinhard_log"])
    ap.add_argument("--alpha", type=float, default=8.0, help="Compression strength")
    ap.add_argument("--threshold", type=float, default=0.7, help="Knee threshold (log_knee only)")
    ap.add_argument("--no-luminance", action="store_true", help="Apply per-channel instead of luminance")
    ap.add_argument("--sky-only", action="store_true", help="Only apply tone mapping in sky region")
    ap.add_argument("--sky-mask", default=None, help="Path to sky mask image (non-zero=sky).")
    ap.add_argument("--sky-heuristic", action="store_true", help="Use heuristic sky detector if no mask provided")
    ap.add_argument("--feather", type=int, default=8, help="Feather amount for sky blending (pixels sigma)")
    ap.add_argument("--write-sky-mask", action="store_true", help="Write detected sky mask alongside output (debug)")

    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    is_folder = in_path.is_dir()
    if is_folder:
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in iter_images(in_path):
        bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            print(f"Skipping unreadable: {p}")
            continue

        if args.sky_only:
            out = apply_sky_only(
                bgr=bgr,
                mode=args.mode,
                alpha=args.alpha,
                threshold=args.threshold,
                use_luminance=(not args.no_luminance),
                mask_path=args.sky_mask,
                use_heuristic=args.sky_heuristic,
                feather=args.feather,
            )

            if args.write_sky_mask:
                if args.sky_mask:
                    sky = load_mask(args.sky_mask, bgr.shape[:2])
                else:
                    sky = heuristic_sky_mask(bgr)
        else:
            out = apply_tonemap(
                bgr,
                mode=args.mode,
                alpha=args.alpha,
                threshold=args.threshold,
                use_luminance=(not args.no_luminance),
            )
            sky = None

        if is_folder:
            rel = p.relative_to(in_path)
            out_file = out_path / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_file = out_path

        # Write output
        ok = cv2.imwrite(str(out_file), out)
        if not ok:
            print(f"Failed to write: {out_file}")

        # Optional debug mask
        if args.write_sky_mask and sky is not None:
            mask_file = out_file.with_name(out_file.stem + "_sky_mask.png")
            cv2.imwrite(str(mask_file), (sky.astype(np.uint8) * 255))

        print(f"Wrote: {out_file}")


if __name__ == "__main__":
    main()
