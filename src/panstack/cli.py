import argparse
from panstack.pipeline import make_panstack


def main() -> None:
    p = argparse.ArgumentParser(prog="panstack")

    p.add_argument("--in", dest="inp", required=True, help="Input burst folder containing ARW files")
    p.add_argument("--out", dest="out", required=True,
                   help="Output file (out.tif/out.png) OR output directory for auto-naming")

    p.add_argument("--k", type=int, default=10)
    p.add_argument("--mode", choices=["symmetric", "trailing", "leading"], default="trailing")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--weight-sigma", type=float, default=0.7)
    p.add_argument("--align-bg", choices=["on", "off"], default="on")

    p.add_argument("--bg-mode", choices=["blur", "stabilize", "stack"], default="blur")
    p.add_argument("--bg-stack", choices=["mean", "median"], default="mean")

    p.add_argument("--blur-scope", choices=["all", "foreground"], default="all")
    p.add_argument("--bg-max-kernel", type=int, default=-1)

    p.add_argument("--fg-motion", choices=["camera", "residual"], default="residual")
    p.add_argument("--kernel-scale", type=float, default=1.0)

    p.add_argument("--motion-roi", choices=["upper", "arms"], default="arms")
    p.add_argument("--motion-feature", choices=["intensity", "edges"], default="edges")

    p.add_argument("--freeze-faces", default="all")
    p.add_argument("--freeze-region", choices=["face", "upper", "full"], default="full")
    p.add_argument("--preview-faces", action="store_true")
    p.add_argument("--face-conf", type=float, default=0.5)

    p.add_argument("--detect-gain", type=float, default=1.6)
    p.add_argument("--detect-gamma", type=float, default=0.6)
    p.add_argument("--detect-clahe", action="store_true")

    p.add_argument("--bps", type=int, choices=[8, 16], default=16)
    p.add_argument("--auto-expose", choices=["on", "off"], default="on")
    p.add_argument("--exposure-gain", type=float, default=1.0)
    p.add_argument("--expose-percentile", type=float, default=99.5)
    p.add_argument("--expose-target", type=float, default=0.98)
    p.add_argument("--min-gain", type=float, default=0.25)
    p.add_argument("--max-gain", type=float, default=8.0)

    p.add_argument("--blur-model", choices=["none", "trajectory"], default="trajectory")
    p.add_argument("--bg-blur", type=float, default=1.0)
    p.add_argument("--body-blur", type=float, default=0.8)
    p.add_argument("--kernel-softness", type=float, default=0.0)
    p.add_argument("--min-kernel", type=int, default=9)

    p.add_argument("--debug-overlay", action="store_true")
    p.add_argument("--debug-out", default="")

    p.add_argument("--segmentation", choices=["off", "auto", "on"], default="auto")
    p.add_argument("--seg-thresh", type=float, default=0.5)
    p.add_argument("--seg-erode", type=int, default=0)
    p.add_argument("--seg-dilate", type=int, default=0)
    p.add_argument("--seg-feather", type=float, default=25.0)

    p.add_argument("--face-opacity", type=float, default=1.0)
    p.add_argument("--body-opacity", type=float, default=1.0)

    p.add_argument("--align-model", choices=["affine", "homography"], default="homography")
    p.add_argument("--align-quality", type=float, default=0.22)
    p.add_argument("--align-ransac", type=float, default=3.0)
    p.add_argument("--align-features", type=int, default=8000)

    args = p.parse_args()

    freeze = args.freeze_faces.strip().lower()
    if freeze == "all":
        freeze_faces = "all"
    else:
        freeze_faces = []
        if freeze:
            for part in freeze.split(","):
                part = part.strip()
                if part:
                    freeze_faces.append(int(part))

    bg_mode = args.bg_mode
    if bg_mode == "stack":
        bg_mode = "stabilize"

    info = make_panstack(
        in_dir=args.inp,
        out_path=args.out,
        k=args.k,
        mode=args.mode,
        scale=args.scale,
        weight_sigma=args.weight_sigma,
        face_conf=args.face_conf,
        align_bg=(args.align_bg == "on"),
        bg_mode=bg_mode,
        bg_stack=args.bg_stack,
        blur_scope=args.blur_scope,
        bg_max_kernel=args.bg_max_kernel,
        fg_motion=args.fg_motion,
        kernel_scale=args.kernel_scale,
        motion_roi=args.motion_roi,
        motion_feature=args.motion_feature,
        freeze_faces=freeze_faces,
        freeze_region=args.freeze_region,
        preview_faces=args.preview_faces,
        detect_gain=args.detect_gain,
        detect_gamma=args.detect_gamma,
        detect_clahe=args.detect_clahe,
        out_bps=args.bps,
        auto_expose=(args.auto_expose == "on"),
        exposure_gain=args.exposure_gain,
        expose_percentile=args.expose_percentile,
        expose_target=args.expose_target,
        min_gain=args.min_gain,
        max_gain=args.max_gain,
        blur_model=args.blur_model,
        bg_blur=args.bg_blur,
        body_blur=args.body_blur,
        kernel_softness=args.kernel_softness,
        min_kernel=args.min_kernel,
        debug_overlay=args.debug_overlay,
        debug_out=args.debug_out,
        segmentation=args.segmentation,
        seg_thresh=args.seg_thresh,
        seg_erode_px=args.seg_erode,
        seg_dilate_px=args.seg_dilate,
        seg_feather_px=args.seg_feather,
        face_opacity=args.face_opacity,
        body_opacity=args.body_opacity,
        align_model=args.align_model,
        align_quality_thresh=args.align_quality,
        align_ransac_thresh=args.align_ransac,
        align_nfeatures=args.align_features,
    )

    print("OK:", info)


if __name__ == "__main__":
    main()
