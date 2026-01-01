import argparse
from panstack.pipeline import make_panstack


def main() -> None:
    parser = argparse.ArgumentParser(prog="panstack")

    parser.add_argument("--in", dest="inp", required=True, help="Input burst folder containing ARW files")

    # out can be a file or a directory
    parser.add_argument(
        "--out",
        dest="out",
        required=True,
        help="Output file (out.tif/out.png) OR output directory for auto-naming",
    )

    # stacking controls
    parser.add_argument("--k", type=int, default=10, help="Stack window size (frames around base)")
    parser.add_argument("--mode", choices=["symmetric", "trailing", "leading"], default="trailing")
    parser.add_argument("--scale", type=float, default=0.5, help="Decode scale (1.0=full res)")
    parser.add_argument("--weight-sigma", type=float, default=0.7, help="Gaussian weight sigma (relative)")
    parser.add_argument("--align-bg", choices=["on", "off"], default="on",
                        help="Align background frames to base before stacking")

    # face controls
    parser.add_argument("--freeze-faces", default="1",
                        help="Faces in base frame to keep sharp: '1' or '1,2' or 'all'")
    parser.add_argument("--preview-faces", action="store_true",
                        help="Write a preview image with numbered face boxes and exit")
    parser.add_argument("--face-conf", type=float, default=0.5, help="Face detection confidence threshold")

    # motion blur controls
    parser.add_argument("--extra-blur", type=float, default=0.0,
                        help="Extra motion blur length in pixels applied to BACKGROUND only (0 disables)")
    parser.add_argument("--blur-angle", default="auto",
                        help="Motion blur angle in degrees, or 'auto' (estimated from burst motion)")

    # detection-only brighten (does NOT change final render)
    parser.add_argument("--detect-gain", type=float, default=1.6,
                        help="Detection proxy gain (>1 brightens). Only affects face detect/alignment.")
    parser.add_argument("--detect-gamma", type=float, default=0.6,
                        help="Detection proxy gamma (<1 brightens shadows). Only affects face detect/alignment.")
    parser.add_argument("--detect-clahe", action="store_true",
                        help="Apply CLAHE to detection proxy (helps low-contrast faces).")

    # output + color management
    parser.add_argument("--bps", type=int, choices=[8, 16], default=16,
                        help="Output bit depth (8 or 16).")
    parser.add_argument("--auto-expose", choices=["on", "off"], default="on",
                        help="Auto exposure on the final (linear) composite before sRGB encoding.")
    parser.add_argument("--exposure-gain", type=float, default=1.0,
                        help="Manual exposure multiplier in linear space (applied after auto-expose if enabled).")
    parser.add_argument("--expose-percentile", type=float, default=99.5,
                        help="Percentile for auto-expose (e.g., 99.5).")
    parser.add_argument("--expose-target", type=float, default=0.98,
                        help="Target value for percentile luminance (0..1).")
    parser.add_argument("--min-gain", type=float, default=0.25, help="Min auto-expose gain clamp.")
    parser.add_argument("--max-gain", type=float, default=8.0, help="Max auto-expose gain clamp.")

    args = parser.parse_args()

    # Parse freeze faces
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

    align_bg = (args.align_bg == "on")

    # Parse blur angle
    blur_angle_s = args.blur_angle.strip().lower()
    blur_angle_val = "auto" if blur_angle_s == "auto" else float(blur_angle_s)

    info = make_panstack(
        in_dir=args.inp,
        out_path=args.out,
        k=args.k,
        mode=args.mode,
        scale=args.scale,
        weight_sigma=args.weight_sigma,
        face_conf=args.face_conf,
        align_bg=align_bg,
        freeze_faces=freeze_faces,
        preview_faces=args.preview_faces,
        extra_blur=args.extra_blur,
        blur_angle=blur_angle_val,
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
    )

    print("OK:", info)


if __name__ == "__main__":
    main()
