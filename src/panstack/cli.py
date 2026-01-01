import argparse
from panstack.pipeline import make_panstack

def main() -> None:
    parser = argparse.ArgumentParser(prog="panstack")
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--mode", choices=["symmetric", "trailing", "leading"], default="symmetric")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--weight-sigma", type=float, default=0.7)
    parser.add_argument("--face-conf", type=float, default=0.5)
    parser.add_argument("--bps", type=int, choices=[8, 16], default=8,
                        help="Bit depth for RAW decode/output. Use 16 for grading.")
    args = parser.parse_args()

    info = make_panstack(
        in_dir=args.inp,
        out_path=args.out,
        k=args.k,
        mode=args.mode,
        scale=args.scale,
        weight_sigma=args.weight_sigma,
        face_conf=args.face_conf,
        bps=args.bps,
    )
    print("OK:", info)

if __name__ == "__main__":
    main()
