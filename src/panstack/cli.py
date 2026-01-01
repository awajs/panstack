import argparse

def main():
    parser = argparse.ArgumentParser(prog="panstack")
    parser.add_argument("--in", dest="inp", required=True,
                        help="Input burst folder containing ARW files")
    parser.add_argument("--out", dest="out", required=True,
                        help="Output image path (png/jpg)")
    parser.add_argument("--k", type=int, default=8,
                        help="Blur strength: number of frames around base")
    parser.add_argument("--mode", choices=["symmetric", "trailing", "leading"],
                        default="symmetric", help="Frame selection mode")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Decode scale for speed (1.0 = full res)")
    args = parser.parse_args()

    # TODO: implement pipeline.make_panstack(...) and call it here.
    raise SystemExit("CLI stub. Next: implement src/panstack/pipeline.py and wire it here.")

if __name__ == "__main__":
    main()
