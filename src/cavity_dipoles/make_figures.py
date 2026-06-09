import argparse
import os
import subprocess
import sys

FIELD_NPZ = os.path.join("out", "field", "field_slice.npz")

AGG = "cavity_dipoles.results.aggregate"
CONV = "cavity_dipoles.plot.convergence"
FLD = "cavity_dipoles.plot.field"

DATA = [[AGG]]
BENCH = [[CONV]]
FIELD = [[FLD, "--mode", "real"],
         [FLD, "--mode", "phase"],
         [FLD, "--mode", "lic"]]
ANIM = [[FLD, "--mode", "real", "--animate"],
        [FLD, "--mode", "phase", "--animate"],
        [FLD, "--mode", "lic", "--animate"]]


def run(step):
    print(f"\n=== {' '.join(step)} ===", flush=True)
    subprocess.run([sys.executable, "-m", step[0], *step[1:]], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-anim", action="store_true")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--png", action="store_true",
                    help="also emit PNG versions of the line plots (default: SVG only)")
    args = ap.parse_args()

    steps = DATA + BENCH
    if args.png:
        steps += [[CONV, "--format", "png"]]
    if not args.skip_field:
        if not os.path.exists(FIELD_NPZ):
            print(f"! {FIELD_NPZ} missing -- generate it with "
                  "'python -m cavity_dipoles.epgp.operators field', or pass --skip-field",
                  file=sys.stderr)
            sys.exit(1)
        steps += FIELD
        if not args.skip_anim:
            steps += ANIM

    for s in steps:
        run(s)
    print(f"\nall figures written to {os.path.join('out', 'figs')}")


if __name__ == "__main__":
    main()
