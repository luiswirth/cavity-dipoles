import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD_NPZ = os.path.join(ROOT, "out", "field", "field_slice.npz")

DATA = [["src/aggregate.py"]]
BENCH = [["src/plot/convergence.py"]]
FIELD = [["src/plot/field.py", "--mode", "mag"],
         ["src/plot/field.py", "--mode", "phase"],
         ["src/plot/field_lic.py"]]
ANIM = [["src/plot/field.py", "--mode", "mag", "--animate"],
        ["src/plot/field.py", "--mode", "phase", "--animate"]]


def run(step):
    print(f"\n=== {' '.join(step)} ===", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, step[0]), *step[1:]], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-anim", action="store_true")
    ap.add_argument("--skip-field", action="store_true")
    args = ap.parse_args()

    steps = DATA + BENCH
    if not args.skip_field:
        if not os.path.exists(FIELD_NPZ):
            print(f"! {FIELD_NPZ} missing -- generate it with maxwellgp "
                  "examples/cavity_field.py, or pass --skip-field", file=sys.stderr)
            sys.exit(1)
        steps += FIELD
        if not args.skip_anim:
            steps += ANIM

    for s in steps:
        run(s)
    print(f"\nall figures written to {os.path.join(ROOT, 'out', 'figs')}")


if __name__ == "__main__":
    main()
