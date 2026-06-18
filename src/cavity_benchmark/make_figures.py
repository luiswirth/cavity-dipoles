import argparse
import os
import subprocess
import sys

FIGS = os.path.join("out", "figs")
MODES = ["real", "phase", "lic"]

# Field slices to render: (npz file, output prefix, required). The ellipsoidal
# cavity is the default benchmark; the spherical cavity is rendered too when its
# slice exists.
FIELD_SLICES = [
    (os.path.join("out", "epgp", "ref", "ellipse", "field.npz"), "epgp_ellipse_field", True),
    (os.path.join("out", "epgp", "ref", "sphere", "field.npz"), "epgp_sphere_field", False),
]

AGG = "cavity_benchmark.results.aggregate"
CONV = "cavity_benchmark.plot.convergence"
FLD = "cavity_benchmark.plot.field"

DATA = [[AGG]]
BENCH = [[CONV]]


def field_steps(npz, prefix, animate):
    ext = "webp" if animate else "png"
    anim = ["--animate"] if animate else []
    suffix = "_anim" if animate else ""
    return [[FLD, npz, "--mode", m, *anim,
             "--out", os.path.join(FIGS, f"{prefix}_{m}{suffix}.{ext}")]
            for m in MODES]


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
        for npz, prefix, required in FIELD_SLICES:
            if not os.path.exists(npz):
                if required:
                    print(f"! {npz} missing -- generate it with "
                          "'uv run epgp-operator field' in cavity-epgp, or pass --skip-field",
                          file=sys.stderr)
                    sys.exit(1)
                continue
            steps += field_steps(npz, prefix, animate=False)
            if not args.skip_anim:
                steps += field_steps(npz, prefix, animate=True)

    for s in steps:
        run(s)
    print(f"\nall figures written to {os.path.join('out', 'figs')}")


if __name__ == "__main__":
    main()
