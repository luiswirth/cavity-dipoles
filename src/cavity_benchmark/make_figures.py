import argparse
import glob
import os
import subprocess
import sys

FIGS = os.path.join("out", "figs")
MODES = ["real", "phase", "lic"]

# UQ figures: the operator value+uncertainty heatmap (from the high-fidelity ref
# operator) and the noise-influence sweep. Each renders only when its data exists.
OPERATOR = "cavity_benchmark.plot.operator"
NOISE = "cavity_benchmark.plot.noise"
KSWEEP = "cavity_benchmark.plot.ksweep"
GEOMETRIES = ["ellipse", "sphere"]

# Field slices to render: (npz file, output prefix, required). The ellipsoidal
# cavity is the default benchmark; the spherical cavity is rendered too when its
# slice exists.
FIELD_SLICES = [
    (os.path.join("out", "epgp", "ref", "ellipse", "field.npz"), "epgp_ellipse_field", True),
    (os.path.join("out", "epgp", "ref", "sphere", "field.npz"), "epgp_sphere_field", False),
]

AGG = "cavity_benchmark.results.aggregate"
CONV = "cavity_benchmark.plot.convergence"
PARETO = "cavity_benchmark.plot.pareto"
FLD = "cavity_benchmark.plot.field"

DATA = [[AGG]]
BENCH = [[CONV], [PARETO]]


def field_steps(npz, prefix, modes, animate):
    ext = "webp" if animate else "png"
    anim = ["--animate"] if animate else []
    suffix = "_anim" if animate else ""
    return [[FLD, npz, "--mode", m, *anim,
             "--out", os.path.join(FIGS, f"{prefix}_{m}{suffix}.{ext}")]
            for m in modes]


def _has_sigma(d):
    return bool(glob.glob(os.path.join(d, "Sigma_*.npy")))


def uq_steps(geometry):
    ref = os.path.join("out", "epgp", "ref", geometry)
    if not _has_sigma(ref):
        return []
    return [[OPERATOR, "--geometry", geometry, "--uq-dir", ref,
             "--out", os.path.join(FIGS, f"{geometry}_uq_operator.png")]]


def noise_steps(geometry):
    if not glob.glob(os.path.join("out", "epgp", "noise", geometry, "ln*")):
        return []
    return [[NOISE, "--geometry", geometry,
             "--out", os.path.join(FIGS, f"{geometry}_noise.svg")]]


def ksweep_steps(geometry):
    if not os.path.exists(os.path.join("out", "epgp", "ksweep", geometry, "ksweep.csv")):
        return []
    return [[KSWEEP, "--geometry", geometry,
             "--out", os.path.join(FIGS, f"{geometry}_ksweep.svg")]]


def run(step):
    print(f"\n=== {' '.join(step)} ===", flush=True)
    subprocess.run([sys.executable, "-m", step[0], *step[1:]], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-anim", action="store_true")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--skip-uq", action="store_true")
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
            steps += field_steps(npz, prefix, MODES + ["std"], animate=False)
            if not args.skip_anim:
                steps += field_steps(npz, prefix, MODES, animate=True)

    if not args.skip_uq:
        for geometry in GEOMETRIES:
            steps += uq_steps(geometry)
            steps += noise_steps(geometry)

    for geometry in GEOMETRIES:
        steps += ksweep_steps(geometry)

    for s in steps:
        run(s)
    print(f"\nall figures written to {os.path.join('out', 'figs')}")


if __name__ == "__main__":
    main()
