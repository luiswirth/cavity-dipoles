# Wavenumber sweep: reads a solver's per-k condition number
# (out/{solver}/ksweep/{geometry}/ksweep.csv) and plots it against k. For BEM
# the spikes mark interior cavity resonances; for EPGP the trend reflects the
# plane-wave basis conditioning.
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .common import FIGS, setup_style


def load(solver, geometry):
    path = os.path.join("out", solver, "ksweep", geometry, "ksweep.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no ksweep data in {path}")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ks = [float(r["k"]) for r in rows]
    cond = [float(r["cond"]) for r in rows]
    return ks, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["epgp", "bem"], default="epgp")
    ap.add_argument("--geometry", choices=["sphere", "ellipse"], default="sphere")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ks, cond = load(args.solver, args.geometry)

    setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.semilogy(ks, cond, "o-", ms=3)
    ax.set_xlabel(r"wavenumber $k$")
    ax.set_ylabel("conditioning number")

    os.makedirs(FIGS, exist_ok=True)
    out = args.out or os.path.join(FIGS, f"{args.geometry}_{args.solver}_ksweep.svg")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
