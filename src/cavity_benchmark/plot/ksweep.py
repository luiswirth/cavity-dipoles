# Wavenumber sweep: reads the solver's per-k condition number
# (out/epgp/ksweep/{geometry}/ksweep.csv) and plots it against k. Spikes mark
# wavenumbers near interior cavity resonances, where conditioning degrades.
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .common import FIGS, setup_style


def load(geometry):
    path = os.path.join("out", "epgp", "ksweep", geometry, "ksweep.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ks = [float(r["k"]) for r in rows]
    cond = [float(r["cond"]) for r in rows]
    return ks, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FIGS, "ksweep.svg"))
    args = ap.parse_args()

    setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for geometry, style in (("ellipse", "o-"), ("sphere", "s--")):
        data = load(geometry)
        if data is None:
            continue
        ks, cond = data
        ax.semilogy(ks, cond, style, ms=3, label=geometry)
    ax.set_xlabel(r"wavenumber $k$")
    ax.set_ylabel("conditioning number")
    ax.legend()

    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
