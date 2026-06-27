import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn

from cavity_epgp import load_config

from ..benchmark import config_path
from .common import FIGS, setup_style


def load(geometry):
    path = os.path.join("out", "epgp", "ksweep", geometry, "ksweep.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no ksweep data in {path}")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ks = [float(r["k"]) for r in rows]
    vals = [float(r["sigma_min"]) for r in rows]
    return ks, vals


def sphere_resonances(R, kmin, kmax, lmax=12):
    def psip(l, x):
        return spherical_jn(l, x) + x * spherical_jn(l, x, derivative=True)

    xs = np.linspace(1e-3, kmax * R + 1.0, 6000)
    out = []
    for l in range(1, lmax + 1):
        for f in (lambda x, l=l: spherical_jn(l, x), lambda x, l=l: psip(l, x)):
            v = np.array([f(x) for x in xs])
            for i in np.where(v[:-1] * v[1:] < 0)[0]:
                k = brentq(f, xs[i], xs[i + 1]) / R
                if kmin <= k <= kmax:
                    out.append(k)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["sphere", "ellipse"], default="sphere")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ks, vals = load(args.geometry)
    k_bench = float(load_config(config_path(args.geometry))[0])

    setup_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")

    if args.geometry == "sphere":
        R = float(load_config(config_path("sphere"))[1][0])
        for j, kr in enumerate(sphere_resonances(R, min(ks), max(ks))):
            ax.axvline(kr, color="0.75", ls="--", lw=0.9, zorder=0,
                       label="analytic resonance" if j == 0 else None)

    ax.axvline(k_bench, color="C1", ls="-", lw=1.5, zorder=1, label=f"$k={k_bench:g}$")
    ax.semilogy(ks, vals, "-", zorder=2)
    ax.set_xlabel(r"wavenumber $k$")
    ax.set_ylabel(r"$\sigma_\mathrm{min}$")
    ax.legend()

    os.makedirs(FIGS, exist_ok=True)
    out = args.out or os.path.join(FIGS, f"{args.geometry}_epgp_ksweep.svg")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
