import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cavity_epgp import load_config

from ..benchmark import config_path, reference_operator
from ..results.uq import entry_std
from .common import FIGS, setup_style

_LN = re.compile(r"ln(-?\d+(?:\.\d+)?)$")


def noise_runs(geometry):
    base = os.path.join("out", "epgp", "noise", geometry)
    rows = []
    for d in glob.glob(os.path.join(base, "ln*")):
        m = _LN.search(os.path.basename(d))
        T = glob.glob(os.path.join(d, "T_*.npy"))
        S = glob.glob(os.path.join(d, "Sigma_*.npy"))
        if m and T and S:
            rows.append((float(m.group(1)), np.load(T[0]), np.load(S[0])))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["sphere", "ellipse"], default="sphere")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = noise_runs(args.geometry)
    if not runs:
        raise SystemExit(f"no noise runs in out/epgp/noise/{args.geometry}")

    k, _, pts, e1, e2 = load_config(config_path(args.geometry))
    T_ref = reference_operator(args.geometry, k, pts, e1, e2)
    nref = np.linalg.norm(T_ref)
    M = T_ref.shape[0]

    sig_n, err, unc = [], [], []
    for ln, T, S in runs:
        sig_n.append(np.exp(ln / 2))
        err.append(float(np.linalg.norm(T - T_ref) / nref))
        unc.append(float(np.sqrt(M * np.sum(entry_std(S) ** 2)) / nref))

    setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.loglog(sig_n, err, "o-", label="reconstruction error")
    ax.loglog(sig_n, unc, "s--", label="predicted uncertainty")
    ax.set_xlabel(r"assumed noise level $\sigma_n$")
    ax.set_ylabel("relative Frobenius norm")
    ax.legend()

    os.makedirs(FIGS, exist_ok=True)
    out = args.out or os.path.join(FIGS, f"{args.geometry}_noise.svg")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
