"""Render BEM convergence figures from out/bem/results.csv into out/figs/.

Figures (SVG):
  1. h-convergence  : err-vs-finest vs #dofs at p=2 (log-log).
  2. p-convergence  : err-vs-finest vs poly_deg at m=3 (semilog-y).
  3. reciprocity    : ||T-T^T||/||T|| vs #dofs at p=2 (log-log).
  4. svd spectrum   : singular values of the finest operator (justifies N).
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from compare import load_bem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEM = os.path.join(ROOT, "out", "bem")
FIGS = os.path.join(ROOT, "out", "figs")


def read_results():
    rows = []
    with open(os.path.join(BEM, "results.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({"p": int(r["p"]), "m": int(r["m"]), "dofs": int(r["dofs"]),
                         "recip": float(r["recip"]), "err": float(r["err_vs_finest"])})
    return rows


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = read_results()
    finest = max(rows, key=lambda r: r["dofs"])

    # 1. h-convergence at p=2 (drop the reference point, err==0)
    h = sorted((r for r in rows if r["p"] == 2 and r is not finest), key=lambda r: r["dofs"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog([r["dofs"] for r in h], [r["err"] for r in h], "o-")
    ax.set_xlabel("# DOFs"); ax.set_ylabel(r"$\|T-T_{\mathrm{ref}}\|/\|T_{\mathrm{ref}}\|$")
    ax.set_title("BEM h-convergence (p=2), ellipsoid"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "h_convergence.svg")); plt.close(fig)

    # 2. p-convergence at m=3
    pser = sorted((r for r in rows if r["m"] == 3), key=lambda r: r["p"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy([r["p"] for r in pser], [max(r["err"], 1e-16) for r in pser], "s-")
    ax.set_xlabel("polynomial degree p"); ax.set_ylabel(r"$\|T-T_{\mathrm{ref}}\|/\|T_{\mathrm{ref}}\|$")
    ax.set_title("BEM p-convergence (m=3), ellipsoid"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "p_convergence.svg")); plt.close(fig)

    # 3. reciprocity at p=2
    h2 = sorted((r for r in rows if r["p"] == 2), key=lambda r: r["dofs"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog([r["dofs"] for r in h2], [r["recip"] for r in h2], "o-")
    ax.set_xlabel("# DOFs"); ax.set_ylabel(r"$\|T-T^{\top}\|/\|T\|$")
    ax.set_title("BEM reciprocity error (p=2), ellipsoid"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "reciprocity.svg")); plt.close(fig)

    # 4. SVD spectrum of finest operator
    Tref = load_bem(os.path.join(BEM, f"T_bem_ell_p{finest['p']}_m{finest['m']}.dat"))
    s = np.linalg.svd(Tref, compute_uv=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(np.arange(1, len(s) + 1), s, ".-")
    ax.set_xlabel("index"); ax.set_ylabel("singular value")
    ax.set_title(f"Operator spectrum (finest: p{finest['p']} m{finest['m']})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "svd_spectrum.svg")); plt.close(fig)

    print(f"wrote 4 SVG figures to {FIGS}")


if __name__ == "__main__":
    main()
