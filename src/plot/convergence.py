import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from common import FIGS, ROOT, save, setup_style

BEM = os.path.join(ROOT, "out", "bem")
EPGP = os.path.join(ROOT, "out", "epgp")
FLOOR = 1e-16

C = {"epgp": "#1f77b4", "recip": "#d62728"}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _grid(ax):
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.margins(x=0.04, y=0.08)


def _epgp_axes(ax, ns, y, color, marker, ylabel, title):
    ax.plot(ns, y, marker + "-", color=color, mec="white", mew=1.0, markersize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_\mathrm{spectral}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _grid(ax)


def fig_epgp_convergence():
    e = sorted(read_csv(os.path.join(EPGP, "results.csv")),
               key=lambda r: int(r["n_spectral"]))
    ns = np.array([int(r["n_spectral"]) for r in e])
    err = np.array([max(float(r["err_vs_bem_ref"]), FLOOR) for r in e])
    rec = np.array([max(float(r["recip"]), FLOOR) for r in e])

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    _epgp_axes(ax, ns, err, C["recip"], "s",
               r"$\|T_{\mathrm{EPGP}}-T_{\mathrm{BEM}}\|/\|T_{\mathrm{BEM}}\|$",
               "EPGP convergence to BEM reference")
    save(fig, "epgp_vs_bem")

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    _epgp_axes(ax, ns, rec, C["recip"], "s",
               r"$\|T-T^{\!\top}\|/\|T\|$",
               "EPGP reciprocity error")
    save(fig, "epgp_reciprocity")


def fig_bem_reciprocity(bem):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    cmap = plt.get_cmap("viridis")

    ps = sorted({r["p"] for r in bem})
    for i, p in enumerate(ps):
        rows = sorted((r for r in bem if r["p"] == p and r["recip"] > 0),
                      key=lambda r: r["dofs"])
        if len(rows) < 2:
            continue
        col = cmap(i / max(len(ps) - 1, 1))
        ax[0].loglog([r["dofs"] for r in rows], [r["recip"] for r in rows],
                     "o-", color=col, mec="white", mew=0.8, label=f"$p={p}$")
    ax[0].set_xlabel("# DOFs"); ax[0].set_ylabel(r"$\|T-T^{\!\top}\|/\|T\|$")
    ax[0].set_title(r"$h$-refinement"); ax[0].legend(frameon=False, ncol=2)
    _grid(ax[0])

    ms = sorted({r["m"] for r in bem})
    for i, m in enumerate(ms):
        rows = sorted((r for r in bem if r["m"] == m and r["recip"] > 0),
                      key=lambda r: r["p"])
        if len(rows) < 2:
            continue
        col = cmap(i / max(len(ms) - 1, 1))
        ax[1].semilogy([r["p"] for r in rows], [r["recip"] for r in rows],
                       "D-", color=col, mec="white", mew=0.8, label=f"$m={m}$")
    ax[1].set_xlabel(r"polynomial degree $p$"); ax[1].set_ylabel(r"$\|T-T^{\!\top}\|/\|T\|$")
    ax[1].set_title(r"$p$-refinement"); ax[1].legend(frameon=False)
    ax[1].set_xticks(sorted({r["p"] for r in bem}))
    _grid(ax[1])

    fig.suptitle("BEM reciprocity error", y=1.02, fontsize=14)
    save(fig, "bem_reciprocity")


def main():
    setup_style()
    os.makedirs(FIGS, exist_ok=True)
    bem = [{"p": int(r["p"]), "m": int(r["m"]), "dofs": int(r["dofs"]),
            "recip": float(r["recip"]), "selfconv_vs_ref": float(r["selfconv_vs_ref"])}
           for r in read_csv(os.path.join(BEM, "results.csv"))]
    ref = min(bem, key=lambda r: r["recip"])

    fig_epgp_convergence()
    fig_bem_reciprocity(bem)
    stale = ("h_convergence", "p_convergence", "reciprocity", "svd_spectrum",
             "bem_validity", "bem_self_convergence", "preview", "all_preview",
             "epgp_convergence", "operator_spectrum")
    for f in os.listdir(FIGS):
        if f.rsplit(".", 1)[0] in stale:
            os.remove(os.path.join(FIGS, f))
    print(f"wrote figures to {FIGS}  (BEM ref p{ref['p']} m{ref['m']}, recip={ref['recip']:.2e})")


if __name__ == "__main__":
    main()
