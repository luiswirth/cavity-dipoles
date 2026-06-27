import argparse
import csv
import os

import matplotlib.pyplot as plt

from .common import FIGS, save, setup_style

BEM_ELLIPSE  = os.path.join("out", "bem",  "grid", "ellipse")
BEM_SPHERE   = os.path.join("out", "bem",  "grid", "sphere")
EPGP_ELLIPSE = os.path.join("out", "epgp", "grid", "ellipse")
EPGP_SPHERE  = os.path.join("out", "epgp", "grid", "sphere")

L_RHO = r"$\rho = \|\mathbf{T}-\mathbf{T}^{\!\top}\|/\|\mathbf{T}\|$"
L_ERR = r"relative error $\varepsilon$"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _grid(ax):
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.margins(x=0.04, y=0.08)


def _epgp_conv_fig(rows, ycol, ylabel, savename, fmt):
    nbs = sorted({int(r["n_boundary"]) for r in rows})
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    for i, nb in enumerate(nbs):
        rs = sorted((r for r in rows if int(r["n_boundary"]) == nb and float(r[ycol]) > 0),
                    key=lambda r: int(r["n_spectral"]))
        if len(rs) < 2:
            continue
        ax.loglog([int(r["n_spectral"]) for r in rs], [float(r[ycol]) for r in rs],
                  "D-", color=cmap(i / max(len(nbs) - 1, 1)), mec="white", mew=0.8,
                  label=fr"$N_b={nb}$")
    ax.set_xlabel(r"$N_s$"); ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2); _grid(ax)
    save(fig, savename, fmt)


def fig_epgp_ellipse_convergence(fmt="svg"):
    path = os.path.join(EPGP_ELLIPSE, "results.csv")
    if not os.path.exists(path):
        return
    _epgp_conv_fig(read_csv(path), "recip", L_RHO, "epgp_ellipse_convergence", fmt)


def fig_epgp_sphere_convergence(fmt="svg"):
    path = os.path.join(EPGP_SPHERE, "results.csv")
    if not os.path.exists(path):
        return
    _epgp_conv_fig(read_csv(path), "err", L_ERR, "epgp_sphere_convergence", fmt)


def _bem_conv_fig(rows, ycol, ylabel, savename, fmt):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    cmap = plt.get_cmap("viridis")
    ms_all = sorted({r["m"] for r in rows})
    ps_all = sorted({r["p"] for r in rows})

    all_x, all_y = [], []
    for i, p in enumerate(ps_all):
        rs = sorted((r for r in rows if r["p"] == p and r[ycol] > 0), key=lambda r: r["m"])
        if len(rs) < 2:
            continue
        xs = [r["dofs"] for r in rs]; ys = [r[ycol] for r in rs]
        all_x += xs; all_y += ys
        ax[0].loglog(xs, ys, "D-", color=cmap(i / max(len(ps_all) - 1, 1)),
                     mec="white", mew=0.8, label=f"$p={p}$")
    if all_x:
        x0, x1 = min(all_x), max(all_x)
        y0 = max(all_y)
        ax[0].plot([x0, x1], [y0, y0 * (x1 / x0) ** -1.5],
                   "k--", lw=1.0, alpha=0.6, label=r"$\propto N^{-3/2}$")
    ax[0].set_xlabel(r"degrees of freedom $N$"); ax[0].set_ylabel(ylabel)
    ax[0].set_title(r"$h$-refinement"); ax[0].legend(frameon=False, ncol=2)
    _grid(ax[0])

    for i, m in enumerate(ms_all):
        rs = sorted((r for r in rows if r["m"] == m and r[ycol] > 0), key=lambda r: r["p"])
        if len(rs) < 2:
            continue
        ax[1].semilogy([r["p"] for r in rs], [r[ycol] for r in rs],
                       "D-", color=cmap(i / max(len(ms_all) - 1, 1)),
                       mec="white", mew=0.8, label=f"$m={m}$")
    ax[1].set_xlabel(r"polynomial degree $p$"); ax[1].set_ylabel(ylabel)
    ax[1].set_title(r"$p$-refinement"); ax[1].legend(frameon=False)
    ax[1].set_xticks(ps_all); _grid(ax[1])

    save(fig, savename, fmt)


def fig_bem_ellipse_convergence(fmt="svg"):
    path = os.path.join(BEM_ELLIPSE, "results.csv")
    if not os.path.exists(path):
        return
    rows = [{"p": int(r["p"]), "m": int(r["m"]), "dofs": int(r["dofs"]), "recip": float(r["recip"])}
            for r in read_csv(path)]
    _bem_conv_fig(rows, "recip", L_RHO, "bem_ellipse_convergence", fmt)


def fig_bem_sphere_convergence(fmt="svg"):
    path = os.path.join(BEM_SPHERE, "results.csv")
    if not os.path.exists(path):
        return
    rows = [{"p": int(r["p"]), "m": int(r["m"]), "dofs": int(r["dofs"]), "err": float(r["err"])}
            for r in read_csv(path)]
    _bem_conv_fig(rows, "err", L_ERR, "bem_sphere_convergence", fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["svg", "png"], default="svg")
    fmt = ap.parse_args().format

    setup_style()
    os.makedirs(FIGS, exist_ok=True)

    fig_bem_ellipse_convergence(fmt)
    fig_bem_sphere_convergence(fmt)
    fig_epgp_ellipse_convergence(fmt)
    fig_epgp_sphere_convergence(fmt)

    stale = ("h_convergence", "p_convergence", "reciprocity", "svd_spectrum",
             "bem_validity", "bem_self_convergence", "preview", "all_preview",
             "epgp_convergence", "operator_spectrum",
             "bem_reciprocity", "epgp_reciprocity", "epgp_vs_bem",
             "bem_convergence", "ellipse_convergence", "sphere_convergence",
             "sphere_multipole",
             "ellipse_bem_convergence", "ellipse_epgp_convergence",
             "sphere_epgp_convergence", "ellipse_epgp_rate", "sphere_epgp_rate",
             "ellipse_epgp_field_real", "ellipse_epgp_field_phase",
             "ellipse_epgp_field_lic", "ellipse_epgp_field_real_anim",
             "ellipse_epgp_field_phase_anim", "ellipse_epgp_field_lic_anim",
             "sphere_epgp_field_real", "sphere_epgp_field_phase",
             "sphere_epgp_field_lic", "sphere_epgp_field_real_anim",
             "sphere_epgp_field_phase_anim", "sphere_epgp_field_lic_anim")
    for f in os.listdir(FIGS):
        if f.rsplit(".", 1)[0] in stale:
            os.remove(os.path.join(FIGS, f))
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
