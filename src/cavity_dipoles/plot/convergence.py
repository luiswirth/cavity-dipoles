import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from .common import FIGS, save, setup_style

BEM = os.path.join("out", "bem")
EPGP = os.path.join("out", "ellipse")
SPHERE = os.path.join("out", "sphere")
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
    ax.set_xlabel(r"$n_\mathrm{spec}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _grid(ax)


def fig_epgp_convergence(fmt="svg"):
    e = sorted(read_csv(os.path.join(EPGP, "results.csv")),
               key=lambda r: int(r["n_spectral"]))
    ns = np.array([int(r["n_spectral"]) for r in e])
    err = np.array([max(float(r["err_vs_bem_ref"]), FLOOR) for r in e])
    rec = np.array([max(float(r["recip"]), FLOOR) for r in e])

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    _epgp_axes(ax, ns, err, C["recip"], "D",
               r"$\varepsilon = \|\mathbf{T}_{\mathrm{EPGP}}-\mathbf{T}_{\mathrm{BEM}}\|/\|\mathbf{T}_{\mathrm{BEM}}\|$",
               "EPGP convergence to BEM reference")
    save(fig, "epgp_vs_bem", fmt)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    _epgp_axes(ax, ns, rec, C["recip"], "D",
               r"$\rho = \|\mathbf{T}-\mathbf{T}^{\!\top}\|/\|\mathbf{T}\|$",
               "EPGP reciprocity error")
    save(fig, "epgp_reciprocity", fmt)


def fig_ksweep(fmt="svg"):
    path = os.path.join(EPGP, "ksweep.csv")
    if not os.path.exists(path):
        return
    from matplotlib.lines import Line2D

    rows = read_csv(path)
    ks = sorted({float(r["k"]) for r in rows})
    R = float(rows[0]["R"])
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
    for i, k in enumerate(ks):
        sel = sorted((r for r in rows if float(r["k"]) == k),
                     key=lambda r: int(r["n_spectral"]))
        ns = np.array([int(r["n_spectral"]) for r in sel])
        rec = np.array([max(float(r["recip"]), FLOOR) for r in sel])
        color = cmap(i / max(len(ks) - 1, 1))
        ax.plot(ns, rec, "D-", color=color, mec="white", mew=0.8,
                markersize=6, label=f"$k={k:g}$")
        # band-limit prediction: drop at sqrt(n_spec) = k R, i.e. n_spec = (k R)^2
        ax.axvline((k * R) ** 2, color=color, ls=":", lw=1.6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_\mathrm{spec}$")
    ax.set_ylabel(r"$\rho = \|\mathbf{T}-\mathbf{T}^{\!\top}\|/\|\mathbf{T}\|$")

    handles = [Line2D([0], [0], color=cmap(i / max(len(ks) - 1, 1)),
                      marker="D", mec="white", mew=0.8, markersize=6)
               for i in range(len(ks))]
    labels = [f"$k={k:g}$" for k in ks]
    handles.append(Line2D([0], [0], color="0.35", ls=":", lw=1.6))
    labels.append(r"$n_\mathrm{spec}=(kR)^2$")
    ax.legend(handles, labels, frameon=False, ncol=2)
    _grid(ax)
    save(fig, "epgp_ksweep", fmt)


def fig_bem_reciprocity(bem, fmt="svg"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    cmap = plt.get_cmap("viridis")

    ms_all = sorted({r["m"] for r in bem})
    ps_all = sorted({r["p"] for r in bem})

    for i, p in enumerate(ps_all):
        rows = sorted((r for r in bem if r["p"] == p and r["recip"] > 0),
                      key=lambda r: r["m"])
        if len(rows) < 2:
            continue
        ax[0].semilogy([r["m"] for r in rows], [r["recip"] for r in rows],
                       "D-", color=cmap(i / max(len(ps_all) - 1, 1)),
                       mec="white", mew=0.8, label=f"$p={p}$")
    ax[0].set_xlabel(r"mesh level $m$"); ax[0].set_ylabel(r"$\rho = \|\mathbf{T}-\mathbf{T}^{\!\top}\|/\|\mathbf{T}\|$")
    ax[0].set_title(r"$h$-refinement"); ax[0].legend(frameon=False, ncol=2)
    ax[0].set_xticks(ms_all)
    _grid(ax[0])

    for i, m in enumerate(ms_all):
        rows = sorted((r for r in bem if r["m"] == m and r["recip"] > 0),
                      key=lambda r: r["p"])
        if len(rows) < 2:
            continue
        ax[1].semilogy([r["p"] for r in rows], [r["recip"] for r in rows],
                       "D-", color=cmap(i / max(len(ms_all) - 1, 1)),
                       mec="white", mew=0.8, label=f"$m={m}$")
    ax[1].set_xlabel(r"polynomial degree $p$"); ax[1].set_ylabel(r"$\rho = \|\mathbf{T}-\mathbf{T}^{\!\top}\|/\|\mathbf{T}\|$")
    ax[1].set_title(r"$p$-refinement"); ax[1].legend(frameon=False)
    ax[1].set_xticks(ps_all)
    _grid(ax[1])

    fig.suptitle("BEM reciprocity error", y=1.02, fontsize=14)
    save(fig, "bem_reciprocity", fmt)


def fig_sphere_convergence(fmt="svg"):
    path = os.path.join(SPHERE, "results.csv")
    if not os.path.exists(path):
        return
    e = sorted(read_csv(path), key=lambda r: int(r["n_spectral"]))
    ns = np.array([int(r["n_spectral"]) for r in e])
    err = np.array([max(float(r["err_vs_analytic"]), FLOOR) for r in e])
    rec = np.array([max(float(r["recip"]), FLOOR) for r in e])
    mp = os.path.join(SPHERE, "multipole.csv")
    kR = float(read_csv(mp)[0]["kR"]) if os.path.exists(mp) else None

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.plot(ns, err, "D-", color=C["recip"], mec="white", mew=1.0, markersize=8,
            label=r"$\|\mathbf{T}_{\mathrm{EPGP}}-\mathbf{T}_\star\|/\|\mathbf{T}_\star\|$")
    ax.plot(ns, rec, "o-", color=C["epgp"], mec="white", mew=1.0, markersize=7,
            label=r"$\rho = \|\mathbf{T}-\mathbf{T}^{\!\top}\|/\|\mathbf{T}\|$")
    if kR is not None:
        ax.axvline(kR**2, color="0.4", ls=":", lw=1.6, label=r"$n_\mathrm{spec}=(kR)^2$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_\mathrm{spec}$")
    ax.set_ylabel("relative error")
    ax.set_title("EPGP convergence to analytic sphere operator")
    ax.legend(frameon=False)
    _grid(ax)
    save(fig, "sphere_convergence", fmt)


def fig_sphere_multipole(fmt="svg"):
    path = os.path.join(SPHERE, "multipole.csv")
    if not os.path.exists(path):
        return
    r = read_csv(path)
    ll = np.array([int(x["l"]) for x in r])
    nrm = np.array([max(float(x["norm"]), FLOOR) for x in r])
    kR = float(r[0]["kR"])

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.semilogy(ll, nrm, "D-", color=C["epgp"], mec="white", mew=1.0, markersize=7)
    ax.axvline(kR, color="0.4", ls=":", lw=1.6, label=r"$l = kR$")
    ax.set_xlabel(r"degree $l$")
    ax.set_ylabel(r"$\|\mathbf{T}_l\|$")
    ax.set_title("Multipole spectrum of the sphere reaction operator")
    ax.legend(frameon=False)
    _grid(ax)
    save(fig, "sphere_multipole", fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["svg", "png"], default="svg")
    fmt = ap.parse_args().format

    setup_style()
    os.makedirs(FIGS, exist_ok=True)
    bem = [{"p": int(r["p"]), "m": int(r["m"]), "dofs": int(r["dofs"]),
            "recip": float(r["recip"]), "selfconv_vs_ref": float(r["selfconv_vs_ref"])}
           for r in read_csv(os.path.join(BEM, "results.csv"))]
    ref = min(bem, key=lambda r: r["recip"])

    fig_epgp_convergence(fmt)
    fig_ksweep(fmt)
    fig_bem_reciprocity(bem, fmt)
    fig_sphere_convergence(fmt)
    fig_sphere_multipole(fmt)
    stale = ("h_convergence", "p_convergence", "reciprocity", "svd_spectrum",
             "bem_validity", "bem_self_convergence", "preview", "all_preview",
             "epgp_convergence", "operator_spectrum")
    for f in os.listdir(FIGS):
        if f.rsplit(".", 1)[0] in stale:
            os.remove(os.path.join(FIGS, f))
    print(f"wrote figures to {FIGS}  (BEM ref p{ref['p']} m{ref['m']}, recip={ref['recip']:.2e})")


if __name__ == "__main__":
    main()
