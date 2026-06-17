import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from ..benchmark import GEOMETRIES
from .common import FIGS, save, setup_style

ELLIPSE_BEM = os.path.join("out", "ellipse_bem")
ELLIPSE_EPGP = os.path.join("out", "ellipse_epgp")
SPHERE_EPGP = os.path.join("out", "sphere_epgp")
FLOOR = 1e-16

# Shared metric styles. Every convergence plot draws whichever of these exist:
# an error against an external reference, the reference-free reciprocity error,
# and the self-convergence distance to the finest run of the family.
ERR = dict(color="#d62728", marker="D", ms=8)    # vs external reference
RECIP = dict(color="#1f77b4", marker="o", ms=7)  # reciprocity rho
SELF = dict(color="#2ca02c", marker="s", ms=7)   # self-convergence delta

L_RHO = r"$\rho = \|\mathbf{T}-\mathbf{T}^{\!\top}\|/\|\mathbf{T}\|$"
L_DELTA = r"$\delta = \|\mathbf{T}-\mathbf{T}_{\mathrm{ref}}\|/\|\mathbf{T}_{\mathrm{ref}}\|$"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _grid(ax):
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.margins(x=0.04, y=0.08)


def _series(ax, x, y, style, label):
    """One metric curve. Non-positive entries (e.g. the reference run's own
    self-convergence, which is exactly zero) are dropped rather than clamped."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = y > 0
    ax.plot(x[m], y[m], style["marker"] + "-", color=style["color"],
            mec="white", mew=1.0, markersize=style["ms"], label=label)


def _conv_ax(ax, x, series, xlabel, xlog2=True):
    """Unified convergence axis: log-y, shared styling, one line per metric.

    series: list of (values, style, label).
    """
    for y, style, label in series:
        _series(ax, x, y, style, label)
    if xlog2:
        ax.set_xscale("log", base=2)
    else:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative error")
    ax.legend(frameon=False)
    _grid(ax)


def fig_sphere_epgp_convergence(fmt="svg"):
    path = os.path.join(SPHERE_EPGP, "results.csv")
    if not os.path.exists(path):
        return
    e = sorted(read_csv(path), key=lambda r: int(r["n_spectral"]))
    ns = [int(r["n_spectral"]) for r in e]
    err = [float(r["err_vs_analytic"]) for r in e]
    rec = [float(r["recip"]) for r in e]
    self = [float(r["selfconv_vs_finest"]) for r in e]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    _conv_ax(ax, ns, [
        (err, ERR, r"$\varepsilon_\star = \|\mathbf{T}_{\mathrm{EPGP}}-\mathbf{T}_\star\|/\|\mathbf{T}_\star\|$"),
        (rec, RECIP, L_RHO),
        (self, SELF, L_DELTA),
    ], r"$n_\mathrm{spec}$")
    save(fig, "sphere_epgp_convergence", fmt)


def fig_ellipse_epgp_convergence(fmt="svg"):
    e = sorted(read_csv(os.path.join(ELLIPSE_EPGP, "results.csv")),
               key=lambda r: int(r["n_spectral"]))
    ns = [int(r["n_spectral"]) for r in e]
    err = [float(r["err_vs_bem_ref"]) for r in e]
    rec = [float(r["recip"]) for r in e]
    self = [float(r["selfconv_vs_finest"]) for r in e]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    _conv_ax(ax, ns, [
        (err, ERR, r"$\varepsilon = \|\mathbf{T}_{\mathrm{EPGP}}-\mathbf{T}_{\mathrm{BEM}}\|/\|\mathbf{T}_{\mathrm{BEM}}\|$"),
        (rec, RECIP, L_RHO),
        (self, SELF, L_DELTA),
    ], r"$n_\mathrm{spec}$")
    save(fig, "ellipse_epgp_convergence", fmt)


def fig_ellipse_bem_convergence(bem, fmt="svg"):
    # BEM is a two-parameter (p, m) study, fundamentally unlike the EPGP sweeps,
    # so it keeps its own h- and p-refinement panels rather than the shared
    # single-axis template. Reciprocity rho is the per-run quantity; delta is in
    # the table.
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
    ax[0].set_xlabel(r"mesh level $m$"); ax[0].set_ylabel(L_RHO)
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
    ax[1].set_xlabel(r"polynomial degree $p$"); ax[1].set_ylabel(L_RHO)
    ax[1].set_title(r"$p$-refinement"); ax[1].legend(frameon=False)
    ax[1].set_xticks(ps_all)
    _grid(ax[1])

    save(fig, "ellipse_bem_convergence", fmt)


def _ksweep_fig(path, yfield, ylabel, savename, fmt, scales):
    """Wavenumber sweep: yfield vs n_spec, one curve per k, with the band-limit
    prediction n_spec = (kR+1)^2 marked per curve. `scales` is a list of
    (R, linestyle, legend_label); anisotropic geometries pass one entry per
    distinct semi-axis."""
    if not os.path.exists(path):
        return
    from matplotlib.lines import Line2D

    rows = read_csv(path)
    ks = sorted({float(r["k"]) for r in rows})
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
    vlines = []
    for i, k in enumerate(ks):
        sel = sorted((r for r in rows if float(r["k"]) == k),
                     key=lambda r: int(r["n_spectral"]))
        ns = np.array([int(r["n_spectral"]) for r in sel])
        y = np.array([max(float(r[yfield]), FLOOR) for r in sel])
        color = cmap(i / max(len(ks) - 1, 1))
        ax.plot(ns, y, "D-", color=color, mec="white", mew=0.8, markersize=6)
        for R, ls, _ in scales:
            vlines.append(((k * R + 1) ** 2, color, ls))

    # Coincident band-limit lines (different k, same n_spec) would hide one
    # another, so nudge each member of a colliding group symmetrically in
    # log-space to keep both visible at essentially the same n_spec.
    groups = {}
    for x, color, ls in vlines:
        groups.setdefault(round(x, 6), []).append((color, ls))
    for x0, members in groups.items():
        m = len(members)
        for j, (color, ls) in enumerate(members):
            off = 0.0 if m == 1 else (j - (m - 1) / 2) * 0.05
            ax.axvline(x0 * 2 ** off, color=color, ls=ls, lw=1.6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_\mathrm{spec}$")
    ax.set_ylabel(ylabel)

    # Two columns: all k entries in the left column, the scale lines in the
    # right. matplotlib fills column-major, so pad the right column with blank
    # entries up to the number of k curves.
    k_handles = [Line2D([0], [0], color=cmap(i / max(len(ks) - 1, 1)),
                        marker="D", mec="white", mew=0.8, markersize=6)
                 for i in range(len(ks))]
    k_labels = [f"$k={k:g}$" for k in ks]
    s_handles = [Line2D([0], [0], color="0.35", ls=ls, lw=1.6) for _, ls, _ in scales]
    s_labels = [lab for _, _, lab in scales]
    pad = len(ks) - len(scales)
    s_handles += [Line2D([], [], linestyle="none")] * pad
    s_labels += [""] * pad
    leg = ax.legend(k_handles + s_handles, k_labels + s_labels, ncol=2,
                    frameon=True, framealpha=1.0, facecolor="white", edgecolor="none")
    leg.set_zorder(5)
    _grid(ax)
    save(fig, savename, fmt)


def fig_ellipse_epgp_ksweep(fmt="svg"):
    a_min, a_max = min(GEOMETRIES["ellipse"]), max(GEOMETRIES["ellipse"])
    _ksweep_fig(os.path.join(ELLIPSE_EPGP, "ksweep.csv"), "recip", L_RHO,
                "ellipse_epgp_ksweep", fmt,
                [(a_min, "-.", r"$n_\mathrm{spec}=(k a_{\min}+1)^2$"),
                 (a_max, ":", r"$n_\mathrm{spec}=(k a_{\max}+1)^2$")])


def fig_sphere_epgp_ksweep(fmt="svg"):
    R = max(GEOMETRIES["sphere"])
    _ksweep_fig(os.path.join(SPHERE_EPGP, "ksweep.csv"), "err_vs_ref",
                r"$\|\mathbf{T}_{\mathrm{EPGP}}-\mathbf{T}_\star\|/\|\mathbf{T}_\star\|$",
                "sphere_epgp_ksweep", fmt,
                [(R, ":", r"$n_\mathrm{spec}=(kR+1)^2$")])


def fig_sphere_analytic_multipole(fmt="svg"):
    path = os.path.join(SPHERE_EPGP, "multipole.csv")
    if not os.path.exists(path):
        return
    r = read_csv(path)
    ll = np.array([int(x["l"]) for x in r])
    nrm = np.array([max(float(x["norm"]), FLOOR) for x in r])
    kR = float(r[0]["kR"])

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.semilogy(ll, nrm, "D-", color=ERR["color"], mec="white", mew=1.0, markersize=7)
    ax.axvline(kR, color="0.4", ls=":", lw=1.6, label=r"$l = kR$")
    ax.set_xlabel(r"degree $l$")
    ax.set_ylabel(r"$\|\mathbf{T}_l\|$")
    ax.legend(frameon=False)
    _grid(ax)
    save(fig, "sphere_analytic_multipole", fmt)


def _rate_fig(path, errcol, label, savename, fmt):
    """Convergence-rate view: the reference error on a log axis against
    sqrt(n_spec) on a linear axis. Root-exponential convergence,
    error ~ exp(-c sqrt(n_spec)), is a straight line here. The plateau and the
    numerical floor are masked so only the spectral descent is shown, and a
    least-squares guide line is overlaid to judge linearity."""
    if not os.path.exists(path):
        return
    rows = sorted(read_csv(path), key=lambda r: int(r["n_spectral"]))
    ns = np.array([int(r["n_spectral"]) for r in rows], float)
    err = np.array([float(r[errcol]) for r in rows], float)

    floor = err[err > 0].min()
    mask = (err > 3 * floor) & (err < 0.5)          # spectral descent only
    x, y = np.sqrt(ns[mask]), err[mask]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.semilogy(x, y, ERR["marker"] + "-", color=ERR["color"], mec="white",
                mew=1.0, markersize=ERR["ms"], label=label)
    if len(x) >= 2:
        a, b = np.polyfit(x, np.log10(y), 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.semilogy(xx, 10 ** (a * xx + b), "--", color="0.4", lw=1.4,
                    label="root-exponential fit")
    ax.set_xlabel(r"$\sqrt{n_\mathrm{spec}}$")
    ax.set_ylabel("relative error")
    ax.legend(frameon=False)
    _grid(ax)
    save(fig, savename, fmt)


def fig_sphere_epgp_rate(fmt="svg"):
    _rate_fig(os.path.join(SPHERE_EPGP, "results.csv"), "err_vs_analytic",
              r"$\varepsilon_\star$", "sphere_epgp_rate", fmt)


def fig_ellipse_epgp_rate(fmt="svg"):
    _rate_fig(os.path.join(ELLIPSE_EPGP, "results.csv"), "err_vs_bem_ref",
              r"$\varepsilon$", "ellipse_epgp_rate", fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["svg", "png"], default="svg")
    fmt = ap.parse_args().format

    setup_style()
    os.makedirs(FIGS, exist_ok=True)
    bem = [{"p": int(r["p"]), "m": int(r["m"]), "dofs": int(r["dofs"]),
            "recip": float(r["recip"]), "selfconv_vs_ref": float(r["selfconv_vs_ref"])}
           for r in read_csv(os.path.join(ELLIPSE_BEM, "results.csv"))]
    ref = min(bem, key=lambda r: r["recip"])

    fig_sphere_epgp_convergence(fmt)
    fig_ellipse_epgp_convergence(fmt)
    fig_ellipse_bem_convergence(bem, fmt)
    fig_ellipse_epgp_ksweep(fmt)
    fig_sphere_epgp_ksweep(fmt)
    fig_sphere_analytic_multipole(fmt)
    fig_sphere_epgp_rate(fmt)
    fig_ellipse_epgp_rate(fmt)
    stale = ("h_convergence", "p_convergence", "reciprocity", "svd_spectrum",
             "bem_validity", "bem_self_convergence", "preview", "all_preview",
             "epgp_convergence", "operator_spectrum",
             "bem_reciprocity", "epgp_reciprocity", "epgp_vs_bem",
             "bem_convergence", "ellipse_convergence", "sphere_convergence",
             "epgp_ksweep", "sphere_ksweep", "sphere_multipole")
    for f in os.listdir(FIGS):
        if f.rsplit(".", 1)[0] in stale:
            os.remove(os.path.join(FIGS, f))
    print(f"wrote figures to {FIGS}  (BEM ref p{ref['p']} m{ref['m']}, recip={ref['recip']:.2e})")


if __name__ == "__main__":
    main()
