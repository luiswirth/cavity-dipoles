import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from .common import FIGS, save, setup_style

BEM_ELLIPSE  = os.path.join("out", "bem",  "grid", "ellipse")
BEM_SPHERE   = os.path.join("out", "bem",  "grid", "sphere")
EPGP_ELLIPSE = os.path.join("out", "epgp", "grid", "ellipse")
EPGP_SPHERE  = os.path.join("out", "epgp", "grid", "sphere")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def bem_staircase(rows, ycol):
    """Pareto staircase for BEM: non-dominated (secs, err) points, monotone descent."""
    pts = [(float(r["secs"]), float(r.get(ycol, 0))) for r in rows
           if float(r.get(ycol, 0)) > 0 and float(r["secs"]) > 0]
    # sort by time, keep only monotone improvements in err
    pts.sort(key=lambda x: x[0])
    front = [pts[0]]
    best = pts[0][1]
    for s, e in pts[1:]:
        if e < best:
            front.append((s, e))
            best = e
    # build staircase
    xs, ys = [], []
    for i, (s, e) in enumerate(front):
        if i > 0:
            xs.append(s); ys.append(front[i - 1][1])
        xs.append(s); ys.append(e)
    return xs, ys


def epgp_envelope(rows, ycol):
    """Best (lowest) ycol per n_spectral, sorted by n_spectral. Returns (secs, err)."""
    by_ns = {}
    for r in rows:
        ns = int(r["n_spectral"])
        e = float(r[ycol])
        s = float(r["secs"])
        if e <= 0 or s <= 0:
            continue
        if ns not in by_ns or e < by_ns[ns][1]:
            by_ns[ns] = (s, e)
    ns_sorted = sorted(by_ns)
    return [by_ns[ns][0] for ns in ns_sorted], [by_ns[ns][1] for ns in ns_sorted]


def _panel(ax, bem_rows, epgp_rows, ycol, ylabel, title):
    C_BEM  = "#1f77b4"
    C_EPGP = "#d62728"

    bem_s  = [float(r["secs"])  for r in bem_rows  if float(r.get(ycol, 0)) > 0 and float(r["secs"]) > 0]
    bem_e  = [float(r[ycol])    for r in bem_rows  if float(r.get(ycol, 0)) > 0 and float(r["secs"]) > 0]
    epgp_s = [float(r["secs"])  for r in epgp_rows if float(r.get(ycol, 0)) > 0 and float(r["secs"]) > 0]
    epgp_e = [float(r[ycol])    for r in epgp_rows if float(r.get(ycol, 0)) > 0 and float(r["secs"]) > 0]

    ax.scatter(bem_s,  bem_e,  s=22, color=C_BEM,  alpha=0.35, zorder=2)
    ax.scatter(epgp_s, epgp_e, s=22, color=C_EPGP, alpha=0.35, zorder=2)

    xs, ys = bem_staircase(bem_rows, ycol)
    if len(xs) >= 2:
        ax.plot(xs, ys, color=C_BEM, lw=2.2, label="BEM", zorder=3)
    env_s, env_e = epgp_envelope(epgp_rows, ycol)
    if len(env_s) >= 2:
        ax.plot(env_s, env_e, "D-", color=C_EPGP, lw=2.2, ms=5,
                mec="white", mew=0.6, label="EP-GP", zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wall time [s]"); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)


def fig_pareto(fmt="svg"):
    paths = [BEM_ELLIPSE, BEM_SPHERE, EPGP_ELLIPSE, EPGP_SPHERE]
    if not all(os.path.exists(os.path.join(p, "results.csv")) for p in paths):
        return

    bem_ellipse  = read_csv(os.path.join(BEM_ELLIPSE,  "results.csv"))
    bem_sphere   = read_csv(os.path.join(BEM_SPHERE,   "results.csv"))
    epgp_ellipse = read_csv(os.path.join(EPGP_ELLIPSE, "results.csv"))
    epgp_sphere  = read_csv(os.path.join(EPGP_SPHERE,  "results.csv"))

    L_RHO = r"$\rho = \|\mathbf{T}-\mathbf{T}^{\!\top}\|/\|\mathbf{T}\|$"
    L_ERR = r"relative error $\varepsilon$"

    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    _panel(axes1[0], bem_sphere, epgp_sphere, "err",   L_ERR, None)
    _panel(axes1[1], bem_sphere, epgp_sphere, "recip", L_RHO, None)
    save(fig1, "pareto_sphere", fmt)

    fig2, ax2 = plt.subplots(1, 1, figsize=(5.5, 4.4), layout="constrained")
    _panel(ax2, bem_ellipse, epgp_ellipse, "recip", L_RHO, None)
    save(fig2, "pareto_ellipse", fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["svg", "png"], default="svg")
    fmt = ap.parse_args().format
    setup_style()
    os.makedirs(FIGS, exist_ok=True)
    fig_pareto(fmt)
    print(f"wrote pareto_sphere.{fmt} and pareto_ellipse.{fmt} to {FIGS}")


if __name__ == "__main__":
    main()
