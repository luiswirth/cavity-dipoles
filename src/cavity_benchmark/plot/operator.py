import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from ..results.uq import available_runs, entry_std, load_run
from .common import FIGS, setup_style


def _heatmap(ax, M, title, cmap, norm=None):
    im = ax.imshow(M, origin="upper", aspect="equal", cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(r"transmitter $j$")
    ax.set_ylabel(r"receiver $i$")
    ax.grid(False)
    return im


def operator_figure(T, Sigma):
    sigma = np.broadcast_to(entry_std(Sigma)[:, None], T.shape)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2))
    im0 = _heatmap(ax[0], np.abs(T), r"reaction operator $|T|$", "viridis")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = _heatmap(ax[1], sigma, r"posterior uncertainty $\sigma$", "magma",
                   norm=mcolors.Normalize(vmin=0.0, vmax=float(sigma.max())))
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="EPGP operator value + uncertainty heatmap")
    ap.add_argument("--geometry", choices=["sphere", "ellipse"], default="sphere")
    ap.add_argument("--uq-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    uq_dir = args.uq_dir or os.path.join("out", "epgp", "ref", args.geometry)
    runs = available_runs(uq_dir)
    if not runs:
        raise SystemExit(f"no Sigma_*.npy in {uq_dir}")
    ns, nb = runs[-1]  # highest-resolution run in the dir
    T, Sigma = load_run(uq_dir, ns, nb)

    setup_style()
    fig = operator_figure(T, Sigma)
    out = args.out or os.path.join(FIGS, f"{args.geometry}_uq_operator.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
