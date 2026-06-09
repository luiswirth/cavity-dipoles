import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from .common import (DEFAULT_NPZ, FIGS, align_phase, clip_vmax,
                     decorate, emag, load_slice, setup_style)

COMP_IN = (0, 2)
FINE = 1100
FINE_ANIM = 560  # lower LIC resolution for animation (per-frame recompute)
FIGSIZE3 = (15.0, 6.2)


def _sample(A, x, y):
    ny, nx = A.shape
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    wx = x - x0; wy = y - y0
    x0c = np.clip(x0, 0, nx - 1); x1c = np.clip(x0 + 1, 0, nx - 1)
    y0c = np.clip(y0, 0, ny - 1); y1c = np.clip(y0 + 1, 0, ny - 1)
    return (A[y0c, x0c] * (1 - wx) * (1 - wy) + A[y0c, x1c] * wx * (1 - wy)
            + A[y1c, x0c] * (1 - wx) * wy + A[y1c, x1c] * wx * wy)


def _upsample(A, nf):
    ny, nx = A.shape
    FX, FY = np.meshgrid(np.linspace(0, nx - 1, nf), np.linspace(0, ny - 1, nf))
    return _sample(A.astype(float), FX, FY)


def lic(u, v, length=45, step=1.0, seed=0):
    ny, nx = u.shape
    mag = np.hypot(u, v); mag[mag == 0] = 1.0
    ux, vy = u / mag, v / mag
    noise = np.random.default_rng(seed).random((ny, nx))
    X, Y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    acc = noise.copy(); w = np.ones_like(noise)
    for sgn in (1.0, -1.0):
        fx, fy = X.copy(), Y.copy()
        for _ in range(length):
            fx = fx + sgn * step * _sample(ux, fx, fy)
            fy = fy + sgn * step * _sample(vy, fx, fy)
            acc += _sample(noise, fx, fy); w += 1
    out = acc / w
    return (out - out.min()) / (np.ptp(out) + 1e-30)


def _equalize(tex, mask):
    t = np.zeros_like(tex)
    order = tex[mask].argsort()
    ranks = np.empty(len(order))
    ranks[order] = np.linspace(0.0, 1.0, len(order))
    t[mask] = ranks
    return t


def lic_rgb(E, mask, fine=FINE):
    """LIC-textured, magnitude-shaded RGB image of the in-plane field E."""
    u = np.where(mask, np.real(E[..., COMP_IN[0]]), 0.0)
    v = np.where(mask, np.real(E[..., COMP_IN[1]]), 0.0)
    uf, vf = _upsample(u, fine), _upsample(v, fine)
    magf = _upsample(emag(E), fine)
    maskf = _upsample(mask.astype(float), fine) > 0.5
    uf[~maskf] = 0.0; vf[~maskf] = 0.0

    tex = _equalize(lic(uf, vf), maskf)
    base = cm.magma(np.clip(magf / clip_vmax(magf, maskf), 0.0, 1.0))[..., :3]
    rgb = base * (0.08 + 0.92 * tex)[..., None]
    rgb[~maskf] = 1.0
    return rgb


def panel(ax, S, key, title, fine=FINE):
    rgb = lic_rgb(align_phase(S[key], S["mask"]), S["mask"], fine)
    ax.imshow(rgb, origin="lower",
              extent=[S["xs"][0], S["xs"][-1], S["zs"][0], S["zs"][-1]],
              aspect="equal", interpolation="bilinear")
    decorate(ax, S["a"], S["c"], S["src"], title)


def main():
    setup_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=DEFAULT_NPZ)
    args = ap.parse_args()
    S = load_slice(args.npz)

    fig, ax = plt.subplots(1, 3, figsize=FIGSIZE3)
    panel(ax[0], S, "Einc", r"incident field $\mathbf{E}^{\mathrm{i}}$")
    panel(ax[1], S, "Escat", r"scattered field $\mathbf{E}^{\mathrm{s}}$")
    panel(ax[2], S, "Etot", r"total field $\mathbf{E}^{\mathrm{tot}}$")
    fig.suptitle("EPGP cavity field LIC", y=0.98, fontsize=14)
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "field_lic.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
