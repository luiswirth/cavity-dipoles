import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .common import (COMP, COMP_LABEL, DEFAULT_NPZ, FIGS, align_phase,
                     clip_vmax, decorate, domain_rgb, emag, grab_frames,
                     load_slice, save_webp, setup_style)
from .field_lic import FINE, FINE_ANIM, lic_rgb

PANELS = [("Einc", "i"), ("Escat", "s"), ("Etot", "tot")]

def _ext(S):
    return [S["xs"][0], S["xs"][-1], S["zs"][0], S["zs"][-1]]


def _re(Ec, mask, theta, fill):
    return np.where(mask, np.real(Ec * np.exp(-1j * theta)), fill)


def real_static(fig, ax, S):
    cl = COMP_LABEL[COMP]
    for a_, (key, sup) in zip(ax, PANELS):
        Ec = align_phase(S[key], S["mask"])[..., COMP]
        v = clip_vmax(np.abs(np.real(Ec)), S["mask"])
        a_.pcolormesh(S["xs"], S["zs"], _re(Ec, S["mask"], 0.0, np.nan),
                      shading="gouraud", cmap="coolwarm", vmin=-v, vmax=v,
                      rasterized=True)
        decorate(a_, S["a"], S["c"], S["src"], rf"$\mathrm{{Re}}\,E_{cl}^{{\mathrm{{{sup}}}}}$")


def real_anim(fig, ax, S, frames):
    cl = COMP_LABEL[COMP]
    arts = []
    for a_, (key, sup) in zip(ax, PANELS):
        Ec = S[key][..., COMP]
        v = clip_vmax(np.abs(Ec), S["mask"])
        im = a_.pcolormesh(S["xs"], S["zs"], _re(Ec, S["mask"], 0.0, 0.0),
                           shading="gouraud", cmap="coolwarm", vmin=-v, vmax=v,
                           rasterized=True)
        decorate(a_, S["a"], S["c"], S["src"], rf"$\mathrm{{Re}}\,E_{cl}^{{\mathrm{{{sup}}}}}(t)$")
        arts.append((im, Ec))

    def update(k):
        th = 2 * np.pi * k / frames
        for im, Ec in arts:
            im.set_array(_re(Ec, S["mask"], th, 0.0).ravel())
    return update


def _phase_build(ax, S):
    cl = COMP_LABEL[COMP]
    ims = []
    for a_, (key, sup) in zip(ax, PANELS):
        Ec = S[key][..., COMP]
        im = a_.imshow(domain_rgb(Ec, S["mask"]), origin="lower", extent=_ext(S),
                       aspect="equal", interpolation="bilinear")
        decorate(a_, S["a"], S["c"], S["src"], rf"$E_{cl}^{{\mathrm{{{sup}}}}}$",
                 marker="black", edge="white")
        ims.append((im, Ec))
    return ims


def phase_static(fig, ax, S):
    _phase_build(ax, S)


def phase_anim(fig, ax, S, frames):
    ims = _phase_build(ax, S)

    def update(k):
        ph = np.exp(-1j * 2 * np.pi * k / frames)
        for im, Ec in ims:
            im.set_data(domain_rgb(Ec * ph, S["mask"]))
    return update


def lic_static(fig, ax, S):
    for a_, (key, sup) in zip(ax, PANELS):
        rgb = lic_rgb(align_phase(S[key], S["mask"]), S["mask"], FINE)
        a_.imshow(rgb, origin="lower", extent=_ext(S), aspect="equal",
                  interpolation="bilinear")
        decorate(a_, S["a"], S["c"], S["src"], rf"$\mathbf{{E}}^{{\mathrm{{{sup}}}}}$")


def lic_anim(fig, ax, S, frames):
    arts = []
    for a_, (key, sup) in zip(ax, PANELS):
        Ec = S[key]
        im = a_.imshow(lic_rgb(Ec, S["mask"], FINE_ANIM), origin="lower",
                       extent=_ext(S), aspect="equal", interpolation="bilinear")
        decorate(a_, S["a"], S["c"], S["src"], rf"$\mathbf{{E}}^{{\mathrm{{{sup}}}}}(t)$")
        arts.append((im, Ec))

    def update(k):
        ph = np.exp(-1j * 2 * np.pi * k / frames)
        for im, Ec in arts:
            im.set_data(lic_rgb(Ec * ph, S["mask"], FINE_ANIM))
    return update


def std_static(fig, ax, S):
    std = emag(S["Escat_std"])
    v = clip_vmax(std, S["mask"])
    im = ax[0].pcolormesh(S["xs"], S["zs"], np.where(S["mask"], std, np.nan),
                          shading="gouraud", cmap="magma", vmin=0.0, vmax=v,
                          rasterized=True)
    decorate(ax[0], S["a"], S["c"], S["src"], r"$\sigma[\mathbf{E}^{\mathrm{s}}]$")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)


MODES = {
    "real": (real_static, real_anim, 3),
    "phase": (phase_static, phase_anim, 3),
    "lic": (lic_static, lic_anim, 3),
    "std": (std_static, None, 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=DEFAULT_NPZ)
    ap.add_argument("--mode", choices=list(MODES), default="real")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_style()
    S = load_slice(args.npz)
    os.makedirs(FIGS, exist_ok=True)
    ext = "webp" if args.animate else "png"
    out = args.out or os.path.join(FIGS, f"field_{args.mode}{'_anim' if args.animate else ''}.{ext}")

    static_fn, anim_fn, ncols = MODES[args.mode]
    fig, ax = plt.subplots(1, ncols, figsize=(5.0 * ncols, 6.2), squeeze=False)
    ax = ax[0]
    if args.animate:
        if anim_fn is None:
            raise SystemExit(f"mode {args.mode!r} has no animation")
        update = anim_fn(fig, ax, S, args.frames)
        fig.tight_layout()
        save_webp(grab_frames(fig, update, args.frames), out, args.fps)
    else:
        static_fn(fig, ax, S)
        fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
