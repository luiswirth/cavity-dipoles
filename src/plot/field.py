import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from common import (COMP, COMP_LABEL, DEFAULT_NPZ, FIGS, FIGSIZE, clip_vmax,
                    colorwheel, decorate, domain_rgb, emag, grab_frames, load_slice,
                    save_webp, setup_style)

PANELS = [("Escat", "s"), ("Etot", "")]


def _re(Ec, mask, theta, fill):
    return np.where(mask, np.real(Ec * np.exp(-1j * theta)), fill)


def mag_static(fig, ax, S):
    for a_, (key, sup) in zip(ax, PANELS):
        F = emag(S[key])
        a_.pcolormesh(S["xs"], S["zs"], np.where(S["mask"], F, np.nan),
                      shading="gouraud", cmap="coolwarm", vmin=0.0,
                      vmax=clip_vmax(F, S["mask"]), rasterized=True)
        decorate(a_, S["a"], S["c"], S["src"], f"$|E_{sup}|$" if sup else "$|E|$")
    fig.suptitle("EPGP cavity field magnitude", y=0.98, fontsize=14)


def mag_anim(fig, ax, S, frames):
    cl = COMP_LABEL[COMP]
    arts = []
    for a_, (key, sup) in zip(ax, PANELS):
        Ec = S[key][..., COMP]
        v = clip_vmax(np.abs(Ec), S["mask"])
        im = a_.pcolormesh(S["xs"], S["zs"], _re(Ec, S["mask"], 0.0, 0.0),
                           shading="gouraud", cmap="coolwarm", vmin=-v, vmax=v,
                           rasterized=True)
        decorate(a_, S["a"], S["c"], S["src"], rf"$\mathrm{{Re}}\,E_{cl}^{{{sup}}}(t)$")
        arts.append((im, Ec))
    fig.suptitle("EPGP cavity field over one period", y=0.98, fontsize=14)
    fig.tight_layout()

    def update(k):
        th = 2 * np.pi * k / frames
        for im, Ec in arts:
            im.set_array(_re(Ec, S["mask"], th, 0.0).ravel())
    return update


def phase_panels(fig, ax, S):
    ext = [S["xs"][0], S["xs"][-1], S["zs"][0], S["zs"][-1]]
    cl = COMP_LABEL[COMP]
    ims = []
    for a_, (key, sup) in zip(ax, PANELS):
        Ec = S[key][..., COMP]
        im = a_.imshow(domain_rgb(Ec, S["mask"]), origin="lower", extent=ext,
                       aspect="equal", interpolation="bilinear")
        decorate(a_, S["a"], S["c"], S["src"], rf"$E_{cl}^{{{sup}}}$",
                 marker="black", edge="white")
        ims.append((im, Ec))
    wax = fig.add_axes([0.46, 0.80, 0.085, 0.085])
    wax.imshow(colorwheel(), origin="lower", extent=[-1, 1, -1, 1])
    wax.set_title("phase", fontsize=9, pad=2); wax.axis("off")
    fig.suptitle("EPGP cavity field phase", y=0.98, fontsize=14)
    return ims


def phase_anim(S, ims, frames):
    def update(k):
        ph = np.exp(-1j * 2 * np.pi * k / frames)
        for im, Ec in ims:
            im.set_data(domain_rgb(Ec * ph, S["mask"]))
    return update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=DEFAULT_NPZ)
    ap.add_argument("--mode", choices=["mag", "phase"], default="mag")
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

    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
    if args.mode == "mag":
        if args.animate:
            update = mag_anim(fig, ax, S, args.frames)
            save_webp(grab_frames(fig, update, args.frames), out, args.fps)
        else:
            mag_static(fig, ax, S)
            fig.savefig(out, dpi=200)
    else:
        ims = phase_panels(fig, ax, S)
        if args.animate:
            update = phase_anim(S, ims, args.frames)
            save_webp(grab_frames(fig, update, args.frames), out, args.fps)
        else:
            fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
