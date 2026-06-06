import os

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(ROOT, "out", "figs")
DEFAULT_NPZ = os.path.join(ROOT, "out", "field", "field_slice.npz")
FIGSIZE = (10.5, 6.2)
COMP = 0
COMP_LABEL = {0: "x", 1: "y", 2: "z"}

_FONT_DIR = os.path.expanduser("~/Library/Fonts")
_FONT_FILES = [
    "NewCMSans10-Regular.otf", "NewCMSans10-Bold.otf",
    "NewCMSans10-Oblique.otf", "NewCMSans10-BoldOblique.otf",
]
_TEXT = "NewComputerModernSans10"


def setup_style():
    for f in _FONT_FILES:
        p = os.path.join(_FONT_DIR, f)
        if os.path.exists(p):
            fm.fontManager.addfont(p)
    plt.rcParams.update({
        "savefig.bbox": "tight",
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "legend.fontsize": 11, "axes.grid": True, "grid.alpha": 0.25,
        "axes.axisbelow": True, "lines.linewidth": 2.2, "lines.markersize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": [_TEXT, "DejaVu Sans"],
        "mathtext.fontset": "cm",
    })


def save(fig, name, fmt="svg", dpi=200):
    fig.savefig(os.path.join(FIGS, f"{name}.{fmt}"), dpi=dpi)
    plt.close(fig)


def save_webp(frames, out, fps, lossless=True):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames[0].save(out, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, lossless=lossless,
                   quality=100, method=6)


def load_slice(path):
    d = np.load(path)
    a, _, c = d["semiaxes"]
    return {"xs": d["xs"], "zs": d["zs"], "mask": d["mask"],
            "a": float(a), "c": float(c), "src": d["source"],
            "Escat": d["Escat"], "Etot": d["Etot"]}


def emag(E):
    return np.sqrt(np.sum(np.abs(E) ** 2, axis=-1))


def clip_vmax(field, mask, pct=98.0):
    return np.nanpercentile(np.where(mask, field, np.nan), pct)


def align_phase(E, mask):
    mag = emag(E)
    cut = np.nanpercentile(np.where(mask, mag, np.nan), 95)
    sel = mask & (mag < cut)
    phi = -0.5 * np.angle(np.sum(E[sel] ** 2))
    return E * np.exp(1j * phi)


def decorate(ax, a, c, src, title, marker="lime", edge="black"):
    ax.add_patch(mpatches.Ellipse((0, 0), 2 * a, 2 * c, fill=False,
                                  edgecolor="black", lw=3.6))
    ax.plot([src[0]], [src[2]], "o", color=marker, ms=7, mec=edge, mew=0.8)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$z$")
    ax.set_title(title); ax.set_aspect("equal")


def domain_rgb(F, mask):
    mag = np.abs(F)
    vmax = clip_vmax(mag, mask)
    H = (np.angle(F) + np.pi) / (2 * np.pi)
    V = np.clip(mag / (vmax + 1e-30), 0.0, 1.0)
    rgb = mcolors.hsv_to_rgb(np.stack([H, np.ones_like(V), V], axis=-1))
    rgb[~mask] = 1.0
    return rgb


def colorwheel(n=220):
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    H = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
    rgb = mcolors.hsv_to_rgb(np.stack([H, np.ones_like(H), np.ones_like(H)], axis=-1))
    rgb[np.hypot(x, y) > 1] = 1.0
    return rgb


def grab_frames(fig, update, n):
    from PIL import Image
    out = []
    for k in range(n):
        update(k)
        fig.canvas.draw()
        out.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
    return out
