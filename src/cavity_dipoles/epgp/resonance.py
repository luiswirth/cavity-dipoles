"""Cheap resonance (eigenvalue) locator for the PEC cavity.

Sweeps the wavenumber k and, at each k, measures how close the cavity is to an
interior PEC resonance by the Betcke--Trefethen subspace-angle method. The
transverse plane-wave modes at wavenumber k are exact Maxwell solutions; a
cavity eigenvalue is a k at which some combination of them has vanishing
tangential trace on the wall while staying nonzero inside. The smallest
singular value sigma_min(k) of the boundary block of an orthonormalized basis
(boundary trace stacked over interior field) is sin of that subspace angle: it
dips to ~0 exactly at the eigenvalues.

Interior normalization is what makes this robust to plane-wave overcompleteness:
without it a plain condition number would report the basis redundancy, not the
resonances. Output is written as a (k, sigma_min) curve; local minima are the
eigenvalue estimates.
"""

import argparse
import csv
import os

import jax
import numpy as np
from maxwellgp.kernel import MaxwellFeatureMap
from maxwellgp.utils import fibonacci_sphere

from ..benchmark import GEOMETRIES, config_path, out_dir
from .operators import boundary_collocation, load_config

jax.config.update("jax_enable_x64", True)


def interior_points(semiaxes, n_per_shell, shells=(0.4, 0.7)):
    pts = [np.asarray(fibonacci_sphere(n_per_shell)) * semiaxes * s for s in shells]
    return np.concatenate(pts, axis=0)


def sigma_min(omega, n_spectral, semiaxes, n_boundary, n_interior):
    bpts, bnor = boundary_collocation(semiaxes, n_boundary)
    Xb = np.concatenate([bpts, bnor], axis=1)
    Xi = interior_points(semiaxes, n_interior)
    fm = MaxwellFeatureMap(n_spectral, omega)
    Ab = np.asarray(fm.tangential(Xb)).T   # (3 n_b, F)  boundary tangential trace
    Ai = np.asarray(fm.full(Xi)).T         # (6 n_i, F)  interior field, for the norm
    M = np.vstack([Ab, Ai])
    Q, _ = np.linalg.qr(M)                  # orthonormal columns spanning the basis
    Qb = Q[: Ab.shape[0], :]                # boundary part of the orthonormal basis
    return float(np.linalg.svd(Qb, compute_uv=False).min())


def local_minima(ks, s):
    out = []
    for i in range(1, len(s) - 1):
        if s[i] < s[i - 1] and s[i] <= s[i + 1]:
            out.append((ks[i], s[i]))
    return out


def main():
    ap = argparse.ArgumentParser(description="PEC cavity resonance locator")
    ap.add_argument("--geometry", choices=list(GEOMETRIES), default="ellipse")
    ap.add_argument("--config", default=None)
    ap.add_argument("--kmin", type=float, default=0.3)
    ap.add_argument("--kmax", type=float, default=3.2)
    ap.add_argument("--dk", type=float, default=0.01)
    # Basis is sized per k to be just complete: n_spec ~ alpha (k R)^2, the
    # multipole count up to degree L_max ~ k R. The boundary is oversampled so
    # the subspace angle is well determined.
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--oversample", type=float, default=3.5)
    ap.add_argument("--nmin", type=int, default=40)
    ap.add_argument("--nmax", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--peaks", default=None)
    ap.add_argument("--thresh", type=float, default=0.03,
                    help="report local minima with sigma_min below this")
    args = ap.parse_args()

    config = args.config or config_path(args.geometry)
    od = out_dir(args.geometry)
    out = args.out or os.path.join(od, "resonances.csv")
    peaks = args.peaks or os.path.join(od, "eigenvalues.csv")
    _k, semiaxes, *_ = load_config(config)
    R = float(np.max(semiaxes))

    ks = np.arange(args.kmin, args.kmax + 0.5 * args.dk, args.dk)
    s = np.empty(len(ks))
    for i, k in enumerate(ks):
        n_spec = int(min(args.nmax, max(args.nmin, round(args.alpha * (k * R) ** 2))))
        n_b = int(max(250, round(args.oversample * 2 * n_spec / 3)))
        n_i = max(40, n_spec // 4)
        s[i] = sigma_min(k, n_spec, semiaxes, n_b, n_i)
        print(f"k={k:6.3f}  n_spec={n_spec:>4}  sigma_min={s[i]:.4e}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "sigma_min"])
        for k, sk in zip(ks, s):
            w.writerow([f"{k:.4f}", f"{sk:.6e}"])
    print(f"wrote {out}: {len(ks)} points")

    mins = [(k, v) for k, v in local_minima(ks, s) if v < args.thresh]
    with open(peaks, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k_eig", "sigma_min"])
        for k, v in mins:
            w.writerow([f"{k:.4f}", f"{v:.6e}"])
    print(f"wrote {peaks}: {len(mins)} eigenvalue estimates")
    for k, v in mins:
        print(f"  resonance near k={k:.3f}  (sigma_min={v:.2e})")


if __name__ == "__main__":
    main()
