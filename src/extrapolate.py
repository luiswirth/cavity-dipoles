import argparse
import os

import numpy as np

from compare import load_bem, reciprocity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEM = os.path.join(ROOT, "out", "bem")


def save_bem(path, T):
    out = np.empty((T.shape[0], 2 * T.shape[1]))
    out[:, 0::2] = T.real
    out[:, 1::2] = T.imag
    np.savetxt(path, out)


def richardson(Ts):
    d1, d2 = Ts[-2] - Ts[-3], Ts[-1] - Ts[-2]
    r = np.linalg.norm(d2) / np.linalg.norm(d1)
    return Ts[-1] + r / (1.0 - r) * d2, r


def rates(Ts):
    s = [np.linalg.norm(Ts[i + 1] - Ts[i]) for i in range(len(Ts) - 1)]
    return [s[i + 1] / s[i] for i in range(len(s) - 1)]


def main(axis, at, vals, rmax=0.5):
    if axis == "p":
        paths = [os.path.join(BEM, f"T_bem_p{v}_m{at}.dat") for v in vals]
        out = os.path.join(BEM, f"T_bem_pinf_m{at}.dat")
        fixed = f"m={at}"
    else:
        paths = [os.path.join(BEM, f"T_bem_p{at}_m{v}.dat") for v in vals]
        out = os.path.join(BEM, f"T_bem_p{at}_minf.dat")
        fixed = f"p={at}"

    Ts = [load_bem(p) for p in paths]
    nref = np.linalg.norm(Ts[-1])
    T_inf, r = richardson(Ts)
    rs = rates(Ts)

    print(f"{axis}-series at {fixed}: {axis}={vals}")
    print(f"step ratios = {[f'{x:.4f}' for x in rs]}  (extrapolation rate r = {r:.4f})")

    if r >= rmax:
        print(f"  SKIP: r >= {rmax}, not in asymptotic regime; not writing {out}")
        return
    if len(rs) >= 2 and not (0.4 < rs[-1] / rs[-2] < 2.5):
        print(f"  SKIP: step ratios inconsistent ({rs[-2]:.3f} -> {rs[-1]:.3f}); "
              f"sequence not cleanly geometric; not writing {out}")
        return
    err = np.linalg.norm(Ts[-1] - T_inf) / nref
    print(f"extrapolation residual ~ r*err = {r * err:.3e}")
    save_bem(out, T_inf)
    print(f"wrote {out}  recip={reciprocity(T_inf):.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["p", "m"], default="p")
    ap.add_argument("--at", type=int, default=4)
    ap.add_argument("--vals", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--rmax", type=float, default=0.5)
    args = ap.parse_args()
    main(args.axis, args.at, args.vals, args.rmax)
