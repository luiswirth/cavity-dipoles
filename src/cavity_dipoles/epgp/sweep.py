"""Wavenumber sweep for the angular-bandwidth hypothesis.

For each wavenumber k and each spectral resolution n_spectral, assemble the
EPGP reaction operator T and record its reciprocity error rho = ||T-T^T||/||T||
(intrinsic to EPGP, no BEM reference needed), its self-convergence against the
finest n_spectral at the same k, and the condition number cond(A) of the
conditioned system. The band-limit hypothesis predicts the staircase drop at
sqrt(n_spectral) ~ k R, that is n_spectral ~ (k R)^2.
"""

import argparse
import csv
import os

import jax
import numpy as np

from .operators import GPConfig, assemble_operator, load_config

jax.config.update("jax_enable_x64", True)

# k sweep: R = 6 (largest semi-axis), so k R ranges 6 .. 18.
K_SWEEP = (1.0, 1.5, 2.0, 2.5, 3.0)
NS_SWEEP = (16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224,
            256, 320, 384, 512, 768)


def main():
    ap = argparse.ArgumentParser(description="EPGP wavenumber sweep")
    ap.add_argument("config", nargs="?", default="out/config.txt")
    ap.add_argument("--n-boundary", type=int, default=1200)
    ap.add_argument("--log-noise", type=float, default=-8.0)
    ap.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--opt-steps", type=int, default=200)
    ap.add_argument("--out", default="out/epgp/ksweep.csv")
    args = ap.parse_args()

    _k_cfg, semiaxes, points, e1, e2 = load_config(args.config)
    cfg = GPConfig.from_args(args)
    R = float(np.max(semiaxes))

    rows = []
    for k in K_SWEEP:
        Ts, conds = {}, {}
        for ns in NS_SWEEP:
            T, post = assemble_operator(cfg, semiaxes, k, points, e1, e2, ns)
            Ts[ns] = T
            conds[ns] = float(np.linalg.cond(np.asarray(post.L @ post.L.conj().T)))
            recip = np.linalg.norm(T - T.T) / np.linalg.norm(T)
            print(f"k={k:>4}  n_spec={ns:>4}  recip={recip:.3e}  "
                  f"cond(A)={conds[ns]:.3e}  drop at n=(kR)^2={int((k * R) ** 2)}")
        Tref = Ts[NS_SWEEP[-1]]
        nref = np.linalg.norm(Tref)
        for ns in NS_SWEEP:
            T = Ts[ns]
            rows.append({
                "k": k,
                "R": R,
                "n_spectral": ns,
                "recip": np.linalg.norm(T - T.T) / np.linalg.norm(T),
                "selfconv": np.linalg.norm(T - Tref) / nref,
                "condA": conds[ns],
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "R", "n_spectral", "recip", "selfconv", "condA"])
        for r in rows:
            w.writerow([r["k"], r["R"], r["n_spectral"],
                        f"{r['recip']:.6e}", f"{r['selfconv']:.6e}",
                        f"{r['condA']:.6e}"])
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
