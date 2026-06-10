"""Wavenumber sweep for the angular-bandwidth hypothesis.

For each wavenumber k and each spectral resolution n_spectral, assemble the EPGP
reaction operator and record the reciprocity error rho (reference-free), the
self-convergence to the finest n_spectral, and the condition number. For the
sphere benchmark the exact reference is available at every k, so the relative
error against the analytic operator is recorded too. The band-limit hypothesis
predicts the staircase drop at sqrt(n_spectral) ~ k R, that is n_spectral ~ (k R)^2.

This is an EPGP-only diagnostic (the BEM reference exists only at the benchmark
k), so generation and analysis are not separated here as they are for the main
convergence study.
"""

import argparse
import csv
import os

import jax
import numpy as np

from ..benchmark import GEOMETRIES, config_path, out_dir, reference_operator
from .operators import GPConfig, assemble_operator, load_config

jax.config.update("jax_enable_x64", True)

K_SWEEP = (1.0, 1.5, 2.0, 2.5, 3.0)
NS_SWEEP = (16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224,
            256, 320, 384, 512, 768)


def main():
    ap = argparse.ArgumentParser(description="EPGP wavenumber sweep")
    ap.add_argument("--geometry", choices=list(GEOMETRIES), default="ellipse")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boundary", type=int, default=1200)
    ap.add_argument("--log-noise", type=float, default=-8.0)
    ap.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--opt-steps", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = args.config or config_path(args.geometry)
    _k, semiaxes, points, e1, e2 = load_config(config)
    cfg = GPConfig.from_args(args)
    R = float(np.max(semiaxes))
    out = args.out or os.path.join(out_dir(args.geometry), "ksweep.csv")

    rows = []
    for k in K_SWEEP:
        Tref = reference_operator(args.geometry, k, points, e1, e2) \
            if args.geometry == "sphere" else None
        nref = np.linalg.norm(Tref) if Tref is not None else None
        Ts = {}
        for ns in NS_SWEEP:
            T, post, _ = assemble_operator(cfg, semiaxes, k, points, e1, e2, ns)
            Ts[ns] = T
            recip = np.linalg.norm(T - T.T) / np.linalg.norm(T)
            cond = float(np.linalg.cond(np.asarray(post.L @ post.L.conj().T)))
            err = np.linalg.norm(T - Tref) / nref if Tref is not None else float("nan")
            rows.append({"k": k, "R": R, "n_spectral": ns, "recip": float(recip),
                         "cond": cond, "err_vs_ref": float(err), "selfconv": None})
            print(f"{args.geometry} k={k:>4} n_spec={ns:>4} recip={recip:.3e} "
                  f"cond={cond:.2e} err_vs_ref={err:.3e}")
        Tfin = Ts[NS_SWEEP[-1]]
        nfin = np.linalg.norm(Tfin)
        for r in rows[-len(NS_SWEEP):]:
            r["selfconv"] = float(np.linalg.norm(Ts[r["n_spectral"]] - Tfin) / nfin)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "R", "n_spectral", "recip", "selfconv", "cond", "err_vs_ref"])
        for r in rows:
            w.writerow([r["k"], r["R"], r["n_spectral"], f"{r['recip']:.6e}",
                        f"{r['selfconv']:.6e}", f"{r['cond']:.6e}", f"{r['err_vs_ref']:.6e}"])
    print(f"wrote {out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
