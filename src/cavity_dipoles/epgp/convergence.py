"""EPGP convergence sweep over n_spectral at the benchmark wavenumber.

Mirrors the BEM convergence study. For each spectral resolution it assembles the
reaction operator, saves T for later aggregation, and records the run-time-only
diagnostics to a manifest: wall-clock seconds, the conditioned-system condition
number, the tuned noise level, the degrees of freedom (2 * n_spectral plane-wave
features), and the operator norm. Self-convergence and cross-validation against
the BEM reference are derived afterward by results.aggregate.
"""

import argparse
import csv
import os
import time

import jax
import numpy as np

from .operators import GPConfig, assemble_operator, load_config

jax.config.update("jax_enable_x64", True)

NS_SWEEP = (16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224,
            256, 320, 384, 512, 768, 1024)


def main():
    ap = argparse.ArgumentParser(description="EPGP n_spectral convergence sweep")
    ap.add_argument("config", nargs="?", default="res/config.txt")
    ap.add_argument("--n-boundary", type=int, default=1200)
    ap.add_argument("--log-noise", type=float, default=-8.0)
    ap.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--opt-steps", type=int, default=200)
    ap.add_argument("--outdir", default="out/epgp")
    args = ap.parse_args()

    k, semiaxes, points, e1, e2 = load_config(args.config)
    cfg = GPConfig.from_args(args)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for ns in NS_SWEEP:
        t0 = time.perf_counter()
        T, post, model = assemble_operator(cfg, semiaxes, k, points, e1, e2, ns)
        secs = time.perf_counter() - t0

        cond = float(np.linalg.cond(np.asarray(post.L @ post.L.conj().T)))
        recip = float(np.linalg.norm(T - T.T) / np.linalg.norm(T))
        norm = float(np.linalg.norm(T))
        log_noise = float(model.log_noise)

        np.save(os.path.join(args.outdir, f"T_epgp_ns{ns}.npy"), T)
        rows.append({
            "n_spectral": ns,
            "dofs": 2 * ns,
            "n_boundary": cfg.n_boundary,
            "secs": secs,
            "log_noise": log_noise,
            "cond": cond,
            "recip": recip,
            "norm": norm,
        })
        print(f"ns={ns:>5}  dofs={2 * ns:>5}  secs={secs:6.1f}  "
              f"log_noise={log_noise:7.3f}  cond={cond:.3e}  recip={recip:.3e}")

    manifest = os.path.join(args.outdir, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_spectral", "dofs", "n_boundary", "secs",
                    "log_noise", "cond", "recip", "norm"])
        for r in rows:
            w.writerow([r["n_spectral"], r["dofs"], r["n_boundary"],
                        f"{r['secs']:.3f}", f"{r['log_noise']:.6f}",
                        f"{r['cond']:.6e}", f"{r['recip']:.6e}", f"{r['norm']:.6e}"])
    print(f"wrote {manifest}: {len(rows)} runs")


if __name__ == "__main__":
    main()
