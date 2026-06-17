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
import datetime
import importlib.metadata
import json
import os
import resource
import socket
import subprocess
import sys
import time

import jax
import numpy as np

from ..benchmark import GEOMETRIES, config_path, out_dir
from .operators import GPConfig, assemble_operator, load_config

jax.config.update("jax_enable_x64", True)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def write_provenance(outdir, cfg, k, geometry, ns_list):
    """Record everything needed to reproduce the run alongside the data."""
    prov = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "geometry": geometry,
        "k": k,
        "n_boundary": cfg.n_boundary,
        "log_noise0": cfg.log_noise,
        "opt_noise": cfg.opt_noise,
        "opt_steps": cfg.opt_steps,
        "n_spectral": list(ns_list),
        "cavity_dipoles_commit": _git_commit(),
        "maxwellgp_version": _pkg_version("maxwellgp"),
        "jax_version": _pkg_version("jax"),
        "python": sys.version.split()[0],
    }
    with open(os.path.join(outdir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

NS_SWEEP = (16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224,
            256, 320, 384, 512, 768, 1024)


def main():
    ap = argparse.ArgumentParser(description="EPGP n_spectral convergence sweep")
    ap.add_argument("--geometry", choices=list(GEOMETRIES), default="ellipse")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boundary", type=int, default=1200)
    ap.add_argument("--log-noise", type=float, default=-8.0)
    ap.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--opt-steps", type=int, default=200)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    config = args.config or config_path(args.geometry)
    outdir = args.outdir or out_dir(args.geometry)
    k, semiaxes, points, e1, e2 = load_config(config)
    cfg = GPConfig.from_args(args)
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for ns in NS_SWEEP:
        t0 = time.perf_counter()
        T, post, model = assemble_operator(cfg, semiaxes, k, points, e1, e2, ns)
        secs = time.perf_counter() - t0

        cond = float(np.linalg.cond(np.asarray(post.L @ post.L.conj().T)))
        recip = float(np.linalg.norm(T - T.T) / np.linalg.norm(T))
        norm = float(np.linalg.norm(T))
        log_noise = float(model.log_noise)

        # Peak resident memory (cumulative high-water mark, KiB on Linux). The
        # sweep runs n_spectral in increasing order, so this is a monotone
        # memory-vs-resolution curve and the final row is the whole-job peak.
        maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        np.save(os.path.join(outdir, f"T_epgp_ns{ns}.npy"), T)
        rows.append({
            "n_spectral": ns,
            "dofs": 2 * ns,
            "n_boundary": cfg.n_boundary,
            "secs": secs,
            "log_noise": log_noise,
            "cond": cond,
            "recip": recip,
            "norm": norm,
            "maxrss_kb": maxrss_kb,
        })
        print(f"ns={ns:>5}  dofs={2 * ns:>5}  secs={secs:6.1f}  "
              f"log_noise={log_noise:7.3f}  cond={cond:.3e}  recip={recip:.3e}  "
              f"maxrss={maxrss_kb / 1048576:.2f}GiB")

    manifest = os.path.join(outdir, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_spectral", "dofs", "n_boundary", "secs",
                    "log_noise", "cond", "recip", "norm", "maxrss_kb"])
        for r in rows:
            w.writerow([r["n_spectral"], r["dofs"], r["n_boundary"],
                        f"{r['secs']:.3f}", f"{r['log_noise']:.6f}",
                        f"{r['cond']:.6e}", f"{r['recip']:.6e}", f"{r['norm']:.6e}",
                        r["maxrss_kb"]])
    print(f"wrote {manifest}: {len(rows)} runs")

    write_provenance(outdir, cfg, k, args.geometry, NS_SWEEP)
    print(f"wrote {os.path.join(outdir, 'provenance.json')}")


if __name__ == "__main__":
    main()
