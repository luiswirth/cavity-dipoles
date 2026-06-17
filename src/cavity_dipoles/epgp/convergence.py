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


def write_provenance(outdir, cfg, k, geometry):
    """Record everything needed to reproduce the run alongside the data."""
    prov = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "geometry": geometry,
        "k": k,
        "log_noise0": cfg.log_noise,
        "opt_noise": cfg.opt_noise,
        "opt_steps": cfg.opt_steps,
        "n_spectral_sweep": list(NS_SWEEP),
        "n_boundary_sweep": list(NB_SWEEP),
        "cavity_dipoles_commit": _git_commit(),
        "maxwellgp_version": _pkg_version("maxwellgp"),
        "jax_version": _pkg_version("jax"),
        "python": sys.version.split()[0],
    }
    with open(os.path.join(outdir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

# EP-GP has two convergence axes, mirroring BEM's (poly-degree p, refinement m):
#   n_spectral  -- plane-wave feature richness (like p)
#   n_boundary  -- boundary-collocation density (like h/m)
# The study is the full grid; the best operator is the high corner of both.
NS_SWEEP = (16, 32, 64, 128, 192, 256, 384, 512, 768, 1024)   # 10, densified descent
NB_SWEEP = (256, 512, 1024, 2048, 4096, 8192)                 # 6, powers of 2
GRID = [(ns, nb) for nb in NB_SWEEP for ns in NS_SWEEP]   # flat 60 pts; ns fastest

# Recorded quantities are only those NOT reconstructible from the saved operator:
# dofs, secs, cond (the conditioned GP system, needs post.L), maxrss. recip and
# ||T|| are functions of T and are computed in post-processing
# (results.aggregate) from the saved T_epgp_*.npy. log_noise is a fixed input
# (opt_noise=False), so it lives in provenance.json, not in every manifest row.
MANIFEST_HEADER = ["n_spectral", "n_boundary", "dofs", "secs",
                   "cond", "maxrss_kb"]
FRAGMENT_DIR = "manifest.d"


def run_one(base_cfg, semiaxes, k, points, e1, e2, ns, nb, outdir, warmup=False):
    """Assemble the operator at one (n_spectral, n_boundary), save it, return row.

    With warmup=True the operator is assembled once (discarded) to trigger JAX/XLA
    compilation, then assembled again under timing, so `secs` is steady-state run
    time with no compile cost folded in. XLA compiles per input shape, so the
    warmup must use the same (ns, nb) as the timed call. Use it for the matched
    runtime/memory comparison; the accuracy grid does not report timing and can
    skip the 2x cost."""
    cfg = GPConfig(nb, base_cfg.log_noise, base_cfg.opt_noise, base_cfg.opt_steps)
    if warmup:
        tc = time.perf_counter()
        assemble_operator(cfg, semiaxes, k, points, e1, e2, ns)
        print(f"  warmup (compile+run) {time.perf_counter() - tc:6.1f}s discarded")
    t0 = time.perf_counter()
    T, post, _model = assemble_operator(cfg, semiaxes, k, points, e1, e2, ns)
    secs = time.perf_counter() - t0

    cond = float(np.linalg.cond(np.asarray(post.L @ post.L.conj().T)))
    recip = float(np.linalg.norm(T - T.T) / np.linalg.norm(T))   # live sanity only
    # Peak resident set (KiB on Linux): the kernel's true high-water mark, the
    # best available peak-memory measure. Exact per grid point only in array mode
    # (one point per process); a whole-sweep process would report just the
    # largest point. /usr/bin/time -v can't replace it here -- it would measure
    # the `uv run` launcher, not this Python process. Fold in children too.
    maxrss_kb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                 + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)

    np.save(os.path.join(outdir, f"T_epgp_ns{ns}_nb{nb}.npy"), T)
    print(f"ns={ns:>5} nb={nb:>5}  dofs={2 * ns:>5}  secs={secs:6.1f}  "
          f"cond={cond:.3e}  recip={recip:.3e}  maxrss={maxrss_kb / 1048576:.2f}GiB")
    return {"n_spectral": ns, "n_boundary": nb, "dofs": 2 * ns, "secs": secs,
            "cond": cond, "maxrss_kb": maxrss_kb}


def _row_values(r):
    return [r["n_spectral"], r["n_boundary"], r["dofs"], f"{r['secs']:.3f}",
            f"{r['cond']:.6e}", r["maxrss_kb"]]


def write_manifest(outdir, rows):
    path = os.path.join(outdir, "manifest.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_HEADER)
        for r in sorted(rows, key=lambda r: (r["n_boundary"], r["n_spectral"])):
            w.writerow(_row_values(r))
    return path


def write_fragment(outdir, row):
    """One file per grid point (array-task safe: no concurrent writers)."""
    d = os.path.join(outdir, FRAGMENT_DIR)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{row['n_spectral']:05d}_{row['n_boundary']:05d}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_HEADER)
        w.writerow(_row_values(row))
    return path


def collect(outdir):
    """Merge per-grid-point fragments (from array tasks) into one manifest.csv."""
    d = os.path.join(outdir, FRAGMENT_DIR)
    rows = {}
    for name in sorted(os.listdir(d)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(d, name)) as f:
            for r in csv.DictReader(f):
                rows[(int(r["n_boundary"]), int(r["n_spectral"]))] = \
                    [r[c] for c in MANIFEST_HEADER]
    path = os.path.join(outdir, "manifest.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_HEADER)
        for key in sorted(rows):
            w.writerow(rows[key])
    print(f"collected {len(rows)} fragments -> {path}")


def main():
    ap = argparse.ArgumentParser(description="EPGP (n_spectral, n_boundary) convergence grid")
    ap.add_argument("--geometry", choices=list(GEOMETRIES), default="ellipse")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boundary", type=int, default=1200)
    ap.add_argument("--log-noise", type=float, default=-12.0)
    ap.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--opt-steps", type=int, default=200)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--index", type=int, default=None,
                    help="run a single GRID entry by flat index (for SLURM array tasks)")
    ap.add_argument("--collect", action="store_true",
                    help="merge per-grid-point fragments into manifest.csv, then exit")
    ap.add_argument("--warmup", action="store_true",
                    help="assemble once to compile before the timed run, so secs "
                         "excludes JAX/XLA compile (for the runtime/memory comparison)")
    args = ap.parse_args()

    outdir = args.outdir or out_dir(args.geometry)
    os.makedirs(outdir, exist_ok=True)
    naff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"jax devices: {jax.devices()}  affinity: {naff} cpus  grid points: {len(GRID)}")

    if args.collect:
        collect(outdir)
        return

    config = args.config or config_path(args.geometry)
    k, semiaxes, points, e1, e2 = load_config(config)
    cfg = GPConfig.from_args(args)

    if args.index is not None:
        ns, nb = GRID[args.index]
        row = run_one(cfg, semiaxes, k, points, e1, e2, ns, nb, outdir, args.warmup)
        write_fragment(outdir, row)
        write_provenance(outdir, cfg, k, args.geometry)
        return

    rows = [run_one(cfg, semiaxes, k, points, e1, e2, ns, nb, outdir, args.warmup)
            for ns, nb in GRID]
    path = write_manifest(outdir, rows)
    print(f"wrote {path}: {len(rows)} grid points")
    write_provenance(outdir, cfg, k, args.geometry)


if __name__ == "__main__":
    main()
