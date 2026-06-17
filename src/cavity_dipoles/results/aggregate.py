"""Analysis step: turn raw generated operators into convergence tables.

Mirrors the generation/analysis split forced by the BEM side. The EPGP operators
and their raw manifest are produced separately (epgp.convergence); here we add the
self-convergence to the finest run and the error against the benchmark reference
(BEM for the ellipse, the exact analytic operator for the sphere).
"""

import argparse
import csv
import os

import numpy as np

from ..benchmark import config_path, out_dir, reference_operator
from ..epgp.operators import load_config
from .compare import load_bem, load_epgp, reciprocity

BEM = os.path.join("out", "ellipse_bem")


def read_bem_manifest():
    info = {}
    with open(os.path.join(BEM, "manifest.csv")) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            # manifest columns: P, M, dofs, secs, maxrss_kb, cond. Geometry is
            # fixed by the dir (ellipse_bem); recip and ||T|| are NOT recorded,
            # they are computed below from the saved T.
            p, m, dofs, secs, maxrss = row[:5]
            cond = float(row[5]) if len(row) > 5 and row[5] else 0.0
            info[(int(p), int(m))] = {"dofs": int(dofs), "secs": int(secs),
                                      "maxrss": int(maxrss) if maxrss else 0,
                                      "cond": cond}
    return info


def aggregate_bem():
    info = read_bem_manifest()
    runs = []
    for (p, m), meta in info.items():
        path = os.path.join(BEM, f"T_bem_p{p}_m{m}.dat")
        if not os.path.exists(path):
            continue
        T = load_bem(path)
        runs.append({"p": p, "m": m, "dofs": meta["dofs"], "secs": meta["secs"],
                     "maxrss": meta["maxrss"], "cond": meta["cond"],
                     "norm": float(np.linalg.norm(T)), "recip": reciprocity(T),
                     "T": T})
    # Reference is the lowest-reciprocity run (p6/m4, computed separately as a
    # dedicated finer reference). It is excluded from the convergence-grid rows:
    # p6 is the reference only, not a point of the (p,m) convergence test.
    ref = min(runs, key=lambda r: r["recip"])
    nref = np.linalg.norm(ref["T"])
    grid = [r for r in runs if r is not ref]
    for r in grid:
        r["selfconv"] = np.linalg.norm(r["T"] - ref["T"]) / nref
    grid.sort(key=lambda r: (r["p"], r["m"]))
    runs = grid

    with open(os.path.join(BEM, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "m", "dofs", "recip", "selfconv_vs_ref", "secs",
                    "maxrss_kb", "norm", "cond"])
        for r in runs:
            w.writerow([r["p"], r["m"], r["dofs"], f"{r['recip']:.6e}",
                        f"{r['selfconv']:.6e}", r["secs"], r["maxrss"],
                        f"{r['norm']:.6e}", f"{r['cond']:.6e}"])
    # Single source of truth for the reference identity: written here, read by
    # make_figures (and anything else) instead of re-deriving min-recip, which
    # would miss the p6/m4 reference now that it is excluded from results.csv.
    with open(os.path.join(BEM, "reference.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "m", "dofs", "recip", "norm", "cond"])
        w.writerow([ref["p"], ref["m"], ref["dofs"], f"{ref['recip']:.6e}",
                    f"{float(np.linalg.norm(ref['T'])):.6e}", f"{ref['cond']:.6e}"])
    print(f"BEM reference (lowest recip): p{ref['p']} m{ref['m']}, "
          f"dofs={ref['dofs']}, recip={ref['recip']:.2e}")
    return ref["T"]


def read_epgp_manifest(epgp_dir):
    rows = []
    with open(os.path.join(epgp_dir, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            rows.append({
                "ns": int(row["n_spectral"]),
                "nb": int(row["n_boundary"]),
                "dofs": int(row["dofs"]),
                "secs": float(row["secs"]),
                "cond": float(row["cond"]),
                "maxrss": int(row["maxrss_kb"]) if row.get("maxrss_kb") else 0,
            })
    return sorted(rows, key=lambda r: (r["nb"], r["ns"]))


def aggregate_epgp(epgp_dir, T_ref, err_col):
    """Combine the EPGP manifest with the saved operators over the
    (n_spectral, n_boundary) grid: self-convergence is measured against the
    finest corner (max n_spectral, max n_boundary); err is vs the reference."""
    nref = np.linalg.norm(T_ref)
    runs = read_epgp_manifest(epgp_dir)
    Ts = {(r["ns"], r["nb"]): load_epgp(
        os.path.join(epgp_dir, f"T_epgp_ns{r['ns']}_nb{r['nb']}.npy")) for r in runs}
    corner = max(Ts)                              # (max ns, max nb) present
    Tfinest = Ts[corner]
    nfinest = np.linalg.norm(Tfinest)

    for r in runs:
        T = Ts[(r["ns"], r["nb"])]
        r["norm"] = float(np.linalg.norm(T))     # derived from saved T
        r["recip"] = reciprocity(T)              # derived from saved T
        r["selfconv"] = np.linalg.norm(T - Tfinest) / nfinest
        r["err"] = np.linalg.norm(T - T_ref) / nref

    path = os.path.join(epgp_dir, "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_spectral", "n_boundary", "dofs", "secs", "cond",
                    "norm", "recip", "selfconv_vs_finest", err_col, "maxrss_kb"])
        for r in runs:
            w.writerow([r["ns"], r["nb"], r["dofs"], f"{r['secs']:.3f}",
                        f"{r['cond']:.6e}", f"{r['norm']:.6e}",
                        f"{r['recip']:.6e}", f"{r['selfconv']:.6e}", f"{r['err']:.6e}",
                        r["maxrss"]])
            print(f"  EP-GP ns={r['ns']:>5} nb={r['nb']:>5}  recip={r['recip']:.3e}  "
                  f"selfconv={r['selfconv']:.3e}  err={r['err']:.3e}")
    print(f"wrote {path}")


def aggregate_multipole(k, points, e1, e2, outdir):
    """Exact per-degree multipole spectrum of the spherical reference operator."""
    from ..benchmark import GEOMETRIES
    from ..sphere import multipole_spectrum

    R = float(max(GEOMETRIES["sphere"]))
    ls, norms = multipole_spectrum(k, R, points, e1, e2)
    path = os.path.join(outdir, "multipole.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["l", "norm", "kR"])
        for l, nrm in zip(ls, norms):
            w.writerow([int(l), f"{nrm:.6e}", f"{k * R:.4f}"])
    print(f"wrote {path}: degrees 1..{int(ls[-1])}, k R = {k * R:.3f}")


def main():
    ap = argparse.ArgumentParser(description="aggregate convergence results")
    ap.add_argument("--geometry", choices=["ellipse", "sphere"], default="ellipse")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    od = out_dir(args.geometry)
    config = args.config or config_path(args.geometry)
    k, _semi, points, e1, e2 = load_config(config)

    if args.geometry == "ellipse":
        T_ref = aggregate_bem()                  # BEM table + lowest-recip reference
        err_col = "err_vs_bem_ref"
    else:
        T_ref = reference_operator(args.geometry, k, points, e1, e2)
        err_col = "err_vs_analytic"

    if os.path.exists(os.path.join(od, "manifest.csv")):
        aggregate_epgp(od, T_ref, err_col)
    else:
        print(f"(no EP-GP manifest in {od}/; run epgp-convergence)")

    if args.geometry == "sphere":
        aggregate_multipole(k, points, e1, e2, od)


if __name__ == "__main__":
    main()
