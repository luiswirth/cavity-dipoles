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

from ..benchmark import bem_reference_path, config_path, out_dir, reference_operator
from cavity_epgp import load_config
from .compare import load_bem, load_epgp, reciprocity

BEM = os.path.join("out", "bem", "grid", "ellipse")


def read_bem_manifest():
    info = {}
    with open(os.path.join(BEM, "manifest.csv")) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            # manifest columns: P, M, dofs, secs, mem_kb, cond. recip and ||T||
            # are NOT recorded; they are computed below from the saved T.
            p, m, dofs, secs, mem = row[:5]
            cond = float(row[5]) if len(row) > 5 and row[5] else 0.0
            info[(int(p), int(m))] = {"dofs": int(dofs), "secs": int(secs),
                                      "mem": int(mem) if mem else 0,
                                      "cond": cond}
    return info


def aggregate_bem():
    info = read_bem_manifest()
    runs = []
    for (p, m), meta in info.items():
        path = os.path.join(BEM, f"T_p{p}_m{m}.dat")
        if not os.path.exists(path):
            continue
        T = load_bem(path)
        runs.append({"p": p, "m": m, "dofs": meta["dofs"], "secs": meta["secs"],
                     "mem": meta["mem"], "cond": meta["cond"],
                     "norm": float(np.linalg.norm(T)), "recip": reciprocity(T),
                     "T": T})
    # Reference is BEM_REFERENCE, declared once in benchmark.py and loaded from
    # its own operator file -- never picked implicitly, and it need NOT be one of
    # the grid runs (it may be a separate, finer run outside the (p,m) grid).
    ref_path = bem_reference_path()
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"BEM reference operator missing: {ref_path}")
    T_ref = load_bem(ref_path)
    nref = np.linalg.norm(T_ref)
    for r in runs:
        r["selfconv"] = np.linalg.norm(r["T"] - T_ref) / nref
    runs.sort(key=lambda r: (r["p"], r["m"]))

    with open(os.path.join(BEM, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "m", "dofs", "recip", "selfconv_vs_ref", "secs",
                    "mem_kb", "norm", "cond"])
        for r in runs:
            w.writerow([r["p"], r["m"], r["dofs"], f"{r['recip']:.6e}",
                        f"{r['selfconv']:.6e}", r["secs"], r["mem"],
                        f"{r['norm']:.6e}", f"{r['cond']:.6e}"])
    print(f"BEM reference: {ref_path}")
    return T_ref


def read_epgp_manifest(epgp_dir):
    # manifest columns (headerless): n_spectral, n_boundary, dofs, secs, mem_kb, cond
    rows = []
    with open(os.path.join(epgp_dir, "manifest.csv")) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            ns, nb, dofs, secs, mem = row[:5]
            rows.append({
                "ns": int(ns), "nb": int(nb), "dofs": int(dofs),
                "secs": float(secs),
                "mem": int(mem) if mem else 0,
                "cond": float(row[5]) if len(row) > 5 and row[5] else 0.0,
            })
    return sorted(rows, key=lambda r: (r["nb"], r["ns"]))


def aggregate_epgp(epgp_dir, T_ref, err_col):
    """Combine the EPGP manifest with the saved operators over the
    (n_spectral, n_boundary) grid: self-convergence is measured against the
    finest corner (max n_spectral, max n_boundary); err is vs the reference."""
    nref = np.linalg.norm(T_ref)
    runs = read_epgp_manifest(epgp_dir)
    Ts = {(r["ns"], r["nb"]): load_epgp(
        os.path.join(epgp_dir, f"T_ns{r['ns']}_nb{r['nb']}.npy")) for r in runs}
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
                    "norm", "recip", "selfconv_vs_finest", err_col, "mem_kb"])
        for r in runs:
            w.writerow([r["ns"], r["nb"], r["dofs"], f"{r['secs']:.3f}",
                        f"{r['cond']:.6e}", f"{r['norm']:.6e}",
                        f"{r['recip']:.6e}", f"{r['selfconv']:.6e}", f"{r['err']:.6e}",
                        r["mem"]])
            print(f"  EP-GP ns={r['ns']:>5} nb={r['nb']:>5}  recip={r['recip']:.3e}  "
                  f"selfconv={r['selfconv']:.3e}  err={r['err']:.3e}")
    print(f"wrote {path}")


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


if __name__ == "__main__":
    main()
