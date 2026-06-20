"""Analysis step: turn raw generated operators into convergence tables.

Mirrors the generation/analysis split forced by the BEM side. The EPGP operators
and their raw manifest are produced separately (epgp.convergence); here we add
derived quantities (norm, recip, err) so downstream plotting never loads operators.
"""

import argparse
import csv
import os

import numpy as np

from ..benchmark import bem_reference_path, config_path, out_dir, reference_operator
from cavity_epgp import load_config
from .compare import load_bem, load_epgp, reciprocity


def bem_grid_dir(geometry):
    return os.path.join("out", "bem", "grid", geometry)


def read_bem_manifest(bem_dir):
    info = {}
    with open(os.path.join(bem_dir, "manifest.csv")) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            p, m, dofs, secs, mem = row[:5]
            cond = float(row[5]) if len(row) > 5 and row[5] else 0.0
            info[(int(p), int(m))] = {"dofs": int(dofs), "secs": int(secs),
                                      "mem": int(mem) if mem else 0,
                                      "cond": cond}
    return info


def aggregate_bem(geometry, T_ref):
    bem_dir = bem_grid_dir(geometry)
    info = read_bem_manifest(bem_dir)
    nref = np.linalg.norm(T_ref)
    runs = []
    for (p, m), meta in info.items():
        path = os.path.join(bem_dir, f"T_p{p}_m{m}.dat")
        if not os.path.exists(path):
            continue
        T = load_bem(path)
        r = {"p": p, "m": m, "dofs": meta["dofs"], "secs": meta["secs"],
             "mem": meta["mem"], "cond": meta["cond"],
             "norm": float(np.linalg.norm(T))}
        r["recip"] = reciprocity(T)
        if geometry != "ellipse":
            r["err"] = np.linalg.norm(T - T_ref) / nref
        runs.append(r)
    runs.sort(key=lambda r: (r["p"], r["m"]))

    with open(os.path.join(bem_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        if geometry == "ellipse":
            w.writerow(["p", "m", "dofs", "secs", "mem_kb", "cond", "norm", "recip"])
            for r in runs:
                w.writerow([r["p"], r["m"], r["dofs"], r["secs"], r["mem"],
                            f"{r['cond']:.6e}", f"{r['norm']:.6e}", f"{r['recip']:.6e}"])
        else:
            w.writerow(["p", "m", "dofs", "secs", "mem_kb", "cond", "norm", "recip", "err"])
            for r in runs:
                w.writerow([r["p"], r["m"], r["dofs"], r["secs"], r["mem"],
                            f"{r['cond']:.6e}", f"{r['norm']:.6e}", f"{r['recip']:.6e}",
                            f"{r['err']:.6e}"])
    print(f"BEM {geometry}: wrote {bem_dir}/results.csv")


def read_epgp_manifest(epgp_dir):
    rows = []
    with open(os.path.join(epgp_dir, "manifest.csv")) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            ns, nb, dofs, secs, mem = row[:5]
            rows.append({
                "ns": int(ns), "nb": int(nb), "dofs": int(dofs),
                "secs": int(secs),
                "mem": int(mem) if mem else 0,
                "cond": float(row[5]) if len(row) > 5 and row[5] else 0.0,
            })
    return sorted(rows, key=lambda r: (r["nb"], r["ns"]))


def aggregate_epgp(epgp_dir, T_ref, is_ellipse):
    """Combine the EPGP manifest with the saved operators. Ellipse: recip + err.
    Sphere: err only (analytic reference, recip unnecessary)."""
    nref = np.linalg.norm(T_ref)
    runs = read_epgp_manifest(epgp_dir)
    Ts = {(r["ns"], r["nb"]): load_epgp(
        os.path.join(epgp_dir, f"T_ns{r['ns']}_nb{r['nb']}.npy")) for r in runs}

    for r in runs:
        T = Ts[(r["ns"], r["nb"])]
        r["norm"] = float(np.linalg.norm(T))
        r["recip"] = reciprocity(T)
        r["err"] = np.linalg.norm(T - T_ref) / nref

    path = os.path.join(epgp_dir, "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_spectral", "n_boundary", "dofs", "secs", "mem_kb",
                    "cond", "norm", "recip", "err"])
        for r in runs:
            print(f"  EP-GP ns={r['ns']:>5} nb={r['nb']:>5}  recip={r['recip']:.3e}  err={r['err']:.3e}")
            w.writerow([r["ns"], r["nb"], r["dofs"], r["secs"], r["mem"],
                        f"{r['cond']:.6e}", f"{r['norm']:.6e}", f"{r['recip']:.6e}",
                        f"{r['err']:.6e}"])
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="aggregate convergence results")
    ap.add_argument("--geometry", choices=["ellipse", "sphere"], default="ellipse")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    config = args.config or config_path(args.geometry)
    k, _semi, points, e1, e2 = load_config(config)
    is_ellipse = args.geometry == "ellipse"

    if is_ellipse:
        ref_path = bem_reference_path()
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"BEM reference operator missing: {ref_path}")
        T_ref = load_bem(ref_path)
        print(f"BEM reference: {ref_path}")
    else:
        T_ref = reference_operator(args.geometry, k, points, e1, e2)

    bem_dir = bem_grid_dir(args.geometry)
    if os.path.exists(os.path.join(bem_dir, "manifest.csv")):
        aggregate_bem(args.geometry, T_ref)
    else:
        print(f"(no BEM manifest in {bem_dir}/; run bem-grid)")

    epgp_dir = out_dir(args.geometry)
    if os.path.exists(os.path.join(epgp_dir, "manifest.csv")):
        aggregate_epgp(epgp_dir, T_ref, is_ellipse)
    else:
        print(f"(no EP-GP manifest in {epgp_dir}/; run epgp-grid)")


if __name__ == "__main__":
    main()
