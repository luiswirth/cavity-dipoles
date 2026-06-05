"""Aggregate BEM convergence runs into out/bem/results.csv.

Reads the per-task manifest (poly_deg, refinement, dofs, recip, secs) pulled from the
cluster and the saved T-matrices, recomputes reciprocity, and computes the
self-convergence error of every run against the finest run (largest #dofs) as a
Richardson proxy for truth.
"""

import csv
import os

import numpy as np

from compare import load_bem, reciprocity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEM = os.path.join(ROOT, "out", "bem")


def read_manifest():
    """(p, m) -> {dofs, secs} from the SLURM manifest (recip recomputed below)."""
    info = {}
    with open(os.path.join(BEM, "manifest.csv")) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            _geom, p, m, dofs, _recip, secs = row
            info[(int(p), int(m))] = {"dofs": int(dofs), "secs": int(secs)}
    return info


def main():
    info = read_manifest()

    runs = []
    for (p, m), meta in info.items():
        T = load_bem(os.path.join(BEM, f"T_bem_ell_p{p}_m{m}.dat"))
        runs.append({"p": p, "m": m, "dofs": meta["dofs"], "secs": meta["secs"],
                     "recip": reciprocity(T), "T": T})

    ref = max(runs, key=lambda r: r["dofs"])
    print(f"reference (finest): p{ref['p']} m{ref['m']}, dofs={ref['dofs']}")
    nref = np.linalg.norm(ref["T"])
    for r in runs:
        r["err"] = np.linalg.norm(r["T"] - ref["T"]) / nref

    runs.sort(key=lambda r: r["dofs"])
    out = os.path.join(BEM, "results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "m", "dofs", "recip", "err_vs_finest", "secs"])
        for r in runs:
            w.writerow([r["p"], r["m"], r["dofs"], f"{r['recip']:.6e}",
                        f"{r['err']:.6e}", r["secs"]])
            print(f"  p{r['p']} m{r['m']}  dofs={r['dofs']:>6}  "
                  f"recip={r['recip']:.3e}  err={r['err']:.3e}  {r['secs']}s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
