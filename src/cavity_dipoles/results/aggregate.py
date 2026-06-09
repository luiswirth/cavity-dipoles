import csv
import os

import numpy as np

from .compare import load_bem, load_epgp, reciprocity

BEM = os.path.join("out", "bem")
EPGP = os.path.join("out", "epgp")


def read_manifest():
    info = {}
    with open(os.path.join(BEM, "manifest.csv")) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            _geom, p, m, dofs, _recip, secs = row
            info[(int(p), int(m))] = {"dofs": int(dofs), "secs": int(secs)}
    return info


def aggregate_bem():
    info = read_manifest()
    runs = []
    for (p, m), meta in info.items():
        path = os.path.join(BEM, f"T_bem_p{p}_m{m}.dat")
        if not os.path.exists(path):
            continue
        T = load_bem(path)
        runs.append({"p": p, "m": m, "dofs": meta["dofs"], "secs": meta["secs"],
                     "recip": reciprocity(T), "T": T})
    ref = min(runs, key=lambda r: r["recip"])
    nref = np.linalg.norm(ref["T"])
    for r in runs:
        r["selfconv"] = np.linalg.norm(r["T"] - ref["T"]) / nref
    runs.sort(key=lambda r: (r["p"], r["m"]))

    with open(os.path.join(BEM, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "m", "dofs", "recip", "selfconv_vs_ref", "secs"])
        for r in runs:
            w.writerow([r["p"], r["m"], r["dofs"], f"{r['recip']:.6e}",
                        f"{r['selfconv']:.6e}", r["secs"]])
    print(f"BEM reference (lowest recip): p{ref['p']} m{ref['m']}, "
          f"dofs={ref['dofs']}, recip={ref['recip']:.2e}")
    return ref["T"], (ref["p"], ref["m"], ref["dofs"])


def read_epgp_manifest():
    rows = []
    with open(os.path.join(EPGP, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            rows.append({
                "ns": int(row["n_spectral"]),
                "dofs": int(row["dofs"]),
                "secs": float(row["secs"]),
                "log_noise": float(row["log_noise"]),
                "cond": float(row["cond"]),
            })
    return sorted(rows, key=lambda r: r["ns"])


def aggregate_epgp(T_bem_ref):
    """Combine the EPGP run manifest with the saved operators: add the
    self-convergence (vs the finest n_spectral) and the cross-validation error
    against the BEM reference, then write the rich convergence table."""
    nref_bem = np.linalg.norm(T_bem_ref)
    runs = read_epgp_manifest()
    Ts = {r["ns"]: load_epgp(os.path.join(EPGP, f"T_epgp_ns{r['ns']}.npy")) for r in runs}
    Tfinest = Ts[runs[-1]["ns"]]
    nfinest = np.linalg.norm(Tfinest)

    for r in runs:
        T = Ts[r["ns"]]
        r["recip"] = reciprocity(T)
        r["selfconv"] = np.linalg.norm(T - Tfinest) / nfinest
        r["err"] = np.linalg.norm(T - T_bem_ref) / nref_bem

    with open(os.path.join(EPGP, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_spectral", "dofs", "secs", "log_noise", "cond",
                    "recip", "selfconv_vs_finest", "err_vs_bem_ref"])
        for r in runs:
            w.writerow([r["ns"], r["dofs"], f"{r['secs']:.3f}",
                        f"{r['log_noise']:.6f}", f"{r['cond']:.6e}",
                        f"{r['recip']:.6e}", f"{r['selfconv']:.6e}", f"{r['err']:.6e}"])
            print(f"  EP-GP ns={r['ns']:>5}  secs={r['secs']:6.1f}  cond={r['cond']:.2e}  "
                  f"recip={r['recip']:.3e}  selfconv={r['selfconv']:.3e}  err_vs_BEM={r['err']:.3e}")
    print(f"wrote {os.path.join(EPGP, 'results.csv')}")


def main():
    T_bem_ref, _ = aggregate_bem()
    if os.path.exists(os.path.join(EPGP, "manifest.csv")):
        aggregate_epgp(T_bem_ref)
    else:
        print("(no EP-GP convergence manifest in out/epgp/; run epgp.convergence)")


if __name__ == "__main__":
    main()
