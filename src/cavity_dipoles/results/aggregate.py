import csv
import glob
import os
import re

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


def aggregate_epgp(T_bem_ref):
    nref = np.linalg.norm(T_bem_ref)
    rows = []
    for f in glob.glob(os.path.join(EPGP, "T_epgp_ns*.npy")):
        ns = int(re.search(r"ns(\d+)", f).group(1))
        T = load_epgp(f)
        rows.append({"ns": ns, "recip": reciprocity(T),
                     "err": np.linalg.norm(T - T_bem_ref) / nref})
    rows.sort(key=lambda r: r["ns"])
    with open(os.path.join(EPGP, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_spectral", "recip", "err_vs_bem_ref"])
        for r in rows:
            w.writerow([r["ns"], f"{r['recip']:.6e}", f"{r['err']:.6e}"])
            print(f"  EP-GP ns={r['ns']:>5}  recip={r['recip']:.3e}  "
                  f"err_vs_BEM={r['err']:.3e}")
    print(f"wrote {os.path.join(EPGP, 'results.csv')}")


def main():
    T_bem_ref, _ = aggregate_bem()
    if os.path.isdir(EPGP) and glob.glob(os.path.join(EPGP, "T_epgp_ns*.npy")):
        aggregate_epgp(T_bem_ref)
    else:
        print("(no EP-GP sweep found in out/epgp/)")


if __name__ == "__main__":
    main()
