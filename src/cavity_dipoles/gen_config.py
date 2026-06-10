import argparse
import os

import numpy as np

from .benchmark import GEOMETRIES, config_path

N = 32


def fibonacci_sphere(n):
    # Defines the benchmark Lambda points: written to the config and consumed by
    # both solvers (the C++ BEM reads them from the config, never regenerating).
    # Kept independent of maxwellgp.utils.fibonacci_sphere, which is the EPGP's
    # private boundary-collocation spiral; the two roles must not be conflated,
    # since changing this spiral would redefine the benchmark.
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=1,
    )


def tangent_frame(n):
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def write_config(k, semiaxes, n, out):
    points = fibonacci_sphere(n)
    rows = []
    for x in points:
        nrm = x
        e1, e2 = tangent_frame(nrm)
        rows.append(np.concatenate([x, e1, e2]))
    rows = np.stack(rows)

    a, b, c = semiaxes

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("# k a b c N\n")
        f.write(f"{k} {a} {b} {c} {n}\n")
        f.write("# x y z  e1x e1y e1z  e2x e2y e2z\n")
        for r in rows:
            f.write(" ".join(f"{v:.15g}" for v in r) + "\n")

    print(f"wrote {out}: {n} points")


def main():
    p = argparse.ArgumentParser(description="generate a benchmark cavity config")
    p.add_argument("--geometry", choices=list(GEOMETRIES), default="ellipse")
    p.add_argument("-n", "--num", type=int, default=N, help="dipole points on Lambda")
    p.add_argument("-k", type=float, default=2.0, help="wavenumber")
    p.add_argument("--semiaxes", type=float, nargs=3, default=None, metavar=("A", "B", "C"))
    p.add_argument("-o", "--out", default=None, help="output config path")
    args = p.parse_args()
    semiaxes = tuple(args.semiaxes) if args.semiaxes else GEOMETRIES[args.geometry]
    out = args.out or config_path(args.geometry)
    write_config(args.k, semiaxes, args.num, out)


if __name__ == "__main__":
    main()
