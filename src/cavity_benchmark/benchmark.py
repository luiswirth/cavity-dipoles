"""Benchmark geometries and their reference reaction operators.

Two benchmarks share one EPGP convergence study, differing only in the reference
the EPGP operator is compared against:
  * ellipse -- semi-axes (4, 4, 6); reference is the deterministic BEM operator
    from out/bem/ref/ellipse/ (produced by cavity-bem/euler/submit_ref.sh);
  * sphere  -- semi-axes (R, R, R); reference is the exact analytic operator of
    sphere.reaction_operator_sphere.

Generation (assembling EPGP operators) and analysis (comparing to the reference)
are kept separate, as forced on the BEM side: epgp.convergence generates the
operators and a raw manifest for a given geometry, and results.aggregate then
compares them to the reference returned here.
"""

import glob
import os

from .results.compare import load_bem
from .sphere import reaction_operator_sphere

GEOMETRIES = {"ellipse": (4.0, 4.0, 6.0), "sphere": (4.0, 4.0, 4.0)}


def bem_reference_path():
    pattern = os.path.join("out", "bem", "ref", "ellipse", "T_p*.dat")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no BEM reference operator found at {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple BEM reference operators found: {matches}; expected exactly one")
    return matches[0]


def semiaxes(name):
    return GEOMETRIES[name]


def config_path(name):
    return os.path.join("res", f"config_{name}.txt")


def out_dir(name):
    return os.path.join("out", "epgp", "grid", name)


def reference_operator(name, k, points, e1, e2):
    """Reference reaction operator for the named benchmark at wavenumber k."""
    if name == "sphere":
        return reaction_operator_sphere(k, float(max(GEOMETRIES[name])), points, e1, e2)
    if name == "ellipse":
        return load_bem(bem_reference_path())
    raise ValueError(f"unknown benchmark {name!r}")
