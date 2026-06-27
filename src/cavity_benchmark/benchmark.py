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


def config_path(name):
    return os.path.join("res", f"config_{name}.txt")


def out_dir(name):
    return os.path.join("out", "epgp", "grid", name)


def reference_operator(name, k, points, e1, e2):
    if name == "sphere":
        return reaction_operator_sphere(k, float(max(GEOMETRIES[name])), points, e1, e2)
    if name == "ellipse":
        return load_bem(bem_reference_path())
    raise ValueError(f"unknown benchmark {name!r}")
