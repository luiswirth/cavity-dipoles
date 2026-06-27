import glob
import os

from .results.compare import load_bem
from .sphere import reaction_operator_sphere

GEOMETRIES = {"ellipse": (4.0, 4.0, 6.0), "sphere": (4.0, 4.0, 4.0)}


def bem_grid_dir(name):
    return os.path.join("out", "bem", "grid", name)


def _parse_pm(path):
    body = os.path.basename(path)[len("T_p"):-len(".dat")]
    p_str, m_str = body.split("_m")
    return int(p_str), int(m_str)


def bem_reference_path(name="ellipse"):
    grid = bem_grid_dir(name)
    cands = glob.glob(os.path.join(grid, "T_p*_m*.dat"))
    if not cands:
        raise FileNotFoundError(f"no BEM grid operators in {grid}")
    return max(cands, key=_parse_pm)


def config_path(name):
    return os.path.join("res", f"config_{name}.txt")


def out_dir(name):
    return os.path.join("out", "epgp", "grid", name)


def reference_operator(name, k, points, e1, e2):
    if name == "sphere":
        return reaction_operator_sphere(k, float(max(GEOMETRIES[name])), points, e1, e2)
    if name == "ellipse":
        return load_bem(bem_reference_path(name))
    raise ValueError(f"unknown benchmark {name!r}")
