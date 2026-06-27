import os
import re

import numpy as np

_RUN = re.compile(r"Sigma_ns(\d+)_nb(\d+)\.npy")


def available_runs(uq_dir):
    runs = [(int(m[1]), int(m[2])) for m in map(_RUN.fullmatch, os.listdir(uq_dir)) if m]
    return sorted(runs)


def load_run(uq_dir, ns, nb):
    T = np.load(os.path.join(uq_dir, f"T_ns{ns}_nb{nb}.npy"))
    Sigma = np.load(os.path.join(uq_dir, f"Sigma_ns{ns}_nb{nb}.npy"))
    return T, Sigma


def entry_std(Sigma):
    return np.sqrt(np.clip(np.real(np.diag(Sigma)), 0.0, None))
