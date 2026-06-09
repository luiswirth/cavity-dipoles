import argparse

import numpy as np


def load_bem(path):
    data = np.loadtxt(path)
    return data[:, 0::2] + 1j * data[:, 1::2]


def load_epgp(path):
    return np.load(path)


def reciprocity(T):
    return np.linalg.norm(T - T.T) / np.linalg.norm(T)


def metrics(Ta, Tb):
    return {
        "rel_err": np.linalg.norm(Ta - Tb) / np.linalg.norm(Tb),
        "recip_a": reciprocity(Ta),
        "recip_b": reciprocity(Tb),
        "norm_a": np.linalg.norm(Ta),
        "norm_b": np.linalg.norm(Tb),
    }


def main(bem_path, epgp_path):
    T_bem = load_bem(bem_path)
    T_epgp = load_epgp(epgp_path)

    assert T_bem.shape == T_epgp.shape, (T_bem.shape, T_epgp.shape)

    m = metrics(T_epgp, T_bem)
    print(f"shape: {T_bem.shape}")
    print(f"||T_bem||   = {m['norm_b']:.4f}")
    print(f"||T_epgp||  = {m['norm_a']:.4f}")
    print(f"||T_epgp - T_bem|| / ||T_bem|| = {m['rel_err']:.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bem", help="path to BEM T_matrix.dat")
    p.add_argument("epgp", help="path to EP-GP T_epgp.npy")
    args = p.parse_args()
    main(args.bem, args.epgp)
