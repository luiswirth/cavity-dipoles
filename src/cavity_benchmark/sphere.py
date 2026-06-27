import numpy as np
from scipy.special import roots_legendre, sph_harm_y, spherical_jn

from cavity_epgp import incident_field_batch


def _quadrature(n_theta, n_phi):
    # Gauss-Legendre nodes are interior, so sin(theta) != 0 at any quadrature point.
    x, wx = roots_legendre(n_theta)
    theta = np.arccos(x)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    W = np.outer(wx, np.full(n_phi, 2.0 * np.pi / n_phi))
    return TH.ravel(), PH.ravel(), W.ravel()


def _frames(theta, phi):
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    er = np.stack([st * cp, st * sp, ct], axis=-1)
    eth = np.stack([ct * cp, ct * sp, -st], axis=-1)
    eph = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
    return er, eth, eph


def _vsh(l, theta, phi):
    npts = theta.size
    ms = np.arange(-l, l + 1)
    Y = np.stack([sph_harm_y(l, int(m), theta, phi) for m in ms])  # (2l+1, npts)

    def Yget(m):
        return Y[m + l] if -l <= m <= l else np.zeros(npts, dtype=complex)

    eip, eim = np.exp(1j * phi), np.exp(-1j * phi)
    dYdth = np.empty_like(Y)
    for i, m in enumerate(ms):
        cpl = np.sqrt((l - m) * (l + m + 1))
        cmi = np.sqrt((l + m) * (l - m + 1))
        dYdth[i] = 0.5 * (eim * cpl * Yget(m + 1) - eip * cmi * Yget(m - 1))

    st = np.sin(theta)
    norm = 1.0 / np.sqrt(l * (l + 1))
    Psi_th = norm * dYdth
    Psi_ph = norm * (1j * ms[:, None] * Y) / st
    # Phi = rhat x Psi : (Phi_th, Phi_ph) = (-Psi_ph, Psi_th)
    _, eth, eph = _frames(theta, phi)
    Psi = Psi_th[..., None] * eth + Psi_ph[..., None] * eph
    Phi = (-Psi_ph)[..., None] * eth + Psi_th[..., None] * eph
    return Psi, Phi


def _psi_prime(l, x):
    """Riccati-Bessel derivative psi_l'(x) = [x j_l(x)]' = j_l(x) + x j_l'(x)."""
    return spherical_jn(l, x) + x * spherical_jn(l, x, derivative=True)


def _tangential(field, normal):
    """Project out the normal component, leaving Pi_t field."""
    return field - np.sum(field * normal, axis=-1, keepdims=True) * normal


def _defaults(k, R, L_max, n_theta, n_phi):
    if L_max is None:
        L_max = int(np.ceil(k * R)) + 12
    if n_theta is None:
        n_theta = L_max + 4
    if n_phi is None:
        n_phi = 2 * L_max + 4
    return L_max, n_theta, n_phi


def _contract(y, e1, e2):
    n_cfg = y.shape[0]
    T = np.empty((n_cfg, n_cfg), dtype=complex)
    T[0::2] = np.einsum("xc,sxc->xs", e1, y)
    T[1::2] = np.einsum("xc,sxc->xs", e2, y)
    return T


def _degree_responses(k, R, points, e1, e2, a, L_max, n_theta, n_phi):
    points = np.asarray(points, dtype=float)
    dirs = points / np.linalg.norm(points, axis=1, keepdims=True)
    N = len(points)

    configs = []
    for i in range(N):
        configs.append((dirs[i] * a, e1[i]))
        configs.append((dirs[i] * a, e2[i]))

    th, ph, W = _quadrature(n_theta, n_phi)
    er, _, _ = _frames(th, ph)
    wall = R * er
    PtE = np.stack([_tangential(incident_field_batch(wall, z, k, p), er)
                    for z, p in configs])  # (2N, nq, 3)

    mth = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    mph = np.arctan2(dirs[:, 1], dirs[:, 0])

    for l in range(1, L_max + 1):
        Psi_g, Phi_g = _vsh(l, th, ph)
        Psi_m, Phi_m = _vsh(l, mth, mph)
        p_lm = np.einsum("q,sqc,mqc->sm", W, PtE, np.conj(Psi_g))
        q_lm = np.einsum("q,sqc,mqc->sm", W, PtE, np.conj(Phi_g))
        # interior PEC reflection: Psi (TM, N-type) fixed by psi_l', Phi (TE,
        # M-type) by j_l; the scattered tangential field on Lambda follows.
        jR, ja = spherical_jn(l, k * R), spherical_jn(l, k * a)
        psR, psa = _psi_prime(l, k * R), _psi_prime(l, k * a)
        y_psi = -(R / a) * (psa / psR) * p_lm
        y_phi = -(ja / jR) * q_lm
        y_l = (np.einsum("sm,mxc->sxc", y_psi, Psi_m)
               + np.einsum("sm,mxc->sxc", y_phi, Phi_m))
        yield l, y_l


def reaction_operator_sphere(k, R, points, e1, e2, a=1.0,
                             L_max=None, n_theta=None, n_phi=None):
    L_max, n_theta, n_phi = _defaults(k, R, L_max, n_theta, n_phi)
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    N = len(points)
    y = np.zeros((2 * N, N, 3), dtype=complex)
    for _, y_l in _degree_responses(k, R, points, e1, e2, a, L_max, n_theta, n_phi):
        y += y_l
    return _contract(y, e1, e2)
