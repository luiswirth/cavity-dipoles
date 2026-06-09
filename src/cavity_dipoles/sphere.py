"""Analytic reaction operator for a PEC spherical cavity (ground truth).

Independent multipole solver. It projects the dipole incident field's tangential
trace on the spherical wall (radius R) onto vector spherical harmonics, applies
the interior PEC reflection relations, and reads the scattered tangential trace
back on the measurement sphere Lambda (radius a). It serves as an exact reference
for the EPGP solver and corroborates the closed-form derivation in the thesis
appendix. A single consistent vector-spherical-harmonic normalization is used
throughout, so the operator is independent of the overall constant the appendix
carries symbolically.

Conventions match the rest of the project: time dependence e^{-i omega t}, so the
fundamental solution is outgoing with e^{+i k r}, and the dipole field is the
dyadic Green's function of analytic.incident_field_batch.
"""

import numpy as np
from scipy.special import roots_legendre, sph_harm_y, spherical_jn

from .analytic import incident_field_batch


def _quadrature(n_theta, n_phi):
    """Product Gauss-Legendre (in cos theta) times uniform phi grid on S^2.

    Returns flattened (theta, phi, weights) with sum(weights f) = integral over
    the unit sphere. Gauss nodes are interior, so sin theta != 0 anywhere.
    """
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
    """Tangential vector spherical harmonics Psi_lm, Phi_lm (Cartesian).

    Returns arrays of shape (2l+1, npts, 3) indexed by m = -l..l. Uses the ladder
    relation for d/dtheta Y_l^m, which is pole-free.
    """
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


def reaction_operator_sphere(k, R, points, e1, e2, a=1.0,
                             L_max=None, n_theta=None, n_phi=None):
    """Exact reaction operator T for a PEC sphere of radius R.

    points are unit directions on Lambda (radius a); e1, e2 the tangent frame.
    Returns the (2N, 2N) complex operator in the same (point, polarization)
    ordering as the EPGP and BEM assemblies.
    """
    if L_max is None:
        L_max = int(np.ceil(k * R)) + 12
    if n_theta is None:
        n_theta = L_max + 4
    if n_phi is None:
        n_phi = 2 * L_max + 4

    points = np.asarray(points, dtype=float)
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    dirs = points / np.linalg.norm(points, axis=1, keepdims=True)
    N = len(points)

    # source / measurement configurations: (point, polarization)
    configs = []
    for i in range(N):
        configs.append((dirs[i] * a, e1[i]))
        configs.append((dirs[i] * a, e2[i]))
    n_cfg = len(configs)

    # wall quadrature points (radius R) and the incident tangential projections
    # there. The benchmark conditions on Pi_t E and measures q . E^s (dipole
    # transmit, dipole receive), so we work with the tangential projection
    # throughout rather than the rotated n x E trace.
    th, ph, W = _quadrature(n_theta, n_phi)
    er, _, _ = _frames(th, ph)
    wall = R * er

    PtE = np.stack([_tangential(incident_field_batch(wall, z, k, p), er)
                    for z, p in configs])  # (n_cfg, nq, 3)

    # measurement directions (radius a) -> angles
    mth = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    mph = np.arctan2(dirs[:, 1], dirs[:, 0])

    y = np.zeros((n_cfg, N, 3), dtype=complex)  # scattered tangential trace on Lambda
    for l in range(1, L_max + 1):
        Psi_g, Phi_g = _vsh(l, th, ph)            # (2l+1, nq, 3)
        Psi_m, Phi_m = _vsh(l, mth, mph)          # (2l+1, N, 3)

        # project the incident tangential projection onto Psi, Phi
        p_lm = np.einsum("q,sqc,mqc->sm", W, PtE, np.conj(Psi_g))
        q_lm = np.einsum("q,sqc,mqc->sm", W, PtE, np.conj(Phi_g))

        # interior PEC reflection: the Psi (TM, N-type) channel is fixed by the
        # Riccati derivative psi_l', the Phi (TE, M-type) channel by j_l. The
        # tangential scattered field on Lambda is then
        jR, ja = spherical_jn(l, k * R), spherical_jn(l, k * a)
        psR, psa = _psi_prime(l, k * R), _psi_prime(l, k * a)
        y_psi = -(R / a) * (psa / psR) * p_lm
        y_phi = -(ja / jR) * q_lm

        y += np.einsum("sm,mxc->sxc", y_psi, Psi_m)
        y += np.einsum("sm,mxc->sxc", y_phi, Phi_m)

    # contract with the measurement tangent frames -> (2N, 2N), interleaving the
    # two polarizations per point to match the (point, polarization) ordering
    T = np.empty((n_cfg, n_cfg), dtype=complex)
    T[0::2] = np.einsum("xc,sxc->xs", e1, y)
    T[1::2] = np.einsum("xc,sxc->xs", e2, y)
    return T
