import numpy as np


def green_scalar(r: float, k: float):
    return np.exp(1j * k * r) / (4 * np.pi * r)


def green_dyadic(r: np.ndarray, k: float) -> np.ndarray:
    rnorm = np.linalg.norm(r)
    rhat = r / rnorm

    Phi = green_scalar(rnorm, k)

    id = np.eye(3)
    rr = np.outer(rhat, rhat)  # r-hat dyad, shape (3,3)

    transverse = k**2 + 1j * k / rnorm - 1 / rnorm**2
    radial = -(k**2) - 3j * k / rnorm + 3 / rnorm**2

    return (1j / k) * Phi * (transverse * id + radial * rr)


def incident_field(x: np.ndarray, z: np.ndarray, k: float, p: np.ndarray) -> np.ndarray:
    return green_dyadic(x - z, k) @ p


ellipsoid_semiaxes = [4, 4, 6]


def ellipsoid_point(theta: float, phi: float) -> np.ndarray:
    a = ellipsoid_semiaxes
    x = np.array(
        [
            a[0] * np.sin(theta) * np.cos(phi),
            a[1] * np.sin(theta) * np.sin(phi),
            a[2] * np.cos(theta),
        ]
    )
    return x


def ellipsoid_normal(x: np.ndarray) -> np.ndarray:
    a = ellipsoid_semiaxes
    n = np.array([2 * x[0] / a[0] ** 2, 2 * x[1] / a[1] ** 2, 2 * x[2] / a[2] ** 2])
    n /= np.linalg.norm(n)
    return n


def compute_boundary_forcing(
    theta: float, phi: float, z: np.ndarray, k: float, p: np.ndarray
) -> np.ndarray:
    """Computes h = -n x E^i on the boundary."""
    x = ellipsoid_point(theta, phi)
    Ei = incident_field(x, z, k, p)
    n = ellipsoid_normal(x)
    return -np.cross(n, Ei)


k = 2
z = np.array([1, 0, 0])
p = np.array([0, 1, 0])

theta, phi = np.pi / 4, np.pi / 4

h = compute_boundary_forcing(theta, phi, z, k, p)
print(f"Forcing term h: {h}")
