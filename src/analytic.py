import numpy as np


def green_scalar(r, k):
    return np.exp(1j * k * r) / (4 * np.pi * r)


def green_dyadic(rv, k):
    r = np.linalg.norm(rv)
    rhat = rv / r
    phi = green_scalar(r, k)
    transverse = k**2 + 1j * k / r - 1 / r**2
    radial = -(k**2) - 3j * k / r + 3 / r**2
    return (1j / k) * phi * (transverse * np.eye(3) + radial * np.outer(rhat, rhat))


def incident_field(x, z, k, p):
    return green_dyadic(x - z, k) @ p


def incident_field_batch(X, z, k, p):
    rv = X - z
    r = np.linalg.norm(rv, axis=1)
    rhat = rv / r[:, None]
    phi = np.exp(1j * k * r) / (4 * np.pi * r)
    transverse = k**2 + 1j * k / r - 1 / r**2
    radial = -(k**2) - 3j * k / r + 3 / r**2
    rhat_p = rhat @ p
    return (1j / k) * phi[:, None] * (
        transverse[:, None] * p + radial[:, None] * rhat_p[:, None] * rhat
    )


def tangential_projection(v, n):
    return v - np.dot(v, n) * n


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    )


def ellipsoid_point(theta, phi, semiaxes):
    a = np.asarray(semiaxes)
    return a * np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def ellipsoid_normal(x, semiaxes):
    n = np.asarray(x) / np.asarray(semiaxes) ** 2
    return n / np.linalg.norm(n)


def boundary_collocation(semiaxes, n):
    semiaxes = np.asarray(semiaxes)
    u = fibonacci_sphere(n)
    points = u * semiaxes
    normals = points / semiaxes**2
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals
