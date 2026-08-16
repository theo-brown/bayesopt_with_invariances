"""Transformation groups acting on the torus T^d = [0,1)^d and the sphere S^2.

Torus groups are given as lists of coordinate permutations (tuples), acting by
x -> x[perm]. These are isometries of the flat torus, so they commute with the
wrapped (periodised) kernel.

Sphere groups are given as lists of 3x3 orthogonal matrices, acting by
x -> g @ x. These are isometries of S^2, so they commute with any zonal kernel
k(x, y) = kappa(x . y).
"""

import itertools

import numpy as np


# ---------------------------------------------------------------------------
# Torus: permutation groups
# ---------------------------------------------------------------------------

def permutation_group(d: int) -> list[tuple[int, ...]]:
    """Full symmetric group S_d acting by permuting the d coordinates."""
    return [tuple(p) for p in itertools.permutations(range(d))]


def cyclic_permutation_group(d: int) -> list[tuple[int, ...]]:
    """Cyclic group C_d acting by cyclic shifts of the d coordinates."""
    return [tuple((i + s) % d for i in range(d)) for s in range(d)]


def apply_permutation(x: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    """Apply a coordinate permutation to an (n, d) array of points."""
    return x[:, list(perm)]


# ---------------------------------------------------------------------------
# Sphere: finite subgroups of O(3)
# ---------------------------------------------------------------------------

def cyclic_rotation_group(n: int) -> list[np.ndarray]:
    """Cyclic group C_n of rotations by 2*pi/n about the z-axis."""
    mats = []
    for k in range(n):
        t = 2.0 * np.pi * k / n
        c, s = np.cos(t), np.sin(t)
        mats.append(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    return mats


def octahedral_rotation_group() -> list[np.ndarray]:
    """Rotation group of the cube/octahedron: the 24 signed permutation
    matrices with determinant +1."""
    mats = []
    for perm in itertools.permutations(range(3)):
        p = np.zeros((3, 3))
        for i, j in enumerate(perm):
            p[i, j] = 1.0
        for signs in itertools.product([1.0, -1.0], repeat=3):
            m = p * np.array(signs)[:, None]
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    assert len(mats) == 24
    return mats


def apply_rotation(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Apply an orthogonal matrix g to an (n, 3) array of unit vectors."""
    return x @ g.T
