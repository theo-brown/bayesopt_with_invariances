"""Kernels on the torus T^d and the sphere S^2, plus plain orbit averaging.

Torus
-----
The base kernel is the *wrapped* (periodised) Matern-5/2:

    k_per(x, y) = sum_{m in Z^d} k_Matern(||x - y + m||),

evaluated by summing over nearby integer images after reducing the
displacement to the fundamental cell [-1/2, 1/2)^d. With lengthscales well
below the cell size the truncated image sum is exact to machine precision.

By the Poisson summation formula, the Fourier coefficients of k_per are
exactly the R^d Matern spectral density sampled at integer frequencies:

    lambda_m = S(m),   k_per(x, y) = sum_m S(m) exp(2*pi*i m.(x - y)),

so RKHS membership and norms of trigonometric polynomials are available in
closed form (see targets.py). This identity is verified numerically in
test_exactness.py.

Sphere
------
The Riemannian Matern kernel on S^2 (Borovitskiy et al., 2020) with the
spectrum truncated at degree L:

    k(x, y) = sum_{l=0}^{L} a_l * (2l + 1) / (4*pi) * P_l(x . y),
    a_l propto (2*nu / kappa^2 + l*(l+1))^{-(nu + 1)},

normalised so that k(x, x) = 1. The truncated kernel is *defined* to be the
benchmark kernel, so the RKHS is exactly the degree-<=L band and target norms
are exact by construction.

Invariance
----------
For a group of isometries the doubly-averaged kernel of Brown et al. (2024),
Eq. (5), reduces to the single average

    k_G(x, y) = (1/|G|) sum_g k(g x, y),

because k(g x, g' y) = k(g'^{-1} g x, y) whenever the base kernel is invariant
under the diagonal action. No renormalisation is applied.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.special import gamma

from groups import apply_permutation, apply_rotation

NU = 2.5  # Matern smoothness used throughout


# ---------------------------------------------------------------------------
# Matern-5/2 in R^d and its spectral density
# ---------------------------------------------------------------------------

def matern52(r: np.ndarray, lengthscale: float) -> np.ndarray:
    a = np.sqrt(5.0) * r / lengthscale
    return (1.0 + a + a**2 / 3.0) * np.exp(-a)


def matern_spectral_density(freqs: np.ndarray, lengthscale: float, d: int,
                            nu: float = NU) -> np.ndarray:
    """Matern spectral density S(s) with the convention
    k(r) = integral S(s) exp(2*pi*i s.r) ds  (Rasmussen & Williams, Eq. 4.15).

    Parameters
    ----------
    freqs : (n, d) array of frequency vectors (cycles per unit length).
    """
    s2 = np.sum(np.asarray(freqs, dtype=float) ** 2, axis=-1)
    const = (2.0**d * np.pi ** (d / 2.0) * gamma(nu + d / 2.0)
             * (2.0 * nu) ** nu) / (gamma(nu) * lengthscale ** (2.0 * nu))
    return const * (2.0 * nu / lengthscale**2 + 4.0 * np.pi**2 * s2) ** (-(nu + d / 2.0))


# ---------------------------------------------------------------------------
# Torus kernel (wrapped Matern-5/2)
# ---------------------------------------------------------------------------

@dataclass
class WrappedMatern52:
    """Wrapped Matern-5/2 on T^d = [0,1)^d."""

    d: int
    lengthscale: float
    n_images: int = 1  # image offsets in {-n_images, ..., n_images}^d
    _offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        rng = range(-self.n_images, self.n_images + 1)
        self._offsets = np.array(list(itertools.product(rng, repeat=self.d)),
                                 dtype=float)

    def _wrapped_dist_sq(self, diff: np.ndarray) -> np.ndarray:
        """Sum the kernel over integer images of the displacement `diff`
        (shape (..., d)); returns k summed over images, shape (...)."""
        diff = diff - np.round(diff)  # reduce to [-1/2, 1/2)^d
        out = 0.0
        for off in self._offsets:
            r = np.sqrt(np.sum((diff + off) ** 2, axis=-1))
            out = out + matern52(r, self.lengthscale)
        return out

    def pair(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Full kernel matrix, shape (n, m)."""
        return self._wrapped_dist_sq(x[:, None, :] - y[None, :, :])

    def elementwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """k(x_i, y_i) for row-aligned inputs, shape (n,)."""
        return self._wrapped_dist_sq(x - y)


# ---------------------------------------------------------------------------
# Sphere kernel (truncated Riemannian Matern on S^2)
# ---------------------------------------------------------------------------

def sphere_matern_coeffs(max_degree: int, kappa: float,
                         nu: float = NU) -> np.ndarray:
    """Per-degree spectral coefficients a_l, l = 0..L, normalised so that
    k(x, x) = sum_l a_l (2l+1)/(4*pi) = 1."""
    ell = np.arange(max_degree + 1, dtype=float)
    a = (2.0 * nu / kappa**2 + ell * (ell + 1.0)) ** (-(nu + 1.0))
    a /= np.sum(a * (2.0 * ell + 1.0) / (4.0 * np.pi))
    return a


@dataclass
class SphereMatern:
    """Truncated-spectrum Matern kernel on S^2."""

    max_degree: int
    kappa: float
    coeffs: np.ndarray = field(init=False, repr=False)  # a_l
    _legendre_c: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.coeffs = sphere_matern_coeffs(self.max_degree, self.kappa)
        ell = np.arange(self.max_degree + 1, dtype=float)
        self._legendre_c = self.coeffs * (2.0 * ell + 1.0) / (4.0 * np.pi)

    def _from_dot(self, t: np.ndarray) -> np.ndarray:
        return legval(np.clip(t, -1.0, 1.0), self._legendre_c)

    def pair(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._from_dot(x @ y.T)

    def elementwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._from_dot(np.sum(x * y, axis=-1))


# ---------------------------------------------------------------------------
# Plain orbit-averaged (invariant) kernel: k_G(x, y) = mean_g k(g x, y)
# ---------------------------------------------------------------------------

@dataclass
class OrbitAveragedKernel:
    """Single-sided group average of a base kernel (equal to the double
    average for isometry groups). No renormalisation."""

    base: object
    group: list        # permutations (torus) or matrices (sphere)
    action: str        # "permutation" or "rotation"

    def _transform(self, x: np.ndarray, g) -> np.ndarray:
        if self.action == "permutation":
            return apply_permutation(x, g)
        return apply_rotation(x, g)

    def pair(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = 0.0
        for g in self.group:
            out = out + self.base.pair(self._transform(x, g), y)
        return out / len(self.group)

    def elementwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = 0.0
        for g in self.group:
            out = out + self.base.elementwise(self._transform(x, g), y)
        return out / len(self.group)
