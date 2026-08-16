"""Synthetic targets with exact RKHS membership and exactly known RKHS norm.

Torus targets
-------------
A truncated Karhunen-Loeve draw in the Fourier basis of the wrapped kernel:

    f(x) = Re sum_{|m|_inf <= M} c_m exp(2*pi*i m.x),
    c_m = sqrt(lambda_m) * z_m,   z_m ~ CN(0, 1),   lambda_m = S(m),

where S is the Matern spectral density (Poisson summation gives lambda_m =
S(m) exactly; see kernels.py). Coefficients are then averaged over the group
orbit of the frequency vectors, which is exactly the projection of f onto the
G-invariant subspace, i.e. a truncated draw from the GP with the
orbit-averaged kernel. The RKHS norm is exact:

    ||f||^2_{H_k} = sum_m |c_m|^2 / lambda_m,

and for invariant f this also equals the norm in H_{k_G} (Kondor 2008; Brown
et al. 2024, Appendix A). The number of active modes (2M+1)^d is chosen to be
far larger than any BO budget, so no small set of clever samples identifies f.

Sphere targets
--------------
A band-limited draw in the real spherical harmonic basis,

    f(x) = sum_{l <= L} sum_j c_{lj} Y_{lj}(x),   c_{lj} = sqrt(a_l) z_{lj},

projected onto the G-invariant subspace by averaging f over the group and
re-expanding: because rotations preserve each degree-l eigenspace, f(g.) is
still band-limited at L, so the Gauss-Legendre x uniform quadrature used for
the re-expansion is exact and the projected coefficients are exact (up to
floating point). The norm ||f||^2 = sum |c_{lj}|^2 / a_l is exact for the
truncated-spectrum kernel of kernels.py.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from kernels import matern_spectral_density, NU


# ---------------------------------------------------------------------------
# Torus
# ---------------------------------------------------------------------------

@dataclass
class TorusTarget:
    modes: np.ndarray        # (n_modes, d) integer frequency vectors
    coeffs: np.ndarray       # (n_modes,) complex coefficients
    lambdas: np.ndarray      # (n_modes,) spectral coefficients lambda_m = S(m)

    @property
    def rkhs_norm(self) -> float:
        return float(np.sqrt(np.sum(np.abs(self.coeffs) ** 2 / self.lambdas)))

    def __call__(self, x: np.ndarray, chunk: int = 2048) -> np.ndarray:
        x = np.atleast_2d(x)
        out = np.empty(x.shape[0])
        for i in range(0, x.shape[0], chunk):
            phase = 2.0j * np.pi * (x[i:i + chunk] @ self.modes.T)
            out[i:i + chunk] = np.real(np.exp(phase) @ self.coeffs)
        return out


def make_torus_target(d: int, lengthscale: float, max_freq: int,
                      group: list[tuple[int, ...]], seed: int) -> TorusTarget:
    """Draw a G-invariant truncated KL sample on T^d.

    `group` is a list of coordinate permutations (see groups.py). The
    coefficient tensor is averaged over the corresponding permutations of the
    frequency axes, which projects the draw onto the invariant subspace.
    """
    rng = np.random.default_rng(seed)
    M = max_freq
    grid_shape = (2 * M + 1,) * d
    axes = [np.arange(-M, M + 1)] * d
    mesh = np.meshgrid(*axes, indexing="ij")
    modes = np.stack([m.ravel() for m in mesh], axis=-1)  # (n_modes, d)

    lambdas = matern_spectral_density(modes, lengthscale, d, NU)

    z = (rng.standard_normal(modes.shape[0])
         + 1j * rng.standard_normal(modes.shape[0])) / np.sqrt(2.0)
    c = np.sqrt(lambdas) * z

    # Hermitian symmetry c_{-m} = conj(c_m) so that f is real-valued.
    ct = c.reshape(grid_shape)
    ct = 0.5 * (ct + np.conj(np.flip(ct)))

    # Project onto the G-invariant subspace: average the coefficient tensor
    # over the group's permutations of the frequency axes. (Averaging over a
    # group equals averaging over its inverses, so the orientation of the
    # axis permutation is immaterial.)
    ct = np.mean([np.transpose(ct, axes=perm) for perm in group], axis=0)

    return TorusTarget(modes=modes, coeffs=ct.ravel(), lambdas=lambdas)


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

def _complex_sph_harm(m: int, l: int, azimuth: np.ndarray,
                      polar: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import sph_harm_y
        return sph_harm_y(l, m, polar, azimuth)
    except ImportError:  # older SciPy
        from scipy.special import sph_harm
        return sph_harm(m, l, azimuth, polar)


def real_sph_harm_basis(points: np.ndarray, max_degree: int) -> np.ndarray:
    """Real orthonormal spherical harmonic basis evaluated at unit vectors.

    Returns an (n_points, (L+1)^2) matrix; column order matches
    `sph_harm_degrees`.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    polar = np.arccos(np.clip(z, -1.0, 1.0))
    azimuth = np.arctan2(y, x)
    cols = []
    for l in range(max_degree + 1):
        cols.append(np.real(_complex_sph_harm(0, l, azimuth, polar)))
        for m in range(1, l + 1):
            ylm = _complex_sph_harm(m, l, azimuth, polar)
            phase = np.sqrt(2.0) * (-1.0) ** m
            cols.append(phase * np.real(ylm))
            cols.append(phase * np.imag(ylm))
    return np.stack(cols, axis=-1)


def sph_harm_degrees(max_degree: int) -> np.ndarray:
    """Degree l of each basis column, aligned with real_sph_harm_basis."""
    degs = []
    for l in range(max_degree + 1):
        degs.extend([l] * (2 * l + 1))
    return np.array(degs)


def sphere_quadrature(max_degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature exact for products of two band-L functions:
    Gauss-Legendre in cos(polar) x uniform in azimuth."""
    L = max_degree
    u, wu = leggauss(L + 1)                # exact up to degree 2L+1 in cos
    n_phi = 2 * L + 2                      # exact for harmonics up to 2L
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    U, PHI = np.meshgrid(u, phi, indexing="ij")
    st = np.sqrt(1.0 - U**2)
    pts = np.stack([st * np.cos(PHI), st * np.sin(PHI), U], axis=-1)
    pts = pts.reshape(-1, 3)
    w = np.repeat(wu, n_phi) * (2.0 * np.pi / n_phi)
    return pts, w


@dataclass
class SphereTarget:
    max_degree: int
    coeffs: np.ndarray       # ((L+1)^2,) real coefficients
    per_degree: np.ndarray   # a_l, l = 0..L

    @property
    def rkhs_norm(self) -> float:
        a = self.per_degree[sph_harm_degrees(self.max_degree)]
        return float(np.sqrt(np.sum(self.coeffs**2 / a)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return real_sph_harm_basis(np.atleast_2d(x), self.max_degree) @ self.coeffs


def make_sphere_target(max_degree: int, per_degree: np.ndarray,
                       group: list[np.ndarray], seed: int) -> SphereTarget:
    """Draw a G-invariant band-limited sample on S^2 with exact norm."""
    rng = np.random.default_rng(seed)
    degs = sph_harm_degrees(max_degree)
    c = np.sqrt(per_degree[degs]) * rng.standard_normal(degs.size)

    nodes, w = sphere_quadrature(max_degree)
    basis = real_sph_harm_basis(nodes, max_degree)

    # Average f over the group, evaluated on the quadrature nodes...
    vals = np.mean([real_sph_harm_basis(nodes @ g.T, max_degree) @ c
                    for g in group], axis=0)
    # ...then re-expand. Exact because the average is band-limited at L.
    c_inv = basis.T @ (w * vals)

    return SphereTarget(max_degree=max_degree, coeffs=c_inv,
                        per_degree=per_degree)
