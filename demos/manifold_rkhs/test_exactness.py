"""Numerical verification of the exactness claims behind the benchmark.

Run with: python test_exactness.py
"""

import numpy as np

from groups import (cyclic_permutation_group, cyclic_rotation_group,
                    octahedral_rotation_group, permutation_group,
                    apply_permutation, apply_rotation)
from kernels import (OrbitAveragedKernel, SphereMatern, WrappedMatern52,
                     matern_spectral_density, sphere_matern_coeffs)
from targets import (make_sphere_target, make_torus_target,
                     real_sph_harm_basis, sphere_quadrature)


def check(name, err, tol):
    status = "ok" if err < tol else "FAIL"
    print(f"[{status}] {name}: err = {err:.3e} (tol {tol:.0e})")
    assert err < tol, name


def test_poisson_summation():
    """Wrapped Matern == its Poisson-summation Fourier series, in 1D and 2D."""
    for d in (1, 2):
        l = 0.15
        kern = WrappedMatern52(d=d, lengthscale=l, n_images=3)
        rng = np.random.default_rng(0)
        x = rng.uniform(size=(40, d))
        y = rng.uniform(size=(40, d))
        k_wrapped = kern.pair(x, y)

        M = 80
        axes = [np.arange(-M, M + 1)] * d
        mesh = np.meshgrid(*axes, indexing="ij")
        modes = np.stack([m.ravel() for m in mesh], axis=-1)
        lam = matern_spectral_density(modes, l, d)
        phase = np.exp(2.0j * np.pi * ((x[:, None, :] - y[None, :, :])
                                       @ modes.T))
        k_spectral = np.real(phase @ lam)
        check(f"Poisson summation d={d}", np.max(np.abs(k_wrapped - k_spectral)),
              1e-7)


def test_torus_target_invariance_and_norm():
    d, l, M = 2, 0.12, 10
    group = permutation_group(d)
    f = make_torus_target(d, l, M, group, seed=1)
    rng = np.random.default_rng(2)
    x = rng.uniform(size=(200, d))
    inv_err = max(np.max(np.abs(f(apply_permutation(x, g)) - f(x)))
                  for g in group)
    check("torus target invariance (S_2)", inv_err, 1e-9)

    # Interpolation-based norm check in 1D: for a band-limited f, the RKHS
    # norm of the kernel interpolant on a fine grid converges to ||f||.
    d1, M1 = 1, 6
    f1 = make_torus_target(d1, l, M1, [(0,)], seed=3)
    kern = WrappedMatern52(d=d1, lengthscale=l, n_images=3)
    xg = np.linspace(0.0, 1.0, 400, endpoint=False)[:, None]
    K = kern.pair(xg, xg)
    fx = f1(xg)
    alpha = np.linalg.solve(K + 1e-10 * np.eye(len(xg)), fx)
    norm_interp = np.sqrt(alpha @ fx)
    rel = abs(norm_interp - f1.rkhs_norm) / f1.rkhs_norm
    print(f"       interp norm {norm_interp:.6f} vs exact {f1.rkhs_norm:.6f}")
    check("torus RKHS norm (interpolation)", rel, 1e-3)


def test_torus_cyclic_invariance():
    d, l, M = 3, 0.15, 4
    group = cyclic_permutation_group(d)
    f = make_torus_target(d, l, M, group, seed=4)
    rng = np.random.default_rng(5)
    x = rng.uniform(size=(200, d))
    inv_err = max(np.max(np.abs(f(apply_permutation(x, g)) - f(x)))
                  for g in group)
    check("torus target invariance (C_3)", inv_err, 1e-9)


def test_invariant_kernel_symmetry():
    d, l = 2, 0.12
    group = permutation_group(d)
    kern = OrbitAveragedKernel(WrappedMatern52(d=d, lengthscale=l),
                               group, "permutation")
    rng = np.random.default_rng(6)
    x, y = rng.uniform(size=(30, d)), rng.uniform(size=(30, d))
    base = kern.pair(x, y)
    err = max(np.max(np.abs(kern.pair(apply_permutation(x, g), y) - base))
              for g in group)
    check("orbit-averaged kernel invariance (torus)", err, 1e-10)


def test_sphere_quadrature_roundtrip():
    L = 20
    rng = np.random.default_rng(7)
    c = rng.standard_normal((L + 1) ** 2)
    nodes, w = sphere_quadrature(L)
    B = real_sph_harm_basis(nodes, L)
    c_back = B.T @ (w * (B @ c))
    check("sphere quadrature round-trip", np.max(np.abs(c_back - c)), 1e-10)


def test_sphere_target_invariance():
    L, kappa = 20, 0.3
    a = sphere_matern_coeffs(L, kappa)
    rng = np.random.default_rng(8)
    x = rng.standard_normal((200, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    for name, group in [("C_5", cyclic_rotation_group(5)),
                        ("octahedral", octahedral_rotation_group())]:
        f = make_sphere_target(L, a, group, seed=9)
        inv_err = max(np.max(np.abs(f(apply_rotation(x, g)) - f(x)))
                      for g in group)
        check(f"sphere target invariance ({name})", inv_err, 1e-9)


def test_sphere_kernel_psd_and_diag():
    L, kappa = 20, 0.3
    kern = SphereMatern(max_degree=L, kappa=kappa)
    rng = np.random.default_rng(10)
    x = rng.standard_normal((150, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    K = kern.pair(x, x)
    check("sphere kernel k(x,x) = 1",
          np.max(np.abs(kern.elementwise(x, x) - 1.0)), 1e-10)
    check("sphere kernel PSD", max(0.0, -np.min(np.linalg.eigvalsh(K))), 1e-9)

    group = octahedral_rotation_group()
    kg = OrbitAveragedKernel(kern, group, "rotation")
    base = kg.pair(x[:20], x[:20])
    err = max(np.max(np.abs(kg.pair(apply_rotation(x[:20], g), x[:20]) - base))
              for g in group)
    check("orbit-averaged kernel invariance (sphere)", err, 1e-10)


def test_sphere_norm_consistency():
    """Interpolation-based check of the sphere target norm."""
    L, kappa = 12, 0.4
    a = sphere_matern_coeffs(L, kappa)
    f = make_sphere_target(L, a, [np.eye(3)], seed=11)
    kern = SphereMatern(max_degree=L, kappa=kappa)
    nodes, _ = sphere_quadrature(2 * L)  # dense-ish, well-conditioned set
    K = kern.pair(nodes, nodes)
    fx = f(nodes)
    alpha = np.linalg.lstsq(K + 1e-10 * np.eye(len(nodes)), fx, rcond=None)[0]
    norm_interp = np.sqrt(max(alpha @ fx, 0.0))
    rel = abs(norm_interp - f.rkhs_norm) / f.rkhs_norm
    print(f"       interp norm {norm_interp:.6f} vs exact {f.rkhs_norm:.6f}")
    check("sphere RKHS norm (interpolation)", rel, 1e-2)


def test_needle_targets():
    """Needle-augmented targets: still exactly invariant, real-valued, and
    peaked on the hidden orbit."""
    import run_demo as rd

    for name in ["torus2_S2_needle", "torus3_S3_needle", "sphere_oct_needle"]:
        exp = rd.build_experiment(name)
        f, group = exp["target"], exp["group"]
        rng = np.random.default_rng(12)
        if exp["manifold"] == "torus":
            x = rng.uniform(size=(200, exp["candidates"].shape[1]))
            inv_err = max(np.max(np.abs(f(apply_permutation(x, g)) - f(x)))
                          for g in group)
        else:
            x = rng.standard_normal((200, 3))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            inv_err = max(np.max(np.abs(f(apply_rotation(x, g)) - f(x)))
                          for g in group)
        check(f"needle target invariance ({name})", inv_err, 1e-8)

        # The global max over the candidate set should be within a
        # lengthscale of the needle orbit, and above the smooth max.
        cands = exp["candidates"]
        x_best = cands[np.argmax(f(cands))]
        orbit = exp["needle_orbit"]
        if exp["manifold"] == "torus":
            diff = x_best[None, :] - orbit
            dist = np.min(np.sqrt(np.sum((diff - np.round(diff))**2, axis=1)))
            tol = 0.15
        else:
            dist = np.min(np.arccos(np.clip(orbit @ x_best, -1.0, 1.0)))
            tol = 0.3
        check(f"needle optimum on hidden orbit ({name})", dist, tol)


if __name__ == "__main__":
    test_poisson_summation()
    test_torus_target_invariance_and_norm()
    test_torus_cyclic_invariance()
    test_invariant_kernel_symmetry()
    test_sphere_quadrature_roundtrip()
    test_sphere_target_invariance()
    test_sphere_kernel_psd_and_diag()
    test_sphere_norm_consistency()
    test_needle_targets()
    print("\nAll exactness checks passed.")
