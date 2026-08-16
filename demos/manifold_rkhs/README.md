# Harder synthetic RKHS benchmarks on the torus and sphere

Demos of a synthetic-objective construction that answers the criticism of the
original synthetic experiments (raised by Bardou et al., 2025): there, the
target was the posterior mean of a GP fitted to `n = 64–512` sampled points,
i.e. an explicit linear combination of `n` kernel atoms — so a learner can in
principle recover the target exactly once those few points are sampled.

Here the targets are still **exactly** elements of the RKHS with **exactly
known norm**, but they have far more active degrees of freedom than any BO
budget, so no small set of clever samples identifies them.

## Construction

**Torus `T^d = [0,1)^d`.** The kernel is the *wrapped* Matérn-5/2,
`k_per(x,y) = Σ_{m∈Z^d} k(‖x−y+m‖)`, evaluated by a truncated image sum that
is exact to machine precision at the lengthscales used. This is the
Riemannian Matérn kernel of Borovitskiy et al. (2020) on the flat torus. By
the **Poisson summation formula** its Fourier coefficients are exactly the
`R^d` Matérn spectral density at integer frequencies, `λ_m = S(m)`, so for a
target built as a truncated Karhunen–Loève draw

```
f(x) = Re Σ_{|m|∞ ≤ M} c_m e^{2πi m·x},   c_m = √λ_m z_m,   z_m ~ CN(0,1),
```

membership in the RKHS is exact and the norm is closed-form:
`‖f‖² = Σ |c_m|²/λ_m`. Invariance is imposed by averaging the coefficient
tensor over the group's action on frequency vectors — exactly the projection
onto the invariant subspace, i.e. a truncated draw from the GP with the
orbit-averaged kernel. For invariant `f`, `‖f‖_{H_kG} = ‖f‖_{H_k}` (Kondor
2008; Brown et al. 2024, App. A).

**Sphere `S^2`.** The truncated-spectrum Riemannian Matérn kernel
`k(x,y) = Σ_{ℓ≤L} a_ℓ (2ℓ+1)/(4π) P_ℓ(x·y)` with
`a_ℓ ∝ (2ν/κ² + ℓ(ℓ+1))^{−(ν+1)}` is *defined* to be the benchmark kernel, so
its RKHS is exactly the degree-≤L band. Targets are band-limited draws
`c_{ℓj} = √a_ℓ z_{ℓj}` projected onto the invariant subspace by averaging over
the group and re-expanding via a Gauss–Legendre × uniform quadrature that is
exact for band-limited functions. Norms `‖f‖² = Σ c²_{ℓj}/a_ℓ` are exact.

The invariant kernel is the plain single-sided orbit average
`k_G(x,y) = |G|⁻¹ Σ_g k(gx, y)` (equal to the double average of Brown et al.
2024, Eq. 5, for isometry groups). **No renormalisation, max-kernels, or
other tricks are used.**

## Experiments

Each experiment runs MVR (query = argmax posterior variance on a large
candidate set; incumbent = argmax posterior mean) with known hyperparameters,
comparing the vanilla kernel against the orbit-averaged kernel. 10 repeats,
5 random initial points, observation noise σ = 0.05.

| Name | Manifold | Group | \|G\| | Active modes | Budget |
|------|----------|-------|------|--------------|--------|
| `torus2_S2` | T² | permutations S₂ | 2 | 625 | 80 |
| `torus3_C3` | T³ | cyclic shifts C₃ | 3 | 2197 | 150 |
| `torus3_S3` | T³ | permutations S₃ | 6 | 2197 | 150 |
| `sphere_C5` | S² | z-rotations C₅ | 5 | 676 | 80 |
| `sphere_oct` | S² | octahedral rotations | 24 | 676 | 80 |

Note the active-mode counts: 625–2197 versus the original 64–512 atoms, and —
unlike atoms at sampleable locations — there is no set of input points whose
observation linearly determines the target with fewer samples than modes.

## Files

- `kernels.py` — wrapped torus Matérn-5/2, truncated sphere Matérn, plain
  orbit averaging
- `targets.py` — invariant KL-draw targets with exact norms
- `groups.py` — permutation groups (torus) and finite rotation groups (sphere)
- `mvr.py` — candidate-set MVR loop
- `run_demo.py` — runs all experiments and writes `plots/` and `results/`
- `test_exactness.py` — numerical verification of every exactness claim
  (Poisson summation identity, invariance, quadrature exactness, closed-form
  norms vs. interpolation-based norms, PSD-ness)

## Run

```bash
pip install numpy scipy matplotlib
python test_exactness.py   # verify the construction (~30 s)
python run_demo.py         # all experiments (~10–20 min)
python run_demo.py sphere_oct   # or a single one
```

## References

- Brown, Cioba, Bogunovic. *Sample-efficient Bayesian optimisation using
  known invariances.* NeurIPS 2024.
- Borovitskiy, Terenin, Mostowsky, Deisenroth. *Matérn Gaussian processes on
  Riemannian manifolds.* NeurIPS 2020.
- Kondor. *Group theoretical methods in machine learning.* PhD thesis, 2008.
- Bardou, Gonon, Ahadinia, Thiran. *Symmetry-aware Bayesian optimization via
  max kernels.* arXiv:2509.25051, 2025 (the criticism this construction
  addresses).
