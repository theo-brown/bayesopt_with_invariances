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
comparing the vanilla kernel against the orbit-averaged kernel. Regret curves
are averaged over **5 independently drawn targets ("worlds")** — for needle
variants the hidden needle placement is also re-drawn per world — **× 4 BO
repeats** per world (20 runs per kernel), with 5 random initial points and
observation noise σ = 0.05, so the comparison is not tied to one particular
target draw. Every world shares the experiment's group, candidate set,
kernels and budget, and each world's smooth target is rescaled to the
experiment's fixed reference norm `B_REF` (12, 19, 15, 12 and 6 for the five
experiments below), so all worlds are exactly equally "hard" in RKHS-norm
terms. The plotted target panel shows world 0.

| Name | Manifold | Group | \|G\| | Active modes | Budget |
|------|----------|-------|------|--------------|--------|
| `torus2_S2` | T² | permutations S₂ | 2 | 625 | 80 |
| `torus3_C3` | T³ | cyclic shifts C₃ | 3 | 2197 | 150 |
| `torus3_S3` | T³ | permutations S₃ | 6 | 2197 | 150 |
| `sphere_C5` | S² | z-rotations C₅ | 5 | 676 | 150 |
| `sphere_oct` | S² | octahedral rotations | 24 | 676 | 150 |

Each experiment also has a `*_needle` variant (same group, budget and smooth
component; see below).

## Planted needle variants

A caveat of the plain construction: the unnormalised orbit-averaged prior has
inflated variance on the group's fixed-point sets (orbit phases add coherently
there), so on the torus the drawn optima land exactly on the symmetry loci.
The `*_needle` variants remove this degeneracy by adding a G-symmetrised
truncated-spectrum kernel atom at a hidden *generic* orbit (rejection-sampled
at least ~a lengthscale away from every fixed-point set):

```
f = f_smooth + β · (1/|G|) Σ_g k_M(g x₀, ·),
```

scaled so the needle peak sits 0.5 above the smooth component's maximum. The
needle lives in the same truncated Fourier/harmonic frame, so exact RKHS
membership and the closed-form norm carry over unchanged (the norm is just
`Σ|c_m|²/λ_m` over the combined coefficients, cross terms included). The
needle location is re-drawn for every world, so the *total* norm of a needle
target varies slightly across worlds — the smooth component's norm is what is
held exactly fixed at `B_REF` — and the figure suptitle reports the mean total
norm over worlds. These
are the strongest rebuttal to the recoverability criticism: the global
optimum is a localised feature at a hidden generic orbit, invisible to the
surrogate until sampled within a lengthscale of one of its |G| copies. The
needle orbit is marked with × on the target plots.

Note the active-mode counts: 625–2197 versus the original 64–512 atoms, and —
unlike atoms at sampleable locations — there is no set of input points whose
observation linearly determines the target with fewer samples than modes.

The regret panels overlay the theoretical MVR rate from Brown et al. (2024),
Theorem 1: the information gain satisfies `γ_T^G = Õ(T^{m/(2ν+m)}/|G|)` (m =
manifold dimension), giving simple regret `r_T = Õ(B |G|^{-1/2}
T^{-ν/(2ν+m)})`. Since constants and polylog factors are not specified by the
theory, the vanilla guide line is anchored to the vanilla curve one-third of
the way through the run, and the invariant guide is placed at exactly the
theoretical `|G|^{-1/2}` offset below it. Both kernels share the same slope
in T; the |G| separation is the theory's testable prediction (an upper
bound, so the empirical invariant curves may — and do — fall well below it).

### Non-asymptotic regret certificate

Because the targets' RKHS norm `B = ‖f‖_{H_k}` is *exactly* known, the demos
also plot a fully computable, non-asymptotic upper bound on simple regret. In
the noise-free RKHS setting, Cauchy–Schwarz in the RKHS gives
`|f(x) − μ_t(x)| ≤ B σ_t(x)` for every `x` — the posterior standard deviation
`σ_t` is precisely the power function, the worst-case interpolation error over
the unit ball of the RKHS. Hence the incumbent's simple regret satisfies

```
r_t ≤ 2 B max_x σ_t(x)
```

over the candidate set. The MVR loop already computes the posterior variance
at every candidate each iteration, so the certificate costs nothing extra;
each run's certificate trace is computed with its own world's exact norm `B`
(which varies slightly across worlds for needle targets), stored per run
(`*_cert` in the saved results) and drawn — averaged over runs — as a dotted
curve in the regret panels. This is only possible here because `B` is exactly
known — with
an estimated or bounded norm the certificate would be heuristic. Being a true
worst-case-over-the-RKHS-ball bound, it necessarily sits well above the
typical-case empirical regret; the point is that it decays and is *rigorous*,
holding for every run rather than on average.

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
python run_demo.py         # all experiments (~30–60 min)
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
