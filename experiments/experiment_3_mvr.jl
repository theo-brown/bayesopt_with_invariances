# Experiment 3.1
# Dimension:            6
# True invariance:      Permutation invariant
# Acquisition function: Maximum variance reduction
# Reporting function:   Maximum posterior mean
# Kernels:              Standard, Fully permutation invariant, 2-block permutation invariant, 3-block permutation invariant

include("run_experiment.jl")

const d = 6
const G = permutation_group(d)

run_experiment(
    seed=42,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_3_mvr",
    n_iterations=512,
    n_repeats=1,
    acquisition_function=mvr,
    acquisition_function_label="MVR",
    reporting_function=maximum_observed_posterior_mean,
    reporting_function_label="Maximum observed posterior mean",
    gp_builders=Dict([
        ("Standard", build_matern52_gp),
        ("Fully permutation invariant", θ -> build_perminvariantmatern52_gp(θ, G)),
        ("2-block permutation invariant", θ -> build_perminvariantmatern52_gp(θ, block_permutation_group(d, 2))),
        ("3-block permutation invariant", θ -> build_perminvariantmatern52_gp(θ, block_permutation_group(d, 3))),]),
    target_gp_builder=θ -> build_perminvariantmatern52_gp(θ, G),
    target_function_seed=5,
    target_function_n_points=512,
    θ=(
        l=0.12,
        σ_f=1.0,
        σ_n=0.1,
    ),
)
