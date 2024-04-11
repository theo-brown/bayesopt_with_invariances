# Experiment 1.1
# Dimension:            2 
# True invariance:      Permutation invariant
# Acquisition function: Maximum variance reduction
# Reporting function:   Maximum observed posterior mean
# Kernels:              Standard, Permutation invariant

include("run_experiment.jl")

const d = 2
const G = permutation_group(d)

run_experiment(
    seed=42,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_1_mvr",
    n_iterations=256,
    n_repeats=32,
    acquisition_function=mvr,
    acquisition_function_label="MVR",
    reporting_function=maximum_observed_posterior_mean,
    reporting_function_label="Maximum observed posterior mean",
    gp_builders=Dict([
        ("Standard", build_matern52_gp),
        ("Permutation invariant", θ -> build_perminvariantmatern52_gp(θ, G))
    ]),
    target_gp_builder=θ -> build_perminvariantmatern52_gp(θ, G),
    target_function_seed=43,
    target_function_n_points=128,
    θ=(
        l=0.12,
        σ_f=1.0,
        σ_n=0.1,
    ),
)
