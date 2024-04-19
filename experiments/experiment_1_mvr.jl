# Experiment 1.1
# Dimension:            2 
# True invariance:      Permutation invariant
# Acquisition function: Maximum variance reduction
# Reporting function:   Maximum posterior mean
# Kernels:              Standard, Permutation invariant

include("../src/permutation_groups.jl")
include("run_experiment.jl")

const d = 2
const T = to_transform.(permutation_group(d))

run_experiment(
    seed=0,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_1_mvr",
    n_iterations=256,
    n_repeats=32,
    acquisition_function=mvr,
    acquisition_function_label="MVR",
    reporting_function=maximum_posterior_mean,
    reporting_function_label="Maximum posterior mean",
    gp_builders=Dict([
        ("Standard", build_gp),
        ("Permutation invariant", θ -> build_invariant_gp(θ, T))
    ]),
    target_gp_builder=θ -> build_invariant_gp(θ, T),
    target_function_seed=43,
    target_function_n_points=128,
    θ=(
        l=0.12,
        σ_f=1.0,
        σ_n=0.1,
    ),
)
