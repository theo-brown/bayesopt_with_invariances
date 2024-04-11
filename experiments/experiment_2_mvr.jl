# Experiment 2.1
# Dimension:            3
# True invariance:      Cyclic invariant
# Acquisition function: Maximum variance reduction
# Reporting function:   Maximum observed posterior mean
# Kernels:              Standard, Cyclic invariant

include("run_experiment.jl")

const d = 3
const G = cyclic_group(d)

run_experiment(
    seed=42,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_2_mvr",
    n_iterations=512,
    n_repeats=32,
    acquisition_function=mvr,
    acquisition_function_label="MVR",
    reporting_function=maximum_observed_posterior_mean,
    reporting_function_label="Maximum observed posterior mean",
    gp_builders=Dict([
        ("Standard", build_matern52_gp),
        ("Cyclic invariant", θ -> build_perminvariantmatern52_gp(θ, G))
    ]),
    target_gp_builder=θ -> build_perminvariantmatern52_gp(θ, G),
    target_function_seed=20,
    target_function_n_points=128,
    θ=(
        l=0.12,
        σ_f=1.0,
        σ_n=0.1,
    ),
)
