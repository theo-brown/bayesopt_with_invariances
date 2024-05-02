# Experiment 1.2
# Dimension:            2 
# True invariance:      Permutation invariant
# Acquisition function: Upper Confidence Bound, β=2.0
# Reporting function:   Most recent point
# Kernels:              Standard, Permutation invariant

include("../../src/permutation_groups.jl")
include("run_experiment.jl")

const d = 2
const T = to_transform.(permutation_group(d))
const β = 2.5

@info "Running on $(Threads.nthreads()) threads"

run_experiment(
    seed=42,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_1_ucb",
    n_iterations=512,
    n_repeats=32,
    acquisition_function=(gp, x) -> ucb(gp, x; beta=β),
    acquisition_function_label="UCB, β=$β",
    acquisition_function_restarts=10,
    reporting_function=latest_point,
    reporting_function_label="Latest point",
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
