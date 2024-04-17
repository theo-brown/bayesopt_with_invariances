# Experiment 3.2
# Dimension:            6
# True invariance:      Permutation invariant
# Acquisition function: Upper Confidence Bound, β=2.5
# Reporting function:   Most recent point
# Kernels:              Standard, Fully permutation invariant, 2-block permutation invariant, 3-block permutation invariant

include("run_experiment_mpi.jl")

const d = 6
const G = permutation_group(d)
const β = 2.5

run_experiment(
    seed=42,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_3_ucb_mpi",
    n_iterations=512,
    n_repeats=32,
    acquisition_function=(gp, x) -> ucb(gp, x; beta=β),
    acquisition_function_label="UCB, β=$β",
    reporting_function=latest_point,
    reporting_function_label="Latest point",
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