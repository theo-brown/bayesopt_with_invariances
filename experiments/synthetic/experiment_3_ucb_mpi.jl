# Experiment 3.2
# Dimension:            6
# True invariance:      Permutation invariant
# Acquisition function: Upper Confidence Bound, β=2.0
# Reporting function:   Most recent point
# Kernels:              Standard, Fully permutation invariant, 2-block permutation invariant, 3-block permutation invariant

include("../../src/permutation_groups.jl")
include("run_experiment_mpi.jl")

const d = 6
const T₁ = to_transform.(permutation_group(d))
const T₂ = to_transform.(block_permutation_group(d, 2))
const T₃ = to_transform.(block_permutation_group(d, 3))
const β = 2.0

run_experiment(
    seed=42,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_3_ucb_mpi",
    n_iterations=256,
    n_repeats=32,
    acquisition_function=(gp, x) -> ucb(gp, x; beta=β),
    acquisition_function_label="UCB, β=$β",
    acquisition_function_restarts=10,
    reporting_function=latest_point,
    reporting_function_label="Latest point",
    gp_builders=Dict([
        ("Standard", build_gp),
        ("Fully permutation invariant", θ -> build_invariant_gp(θ, T₁)),
        ("2-block permutation invariant", θ -> build_invariant_gp(θ, T₂)),
        ("3-block permutation invariant", θ -> build_invariant_gp(θ, T₃))
    ]),
    # target_gp_builder=θ -> build_invariant_gp(θ, T₁),
    target_gp_builder=build_gp,
    target_function_seed=5,
    target_function_n_points=128,
    θ=(
        l=0.12,
        σ_f=1.0,
        σ_n=0.1,
    ),
)
