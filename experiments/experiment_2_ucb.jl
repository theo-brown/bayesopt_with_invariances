# Experiment 2.2
# Dimension:            3
# True invariance:      Cyclic invariant
# Acquisition function: Upper Confidence Bound, β=2.5
# Reporting function:   Most recent point
# Kernels:              Standard, Cyclic invariant

include("run_experiment.jl")

const d = 3
const G = cyclic_group(d)
const β = 2.5

run_experiment(
    seed=42,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_2_ucb",
    n_iterations=512,
    n_repeats=5,
    acquisition_function=(gp, x) -> ucb(gp, x; beta=β),
    acquisition_function_label="UCB, β=$β",
    reporting_function=latest_point,
    reporting_function_label="Latest point",
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
