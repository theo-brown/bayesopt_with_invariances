# Experiment 4.2
# Dimension:            6
# True invariance:      Permutation invariant
# Acquisition function: Upper Confidence Bound, β=2.5
# Reporting function:   Most recent point
# Kernels:              Standard, Fully permutation invariant, 2-block permutation invariant, 3-block permutation invariant

include("run_experiment.jl")

const d = 6
const G = permutation_group(d)
const β = 2.5

run_experiment(
    seed=42,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_3_mvr",
    n_iterations=512,
    n_repeats=1,
    acquisition_function=(gp, x) -> ucb(gp, x; beta=β),
    acquisition_function_label="UCB, β=$β",
    reporting_function=latest_point,
    reporting_function_label="Latest point",
    gp_builders=Dict([
        ("Random subgroup (N>2)", θ -> build_approx_perminvariantmatern52_gp(θ, permutation_group(d), 2)),
        ("Random subgroup (N>10)", θ -> build_approx_perminvariantmatern52_gp(θ, permutation_group(d), 10)),
        ("Random subgroup (N>25)", θ -> build_approx_perminvariantmatern52_gp(θ, permutation_group(d), 25)),
        ("Random subgroup (N>45)", θ -> build_approx_perminvariantmatern52_gp(θ, permutation_group(d), 45)),
        ("Random subgroup (N>180)", θ -> build_approx_perminvariantmatern52_gp(θ, permutation_group(d), 180)),
    ]),
    target_gp_builder=θ -> build_perminvariantmatern52_gp(θ, G),
    target_function_seed=5,
    target_function_n_points=512,
    θ=(
        l=0.12,
        σ_f=1.0,
        σ_n=0.1,
    ),
)
