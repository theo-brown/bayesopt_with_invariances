include("run_experiment.jl")

seed = 42
d = 6
bounds = [(0.0, 1.0) for _ in 1:d]
output_directory = "data/experiment_3_ucb"
n_iterations = 512
n_repeats = 16
β = 12.0
acquisition_function(gp, x) = ucb(gp, x; beta=β)
acquisition_function_label = "UCB, β=$β"
reporting_function = latest_point
reporting_function_label = "Latest point"
gp_builders = Dict([
    ("Standard", build_matern52_gp),
    ("Invariant", θ -> build_invariantmatern52_gp(θ, permutation_group(d))),
    ("Random subgroup (W=2)", θ -> build_approx_invariantmatern52_gp(θ, permutation_group(d), 2)),
    ("Random subgroup (W=3)", θ -> build_approx_invariantmatern52_gp(θ, permutation_group(d), 3)),
    ("Random subgroup (W=4)", θ -> build_approx_invariantmatern52_gp(θ, permutation_group(d), 4)),
    ("Random subgroup (W=5)", θ -> build_approx_invariantmatern52_gp(θ, permutation_group(d), 5)),
])
target_gp_builder = gp_builders["Invariant"]
target_function_seed = 1
target_function_n_points = 512
θ = (
    l=0.12,
    σ_f=1.0,
    σ_n=0.1,
)

println("Initial setup completed")
flush(stdout)

run_experiment(
    seed,
    bounds,
    output_directory,
    n_iterations,
    n_repeats,
    acquisition_function,
    acquisition_function_label,
    reporting_function,
    reporting_function_label,
    gp_builders,
    target_gp_builder,
    target_function_seed,
    target_function_n_points,
    θ,
)