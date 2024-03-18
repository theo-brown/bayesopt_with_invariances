include("run_experiment.jl")

seed = 42
d = 3
bounds = [(0.0, 1.0) for _ in 1:d]
output_directory = "data/experiment_2_ucb"
n_iterations = 512
n_repeats = 16
β = 12.0
acquisition_function(gp, x) = ucb(gp, x; beta=β)
acquisition_function_label = "UCB, β=$β"
reporting_function = latest_point
reporting_function_label = "Latest point"
gp_builders = Dict([
    ("Standard", build_matern52_gp),
    ("Cyclic invariant", θ -> build_perminvariantmatern52_gp(θ, cyclic_group(d)))
])
target_gp_builder = gp_builders["Cyclic invariant"]
target_function_seed = 20
target_function_n_points = 128
θ = (
    l=0.12,
    σ_f=1.0,
    σ_n=0.1,
)

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