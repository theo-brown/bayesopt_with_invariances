include("run_experiment.jl")

seed = 42
d = 2
bounds = [(0.0, 1.0) for _ in 1:d]
output_directory = "data/experiment_1"
n_iterations = 32
n_repeats = 16
β = 12.0
acquisition_function(gp, x) = ucb(gp, x; beta=β)
acquisition_function_label = "UCB, β=$β"
reporting_function = latest_point
reporting_function_label = "Latest point"
vanilla_gp_builder = build_matern52_gp
invariant_gp_builder(θ) = build_invariantmatern52_gp(θ, permutation_group(d))
invariant_gp_label = "Permutation invariant"
θ = (
    l=0.12,
    σ_f=1.0,
    σ_n=0.1,
)
latent_function_seed = 18
latent_function_n_points = 128

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
    vanilla_gp_builder,
    invariant_gp_builder,
    invariant_gp_label,
    θ,
    latent_function_seed,
    latent_function_n_points,
)