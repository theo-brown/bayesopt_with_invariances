include("run_experiment.jl")

seed = 42
d = 6
bounds = [(0.0, 1.0) for _ in 1:d]
output_directory = "data/experiment_3_mvr"
n_iterations = 512
n_repeats = 16
acquisition_function = mvr
acquisition_function_label = "MVR"
reporting_function = maximum_observed_posterior_mean
reporting_function_label = "Maximum observed posterior mean"
gp_builders = Dict([
    ("Standard", build_matern52_gp),
    ("Fully permutation invariant", θ -> build_perminvariantmatern52_gp(θ, permutation_group(d))),
    ("2-block permutation invariant", θ -> build_perminvariantmatern52_gp(θ, block_permutation_group(d, 2))),
    ("3-block permutation invariant", θ -> build_perminvariantmatern52_gp(θ, block_permutation_group(d, 3))),
])
target_gp_builder = gp_builders["Fully permutation invariant"]
target_function_seed = 1
target_function_n_points = 512
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