using ArgParse
using Random
using Plots
using LaTeXStrings
using HDF5
using KernelFunctions # For Transform type 

include("../../src/gp_utils.jl")
include("../../src/synthetic_objective.jl")
include("../../src/bayesopt.jl")
include("../../src/reporting.jl")
include("render.jl")
include("regret_plot.jl")

# Setup
Random.seed!(0)
output_directory = "data/dihedral_progress"
mkdir(output_directory)
output_file = joinpath(output_directory, "results.h5")

# Define the target function
function dihedral_group(n::Int)
    r_k = [[cos(2π * k / n) -sin(2π * k / n); sin(2π * k / n) cos(2π * k / n)] for k in 0:n-1]
    s_k = [[cos(2π * k / n) sin(2π * k / n); sin(2π * k / n) -cos(2π * k / n)] for k in 0:n-1]
    return [r_k; s_k]
end

G = dihedral_group(5)
T = Tuple(LinearTransform(σ) for σ in G)
θ = (
    l=0.2,
    σ_f=1.0,
    σ_n=0.1,
)
bounds = [(-1.0, 1.0), (-1.0, 1.0)]
target_function_n_points = 64
target_function_seed = 11
gp_builders = Dict([
    "Standard" => build_gp,
    "Invariant" => (θ) -> build_invariant_gp(θ, T),
])

# Define the target function
@info "Generating target function..."
f = build_synthetic_objective(
    gp_builders["Invariant"],
    θ,
    target_function_n_points,
    bounds,
    target_function_seed,
)
f_noisy(x) = f(x) + randn() * θ.σ_n

for (label, gp_builder) in gp_builders
    @info "Running BO with $label kernel..."

    h5open(output_file, isfile(output_file) ? "r+" : "w") do file
        create_group(file, label)
    end

    # Run BO
    observed_x, observed_y, reported_x, reported_y = run_bayesopt(
        f_noisy,
        bounds,
        512,
        gp_builder,
        (gp, x) -> ucb(gp, x; beta=2.0),
        latest_point,
        θ;
        n_restarts=10,
    )

    # Save data to file
    h5open(output_file, "r+") do file
        file["$label/observed_x"] = observed_x
        file["$label/observed_y"] = observed_y
        file["$label/observed_f"] = f(observed_x)

        file["$label/reported_x"] = reported_x
        file["$label/reported_y"] = reported_y
        file["$label/reported_f"] = f(reported_x)
    end
end

