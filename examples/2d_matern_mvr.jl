using ArgParse
using Random
using Plots, LaTeXStrings
using HDF5

include("../src/gp_utils.jl")
include("../src/objective_functions/kernel_objective_function.jl")
include("../src/bayesopt.jl")
include("../src/invariant_kernels.jl")
include("../src/groups/permutation_group.jl")


################
# 0. Arguments #
################
s = ArgParseSettings()
@add_arg_table s begin
    "--output_dir"
    arg_type = String
    default = "output"

    "--seed"
    arg_type = Int
    default = 42

    "--n_iterations"
    arg_type = Int
    default = 10

    "--n_repeats"
    arg_type = Int
    default = 2

    "--latent_function_seed"
    arg_type = Int
    default = 1
end
args = parse_args(s)

# Set seed
Random.seed!(args["seed"])

# Create output directory
mkdir(args["output_dir"])
mkdir(joinpath(args["output_dir"], "figures"))
mkdir(joinpath(args["output_dir"], "figures", "gp_visualisation"))


######################
# 1. Target function #
######################
# Target function is a draw from a 2D symmetric Matern-5/2 GP on [0, 1]^2
d = 2
bounds = [(0.0, 1.0) for i in 1:d]
θ = (
    l=0.2,
    σ_f=1.0,
    σ_n=0.1,
)

# Define a convenience function for building the GP
function build_2d_invariant_matern52_gp(θ)
    return build_permutationinvariantmatern52_gp(θ, d)
end

# Generate a latent function by evaluating a symmetric Matern-5/2 GP at a grid of points
# We use a fixed seed to get the same function every time
latent_function_n_points = 128
f = build_latent_function(
    build_2d_invariant_matern52_gp,
    θ,
    latent_function_n_points,
    bounds,
    args["latent_function_seed"]
)
# Define a noisy version of the function
f_noisy(x) = f(x) + randn() * θ.σ_n

# Evaluate the target function on a grid 
x_range = range(0.0, 1.0, length=100)
x_grid = [[x₁, x₂] for x₁ in x_range, x₂ in x_range]
y_grid = [f(x) for x in x_grid]

# Initial guess of maximiser
maximiser_guess = x_grid[argmax(y_grid)]

# Fine-tune with BFGS 
x0 = [
    bounded(maximiser_guess[1], bounds[1][1], bounds[1][2]),
    bounded(maximiser_guess[2], bounds[2][1], bounds[2][2]),
]
x0_flat, unflatten = value_flatten(x0)
result = optimize(x -> -f(unflatten(x)), x0_flat, LBFGS(), Optim.Options(iterations=1000))
x_opt = unflatten(Optim.minimizer(result))
y_opt = f(x_opt)

println("Maximiser: ", x_opt)
println("Max value: ", y_opt)

# Visualise
contourf(
    x_range,
    x_range,
    (x₁, x₂) -> f([x₁, x₂]),
    levels=32,
    color=:viridis,
    cbar=true,
    xlabel=L"$x_1$",
    ylabel=L"$x_2$",
    colorbar_title=L"$f(x)$"
)
plot!(
    [x_opt[1], x_opt[2]],
    [x_opt[2], x_opt[1]],
    marker=:star,
    markersize=10,
    color=:red,
    label="Maximiser",
    seriestype=:scatter
)
plot!(size=(600, 600), aspect_ratio=:equal, xlims=(0, 1), ylims=(0, 1))
savefig(joinpath(args["output_dir"], "figures", "target_function.pdf"))


############################
# 2. Bayesian optimisation #
############################
output_file = joinpath(args["output_dir"], "results.h5")

h5open(output_file, "w") do file
    # Save metadata
    attrs(file)["n_repeats"] = args["n_repeats"]
    attrs(file)["n_iterations"] = args["n_iterations"]
    attrs(file)["seed"] = args["seed"]
    attrs(file)["target_function"] = "Permutation invariant Matern-5/2 GP on [0, 1]^2"
    attrs(file)["latent_function_seed"] = args["latent_function_seed"]
    attrs(file)["latent_function_n_points"] = latent_function_n_points
    attrs(file)["acquisition_function"] = "MVR"

    for gp_builder in [build_2d_invariant_matern52_gp, build_matern52_gp]
        # Create a group for the GP builder
        if gp_builder == build_2d_invariant_matern52_gp
            gp_group = create_group(file, "permutation_invariant")
        else
            gp_group = create_group(file, "standard")
        end

        for i in 1:args["n_repeats"]
            println("# Repeat $i")

            # Set seed
            Random.seed!(args["seed"] + i)

            # Run BO
            observed_x, observed_y = run_bayesopt(
                f_noisy,
                bounds,
                args["n_iterations"],
                gp_builder,
                mvr,
                θ;
                optimise_hyperparameters=false
            )

            # Generate the reported values 
            # MVR's reporting rule is max posterior mean 
            reported_x = Matrix{Float64}(undef, args["n_iterations"], 2)
            reported_f = Vector{Float64}(undef, args["n_iterations"])
            for j in 1:args["n_iterations"]
                # Generate the GP posterior based on the observations to date
                gp_posterior = get_posterior_gp(
                    gp_builder,
                    eachrow(observed_x[1:j, :]), # Convert from Matrix to Vector
                    observed_y[1:j],
                    θ;
                    optimise_hyperparameters=false
                )

                # Function to minimise
                function target(x)
                    return -only(mean(gp_posterior([x]))) # Extra [] required to turn into vector of vectors, only() required to turn it back
                end
                # Start at our best guess
                maximiser_guess = observed_x[argmax(observed_y), :]
                # Bound it using ParameterHandling.jl
                x0 = [
                    bounded(maximiser_guess[1], bounds[1][1], bounds[1][2]),
                    bounded(maximiser_guess[2], bounds[2][1], bounds[2][2]),
                ]
                x0_flat, unflatten = value_flatten(x0)
                # Minimise the target
                result = optimize(target ∘ unflatten, x0_flat, inplace=false)

                # Store the reported value
                reported_x[j, :] = unflatten(result.minimizer)
                reported_f[j] = f(reported_x[j, :])
            end

            # Create a group for this repeat
            repeat_group = create_group(gp_group, string(i))
            write_dataset(repeat_group, "observed_x", observed_x)
            write_dataset(repeat_group, "observed_y", observed_y)
            write_dataset(repeat_group, "reported_x", reported_x)
            write_dataset(repeat_group, "reported_f", reported_f)
        end
    end
end


###################
# 3. Regret plots #
###################
println("Plotting simple regret...")
figure = plot(legend=:topright, xlabel="Iteration", ylabel="Simple regret")
h5open(output_file, "r") do file
    n_repeats = attrs(file)["n_repeats"]
    n_iterations = attrs(file)["n_iterations"]

    for group_name in ["permutation_invariant", "standard"]
        gp_group = file[group_name]

        simple_regret = zeros(n_repeats, n_iterations)
        for i in 1:n_repeats
            repeat_group = gp_group[string(i)]
            reported_x = read_dataset(repeat_group, "reported_x")
            reported_f = read_dataset(repeat_group, "reported_f")

            for j in 1:n_iterations
                simple_regret[i, j] = y_opt - reported_f[j]
            end
        end

        mean_regret = vec(mean(simple_regret, dims=1))
        var_regret = vec(var(simple_regret, dims=1))

        # Plot the simple regret with error bars
        plot!(
            1:n_iterations,
            mean_regret,
            ribbon=2 * sqrt.(var_regret),
            fillalpha=0.1,
            label=group_name == "permutation_invariant" ? L"$k_G$" : L"$k$",
            xlabel="Iteration",
            ylabel="Simple regret",
            legend=:bottomright,
            xlims=(1, n_iterations),
        )
    end
end
savefig(joinpath(args["output_dir"], "figures", "simple_regret.pdf"))


#######################
# 4. GP visualisation #
#######################
println("Visualising GP predictions...")

h5open(output_file, "r") do file
    n_repeats = attrs(file)["n_repeats"]
    n_iterations = attrs(file)["n_iterations"]
    for group_name in ["permutation_invariant", "standard"]
        gp_group = file[group_name]

        gp_builder = group_name == "permutation_invariant" ? build_2d_invariant_matern52_gp : build_matern52_gp

        for i in 1:n_repeats
            repeat_group = gp_group[string(i)]
            observed_x = read_dataset(repeat_group, "observed_x")
            observed_y = read_dataset(repeat_group, "observed_y")
            for j in 1:n_iterations
                println("($group_name) repeat [$i / $n_repeats] iteration [$j / $n_iterations]")

                x = eachrow(observed_x[1:j, :])
                y = observed_y[1:j]

                # Fit a GP to the observed data
                gp = get_posterior_gp(gp_builder, x, y, θ; optimise_hyperparameters=false)

                # Evaluate on a grid
                x_range = range(0.0, 1.0, length=100)
                x_grid = [[x₁, x₂] for x₁ in x_range, x₂ in x_range]
                gpx = gp([x_grid[i] for i in eachindex(x_grid)], 1e-6)
                μ = mean(gpx)
                σ² = var(gpx)

                function plot_with_observations(z, title)
                    figure = contourf(
                        x_range,
                        x_range,
                        reshape(z, length(x_range), length(x_range)),
                        levels=32,
                        color=:viridis,
                        cbar=false,
                    )
                    scatter!(
                        [xᵢ[1] for xᵢ in x],
                        [xᵢ[2] for xᵢ in x],
                        seriestype=:scatter,
                        color=:white,
                        label="Observations",
                        markersize=3,
                    )
                    scatter!(
                        [x[end][1]],
                        [x[end][2]],
                        marker=:star,
                        markersize=10,
                        color=:red,
                        label="Last observation"
                    )
                    plot!(
                        legend=false,
                        title=title,
                        xlabel=L"$x_1$",
                        ylabel=L"$x_2$",
                        size=(300, 300),
                        aspect_ratio=:equal,
                        xlims=(0, 1),
                        ylims=(0, 1),
                    )
                    return figure
                end

                # Plot the GP and acquisition functions
                μ_figure = plot_with_observations(μ, "Mean")
                σ²_figure = plot_with_observations(σ², "Variance")
                gp_figure = plot(μ_figure, σ²_figure, layout=(1, 2), size=(600, 300))
                savefig(
                    joinpath(
                        args["output_dir"],
                        "figures",
                        "gp_visualisation",
                        "$(group_name)_repeat_$(i)_iteration_$(j).png", # Save as PNG because it's easier to flick through
                    )
                )
            end
        end
    end
end
