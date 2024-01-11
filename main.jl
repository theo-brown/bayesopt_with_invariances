using ArgParse
using Plots
using Random
using Distributions
using HDF5

include("gp_utils.jl")
include("invariant_gps.jl")
include("acquisition_functions.jl")
include("plot_utils.jl")

include("ackley.jl")
include("kernel_objective_function.jl")


# Parse command line arguments
settings = ArgParseSettings()

supported_functions = ["ackley", "matern"]

@add_arg_table settings begin
    "function"
    help = "Function to maximise (supports: $supported_functions). Note that these functions are transformed so that if they were originally designed as minimisation tasks, they are multiplied by -1 to make them a maximisation task."
    arg_type = String
    default = "matern"
    range_tester = (x -> x in supported_functions)
    required = true

    "acqf"
    help = "Acquisition function to use (supports: ucb, mvr)"
    arg_type = String
    default = "mvr"
    range_tester = (x -> x in ["ucb", "mvr"])
    required = true

    "--seed"
    help = "Random seed"
    arg_type = Int
    default = 2024

    "--dimension"
    help = "Number of dimensions"
    arg_type = Int
    default = 2

    "--noise_variance"
    help = "Variance of Gaussian noise for noisy function value observations"
    arg_type = Float64
    default = 0.0

    "--n_steps"
    help = "Number of steps to run the optimisation for"
    arg_type = Int
    default = 256

    "--n_restarts"
    help = "Number of restarts for the acquisition function maximisation"
    arg_type = Int
    default = 64

    "--output"
    help = "Directory to save the plots to"
    arg_type = String
    default = "output"
end

args = parse_args(settings)

# Set the random seed
Random.seed!(args["seed"])

# Define the function to optimise
if args["function"] == "ackley"
    f = x -> -ackley(x)
    bounds = [(-5.0, 5.0) for _ in 1:args["dimension"]]
    f_maximum = 0.0
    f_maximiser = [0.0 for _ in 1:args["dimension"]]
elseif args["function"] == "matern"
    if args["dimension"] != 2
        error("Symmetric Matern52 function only supports dimension=2")
    end
    bounds = [(0.0, 1.0) for i in 1:2]
    θ_true = (
        l=0.2,
        σ_f=1.0,
        σ_n=0.1,
    )
    f = build_latent_function(build_permutationinvariantmatern52_gp, θ_true, 64, bounds, 22)

    # Find the maximum
    x0 = bounded.([0.79, 0.38], [b[1] for b in bounds], [b[2] for b in bounds]) # Start near the true maximum
    x0_transformed, untransform = value_flatten(x0)
    result = optimize(
        x_transformed -> -f(untransform(x_transformed)),
        x0_transformed,
        inplace=false,
    )
    f_maximum = -result.minimum
    f_maximiser = untransform(result.minimizer)
else
    error("Unsupported function: $args['function']")
end

# Define the noisy function observation protocol
if args["noise_variance"] > 0.0
    noise = Normal(0.0, sqrt(args["noise_variance"]))
    f_noisy = x -> f(x) + rand(noise)
else
    f_noisy = f
end

# Define the acquisition function
if args["acqf"] == "ucb"
    acqf = ucb
elseif args["acqf"] == "mvr"
    acqf = mvr
else
    error("Unsupported acquisition function: $args['acqf']")
end

# Define the GPs we're comparing
gp_builders = Dict(
    "standard" => build_matern52_gp,
    "permutation_invariant" => build_permutationinvariantmatern52_gp
)

# Make the output directory
mkdir(args["output"])

# Create the plot canvas
simple_regret_figure = plot(
    xlabel="Iteration",
    ylabel="Simple regret",
    legend=:outertopright,
    size=(800, 600),
)


# Plot the simple regret for each GP
for (label, builder) in gp_builders
    # Make an output directory for this GP
    mkdir(joinpath(args["output"], label))

    # Preallocate arrays for the samples
    observed_x = Vector{Vector{Float64}}(undef, 1 + args["n_steps"])
    observed_y = Vector{Float64}(undef, 1 + args["n_steps"])
    simple_regret = Vector{Float64}(undef, 1 + args["n_steps"])

    # Generate the initial observation
    observed_x[1] = [
        rand(Uniform(lower, upper))
        for (lower, upper) in bounds
    ]
    observed_y[1] = f_noisy(observed_x[1])
    simple_regret[1] = abs(f_maximum - observed_y[1])

    # Prior on the GP hyperparameters
    θ_0 = (
        σ_f=1.0,
        l=[1.0 for _ in 1:args["dimension"]],
        σ_n=0.1
    )

    # Build the GP
    gp = get_posterior_gp(builder, observed_x[1:1], observed_y[1:1], θ_0; optimise_hyperparameters=true)

    # Run the optimisation
    for t in 2:args["n_steps"]+1
        # Maximise the acquisition function
        x_next = maximise_acqf(gp, acqf, bounds, args["n_restarts"])
        observed_x[t] = x_next
        observed_y[t] = f_noisy(x_next)
        println("[$label] ($(t-1)/$(args["n_steps"])): ", observed_x[t], " -> ", observed_y[t])

        # Update the GP
        gp = get_posterior_gp(builder, observed_x[1:t], observed_y[1:t], θ_0; optimise_hyperparameters=true)

        # Compute the simple regret
        simple_regret[t] = abs(f_maximum - maximum(observed_y[1:t]))
        println("Regret: $(simple_regret[t])")

        # Plot the GP
        if args["dimension"] == 2
            plot_2d_gp_with_observations(gp, observed_x[1:t], bounds)
            savefig(joinpath(args["output"], label, "$t.png"))
        end
    end

    # Plot the simple regret
    Plots.plot!(
        simple_regret_figure,
        collect(0:args["n_steps"]),
        simple_regret,
        label=label,
    )

    # Save the data
    h5open(joinpath(args["output"], "data.h5"), "cw") do output_file
        create_group(output_file, label)
        output_file[label]["observed_x"] = stack(observed_x; dims=1)
        output_file[label]["observed_y"] = observed_y
        output_file[label]["simple_regret"] = simple_regret
    end
end

# Save the plot
savefig(simple_regret_figure, joinpath(args["output"], "simple_regret.png"))