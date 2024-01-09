using ArgParse
using Plots
using Random
using Distributions

include("ackley.jl")
include("acquisition_functions.jl")
include("gp_utils.jl")
include("invariant_gps.jl")


# Parse command line arguments
settings = ArgParseSettings()

@add_arg_table settings begin
    "function"
    help = "Function to maximise (supports: ackley). Note that the ackley function is multiplied by -1, so that it is a maximisation task."
    arg_type = String
    default = "ackley"
    range_tester = (x -> x in ["ackley"])
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
    help = "Noise variance"
    arg_type = Float64
    default = 1e-3

    "--n_steps"
    help = "Number of steps to run the optimisation for"
    arg_type = Int
    default = 256

    "--n_restarts"
    help = "Number of restarts for the acquisition function maximisation"
    arg_type = Int
    default = 64

    "--output"
    help = "Filename to save the plot to"
    arg_type = String
    default = "plot.png"
end

args = parse_args(settings)

# Set the random seed
Random.seed!(args["seed"])

# Define the function to optimise
noise = Normal(0, sqrt(args["noise_variance"]))
if args["function"] == "ackley"
    f = x -> -ackley(x)
    bounds = [(-5.0, 5.0) for _ in 1:args["dimension"]]
    maximum = 0.0
else
    error("Unsupported function: $args['function']")
end
f_noisy(x) = f(x) + rand(noise)

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
    "Standard" => build_matern52_gp,
    "Permutation invariant" => build_permutationinvariantmatern52_gp
)

# Create the plot canvas
figure = plot(
    xlabel="Iteration",
    ylabel="Simple regret",
    legend=:outertopright,
    size=(800, 600),
)

# Plot the simple regret for each GP
for (label, builder) in gp_builders
    # Preallocate arrays for the samples
    observed_x = Vector{Vector{Float64}}(undef, 1 + args["n_steps"])
    observed_y = Vector{Float64}(undef, 1 + args["n_steps"])
    simple_regret = Vector{Float64}(undef, 1 + args["n_steps"])

    # Generate the initial observation
    observed_x[1] = [
        rand(Uniform(lower, upper))
        for (lower, upper) in bounds
    ] #TODO: tidy up
    observed_y[1] = f_noisy(observed_x[1])
    simple_regret[1] = abs(observed_y[1])

    # Prior on the GP hyperparameters
    θ_0 = (
        σ_f=1.0,
        l=[1.0 for _ in 1:args["dimension"]],
        σ_n=0.1
    )

    # Build the GP
    gp = get_posterior_gp(builder, observed_x[1:1], observed_y[1:1], θ_0; optimise_hyperparameters=true)

    # Run the optimisation
    for t in 1:args["n_steps"]
        # Maximise the acquisition function
        x_next = maximise_acqf(gp, acqf, bounds, args["n_restarts"])

        # Evaluate the function at the new point
        y_next = f_noisy(x_next)

        println("Queried x=$(x_next), got y=$(y_next) ")

        # Update the GP
        observed_x[t+1] = x_next
        observed_y[t+1] = y_next
        gp = get_posterior_gp(builder, observed_x[1:1+t], observed_y[1:1+t], θ_0; optimise_hyperparameters=true)

        # Report the best observation so far
        reported_y, i = findmax(observed_y[1:1+t])

        # Compute the simple regret
        simple_regret[t+1] = abs(maximum - reported_y)
        println("[$label] r($t) = $(simple_regret[t+1])")
    end

    # Plot the simple regret
    plot!(
        figure,
        collect(1:args["n_steps"]+1),
        simple_regret,
        label=label,
    )
end

# Save the plot
savefig(figure, args["output"])
