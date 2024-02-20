using Random # For generation of the initial point
include("acquisition.jl")
include("gp_utils.jl")


function run_bayesopt(
    f::Function,
    input_bounds::Vector{Tuple{Float64,Float64}},
    n_steps::Int,
    gp_builder::Function,
    acquisition_function::Function,
    reporting_function::Function,
    θ_0::NamedTuple;
    optimise_hyperparameters::Bool=true,
    n_restarts::Int=256,
)
    # Create the output arrays
    d = length(input_bounds)
    observed_x = Matrix{Float64}(undef, n_steps, d)
    observed_y = Vector{Float64}(undef, n_steps)
    reported_x = Matrix{Float64}(undef, n_steps, d)
    reported_y = Vector{Float64}(undef, n_steps)

    # Initial sample
    observed_x[1, :] = [
        rand(Uniform(lower, upper))
        for (lower, upper) in bounds
    ]
    observed_y[1] = f(observed_x[1, :])

    for i in 1:n_steps-1
        # Update the GP
        gp = get_posterior_gp(
            gp_builder,
            eachrow(observed_x[1:i, :]), # Use eachrow to turn Matrix into an AbstractVector, which is required by AbstractGPs
            observed_y[1:i],
            θ_0;
            optimise_hyperparameters=optimise_hyperparameters
        )

        # Generate the next observation
        x_next = maximise_acqf(gp, acquisition_function, bounds, n_restarts)
        observed_x[i+1, :] = x_next
        observed_y[i+1] = f(x_next)

        # Report
        reported_x[i+1, :] = reporting_function(gp, observed_x[1:i+1, :], observed_y[1:i+1], bounds)
        reported_y[i+1] = f(reported_x[i+1, :])
    end

    return observed_x, observed_y, reported_x, reported_y
end