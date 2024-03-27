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
    θ::NamedTuple;
    n_restarts::Int=1,
    use_autograd::Bool=false,
    time_limit::Int=150,
    tolerance::Float=1e-4,
)
    # Create the output arrays
    d = length(input_bounds)
    observed_x = Matrix{Float64}(undef, d, n_steps)
    observed_y = Vector{Float64}(undef, n_steps)
    reported_x = Matrix{Float64}(undef, d, n_steps)
    reported_y = Vector{Float64}(undef, n_steps)

    # Initial sample
    observed_x[:, 1] = [
        rand(Uniform(lower, upper))
        for (lower, upper) in bounds
    ]
    observed_y[1] = f(observed_x[:, 1])
    gp = get_posterior_gp(
        gp_builder,
        ColVecs(observed_x[:, 1:1]),
        observed_y[1:1], # Use 1:1 to make sure it's a Vector{Float64} and not a scalar Float64
        θ,
    )

    for i in 1:n_steps-1
        # Update the GP
        @debug "Updating GP with $(i) observations"
        gp = get_posterior_gp(
            gp,
            ColVecs(observed_x[:, 1:i]),
            observed_y[1:i],
            θ,
        )

        # Generate the next observation
        @debug "Maximising acquisition function"
        x_next = maximise_acqf(gp, acquisition_function, bounds, n_restarts; use_autograd=use_autograd, time_limit=time_limit, tolerance=tolerance)
        observed_x[:, i+1] = x_next
        @debug "Evaluating function"
        observed_y[i+1] = f(x_next)

        # Report
        @debug "Reporting"
        reported_x[:, i+1] = reporting_function(gp, observed_x[:, 1:i+1], observed_y[1:i+1], bounds)
        reported_y[i+1] = f(reported_x[:, i+1])

        @info "Step $(i+1): $(reported_x[:, i+1]) -> $(reported_y[i+1])"
    end

    return observed_x, observed_y, reported_x, reported_y
end