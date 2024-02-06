using Random
include("acquisition.jl")
include("gp_utils.jl")


function run_bayesopt(f, input_bounds, n_steps, gp_builder, acquisition_function; θ=nothing)
    # Create the output arrays
    d = length(input_bounds)
    observed_x = Matrix{Float64}(undef, n_steps, d)
    observed_y = Vector{Float64}(undef, n_steps)

    # Initial GP hyperparameters
    if isnothing(θ)
        θ_0 = (
            σ_f=1.0,
            l=[1.0 for _ in 1:d],
            σ_n=0.1
        )
    else
        θ_0 = θ
    end

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
            eachrow(observed_x[1:i, :]), # Use eachrow to turn into an AbstractVector, which is required by AbstractGPs
            observed_y[1:i],
            θ_0;
            optimise_hyperparameters=isnothing(θ) # If θ is provided, don't optimise the hyperparameters
        )

        # Generate the next observation
        x_next = maximise_acqf(gp, acquisition_function, bounds, 256)
        observed_x[i+1, :] = x_next
        observed_y[i+1] = f(x_next)
        println("[$i/$n_steps]: ", observed_x[i], " -> ", observed_y[i])
    end

    return observed_x, observed_y
end