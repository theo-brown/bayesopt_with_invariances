using AbstractGPs # For general GP functionality
using Optim, Zygote, ParameterHandling # For optimisation
using Random, Distributions, QuasiMonteCarlo # For generating spatial samples


"""
    mvr(posterior_gp, x)

Compute the uncertainty of a GP posterior at a point x.

# Arguments
- `posterior_gp::AbstractGPs.AbstractGP`: A GP posterior.
- `x::Vector{Float64}`: A single point at which to compute the uncertainty.

# Returns
- `Float64`: The uncertainty of the GP posterior at x.
"""
function mvr(posterior_gp::AbstractGPs.AbstractGP, x::Vector{Float64})::Float64
    # Compute the variance of the GP posterior at x
    # Note that posterior_gp expects a vector of inputs, so we need to wrap x in a vector
    σ² = var(posterior_gp([x]))

    # var is a Vector{Float64}, but we only want a single Float64
    return only(σ²)
end

"""
    ucb(posterior_gp, x, β)

Compute the upper confidence bound of a GP posterior at a point x.

# Arguments
- `posterior_gp::AbstractGPs.AbstractGP`: A GP posterior.
- `x::Vector{Float64}`: A single point at which to compute the upper confidence bound.
- `beta`::Float64: The hyperparameter that determines the explore/exploite tradeoff
# Returns
- `Float64`: The upper confidence bound of the GP posterior at x.
"""
function ucb(posterior_gp::AbstractGPs.AbstractGP, x::Vector{Float64}; beta::Float64=2.0)::Float64
    # Compute the mean and variance of the GP posterior at x
    # Note that posterior_gp expects a vector of inputs, so we need to wrap x in a vector
    finite_gp = posterior_gp([x])
    σ² = only(var(finite_gp))
    μ = only(mean(finite_gp))

    if σ² < 1e-6
        # If the variance is very small, return the mean. This prevents numerical underflow in the sqrt.
        return μ
    end

    return μ + beta * sqrt(σ²)
end


"""
    maximise_acqf(posterior_gp, acqf, bounds, n_restarts)

Given the GP model, maximise the given acquisition function using multi-start optimisation
"""
function maximise_acqf(
    posterior_gp::AbstractGPs.AbstractGP,
    acqf::Function,
    bounds::Vector{Tuple{Float64,Float64}},
    n_restarts::Int;
    time_limit::Int=150,
    acqf_tolerance::Float64=1e-8,
    acqf_x_tolerance::Float64=1e-8,
)
    function objective(x)
        return -acqf(posterior_gp, x)
    end

    function debug_callback(state::Optim.OptimizationState)
        @debug "- step $(state.iteration): acqf = $(state.value)"
        return false
    end

    lower_bounds = [b[1] for b in bounds]
    upper_bounds = [b[2] for b in bounds]
    d = length(bounds)

    start_points = Matrix{Float64}(undef, d, n_restarts)
    if n_restarts == 1
        # Select a point uniformly at random
        start_points .= rand.(Uniform(lower, upper) for (lower, upper) in bounds)
    else
        # Select a batch of points using QMC sampling
        # Note: LatinHypercubeSample and SobolSample are type unstable at the time of writing
        start_points .= QuasiMonteCarlo.sample(n_restarts, lower_bounds, upper_bounds, QuasiMonteCarlo.HaltonSample())
    end
    bounded_start_points = bounded.(start_points, lower_bounds, upper_bounds)

    candidate_x = Matrix{Float64}(undef, d, n_restarts)
    candidate_y = Vector{Float64}(undef, n_restarts)

    Threads.@threads for i in 1:n_restarts
        #for i in 1:n_restarts
        x0_transformed, untransform = value_flatten(bounded_start_points[:, i])

        # Maximise the acquisition function by minimising its negative
        result = optimize(
            objective ∘ untransform,
            # x_transformed -> only(Zygote.gradient(objective ∘ untransform, x_transformed)),
            x0_transformed,
            LBFGS(
                alphaguess=Optim.LineSearches.InitialStatic(scaled=true),
                linesearch=Optim.LineSearches.BackTracking(),
            ),
            Optim.Options(
                show_every=10,
                callback=debug_callback,
                time_limit=time_limit,
                f_tol=acqf_tolerance,
                x_tol=acqf_x_tolerance,
            ),
            # inplace=false,
        )

        candidate_x[:, i] = untransform(result.minimizer)
        candidate_y[i] = -result.minimum # Objective was to minimise the negative of the acquisition function, so flip the sign back
    end

    # Return the best x
    return candidate_x[:, argmax(candidate_y)]
end
