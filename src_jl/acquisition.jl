using AbstractGPs # For general GP functionality
using Optim, Zygote, ParameterHandling # For optimisation
using Random, Distributions, QuasiMonteCarlo, StatsBase # For generating spatial samples


"""
    mvr(posterior_gp, x)

Compute the uncertainty of a GP posterior at point x.

# Arguments
- `posterior_gp::AbstractGPs.AbstractGP`: A GP posterior.
- `x::Vector{Float64}`: A single point at which to compute the uncertainty.

# Returns
- `Float64`: The uncertainty of the GP posterior at x.
"""
function mvr(posterior_gp::AbstractGPs.AbstractGP, x::AbstractVector)
    return var(posterior_gp(x))
end

"""
    ucb(posterior_gp, x, β)

Compute the upper confidence bound of a GP posterior at x.

# Arguments
- `posterior_gp::AbstractGPs.AbstractGP`: A GP posterior.
- `x::Vector{Float64}`: A single point at which to compute the upper confidence bound.
- `beta`::Float64: The hyperparameter that determines the explore/exploite tradeoff
# Returns
- `Float64`: The upper confidence bound of the GP posterior at x.
"""
function ucb(posterior_gp::AbstractGPs.AbstractGP, x::AbstractVector; beta::Float64=2.0)
    finite_gp = posterior_gp(x)
    return mean(finite_gp) .+ beta .* sqrt.(var(finite_gp))
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
    n_initial_samples::Int=1024,
    eta::Float64=2.0,
    time_limit::Int=150,
    acqf_tolerance::Float64=1e-8,
    acqf_x_tolerance::Float64=1e-8,
)
    function objective(x)
        # Optim.jl only supports scalars
        return -only(acqf(posterior_gp, [x]))
    end

    function debug_callback(state::Optim.OptimizationState)
        @debug "- step $(state.iteration): acqf = $(state.value)"
        return false
    end

    lower_bounds = [b[1] for b in bounds]
    upper_bounds = [b[2] for b in bounds]
    d = length(bounds)

    if n_restarts == 1
        # Select a point uniformly at random
        start_points = rand.(Uniform(lower, upper) for (lower, upper) in bounds)
    else
        # This initialisation heuristic is based on the one in BoTorch
        # 1. Select a large batch of points using QMC sampling
        # Note: LatinHypercubeSample and SobolSample are type unstable at the time of writing
        candidate_start_points = QuasiMonteCarlo.sample(n_initial_samples, lower_bounds, upper_bounds, QuasiMonteCarlo.HaltonSample())

        # 2. Compute the acquisition function at each point
        candidate_acqf_values = acqf(posterior_gp, ColVecs(candidate_start_points))

        # 3. Sample from a multinomial distribution with probabilities proportional to the acquisition function values
        zero_mask = candidate_acqf_values .≈ 0
        μ = mean(candidate_acqf_values)
        σ = std(candidate_acqf_values)
        normalised_candidate_acqf_values = (candidate_acqf_values .- μ) ./ σ
        weights = exp.(eta .* normalised_candidate_acqf_values)
        weights[zero_mask] .= 0
        indices = sample(1:n_initial_samples, Weights(weights), n_restarts, replace=false)
        start_points = candidate_start_points[:, indices]
    end

    bounded_start_points = bounded.(start_points, lower_bounds, upper_bounds)

    candidate_x = Matrix{Float64}(undef, d, n_restarts)
    candidate_y = Vector{Float64}(undef, n_restarts)

    # TODO: we can definitely batch this, which would be wayy faster
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
