using AbstractGPs
using Optim, Zygote, ParameterHandling
using Random, Distributions, QuasiMonteCarlo

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
    # Expand the input point to a vector of vectors, to make it compatible with the GP
    x_vec = [x]

    # Compute the variance of the GP posterior at x
    # Add jitter to improve stability
    σ2 = var(posterior_gp(x_vec, 1e-6))

    # var is a Vector{Float64}, but we only want a single Float64
    return only(σ2)
end

"""
    ucb(posterior_gp, x)

Compute the upper confidence bound of a GP posterior at a point x.

# Arguments
- `posterior_gp::AbstractGPs.AbstractGP`: A GP posterior.
- `x::Vector{Float64}`: A single point at which to compute the upper confidence bound.

# Returns
- `Float64`: The upper confidence bound of the GP posterior at x.
"""
function ucb(posterior_gp::AbstractGPs.AbstractGP, x::Vector{Float64})::Float64
    # Expand the input point to a vector of vectors, to make it compatible with the GP
    x_vec = [x]

    # Compute the mean and variance of the GP posterior at x
    finite_gp = posterior_gp(x_vec, 1e-6)
    σ2 = only(var(finite_gp))
    μ = only(mean(finite_gp))

    if σ2 < 1e-6
        # If the variance is very small, return the mean. This prevents numerical underflow in the sqrt.
        return μ
    end

    return μ + 2 * sqrt(σ2)
end

"""
    maximise_acqf(posterior_gp, acqf, bounds, n_restarts)

Given the GP model, maximise the given acquisition function using multi-start optimisation
"""
# function maximise_acqf(posterior_gp::AbstractGPs.AbstractGP, acqf::Function, bounds::Vector{Tuple{Float64,Float64}}, n_restarts::Int)


#     lower_bounds = [b[1] for b in bounds]
#     upper_bounds = [b[2] for b in bounds]

#     # Multistart optimisation
#     start_points = QuasiMonteCarlo.sample(n_restarts, lower_bounds, upper_bounds, LatinHypercubeSample())
#     # Threads.@threads for k in 1:n_restarts
#     for k in 1:n_restarts
#         # Enforce bounds on elements of x0
#         x0 = [
#             bounded(start_points[:, k][i], lower_bounds[i], upper_bounds[i])
#             for i in eachindex(lower_bounds)
#         ]
#         # "_transformed" is values transformed into an unbounded space, "_untransformed" is the values in the original bounded space
#         x0_transformed, untransform = value_flatten(x0)


#         function objective(x_untransformed)
#             return -acqf(posterior_gp, x_untransformed)
#         end

#         # Maximise the acquisition function by minimising its negative
#         result = optimize(
#             objective ∘ untransform,
#             # x_transformed -> only(Zygote.gradient(objective ∘ untransform, x_transformed)), # Permutation invariant GP is not differentiable
#             x0_transformed,
#             inplace=false,
#             BFGS(
#                 alphaguess=Optim.LineSearches.InitialStatic(scaled=true),
#                 linesearch=Optim.LineSearches.BackTracking(),
#             ),
#         )

#         candidate_xs[k] = untransform(result.minimizer)
#         candidate_ys[k] = result.minimum
#     end

#     # println("candidate_xs = ", candidate_xs)
#     # println("candidate_ys = ", candidate_ys)

#     # Return the best x
#     _, i = findmax(candidate_ys)
#     return candidate_xs[i]
# end
function maximise_acqf(posterior_gp::AbstractGPs.AbstractGP, acqf::Function, bounds::Vector{Tuple{Float64,Float64}}, n_restarts::Int)
    candidate_x = nothing
    candidate_y = nothing

    # Multistart optimisation
    lower_bounds = [b[1] for b in bounds]
    upper_bounds = [b[2] for b in bounds]
    start_points = [
        bounded.(x0, lower_bounds, upper_bounds)
        for x0 in eachcol(QuasiMonteCarlo.sample(n_restarts, lower_bounds, upper_bounds, LatinHypercubeSample()))
    ]

    for k in 1:n_restarts
        x0_transformed, untransform = value_flatten(start_points[k])

        function objective(x_untransformed)
            return -acqf(posterior_gp, x_untransformed)
        end

        # Maximise the acquisition function by minimising its negative
        # TODO: This often fails due to Cholesky / not p.d.
        result = optimize(
            objective ∘ untransform,
            # x_transformed -> only(Zygote.gradient(objective ∘ untransform, x_transformed)),
            x0_transformed,
            BFGS(
                alphaguess=Optim.LineSearches.InitialStatic(scaled=true),
                linesearch=Optim.LineSearches.BackTracking(),
            ),
            inplace=false,
        )

        # If this is the first iteration, or if the result is better than the previous best, save it
        if isnothing(candidate_x) || result.minimum < candidate_y
            candidate_x = untransform(result.minimizer)
            candidate_y = result.minimum
        end

        #TODO: Add a stopping criterion
    end

    # Return the best x
    return candidate_x
end