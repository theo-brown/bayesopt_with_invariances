using AbstractGPs # For the GP type
using ParameterHandling, Optim # For optimisation


"""
    latest_point(posterior_gp, observed_x, observed_y, bounds)

Return the latest observed point.
"""
function latest_point(posterior_gp::AbstractGPs.AbstractGP, observed_x::Matrix{Float64}, observed_y::Vector{Float64}, bounds::Vector{Tuple{Float64,Float64}})
    return observed_x[:, end]
end


""" 
    maximum_observed_posterior_mean(posterior_gp, observed_x, observed_y, bounds)

Return the observed point that has the highest posterior mean.
"""
function maximum_observed_posterior_mean(posterior_gp::AbstractGPs.AbstractGP, observed_x::Matrix{Float64}, observed_y::Vector{Float64}, bounds::Vector{Tuple{Float64,Float64}})
    # Compute the posterior mean at each observed_x 
    # Use eachrow to turn Matrix into an AbstractVector, which is required by AbstractGPs
    μ = mean(posterior_gp(ColVecs(observed_x)))
    # Return the point with the maximum posterior mean
    return observed_x[:, argmax(μ)]
end


""" 
    maximum_posterior_mean(posterior_gp, observed_x, observed_y, bounds)

Return the point in the entire domain that has the highest posterior mean.
"""
function maximum_posterior_mean(posterior_gp::AbstractGPs.AbstractGP, observed_x::Matrix{Float64}, observed_y::Vector{Float64}, bounds::Vector{Tuple{Float64,Float64}})
    # Function to minimise
    function target(x)
        return -only(mean(posterior_gp([x])))
    end

    # Start at our best guess
    maximiser_guess = observed_x[:, argmax(observed_y)]
    # Bound it using ParameterHandling.jl
    x0 = [
        bounded(maximiser_guess[i], bounds[i][1], bounds[i][2])
        for i in eachindex(bounds)
    ]
    x0_flat, unflatten = value_flatten(x0)
    # Minimise the target
    result = optimize(target ∘ unflatten, x0_flat)
    return unflatten(result.minimizer)
end