using AbstractGPs # For the GP type
using ParameterHandling, Optim # For optimisation

include("acquisition.jl")

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
    function acqf(posterior_gp::AbstractGPs.AbstractGP, x::AbstractVector)
        return mean(posterior_gp(x))
    end

    return maximise_acqf(posterior_gp, acqf, bounds, 10)
end