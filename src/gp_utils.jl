using AbstractGPs # For general GP functionality
using ParameterHandling
using Random, Distributions # For generating spatial samples

include("invariant_kernels.jl")
include("permutation_groups.jl")

"""
    get_posterior_gp(gp_builder_function, x_train, y_train, θ)

Get the posterior GP given the data (x_train, y_train).

# Arguments
- `gp_builder_function::Function`: A function that takes a hyperparameter vector θ and returns a GP.
- `x_train::Vector{Vector{Float64}}`: A vector of N d-dimensional input points.
- `y_train::Vector{Float64}`: A vector of N 1-dimensional output points.
- `θ::NamedTuple`: GP hyperparameters.

# Returns
- `posterior_gp::AbstractGPs.AbstractGP`: The posterior GP.
"""
function get_posterior_gp(gp_builder_function::Function, x_train::AbstractVector, y_train::AbstractVector, θ::NamedTuple)
    return get_posterior_gp(gp_builder_function(θ), x_train, y_train, θ)
end

"""
    get_posterior_gp(gp, x_train, y_train)

Get the posterior GP given the data (x_train, y_train).

Can be used for sequential updates, by passing the posterior GP from the previous iteration as the GP argument.

# Arguments
- `gp::AbstractGPs.AbstractGP`: A GP prior.
- `x_train::Vector{Vector{Float64}}`: A vector of N d-dimensional input points.
- `y_train::Vector{Float64}`: A vector of N 1-dimensional output points.

# Returns
- `posterior_gp::AbstractGPs.AbstractGP`: The posterior GP.
"""
function get_posterior_gp(gp::AbstractGPs.AbstractGP, x_train::AbstractVector, y_train::AbstractVector, θ::NamedTuple)
    return posterior(gp(x_train, θ.σ_n^2 + 1e-6), y_train)
end

"""
    build_gp(θ::NamedTuple)

Build a Matern 5/2 GP with the given hyperparameters.

# Arguments
- `θ::NamedTuple`: A named tuple containing the hyperparameters σ_f, l, and σ_n.
"""
function build_gp(θ::NamedTuple)::AbstractGPs.AbstractGP
    kernel = θ.σ_f^2 * with_lengthscale(Matern52Kernel(), θ.l)
    return GP(kernel)
end


"""
    build_invariant_gp(θ::NamedTuple, T::Tuple{Vararg{Transform}})

Build a Matern 5/2 GP with the given hyperparameters that is invariant under the action of transformations T.

# Arguments
- `θ::NamedTuple`: A named tuple containing the hyperparameters σ_f, l, and σ_n.
- `T::Tuple{Vararg{Transform}}`: A collection of transformations
"""
function build_invariant_gp(θ::NamedTuple, T::Tuple{Vararg{Transform}})
    base_kernel = θ.σ_f^2 * with_lengthscale(Matern52Kernel(), θ.l)
    if applicable(length, θ.l)
        # ARD case
        kernel = invariant_kernel(base_kernel, T)
    else
        # Isotropic case
        kernel = isotropic_invariant_kernel(base_kernel, T)
    end
    return GP(kernel)
end
