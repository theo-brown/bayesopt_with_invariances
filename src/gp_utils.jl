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
    build_matern52_gp(θ::NamedTuple)

Build a Matern 5/2 GP with the given hyperparameters.

# Arguments
- `θ::NamedTuple`: A named tuple containing the hyperparameters σ_f, l, and σ_n.
"""
function build_matern52_gp(θ::NamedTuple)::AbstractGPs.AbstractGP
    kernel = θ.σ_f^2 * with_lengthscale(Matern52Kernel(), θ.l)
    return GP(kernel)
end


"""
    build_perminvariantmatern52_gp(θ::NamedTuple, G::Tuple{Vararg{PermutationGroupElement}})

Build a GP with the given hyperparameters that is invariant under the action of the permutations in G.

# Arguments
- `θ::NamedTuple`: A named tuple containing the hyperparameters σ_f, l, and σ_n.
- `G::Tuple{Vararg{PermutationGroupElement}}`: A collection of permutations.
"""
function build_perminvariantmatern52_gp(θ::NamedTuple, G::Tuple{Vararg{PermutationGroupElement}})
    base_kernel = θ.σ_f^2 * with_lengthscale(Matern52Kernel(), θ.l)
    kernel = invariantkernel(base_kernel, G)
    return GP(kernel)
end


"""
    build_approx_perminvariantmatern52_gp(θ::NamedTuple, G::Tuple{Vararg{PermutationGroupElement}}, n::Int)

Build a GP with the given hyperparameters that is a random subgroup approximation to the kernel invariant to G.

# Arguments
- `θ::NamedTuple`: A named tuple containing the hyperparameters σ_f, l, and σ_n.
- `G::Tuple{Vararg{PermutationGroupElement}}`: A collection of permutations.
- `n::Int`: The size of the random subgroup.
"""
function build_approx_perminvariantmatern52_gp(θ::NamedTuple, G::Tuple{Vararg{PermutationGroupElement}}, n::Int)
    base_kernel = θ.σ_f^2 * with_lengthscale(Matern52Kernel(), θ.l)
    subgroup = random_subgroup(G, n)
    kernel = invariantkernel(base_kernel, subgroup)
    return GP(kernel)
end
