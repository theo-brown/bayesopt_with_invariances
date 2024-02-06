using AbstractGPs # For general GP functionality
using ParameterHandling, Optim # For fitting GPs
using Random, Distributions # For generating spatial samples

include("symmetrisation.jl")

"""
    maximise_mll(gp_builder_function, θ_0, x, y)

Find the hyperparameters θ that maximise the marginal log-likelihood of a GP with the given kernel, given data (x, y).

Note that currently only scalar-output GPs are supported.

# Arguments
- `gp_builder_function::Function`: A function that takes a hyperparameter vector θ and returns a GP.
- `θ_0::NamedTuple`: An initial guess for the hyperparameters.
- `x::AbstractVector`: A vector of N d-dimensional input points.
- `y::AbstractVector`: A vector of N 1-dimensional output points.

# Returns
- `θ::NamedTuple`: The hyperparameters that maximise the marginal log-likelihood.
"""
function maximise_mll(gp_builder_function::Function, θ_0::NamedTuple, x::AbstractVector, y::AbstractVector)::NamedTuple
    # Flatten the hyperparameters
    # See ParameterHandling.jl docs for more info
    θ_flat, unflatten = value_flatten(θ_0)

    # Objective: negative marginal log-likelihood
    function nmll(θ::NamedTuple)
        gp = gp_builder_function(θ)
        return -logpdf(gp(x, θ.σ_n^2 + 1e-6), y) # Add jitter to ensure p.d.
    end

    result = optimize(
        nmll ∘ unflatten, # Function to optimize
        θ_flat, # Initial guess
        BFGS(
            alphaguess=Optim.LineSearches.InitialStatic(scaled=true), #TODO Need to look up what this does
            linesearch=Optim.LineSearches.BackTracking(), #TODO Need to look up what this does
        ),
        inplace=false, # TODO need to look up what this does
    )
    return unflatten(result.minimizer)
end

"""
    get_posterior_gp(gp_builder_function, x_train, y_train, θ_0; optimise_hyperparameters=true)

Get the posterior GP given the data (x_train, y_train).

# Arguments
- `gp_builder_function::Function`: A function that takes a hyperparameter vector θ and returns a GP.
- `x_train::Vector{Vector{Float64}}`: A vector of N d-dimensional input points.
- `y_train::Vector{Float64}`: A vector of N 1-dimensional output points.
- `θ_0::NamedTuple`: An initial guess for the hyperparameters.
- `optimise_hyperparameters::Bool=true`: Whether to optimise the hyperparameters. If false, θ_0 is used for the hyperparameters.

# Returns
- `posterior_gp::AbstractGPs.AbstractGP`: The posterior GP.
"""
function get_posterior_gp(gp_builder_function::Function, x_train::AbstractVector, y_train::AbstractVector, θ_0::NamedTuple; optimise_hyperparameters=true)::AbstractGPs.AbstractGP
    if optimise_hyperparameters
        θ_opt = maximise_mll(gp_builder_function, θ_0, x_train, y_train)
    else
        θ_opt = θ_0
    end
    gp = gp_builder_function(θ_opt)
    posterior_gp = posterior(gp(x_train, θ_opt.σ_n^2 + 1e-6), y_train)
    return posterior_gp
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
    build_permutationinvariantmatern52_gp(θ::NamedTuple, d::Int)

Build a permutation-invariant GP with the given hyperparameters.

# Arguments
- `θ::NamedTuple`: A named tuple containing the hyperparameters σ_f, l, and σ_n.
- `d::Int`: The dimensionality of the input space.
"""
function build_permutationinvariantmatern52_gp(θ::NamedTuple, d::Int)::AbstractGPs.AbstractGP
    base_kernel = θ.σ_f^2 * with_lengthscale(Matern52Kernel(), θ.l)
    kernel = SymmetrisedKernel(base_kernel, permutation_group(d))
    return GP(kernel)
end