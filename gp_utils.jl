using AbstractGPs # For general GP functionality
using ParameterHandling, Optim # For fitting GPs
using Random, Distributions # For generating spatial samples

"""
    maximise_mll(gp_builder_function, θ_0, x, y)

Find the hyperparameters θ that maximise the marginal log-likelihood of a GP given data (x, y).

Note that currently only scalar-output GPs are supported.
TODO: There's currently a bug in providing the gradient to the optimisation routine

# Arguments
- `gp_builder_function::Function`: A function that takes a hyperparameter vector θ and returns a GP.
- `θ_0::NamedTuple`: An initial guess for the hyperparameters.
- `x::Vector{Vector{Float64}}`: A vector of input points (each input point is a vector).
- `y::Vector{Float64}`: A vector of output points (each output point is a float).

# Returns
- `θ::NamedTuple`: The hyperparameters that maximise the marginal log-likelihood.
"""
function maximise_mll(gp_builder_function::Function, θ_0::NamedTuple, x::Vector{Vector{Float64}}, y::Vector{Float64})::NamedTuple
    θ_flat, unflatten = value_flatten(θ_0)

    # Objective: negative marginal log-likelihood
    function nmll(θ::NamedTuple)
        gp = gp_builder_function(θ)
        return -logpdf(gp(x, θ.σ_n^2 + 1e-6), y) # Add jitter
    end

    result = optimize(
        nmll ∘ unflatten, # Function to optimize
        # θ -> only(gradient(nmll ∘ unflatten, θ)), # Gradient of function to optimize #TODO this doesn't work
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
- `x_train::Vector{Vector{Float64}}`: A vector of input points (each input point is a vector).
- `y_train::Vector{Float64}`: A vector of output points (each output point is a float).
- `θ_0::NamedTuple`: An initial guess for the hyperparameters.
- `optimise_hyperparameters::Bool=true`: Whether to optimise the hyperparameters. If false, θ_0 is used for the hyperparameters.

# Returns
- `posterior_gp::AbstractGPs.AbstractGP`: The posterior GP.
"""
function get_posterior_gp(gp_builder_function::Function, x_train::Vector{Vector{Float64}}, y_train::Vector{Float64}, θ_0::NamedTuple; optimise_hyperparameters=true)::AbstractGPs.AbstractGP
    if optimise_hyperparameters
        θ_opt = maximise_mll(gp_builder_function, θ_0, x_train, y_train)
    else
        θ_opt = θ_0
    end
    gp = gp_builder_function(θ_opt)
    posterior_gp = posterior(gp(x_train, θ_opt.σ_n^2 + 1e-6), y_train)
    return posterior_gp
end
