using AbstractGPs # For general GP functionality
using ParameterHandling
using Random, Distributions

"""
    build_latent_function(gp_builder_function, θ, D, n_points, seed)

Create a persistent function that is generated by a one-off random draw from a GP with the specified architecture and hyperparameters.

# Arguments
- `gp_builder_function::Function`: A function that takes a hyperparameter vector θ and returns a GP.
- `θ::NamedTuple`: The hyperparameters of the GP.
- `n_points::UInt`: The number of points to use to generate the latent function. Higher values will give more complex functions, at the cost of increased computational time.
- `bounds::Vector{Tuple{Float64, Float64}}`: The bounds used to generate the sample points. bounds should be a vector of length D, where D is the dimensionality of the input space. Each element of bounds should be a tuple of the form (lower_bound, upper_bound).
- `seed::UInt`: The seed for the random number generator.

# Returns
- `f::Function`: A function that maps from R^D to R.
"""
function build_latent_function(gp_builder_function::Function, θ::NamedTuple, n_points::Int, bounds::Vector{Tuple{Float64,Float64}}, seed::Int; jitter=0.0)::Function

    # Create a GP with the specified architecture and hyperparameters
    base_gp = gp_builder_function(θ)

    # Create a custom RNG that is specific to this scope
    rng = MersenneTwister(seed)

    # Generate the sample points
    distribution = [Uniform(lower, upper) for (lower, upper) in bounds]
    x = [rand.(rng, distribution) for _ in 1:n_points]

    # Evaluate the GP at the points 
    finite_gp = base_gp(x, jitter)

    # Observe values of a random sample from the GP evaluated at the sample locations
    # We do this to ensure that the resulting function can be represented by the GP
    y = rand(rng, finite_gp)

    # Condition the GP on the values at those points
    true_gp = posterior(finite_gp, y)

    # Define the latent function
    function f(x::AbstractMatrix)::Vector{Float64}
        return mean(true_gp(ColVecs(x), jitter))
    end
    f(x::Vector{Float64}) = only(mean(true_gp([x], jitter)))
    return f
end