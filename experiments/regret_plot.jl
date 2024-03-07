using Plots
using LaTeXStrings
using Statistics
using ParameterHandling
using Optim
using Random, Distributions, QuasiMonteCarlo # For generating spatial samples


function simple_regret(
    optimal_f::Number,
    reported_f::Number
)
    return optimal_f - reported_f
end

# This is achievable with broadcasting: simple_regret.(x, y_vec)
# However, it is not possible to broadcast the cumulative regret in the same way
# Defining a new dispatch for Vector{T} is necessary to have compatible simple_regret and cumulative_regret functions
function simple_regret(
    optimal_f::Number,
    reported_f::Vector{T} where {T<:Number}
)
    return optimal_f .- reported_f
end


function cumulative_regret(
    optimal_f::Number,
    reported_f::Vector{T} where {T<:Number}
)
    return cumsum(simple_regret.(optimal_f, reported_f), dims=1)
end


function cumulative_regret(
    optimal_f::Number,
    reported_f::Matrix{T} where {T<:Number}
)
    return cumsum(simple_regret.(optimal_f, reported_f), dims=2)
end


function get_approximate_maximum(
    f::Function,
    bounds::Vector{Tuple{Float64,Float64}};
    n_iterations::Int=1000,
    n_restarts::Int=16,
)
    # Multistart optimisation
    candidate_x = Vector{Vector{Float64}}(undef, n_restarts)
    candidate_y = Vector{Float64}(undef, n_restarts)

    lower_bounds = [b[1] for b in bounds]
    upper_bounds = [b[2] for b in bounds]
    start_points = [
        bounded.(x0, lower_bounds, upper_bounds)
        for x0 in eachcol(QuasiMonteCarlo.sample(n_restarts, lower_bounds, upper_bounds, LatinHypercubeSample()))
    ]

    Threads.@threads for i in 1:n_restarts
        x0_transformed, untransform = value_flatten(start_points[i])

        function objective(x_untransformed)
            return -f(x_untransformed)
        end

        # Maximise the acquisition function by minimising its negative
        result = optimize(
            objective ∘ untransform,
            # x_transformed -> only(Zygote.gradient(objective ∘ untransform, x_transformed)), # TODO: We might've fixed this so that our kernels can be differentiable
            x0_transformed,
            inplace=false,
        )

        candidate_x[i] = untransform(result.minimizer)
        candidate_y[i] = -result.minimum # Objective was to minimise the negative of the acquisition function, so flip the sign back
    end

    # Return the best x
    max_y, max_idx = findmax(candidate_y)
    return candidate_x[max_idx], max_y
end


function plot_with_ribbon(
    y::Matrix{T} where {T<:Number},
    trace_label::String,
    yaxis_label::String;
    output_filename::String="",
)
    μ = vec(mean(y, dims=1))
    σ² = vec(var(y, dims=1))

    figure = Plots.plot(
        1:length(μ),
        μ,
        ribbon=2 * sqrt.(σ²),
        fillalpha=0.1,
        label=trace_label,
        xlabel="Number of evaluations",
        ylabel=yaxis_label,
        xlims=(1, length(μ)),
    )
    if output_filename != ""
        savefig(output_filename)
    end
    return figure
end


function plot_with_ribbon!(
    p::Plots.Plot,
    y::Matrix{T} where {T<:Number},
    trace_label::String,
    yaxis_label::String;
    output_filename::String="",
)
    μ = vec(mean(y, dims=1))
    σ² = vec(var(y, dims=1))
    figure = Plots.plot!(
        p,
        1:length(μ),
        μ,
        ribbon=2 * sqrt.(σ²),
        fillalpha=0.1,
        label=trace_label,
        xlabel="Number of evaluations",
        ylabel=yaxis_label,
        xlims=(1, length(μ)),
    )
    if output_filename != ""
        savefig(output_filename)
    end
    return figure
end
