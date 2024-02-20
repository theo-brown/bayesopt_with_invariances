using Plots
using LaTeXStrings
using Statistics
using ParameterHandling
using Optim


function simple_regret(
    optimal_f::Number,
    reported_f::Number
)
    return optimal_f - reported_f
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
    n_iterations::Int=1000
)
    # Start in the middle of the domain 
    maximiser_guess = [
        (lower + upper) / 2
        for (lower, upper) in bounds
    ]
    x0 = [
        bounded(maximiser_guess[i], bounds[i][1], bounds[i][2])
        for i in eachindex(bounds)
    ]
    x0_flat, unflatten = value_flatten(x0)
    result = optimize(x -> -f(unflatten(x)), x0_flat, LBFGS(), Optim.Options(iterations=n_iterations))
    x_opt = unflatten(Optim.minimizer(result))
    f_opt = f(x_opt)
    return x_opt, f_opt
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
