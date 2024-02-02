@doc raw"""
    rosenbrock(x)

Compute the d-dimensional Rosenbrock function at a point x ∈ R^d.

The d-dimensional Rosenbrock function is defined as:
```math
    f(x) = \sum_{i=1}^{d-1} (100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
```
where i = 1, ..., d-1.
The global minimum is at x = (1, ..., 1), where the function value is 0.
"""
function rosenbrock(x::Vector{Float64})
    return sum(100 .* (x[2:end] .- x[1:end-1] .^ 2) .^ 2 .+ (1 .- x[1:end-1]) .^ 2)
end

rosenbrock_maximum = 0.0
rosenbrock_maximiser(d::Int) = ones(d)