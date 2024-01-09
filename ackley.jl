@doc raw"""
    ackley(x::Vector{Float64})

Compute the d-dimensional Ackley function at a point x ∈ R^d. 

The d-dimensional Ackley function is defined as:
```math
    f(x) = 20 + e - 20 * \exp \left( -0.2 \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2} \right) - \exp\left(\frac{1}{d} \sum_{i=1}^d \cos(2 * pi * x_i\right)
```
where i = 1, ..., d.
The global minimum is at x = 0, where the function value is 0.
"""
function ackley(x::Vector{Float64})
    n = length(x)
    return 20 + exp(1) - 20 * exp(-0.2 * sqrt(sum(x .^ 2) / n)) - exp(sum(cos.(2 * pi * x)) / n)
end
