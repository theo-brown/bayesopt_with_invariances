@doc raw"""
    forrester(x::Float64)

Compute the 1-dimensional Forrester function at a point x ∈ R.

The 1-dimensional Forrester function is defined as:
```math
    f(x) = (6x - 2)^2 \sin(12x - 4)
```
where x ∈ [0, 1].
The global minimum is at x = 0.78, where the function value is -6.02.
"""
function forrester(x::Float64)
    return (6 * x - 2)^2 * sin(12 * x - 4)
end