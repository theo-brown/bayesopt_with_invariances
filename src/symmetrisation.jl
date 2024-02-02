using KernelFunctions
using Combinatorics
import Base.+
import Base.*

"""
    +(f, g)

Create a new function `x ↦ f(x) + g(x)`.
"""
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...)

"""
    *(c, f)

Create a new function `x ↦ c f(x)`.
"""
*(c::Number, f::Function) = (x...) -> c * f(x...)

@doc raw"""
    symmetrise(f, G)

Symmetrise a function `f` with respect to a group `G`.
    
The group `G` is a collection of scale-preserving transformations (isometries) that act on the domain of `f`.
The symmetrised function is the average of `f` over the group `G`:
.. math::
    f_{\text{sym}}(x) = \frac{1}{|G|} \sum_{\sigma \in G} f(\sigma(x)),
        
where |G| is the number of elements in the group.
"""
function symmetrise(f::Function, G::Vector{Function})
    return 1 / length(G) * sum([f ∘ σ for σ ∈ G])
end

function symmetrise(k::KernelFunctions.Kernel, G::Vector{Function})
    # Need to convert σ ∈ G into a KernelFunctions.Transform object
    # Then we can use the ∘ operator to compose the kernel with the transformation
    return 1 / (length(G)^2) * sum([k ∘ FunctionTransform(σ) for σ in G])
end

"""
    permutation_group(d)

Generate the permutation group in d dimensions.
"""
function permutation_group(d::Int)::Vector{Function}
    indices = collect(1:d)
    return [x -> x[σ] for σ in permutations(indices)]
end

;
