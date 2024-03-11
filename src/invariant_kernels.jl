using KernelFunctions
using StatsBase


@doc raw"""
    GroupInvariantKernel(base_kernel, transformation_group)

A kernel that is invariant to the action of a transformation group.

The invariant kernel is given by

```math  
k_G = \frac{1}{|G|} \sum_{\sigma \in G} k(\sigma(x), y)
```

where k is the base kernel and G is the transformation group.
"""
struct GroupInvariantKernel <: Kernel
    base_kernel::Kernel
    transformation_group::Vector{Function}
end

function (k::GroupInvariantKernel)(x, y)
    G = k.transformation_group
    κ = k.base_kernel
    # Incremental sum for memory efficiency
    K = 0
    for σ in G
        K = K .+ κ(σ(x), y)
    end
    K = K ./ length(G)
    return K
end

