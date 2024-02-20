using KernelFunctions

@doc raw"""
    GroupInvariantKernel(base_kernel, transformation_group)

A kernel that is invariant to the action of a transformation group.

The symmetrised kernel is given by

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
    return 1 / length(G) * sum([κ(σ(x), y) for σ in G])
end
