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


@doc raw"""
    RandomSubgroupInvariantKernel(base_kernel, transformation_group, subgroup_size)

An random subgroup approximation to an invariant kernel.

The approximate invariant kernel is given by

```math  
k_G^W = \frac{1}{|W|} \sum_{\sigma \in W} k(\sigma(x), y)
```

where k is the base kernel and $W \subset G$ is a subgroup_size subset of the transformation group G.
W is chosen randomly at each kernel evaluation.
"""
struct RandomSubgroupInvariantKernel <: Kernel
    base_kernel::Kernel
    transformation_group::Vector{Function}
    subgroup_size::Integer
end

function (k::RandomSubgroupInvariantKernel)(x, y)
    G = k.transformation_group
    κ = k.base_kernel
    W_size = k.subgroup_size

    # Select a random subgroup of size W
    W = sample(G, W_size; replace=false)

    # Incremental sum for memory efficiency
    K = 0
    for σ in W
        K = K .+ κ(σ(x), y)
    end
    K = K ./ W_size
    return K
end
