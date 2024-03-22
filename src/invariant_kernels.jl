using KernelFunctions
include("permutation_groups.jl")
include("onesided_transformedkernel.jl")


function invariantkernel(k::Kernel, G::Tuple{Vararg{PermutationGroupElement}})
    # Check that the inverse of each element is in the group
    for g in G
        if !(inv(g) in G)
            throw(ArgumentError("Inverse of $g is not in the group"))
        end
    end

    return 1 / length(G) * KernelSum(
        collect(
            OneSidedTransformedKernel(k, to_transform(σ))
            for σ in G
        )
    )
end

function approx_invariantkernel(k::Kernel, G::Tuple{Vararg{PermutationGroupElement}}, n::Int)
    return invariantkernel(k, random_subgroup(G, n))
end