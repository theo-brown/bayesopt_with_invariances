using KernelFunctions
include("permutation_groups.jl")

struct OneSidedTransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

(k::OneSidedTransformedKernel)(x, y) = k.kernel(k.transform(x), y)

function invariantkernel(k::Tk, G::NTuple{N,PermutationGroupElement}) where {Tk<:Kernel,N}
    # Check that the inverse of each element is in the group
    for g in G
        if !(inv(g) in G)
            throw(ArgumentError("Inverse of $g is not in the group"))
        end
    end

    return 1 / length(G) * KernelSum([OneSidedTransformedKernel(k, FunctionTransform(x -> σᵢ(x))) for σᵢ in G])
end

function approx_invariantkernel(k::Tk, G::NTuple{N,PermutationGroupElement}, n::Int) where {Tk<:Kernel,N}
    return invariantkernel(k, random_subgroup(G, n))
end