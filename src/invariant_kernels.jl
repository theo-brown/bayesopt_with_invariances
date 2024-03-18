using KernelFunctions
include("permutation_groups.jl")

struct OneSidedTransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

(k::OneSidedTransformedKernel)(x, y) = k.kernel(k.transform(x), y)

function invariantkernel(k::Tk, G::Vector{PermutationGroupElement}) where {Tk<:Kernel}
    # Check that the inverse of each element is in the group
    for g in G
        if !(inv(g) in G)
            throw(ArgumentError("Inverse of $g is not in the group"))
        end
    end

    G_functions = Function[x -> x[σ.permutation] for σ in G]
    return 1 / length(G) * sum([OneSidedTransformedKernel(k, FunctionTransform(σᵢ)) for σᵢ in G_functions])
end

function approx_invariantkernel(k::Tk, G::Vector{PermutationGroupElement}, n::Int) where {Tk<:Kernel}
    return invariantkernel(k, random_subgroup(G, n))
end