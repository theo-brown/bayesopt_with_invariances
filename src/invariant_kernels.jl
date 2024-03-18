using KernelFunctions
include("permutation_groups.jl")

struct PartialTransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

(k::PartialTransformedKernel)(x, y) = k.kernel(k.transform(x), y)


function invariantkernel(k::Tk, G::Vector{PermutationGroupElement}) where {Tk<:Kernel}
    # Check that the inverse of each element is in the group
    for g in G
        if !(inv(g) in G)
            throw(ArgumentError("Inverse of $g is not in the group"))
        end
    end

    G_functions = Function[x -> x[σ.permutation] for σ in G]

    return 1 / length(G) * sum([PartialTransformedKernel(k, FunctionTransform(σᵢ)) for σᵢ in G_functions])
end

function quasiinvariantkernel(k::Tk, G::Vector{PermutationGroupElement}, w::Vector{Tw}) where {Tk<:Kernel,Tw<:Number}
    for (g, wᵢ) in zip(G, w)
        # Check that the inverse of each element is in the group
        if !(inv(g) in G)
            throw(ArgumentError("Inverse of $g is not in the group"))
        end
        # Check that the weights of element g and its inverse are the same
        if wᵢ != w[findfirst(x -> x == inv(g), G)]
            throw(ArgumentError("Weight of $g and its inverse are not the same"))
        end
    end
    if sum(w) != 1
        throw(ArgumentError("Weights do not sum to 1"))
    end

    G_functions = Function[x -> x[σ.permutation] for σ in G]

    return sum([wᵢ * PartialTransformedKernel(k, FunctionTransform(σᵢ)) for (σᵢ, wᵢ) in zip(G_functions, w)])
end

function softmax(x::Vector{Float64})
    exp_x = exp.(x)
    return exp_x / sum(exp_x)
end
