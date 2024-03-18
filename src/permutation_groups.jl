using Combinatorics
using StatsBase

apply_permutation(x::AbstractVector, perm::Vector{Int}) = x[perm]

struct PermutationGroupElement
    is_identity::Bool
    permutation::Vector{Int}
    inverse_permutation::Vector{Int}
end

function PermutationGroupElement(permutation::Vector{Int})
    inverse_permutation = sortperm(permutation)
    is_identity = permutation == collect(1:length(permutation))

    return PermutationGroupElement(is_identity, permutation, inverse_permutation)
end

inv(g::PermutationGroupElement) = PermutationGroupElement(g.is_identity, g.inverse_permutation, g.permutation)

is_identity(g::PermutationGroupElement) = g.is_identity

Base.:(==)(g1::PermutationGroupElement, g2::PermutationGroupElement) = (g1.permutation == g2.permutation)

Base.:(∘)(g1::PermutationGroupElement, g2::PermutationGroupElement) = PermutationGroupElement(g1.permutation[g2.permutation])

function (g::PermutationGroupElement)(x::T) where {T<:AbstractVector}
    return apply_permutation(x, g.permutation)
end

function Base.show(io::IO, g::PermutationGroupElement)
    d = length(g.permutation)
    x = collect(1:d)
    gx = g(x)
    if g.is_identity
        print(io, "$x -> $gx (identity)")
    else
        print(io, "$x -> $gx")
    end
end

function permutation_group(d::Int)
    return [PermutationGroupElement(p) for p in permutations(1:d)]
end

function block_permutation_group(d::Int, block_size::Int)
    if d % l != 0
        throw(ArgumentError("d must be divisible by l"))
    end
    # Group the indices into blocks
    block_indices = [collect(i:i+l-1) for i in 1:l:d]
    # Permute the block indices
    permuted_block_indices = [reduce(vcat, p) for p in permutations(block_indices)]
    return [PermutationGroupElement(p) for p in permuted_block_indices]
end

function cyclic_group(d::Int)
    return [PermutationGroupElement(circshift(collect(1:d), i)) for i in 1:d]
end

function block_cyclic_group(d::Int, l::Int)
    if d % l != 0
        throw(ArgumentError("d must be divisible by l"))
    end
    n_blocks = d / l
    return [PermutationGroupElement(circshift(collect(1:d), i * l)) for i in 1:n_blocks]
end

function random_subset(G::Vector{PermutationGroupElement}, n::Int)
    if n > length(G)
        throw(ArgumentError("n must be less than or equal to the number of elements in G"))
    end

    # Select n random elements from G
    set = sample(G, n, replace=false)

    # Add their inverses, except if they are already in the set
    set_with_inverses = copy(set)
    for g in set
        if !(inv(g) in set_with_inverses)
            push!(set_with_inverses, inv(g))
        end
    end
    return set_with_inverses
end
