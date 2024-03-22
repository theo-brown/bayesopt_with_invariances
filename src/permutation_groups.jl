using Combinatorics
using StatsBase

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
    return x[g.permutation]
end

function to_matrix(g::PermutationGroupElement)
    d = length(g.permutation)
    M = zeros(Int, d, d)
    for i in 1:d
        M[i, g.permutation[i]] = 1
    end
    return M
end

function to_transform(g::PermutationGroupElement)
    return LinearTransform(to_matrix(g))
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
    return Tuple(PermutationGroupElement(p) for p in permutations(1:d))
end

function block_permutation_group(d::Int, block_size::Int)
    if d % block_size != 0
        throw(ArgumentError("d must be divisible by block_size"))
    end
    # Group the indices into blocks
    block_indices = [collect(i:i+block_size-1) for i in 1:block_size:d]
    # Permute the block indices
    permuted_block_indices = [reduce(vcat, p) for p in permutations(block_indices)]
    return Tuple(PermutationGroupElement(p) for p in permuted_block_indices)
end

function cyclic_group(d::Int)
    return Tuple(PermutationGroupElement(circshift(collect(1:d), i)) for i in 1:d)
end

function block_cyclic_group(d::Int, block_size::Int)
    if d % block_size != 0
        throw(ArgumentError("d must be divisible by block_size"))
    end
    n_blocks = d / block_size
    return Tuple(PermutationGroupElement(circshift(collect(1:d), i * block_size)) for i in 1:n_blocks)
end

function random_subgroup(G::NTuple{N,PermutationGroupElement}, n::Int) where {N}
    if n > length(G)
        throw(ArgumentError("n must be less than or equal to the number of elements in G"))
    end

    identity_element = G[[g.is_identity for g in G]][1]
    H = [identity_element]

    function add_closure!(H::Vector{PermutationGroupElement}, h::PermutationGroupElement)
        for other_h in H
            g = h ∘ other_h
            if !(g in H)
                push!(H, g)
                add_closure!(H, g)
            end
        end
    end

    function add_element!(H::Vector{PermutationGroupElement})
        g = rand(G)
        if !(g in H)
            # Add element
            push!(H, g)
            # Αdd its inverse
            if !(inv(g) in H)
                push!(H, inv(g))
            end
            # Add its closure 
            add_closure!(H, g)
        end
    end

    while length(H) < n
        add_element!(H)
    end

    return Tuple(H)
end
