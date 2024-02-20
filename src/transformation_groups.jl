using Combinatorics


"""
    permutation_group(d::Int)

Generate the permutation group in `d` dimensions.

Example:
```julia-repl
julia> G = permutation_group(3);
julia> x = [1, 2, 3];
julia> [σ(x) for σ in G]
6-element Vector{Vector{Int64}}:
 [1, 2, 3]
 [1, 3, 2]
 [2, 1, 3]
 [2, 3, 1]
 [3, 1, 2]
 [3, 2, 1]
```
"""
function permutation_group(d::Int)
    return Function[x -> x[p] for p in permutations(1:d)]
end


"""
    blockpermutation_group(d::Int, l::Int)

Generate the block-permutation group in `d` dimensions with blocks of length `l`.

Throws an error if `d` is not divisible by `l`.

Example:
```julia-repl
julia> G = blockpermutation_group(6, 2);
julia> x = [1, 2, 3, 4, 5, 6];
julia> [σ(x) for σ in G]
6-element Vector{Vector{Int64}}:
 [1, 2, 3, 4, 5, 6]
 [1, 2, 5, 6, 3, 4]
 [3, 4, 1, 2, 5, 6]
 [3, 4, 5, 6, 1, 2]
 [5, 6, 1, 2, 3, 4]
 [5, 6, 3, 4, 1, 2]
"""
function blockpermutation_group(d::Int, l::Int)
    if d % l != 0
        throw(ArgumentError("d must be divisible by l"))
    end
    # Group the indices into blocks
    block_indices = [collect(i:i+l-1) for i in 1:l:d]
    # Permute the block indices
    permuted_block_indices = [reduce(vcat, p) for p in permutations(block_indices)]
    # Generate the transformations
    return Function[x -> x[p] for p in permuted_block_indices]
end


"""
    cyclic_group(d)

Generate the cyclic group in `d` dimensions.

Example:
```julia-repl
julia> G = cyclic_group(3);
julia> x = [1, 2, 3];
julia> [σ(x) for σ in G]
3-element Vector{Vector{Int64}}:
 [3, 1, 2]
 [2, 3, 1]
 [1, 2, 3]
"""
function cyclic_group(d::Int)
    return Function[x -> circshift(x, i) for i in 1:d]
end


"""
    blockcyclic_group(d, l)

Generate the block-cyclic group in `d` dimensions with blocks of length `l`.

Throws an error if `d` is not divisible by `l`.

Example:
```julia-repl
julia> G = blockcyclic_group(6, 2);
julia> x = [1, 2, 3, 4, 5, 6];
julia> [σ(x) for σ in G]
3-element Vector{Vector{Int64}}:
[5, 6, 1, 2, 3, 4]
[3, 4, 5, 6, 1, 2]
[1, 2, 3, 4, 5, 6]
"""
function blockcyclic_group(d::Int, l::Int)
    if d % l != 0
        throw(ArgumentError("d must be divisible by l"))
    end
    n_blocks = d / l
    return Function[x -> circshift(x, i * l) for i in 1:n_blocks]
end

"""
    dihedral_group(n::Int)

Generate the dihedral group D_n in 2 dimensions.
"""
function dihedral_group(n::Int)
    r_k = Function[x -> [cos(2π * k / n) -sin(2π * k / n); sin(2π * k / n) cos(2π * k / n)] * x for k in 0:n-1]
    s_k = Function[x -> [cos(2π * k / n) sin(2π * k / n); sin(2π * k / n) -cos(2π * k / n)] * x for k in 0:n-1]
    return [r_k; s_k]
end