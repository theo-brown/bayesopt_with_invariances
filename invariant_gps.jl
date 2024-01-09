using AbstractGPs # For general GP functionality
using KernelFunctions, Combinatorics # For defining invariant kernels


struct PermutationInvariantKernel <: KernelFunctions.Kernel
    base_kernel::KernelFunctions.Kernel
end

#TODO: accept composed functions as base kernels
#TODO: This doesn't work with Zygote autograd
function (k::PermutationInvariantKernel)(x, y)
    x_permutations = permutations(x, length(x))
    y_permutations = permutations(y, length(y))
    return sum(
        [
        k.base_kernel(x_perm, y_perm)
        for (x_perm, y_perm) in Iterators.product(x_permutations, y_permutations)
    ]
    ) / (length(x_permutations) * length(y_permutations))
end

function build_permutationinvariantmatern52_gp(θ::NamedTuple)::AbstractGPs.AbstractGP
    base_kernel = θ.σ_f^2 * with_lengthscale(Matern52Kernel(), θ.l)
    invariant_kernel = PermutationInvariantKernel(base_kernel)
    return GP(invariant_kernel)
end

function build_matern52_gp(θ::NamedTuple)::AbstractGPs.AbstractGP
    kernel = θ.σ_f^2 * with_lengthscale(Matern52Kernel(), θ.l)
    return GP(kernel)
end