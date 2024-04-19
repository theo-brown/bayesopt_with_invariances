using KernelFunctions
import KernelFunctions.kernelmatrix_diag
import KernelFunctions.kernelmatrix_diag!
import KernelFunctions.kernelmatrix
import KernelFunctions.kernelmatrix!
import KernelFunctions.printshifted

struct AsymmetricTransformedKernel{Tk<:Kernel,Tt1<:Transform,Tt2<:Transform} <: Kernel
    kernel::Tk
    x_transform::Tt1
    y_transform::Tt2
end

# Basic call
(k::AsymmetricTransformedKernel)(x, y) = k.kernel(k.x_transform(x), k.y_transform(y))

# Kernel matrix operations
function KernelFunctions.kernelmatrix_diag!(K::AbstractVector, κ::AsymmetricTransformedKernel, x::AbstractVector)
    return KernelFunctions.kernelmatrix_diag!(K, κ.kernel, map(κ.x_transform, x), map(κ.y_transform, x))
end

function KernelFunctions.kernelmatrix_diag!(
    K::AbstractVector, κ::AsymmetricTransformedKernel, x::AbstractVector, y::AbstractVector
)
    return KernelFunctions.kernelmatrix_diag!(K, κ.kernel, map(κ.x_transform, x), map(κ.y_transform, y))
end

function KernelFunctions.kernelmatrix!(K::AbstractMatrix, κ::AsymmetricTransformedKernel, x::AbstractVector)
    return KernelFunctions.kernelmatrix!(K, κ.kernel, map(κ.x_transform, x), map(κ.y_transform, y))
end

function KernelFunctions.kernelmatrix!(
    K::AbstractMatrix, κ::AsymmetricTransformedKernel, x::AbstractVector, y::AbstractVector
)
    return KernelFunctions.kernelmatrix!(K, κ.kernel, map(κ.x_transform, x), map(κ.y_transform, y))
end

function KernelFunctions.kernelmatrix_diag(κ::AsymmetricTransformedKernel, x::AbstractVector)
    return KernelFunctions.kernelmatrix_diag(κ.kernel, map(κ.x_transform, x), map(κ.y_transform, x))
end

function KernelFunctions.kernelmatrix_diag(κ::AsymmetricTransformedKernel, x::AbstractVector, y::AbstractVector)
    return KernelFunctions.kernelmatrix_diag(κ.kernel, map(κ.x_transform, x), map(κ.y_transform, y))
end

function KernelFunctions.kernelmatrix(κ::AsymmetricTransformedKernel, x::AbstractVector)
    return KernelFunctions.kernelmatrix(κ.kernel, map(κ.x_transform, x), map(κ.y_transform, x))
end

function KernelFunctions.kernelmatrix(κ::AsymmetricTransformedKernel, x::AbstractVector, y::AbstractVector)
    return KernelFunctions.kernelmatrix(κ.kernel, map(κ.x_transform, x), map(κ.y_transform, y))
end

# Print 
function Base.show(io::IO, κ::AsymmetricTransformedKernel)
    return printshifted(io, κ, 0)
end

function KernelFunctions.printshifted(io::IO, κ::AsymmetricTransformedKernel, shift::Int)
    KernelFunctions.printshifted(io, κ.kernel, shift)
    print(io, "\n" * ("\t"^(shift + 1)) * "- Transform applied to x: $(κ.x_transform)")
    print(io, "\n" * ("\t"^(shift + 1)) * "- Transform applied to y: $(κ.y_transform)")
    return
end
