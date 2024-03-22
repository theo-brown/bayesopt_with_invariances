using KernelFunctions
import KernelFunctions.kernelmatrix_diag
import KernelFunctions.kernelmatrix_diag!
import KernelFunctions.kernelmatrix
import KernelFunctions.kernelmatrix!
import KernelFunctions.printshifted

struct OneSidedTransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end


# @functor OneSidedTransformedKernel

# Basic call
(k::OneSidedTransformedKernel)(x, y) = k.kernel(k.transform(x), y)

# Kernel matrix operations
function KernelFunctions.kernelmatrix_diag!(K::AbstractVector, κ::OneSidedTransformedKernel, x::AbstractVector)
    return KernelFunctions.kernelmatrix_diag!(K, κ.kernel, map(κ.transform, x), x)
end

function KernelFunctions.kernelmatrix_diag!(
    K::AbstractVector, κ::OneSidedTransformedKernel, x::AbstractVector, y::AbstractVector
)
    return KernelFunctions.kernelmatrix_diag!(K, κ.kernel, map(κ.transform, x), y)
end

function KernelFunctions.kernelmatrix!(K::AbstractMatrix, κ::OneSidedTransformedKernel, x::AbstractVector)
    return KernelFunctions.kernelmatrix!(K, κ.kernel, map(κ.transform, x), x)
end

function KernelFunctions.kernelmatrix!(
    K::AbstractMatrix, κ::OneSidedTransformedKernel, x::AbstractVector, y::AbstractVector
)
    return KernelFunctions.kernelmatrix!(K, κ.kernel, map(κ.transform, x), y)
end

function KernelFunctions.kernelmatrix_diag(κ::OneSidedTransformedKernel, x::AbstractVector)
    return KernelFunctions.kernelmatrix_diag(κ.kernel, map(κ.transform, x), x)
end

function KernelFunctions.kernelmatrix_diag(κ::OneSidedTransformedKernel, x::AbstractVector, y::AbstractVector)
    return KernelFunctions.kernelmatrix_diag(κ.kernel, map(κ.transform, x), y)
end

function KernelFunctions.kernelmatrix(κ::OneSidedTransformedKernel, x::AbstractVector)
    return KernelFunctions.kernelmatrix(κ.kernel, map(κ.transform, x), x)
end

function KernelFunctions.kernelmatrix(κ::OneSidedTransformedKernel, x::AbstractVector, y::AbstractVector)
    return KernelFunctions.kernelmatrix(κ.kernel, map(κ.transform, x), y)
end

# Print 
function Base.show(io::IO, κ::OneSidedTransformedKernel)
    return printshifted(io, κ, 0)
end

function KernelFunctions.printshifted(io::IO, κ::OneSidedTransformedKernel, shift::Int)
    KernelFunctions.printshifted(io, κ.kernel, shift)
    return print(io, "\n" * ("\t"^(shift + 1)) * "- $(κ.transform) (one-sided)")
end