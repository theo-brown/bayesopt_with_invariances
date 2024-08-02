using KernelFunctions
include("onesided_transformed_kernel.jl")
include("asymmetric_transformed_kernel.jl")


function isotropic_invariant_kernel(k::Kernel, T::Tuple{Vararg{Transform}})
    return 1 / length(T) * KernelSum(
        collect( # Collect is required because KernelSum can't take generators currently
            OneSidedTransformedKernel(k, t)
            for t in T
        )
    )
end


function invariant_kernel(k::Kernel, T::Tuple{Vararg{Transform}})
    return 1 / length(T)^2 * KernelSum(
        collect( # Collect is required because KernelSum can't take generators currently
            AsymmetricTransformedKernel(k, t₁, t₂)
            for t₁ in T
            for t₂ in T
        )
    )
end
