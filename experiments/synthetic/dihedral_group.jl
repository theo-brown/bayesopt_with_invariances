using KernelFunctions, AbstractGPs
using Random
using Plots, LaTeXStrings

include("../../src/onesided_transformedkernel.jl")

function dihedral_group(n::Int)
    r_k = [[cos(2π * k / n) -sin(2π * k / n); sin(2π * k / n) cos(2π * k / n)] for k in 0:n-1]
    s_k = [[cos(2π * k / n) sin(2π * k / n); sin(2π * k / n) -cos(2π * k / n)] for k in 0:n-1]
    return [r_k; s_k]
end

Random.seed!(2)

G = dihedral_group(5)
k = with_lengthscale(Matern52Kernel(), 0.4)
k = 1 / length(G) * KernelSum(collect(OneSidedTransformedKernel(k, LinearTransform(σ)) for σ in G))
gp = GP(k)

x_range = -1:0.05:1
X = Matrix{Float64}(undef, 2, length(x_range)^2)
i = 1
for (x, y) in Iterators.product(x_range, x_range)
    X[:, i] .= [x, y]
    global i += 1
end

gp_x = gp(ColVecs(X), 1e-6)
y = rand(gp_x)

contourf(
    x_range,
    x_range,
    y,
    levels=32,
    color=:viridis,
    cbar=true,
    xlabel=L"$x_1$",
    ylabel=L"$x_2$",
    colorbar_title=L"$f(\mathbf{x})$",
    size=(600, 600),
    aspect_ratio=:equal,
    xlims=(-1, 1),
    ylims=(-1, 1),
    xticks=nothing,
    yticks=nothing
)
savefig("dihedral_group.pdf")
savefig("dihedral_group.png")