using Plots
using LaTeXStrings

include("../../src/gp_utils.jl")
include("../../src/permutation_groups.jl")
include("../../src/synthetic_objective.jl")

Plots.default(
    titlefont=font(14),
    guidefont=font(12),
    tickfont=font(10),
    legendfont=font(10),
    linewidth=2,
    size=(400, 300),
    grid=false,
)

function dihedral_group(n::Int)
    r_k = [[cos(2π * k / n) -sin(2π * k / n); sin(2π * k / n) cos(2π * k / n)] for k in 0:n-1]
    s_k = [[cos(2π * k / n) sin(2π * k / n); sin(2π * k / n) -cos(2π * k / n)] for k in 0:n-1]
    return [r_k; s_k]
end

G = Tuple(collect(LinearTransform(σ) for σ in dihedral_group(5)))
θ = (
    σ_f=1.0,
    l=0.2,
    σ_n=0.1,
)
gp = build_invariant_gp(θ, G)

resolution = 32
x_ranges = [range(-1, 1, length=resolution) for _ in 1:2]
x_grid = [[xi, xj] for xi in x_ranges[1] for xj in x_ranges[2]]
finite_gp = gp(x_grid, 1e-6)

for i in 1:3
    y = rand(finite_gp)
    figure = contourf(
        x_ranges[1],
        x_ranges[2],
        reshape(y, resolution, resolution),
        levels=10,
        color=:viridis,
        cbar=false,
        linewidth=0,
    )
    plot!(
        aspect_ratio=:equal,
        xlims=(-1, 1),
        ylims=(-1, 1),
        xticks=nothing,
        yticks=nothing,
        size=(1000, 1000),
        dpi=500,
    )
    savefig("dihedral_sample_$i.png")
end