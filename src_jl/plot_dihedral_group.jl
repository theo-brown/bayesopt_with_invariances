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

f = build_synthetic_objective(
    θ -> build_invariant_gp(θ, G),
    (
        σ_f=1.0,
        l=0.2,
        σ_n=0.1,
    ),
    64,
    [(-1.0, 1.0) for _ in 1:2],
    11,
)

x_ranges = [range(-1, 1, length=100) for _ in 1:2]
x_grid = Iterators.product(x_ranges...)
y_grid = [f(collect(x)) for x in x_grid]

x_observed = [0.4, 0.4]
extra_observations = [σ(x_observed) for σ in G if σ(x_observed) != x_observed]


figure = heatmap(
    x_ranges[1],
    x_ranges[2],
    y_grid,
    # levels=32,
    color=:viridis,
    cbar=true,
    xlabel=L"$x_1$",
    ylabel=L"$x_2$",
    colorbar_title=" \n" * L"$f(\mathbf{x})$",
    # lw=0,
    # linewidth=0,
)
scatter!(
    [x_observed[2]],
    [x_observed[1]],
    seriestype=:scatter,
    color=:red,
    label=nothing,
    markersize=5,
)
scatter!(
    [x_extra[2] for x_extra in extra_observations],
    [x_extra[1] for x_extra in extra_observations],
    seriestype=:scatter,
    color=:white,
    label=nothing,
    markersize=5,
)
plot!(
    aspect_ratio=:equal,
    xlims=(-1, 1),
    ylims=(-1, 1),
    xticks=nothing,
    yticks=nothing
)
savefig("dihedral_group_new.pdf")
savefig("dihedral_group_new.png")
