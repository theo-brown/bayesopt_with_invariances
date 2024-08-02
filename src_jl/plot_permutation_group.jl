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

f = build_synthetic_objective(
    θ -> build_invariant_gp(θ, to_transform.(permutation_group(2))),
    (
        σ_f=1.0,
        l=0.12,
        σ_n=0.1,
    ),
    128,
    [(0.0, 1.0) for _ in 1:2],
    43,
)

x_ranges = [range(0, 1, length=100) for _ in 1:2]
x_grid = Iterators.product(x_ranges...)
y_grid = [f(collect(x)) for x in x_grid]
x_observed = [0.2, 0.9]
extra_observations = [σ(x_observed) for σ in permutation_group(2) if ~is_identity(σ)]

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
    [x_observed[1]],
    [x_observed[2]],
    seriestype=:scatter,
    color=:red,
    label=nothing,
    markersize=5,
)
scatter!(
    [x_extra[1] for x_extra in extra_observations],
    [x_extra[2] for x_extra in extra_observations],
    seriestype=:scatter,
    color=:white,
    label=nothing,
    markersize=5,
)
plot!(
    aspect_ratio=:equal,
    xlims=(0, 1),
    ylims=(0, 1),
    xticks=nothing,
    yticks=nothing
)

savefig("permutation_group.pdf")
savefig("permutation_group.png")
display(figure)