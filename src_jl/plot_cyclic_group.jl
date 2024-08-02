using GLMakie
using LaTeXStrings

include("gp_utils.jl")
include("permutation_groups.jl")
include("synthetic_objective.jl")

f = build_synthetic_objective(
    θ -> build_invariant_gp(θ, to_transform.(cyclic_group(3))),
    (
        σ_f=1.0,
        l=0.12,
        σ_n=0.1,
    ),
    128,
    [(0.0, 1.0) for _ in 1:3],
    20,
)

resolution = 100
x_ranges = [range(0, 1, length=resolution) for _ in 1:3]
x_grid = Iterators.product(x_ranges...)
y_grid = [f(collect(x)) for x in x_grid]
x_observed = [0.3, 0.8, 0.5]
extra_observations = [σ(x_observed) for σ in cyclic_group(3) if ~is_identity(σ)]

figure = GLMakie.Figure(size=(600, 400), fontsize=24, figure_padding=1)
ax = GLMakie.Axis3(
    figure[1, 1],
    aspect=:equal,
    azimuth=π / 3,
    limits=((0, 1), (0, 1), (0, 1)),
    xlabel=L"$x_1$",
    ylabel=L"$x_2$",
    zlabel=L"$x_3$",
    xlabeloffset=5,
    ylabeloffset=5,
    zlabeloffset=5,
    xlabelsize=28,
    ylabelsize=28,
    zlabelsize=28,
)
# Latent function
volplot = GLMakie.volume!(
    ax,
    x_ranges[1],
    x_ranges[2],
    x_ranges[3],
    y_grid,
    levels=32,
)
# Visible edges
GLMakie.linesegments!(ax, [[1.0, 0.0, 0.0] [1.0, 1.0, 0.0]], color=:black)
GLMakie.linesegments!(ax, [[1.0, 0.0, 0.0] [1.0, 0.0, 1.0]], color=:black)
GLMakie.linesegments!(ax, [[1.0, 0.0, 1.0] [1.0, 1.0, 1.0]], color=:black)
GLMakie.linesegments!(ax, [[1.0, 1.0, 1.0] [1.0, 1.0, 0.0]], color=:black)
GLMakie.linesegments!(ax, [[1.0, 1.0, 1.0] [0.0, 1.0, 1.0]], color=:black)
GLMakie.linesegments!(ax, [[0.0, 1.0, 0.0] [0.0, 1.0, 1.0]], color=:black)
GLMakie.linesegments!(ax, [[0.0, 1.0, 0.0] [1.0, 1.0, 0.0]], color=:black)
GLMakie.linesegments!(ax, [[0.0, 0.0, 1.0] [0.0, 1.0, 1.0]], color=:black)
GLMakie.linesegments!(ax, [[0.0, 0.0, 1.0] [1.0, 0.0, 1.0]], color=:black)
# Hidden edges
GLMakie.linesegments!(ax, [[0.0, 0.0, 0.0] [0.0, 0.0, 1.0]], color=:black, linestyle=:dot, overdraw=true)
GLMakie.linesegments!(ax, [[0.0, 0.0, 0.0] [0.0, 1.0, 0.0]], color=:black, linestyle=:dot, overdraw=true)
GLMakie.linesegments!(ax, [[0.0, 0.0, 0.0] [1.0, 0.0, 0.0]], color=:black, linestyle=:dot, overdraw=true)
# Hide axes ticks
GLMakie.hidexdecorations!(ax; label=false)
GLMakie.hideydecorations!(ax; label=false)
GLMakie.hidezdecorations!(ax; label=false)
# Add in a colorbar
GLMakie.Colorbar(figure[1, 2], volplot; height=325, tellheight=false, label=L"f(\mathbf{x})")

# Observations
scatter!(
    ax,
    [x_observed[1]],
    [x_observed[2]],
    [x_observed[3]],
    color=:red,
    markersize=16,
    strokewidth=1,
)
scatter!(
    ax,
    [x_extra[1] for x_extra in extra_observations],
    [x_extra[2] for x_extra in extra_observations],
    [x_extra[3] for x_extra in extra_observations],
    color=:white,
    markersize=16,
    strokewidth=1,
)
colsize!(figure.layout, 1, Aspect(1, 1))
save("cyclic_group.png", figure, px_per_unit=16)
# save("cyclic_group.png", figure, px_per_unit=5)