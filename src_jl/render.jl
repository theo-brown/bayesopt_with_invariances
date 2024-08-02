using GLMakie
using Plots
using LaTeXStrings

include("gp_utils.jl")
include("synthetic_objective.jl")


function render(
    f::Function,
    input_bounds::Vector{Tuple{Float64,Float64}};
    resolution::Int=100,
    output_filename::String=""
)
    # Evaluate the function on a grid
    x_ranges = [range(lower, upper, length=resolution) for (lower, upper) in input_bounds]
    x_grid = Iterators.product(x_ranges...)
    y_grid = [f(collect(x)) for x in x_grid]

    if length(input_bounds) == 2
        figure = Plots.contourf(
            x_ranges[1],
            x_ranges[2],
            y_grid,
            levels=32,
            color=:viridis,
            cbar=true,
            xlabel=L"$x_1$",
            ylabel=L"$x_2$",
            colorbar_title=L"$f(\mathbf{x})$"
        )
        Plots.plot!(
            size=(600, 600),
            aspect_ratio=:equal,
            xlims=input_bounds[1],
            ylims=input_bounds[2],
            xticks=nothing,
            yticks=nothing
        )

        if output_filename != ""
            Plots.savefig("$output_filename.pdf")
            Plots.savefig("$output_filename.png")
        end

        return figure

    elseif length(input_bounds) == 3
        figure = GLMakie.Figure(size=(600, 600))
        ax = GLMakie.Axis3(
            figure[1, 1],
            aspect=:equal,
            azimuth=π / 3,
            limits=(input_bounds[1], input_bounds[2], input_bounds[3]),
            xlabel=L"$x_1$",
            ylabel=L"$x_2$",
            zlabel=L"$x_3$",
            xlabeloffset=5,
            ylabeloffset=5,
            zlabeloffset=5,
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
        # Hide axes
        GLMakie.hidexdecorations!(ax; label=false)
        GLMakie.hideydecorations!(ax; label=false)
        GLMakie.hidezdecorations!(ax; label=false)
        # Add in a colorbar
        GLMakie.Colorbar(figure[1, 2], volplot; height=500, tellheight=false, label=L"f(\mathbf{x})")

        if output_filename != ""
            Makie.save("$output_filename.png", figure, px_per_unit=12)
        end

        return figure
    else
        throw(ArgumentError("Only 2D and 3D input spaces are supported"))
    end
end

function render(
    f::Function,
    input_bounds::Vector{Tuple{Float64,Float64}},
    observed_inputs::Matrix{Float64};
    resolution::Int=100,
    output_filename::String=""
)
    figure = render(f, input_bounds; resolution=resolution)
    if length(input_bounds) == 2
        Plots.scatter!(
            observed_inputs[1, :],
            observed_inputs[1, :],
            seriestype=:scatter,
            color=:white,
            label="Observations",
            markersize=3
        )
        if output_filename != ""
            Plots.savefig("$output_filename.pdf")
        end
        return figure
    elseif length(input_bounds) == 3
        GLMakie.scatter!(
            figure[1, 1],
            observed_inputs[1, :],
            observed_inputs[2, :],
            observed_inputs[3, :],
            color=:white,
            markersize=3,
            label="Observations"
        )
        if output_filename != ""
            Makie.save("$output_filename.png", figure, px_per_unit=12)
        end
        return figure
    else
        throw(ArgumentError("Only 2D and 3D input spaces are supported"))
    end
end
