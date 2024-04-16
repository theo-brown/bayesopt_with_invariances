using ArgParse
using Random
using Plots
using LaTeXStrings
using HDF5
using Base.Threads

include("../src/gp_utils.jl")
include("../src/synthetic_objective.jl")
include("../src/bayesopt.jl")
include("../src/reporting.jl")
include("render.jl")
include("regret_plot.jl")


function run_experiment(
    ;
    seed::Int,
    bounds::Vector{Tuple{Float64,Float64}},
    output_directory::String,
    n_iterations::Int,
    n_repeats::Int,
    acquisition_function::Function,
    acquisition_function_label::String,
    reporting_function::Function,
    reporting_function_label::String,
    gp_builders::Dict{String,T} where {T<:Function},
    target_gp_builder::Function,
    target_function_seed::Int,
    target_function_n_points::Int,
    θ::NamedTuple,
)
    # Setup
    Random.seed!(seed)
    mkdir(output_directory)
    output_file = joinpath(output_directory, "results.h5")

    # Define the target function
    @info "Generating target function..."
    flush(stdout)
    f = build_latent_function(
        target_gp_builder,
        θ,
        target_function_n_points,
        bounds,
        target_function_seed,
    )
    f_noisy(x) = f(x) + randn() * θ.σ_n
    if length(bounds) in [2, 3]
        render(f, bounds; output_filename=joinpath(output_directory, "latent_function"))
    end

    @info "Finding approximate maximum..."
    x_opt, f_opt = get_approximate_maximum(f, bounds)

    # Regret plot setup
    if acquisition_function_label == "MVR"
        regret_function = simple_regret
        regret_label = "Simple regret"
    else
        regret_function = cumulative_regret
        regret_label = "Cumulative regret"
    end
    figure = Plots.plot()

    # Save metadata
    h5open(output_file, "w") do file
        attrs(file)["acquisition_function"] = acquisition_function_label
        attrs(file)["reporting_function"] = reporting_function_label
        attrs(file)["n_iterations"] = n_iterations
        attrs(file)["n_repeats"] = n_repeats
        attrs(file)["seed"] = seed
        attrs(file)["target_function_seed"] = target_function_seed
        attrs(file)["target_function_n_points"] = target_function_n_points
        attrs(file)["sigma_n"] = θ.σ_n
        attrs(file)["sigma_f"] = θ.σ_f
        attrs(file)["l"] = θ.l
        attrs(file)["x_opt"] = x_opt
        attrs(file)["f_opt"] = f_opt
    end

    # Run experiment
    for (label, gp_builder) in gp_builders
        @info "Running BO with $label kernel..."

        h5open(output_file, "r+") do file
            create_group(file, label)
        end

        regret = zeros(n_repeats, n_iterations)

        for i in 1:n_repeats
            @info "Repeat $i/$n_repeats"

            # Set seed
            Random.seed!(seed + i)

            # Run BO
            observed_x, observed_y, reported_x, reported_y = run_bayesopt(
                f_noisy,
                bounds,
                n_iterations,
                gp_builder,
                acquisition_function,
                reporting_function,
                θ;
                n_restarts=nthreads(),
            )

            # Compute the true function values
            observed_f = f(observed_x)
            reported_f = f(reported_x)

            # Compute the regret
            regret[i, :] = regret_function(f_opt, reported_f)

            @info "Final regret: $(regret[i, end])"

            # Save data to file
            h5open(output_file, "r+") do file
                file["$label/$i/observed_x"] = observed_x
                file["$label/$i/observed_y"] = observed_y
                file["$label/$i/observed_f"] = observed_f

                file["$label/$i/reported_x"] = reported_x
                file["$label/$i/reported_y"] = reported_y
                file["$label/$i/reported_f"] = reported_f
            end
        end

        # Add to the regret plot
        plot_with_ribbon!(
            figure,
            regret,
            label,
            regret_label;
            ylims=regret_label == "Simple regret" ? (0.0, 1.5) : nothing,
        )
    end

    # Save the regret plot
    regret_plot_path = joinpath(output_directory, "regret_plot")
    savefig(figure, "$regret_plot_path.pdf")
    savefig(figure, "$regret_plot_path.png")
end
