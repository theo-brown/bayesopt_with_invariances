using ArgParse
using Random
using Plots
using LaTeXStrings
using HDF5

include("../src/gp_utils.jl")
include("../src/objective_functions/kernel_objective_function.jl")
include("../src/bayesopt.jl")
include("../src/reporting.jl")
include("render.jl")
include("regret_plot.jl")


function run_experiment(
    seed::Int,
    bounds::Vector{Tuple{Float64,Float64}},
    output_directory::String,
    n_iterations::Int,
    n_repeats::Int,
    acquisition_function::Function,
    acquisition_function_label::String,
    reporting_function::Function,
    reporting_function_label::String,
    vanilla_gp_builder::Function,
    invariant_gp_builder::Function,
    invariant_gp_label::String,
    θ::NamedTuple,
    latent_function_seed::Int,
    latent_function_n_points::Int,
)
    # Setup
    Random.seed!(seed)
    mkdir(output_directory)
    output_file = joinpath(output_directory, "results.h5")

    # Define the target function
    println("Generating target function...")
    f = build_latent_function(
        invariant_gp_builder,
        θ,
        latent_function_n_points,
        bounds,
        latent_function_seed,
    )
    f_noisy(x) = f(x) + randn() * θ.σ_n
    render(f, bounds; output_filename=joinpath(output_directory, "latent_function"))

    println("Finding approximate maximum...")
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
        attrs(file)["latent_function_seed"] = latent_function_seed
        attrs(file)["latent_function_n_points"] = latent_function_n_points
        attrs(file)["sigma_n"] = θ.σ_n
        attrs(file)["sigma_f"] = θ.σ_f
        attrs(file)["l"] = θ.l
        attrs(file)["x_opt"] = x_opt
        attrs(file)["f_opt"] = f_opt
    end

    # Run experiment
    for (gp_builder, label) in [
        (vanilla_gp_builder, "Standard"),
        (invariant_gp_builder, invariant_gp_label),
    ]
        println("Running BO with $label kernel...")

        h5open(output_file, "r+") do file
            create_group(file, label)
        end

        regret = zeros(n_repeats, n_iterations)

        for i in 1:n_repeats
            println("Repeat $i/$n_repeats")

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
                optimise_hyperparameters=false
            )

            # Compute the true function values
            observed_f = f([collect(xᵢ) for xᵢ in eachrow(observed_x)])
            reported_f = f([collect(xᵢ) for xᵢ in eachrow(reported_x)])

            # Compute the regret
            regret[i, :] = regret_function(f_opt, reported_f)

            println("Final regret: $(regret[i, end])")
            flush(stdout)

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
            regret_label
        )
    end

    # Save the regret plot
    regret_plot_path = joinpath(output_directory, "regret_plot")
    savefig(figure, "$regret_plot_path.pdf")
    savefig(figure, "$regret_plot_path.png")
end
