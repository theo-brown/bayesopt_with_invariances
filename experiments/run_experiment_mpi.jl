using MPI
using HDF5
using Random
using Plots, LaTeXStrings

include("../src/gp_utils.jl")
include("../src/synthetic_objective.jl")
include("../src/bayesopt.jl")
include("../src/reporting.jl")
include("render.jl")
include("regret_plot.jl")

@assert HDF5.has_parallel()


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
    # Initialise MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    mpi_info = MPI.Info()
    mpi_rank = MPI.Comm_rank(comm)
    mpi_size = MPI.Comm_size(comm)

    # Check we have sufficient parallelism
    n_required_tasks = n_repeats * length(gp_builders)
    if n_required_tasks > mpi_size
        error("Not enough MPI tasks to run the experiment (need $n_required_tasks, got $mpi_size)")
    end

    # Set up output 
    if mpi_rank == 0
        mkdir(output_directory)
    end
    output_file = joinpath(output_directory, "results.h5")


    # All tasks need a copy of the target function - this is deterministic given target_function_seed
    d = length(bounds)
    f = build_latent_function(
        target_gp_builder,
        θ,
        target_function_n_points,
        bounds,
        target_function_seed,
    )

    # Find the approximate maximum of the target function
    ## TODO: We can do this in parallel
    x_opt, f_opt = get_approximate_maximum(f, bounds)

    # Initialise output file
    ## This must be done together, as the file is shared
    # MPI.Barrier(comm)
    h5_file = h5open(output_file, "w", comm, mpi_info)
    ## Save metadata
    attrs(h5_file)["acquisition_function"] = acquisition_function_label
    attrs(h5_file)["reporting_function"] = reporting_function_label
    attrs(h5_file)["n_iterations"] = n_iterations
    attrs(h5_file)["n_repeats"] = n_repeats
    attrs(h5_file)["seed"] = seed
    attrs(h5_file)["target_function_seed"] = target_function_seed
    attrs(h5_file)["target_function_n_points"] = target_function_n_points
    attrs(h5_file)["sigma_n"] = θ.σ_n
    attrs(h5_file)["sigma_f"] = θ.σ_f
    attrs(h5_file)["l"] = θ.l
    attrs(h5_file)["x_opt"] = x_opt
    attrs(h5_file)["f_opt"] = f_opt
    attrs(h5_file)["d"] = d
    attrs(h5_file)["bounds"] = bounds

    ## Initialise data groups
    ## Note we don't bother chunking as no two processes write the same dataset
    for label in keys(gp_builders)
        create_group(h5_file, label)

        for repeat in 1:n_repeats
            create_group(h5_file, "$label/$repeat")

            create_dataset(h5_file, "$label/$repeat/observed_x", Float64, (d, n_iterations))
            create_dataset(h5_file, "$label/$repeat/observed_y", Float64, (n_iterations,))
            create_dataset(h5_file, "$label/$repeat/observed_f", Float64, (n_iterations,))

            create_dataset(h5_file, "$label/$repeat/reported_x", Float64, (d, n_iterations))
            create_dataset(h5_file, "$label/$repeat/reported_y", Float64, (n_iterations,))
            create_dataset(h5_file, "$label/$repeat/reported_f", Float64, (n_iterations,))
        end
    end

    ## Sync before starting the experiment
    # MPI.Barrier(comm)

    # Distribute tasks 
    tasks = [(label, gp_builder, repeat) for (label, gp_builder) in gp_builders for repeat in 1:n_repeats]
    label, gp_builder, repeat = tasks[mpi_rank+1]
    @info "Worker $mpi_rank starting $label/$repeat"
    Random.seed!(seed + mpi_rank % n_repeats) # Repeat i should have the same seeds for all the labels/gp_builders 

    # Noisy observations of the functions should be dependent on this task's seed
    f_noisy(x) = f(x) + randn() * θ.σ_n

    # Run BO
    observed_x, observed_y, reported_x, reported_y = run_bayesopt(
        f_noisy,
        bounds,
        n_iterations,
        gp_builder,
        acquisition_function,
        reporting_function,
        θ;
        n_restarts=10,
    )

    # Compute the true function values
    observed_f = f(observed_x)
    reported_f = f(reported_x)

    # Save to the file
    h5_file["$label/$repeat/observed_x"][:, :] = observed_x
    h5_file["$label/$repeat/observed_y"][:] = observed_y
    h5_file["$label/$repeat/observed_f"][:] = observed_f

    h5_file["$label/$repeat/reported_x"][:, :] = reported_x
    h5_file["$label/$repeat/reported_y"][:] = reported_y
    h5_file["$label/$repeat/reported_f"][:] = reported_f

    # Close the file
    # MPI.Barrier(comm)
    close(h5_file)
    @info "Worker $mpi_rank finished"
end