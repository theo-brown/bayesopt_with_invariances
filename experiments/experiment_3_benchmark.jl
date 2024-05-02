using MPI
using HDF5
using Random, QuasiMonteCarlo
using Plots, LaTeXStrings
using BenchmarkTools

include("../src/gp_utils.jl")
include("../src/synthetic_objective.jl")
include("../src/acquisition.jl")
include("../src/permutation_groups.jl")

# Set global constants
const n_repeats = 32
const n_observed_x = 100
const d = 6
const T₁ = to_transform.(permutation_group(d))
const T₂ = to_transform.(block_permutation_group(d, 2))
const T₃ = to_transform.(block_permutation_group(d, 3))
const β = 2.0
const seed = 42
const target_function_seed = 5
const target_function_n_points = 512
const bounds = [(0.0, 1.0) for _ in 1:d]
const θ = (
    l=0.12,
    σ_f=1.0,
    σ_n=0.1,)
const f = build_synthetic_objective(
    θ -> build_invariant_gp(θ, T₁),
    θ,
    target_function_n_points,
    bounds,
    target_function_seed,
)
const gp_builders = Dict([
    ("Standard", build_gp),
    ("Fully permutation invariant", θ -> build_invariant_gp(θ, T₁)),
    ("2-block permutation invariant", θ -> build_invariant_gp(θ, T₂)),
    ("3-block permutation invariant", θ -> build_invariant_gp(θ, T₃))
])

# Initialise MPI
MPI.Init()
comm = MPI.COMM_WORLD
mpi_info = MPI.Info()
mpi_rank = MPI.Comm_rank(comm)
mpi_size = MPI.Comm_size(comm)

# Initialise output file 
output_directory = "data/experiment_3_benchmark"
if mpi_rank == 0
    mkdir(output_directory)
end
output_file = joinpath(output_directory, "results.h5")
h5open(output_file, "w", comm, mpi_info) do file
    # Metadata
    attrs(file)["seed"] = seed
    attrs(file)["n_repeats"] = n_repeats
    attrs(file)["n_observed_x"] = n_observed_x
    attrs(file)["d"] = d
    attrs(file)["beta"] = β
    attrs(file)["target_function_seed"] = target_function_seed
    attrs(file)["target_function_n_points"] = target_function_n_points
    attrs(file)["bounds"] = bounds
    attrs(file)["sigma_n"] = θ.σ_n
    attrs(file)["sigma_f"] = θ.σ_f
    attrs(file)["l"] = θ.l

    # Datasets
    for label in keys(gp_builders)
        create_group(file, label)
        create_dataset(file, "$label/observed_x", Float64, (d, n_repeats, n_observed_x))
        file["$label/observed_x"][:, :, :] = NaN
        create_dataset(file, "$label/observed_y", Float64, (n_repeats, n_observed_x))
        file["$label/observed_y"][:, :] = NaN
        for subgroup in ("fit_gp", "maximise_acqf")
            create_group(file, "$label/$subgroup")
            create_dataset(file, "$label/$subgroup/allocs", Int64, n_repeats)
            create_dataset(file, "$label/$subgroup/gctimes", Float64, n_repeats)
            create_dataset(file, "$label/$subgroup/memory", Int64, n_repeats)
            create_dataset(file, "$label/$subgroup/times", Float64, n_repeats)

            for dataset in ("gctimes", "times")
                file["$label/$subgroup/$dataset"][:] = NaN
            end
            for dataset in ("allocs", "memory")
                file["$label/$subgroup/$dataset"][:] = 0
            end
        end
    end
end

# Distribute tasks
tasks = [(label, gp_builder, repeat) for (label, gp_builder) in gp_builders for repeat in 1:n_repeats]
label, gp_builder, repeat = tasks[mpi_rank+1]
@info "[Worker $mpi_rank] Starting $label/$repeat"
flush(stdout)
Random.seed!(seed + mpi_rank % n_repeats) # Repeat i should have the same seeds for all the labels/gp_builders

# Noisy observations of the function should be dependent on this task's seed
f_noisy(x) = f(x) .+ randn() * θ.σ_n

# Generate a set of random samples from f_noisy
@info "Generating observations"
lower_bounds = [b[1] for b in bounds]
upper_bounds = [b[2] for b in bounds]
x = QuasiMonteCarlo.sample(n_observed_x, lower_bounds, upper_bounds, QuasiMonteCarlo.RandomSample())
y = f_noisy(x)

# Save the samples
h5open(output_file, "r+", comm, mpi_info) do file
    file["$label/observed_x"][:, repeat, :] = x
    file["$label/observed_y"][repeat, :] = y
end

# Trigger precompilation
gp = get_posterior_gp(gp_builder, ColVecs(x), y, θ)
maximise_acqf(gp, mvr, bounds, 1; time_limit=1)

# Benchmark GP fit 
@info "Worker $mpi_rank] Benchmarking GP fit"
gp_bench = @benchmark get_posterior_gp($gp_builder, $(ColVecs(x)), $y, $θ) samples = 1
# Save the output 
h5open(output_file, "r+", comm, mpi_info) do file
    file["$label/fit_gp/allocs"][repeat] = gp_bench.allocs
    file["$label/fit_gp/gctimes"][repeat] = gp_bench.gctimes[1]
    file["$label/fit_gp/memory"][repeat] = gp_bench.memory
    file["$label/fit_gp/times"][repeat] = gp_bench.times[1]
end

# Benchmark maximise_acqf
@info "[Worker $mpi_rank] Benchmarking maximise_acqf"
acqf_bench = @benchmark maximise_acqf($gp, $mvr, $bounds, $1) samples = 1
# Save the output 
h5open(output_file, "r+", comm, mpi_info) do file
    file["$label/maximise_acqf/allocs"][repeat] = acqf_bench.allocs
    file["$label/maximise_acqf/gctimes"][repeat] = acqf_bench.gctimes[1]
    file["$label/maximise_acqf/memory"][repeat] = acqf_bench.memory
    file["$label/maximise_acqf/times"][repeat] = acqf_bench.times[1]
end

