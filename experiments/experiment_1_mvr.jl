
using Dates, Logging, LoggingExtras
include("run_experiment.jl")

const t0 = now()

timestamp_logger(logger) =
    TransformerLogger(logger) do log
        elapsed = canonicalize(now() - t0)
        elapsed_time = Dates.Time(elapsed.periods...)
        merge(log, (; message="$(Dates.format(elapsed_time, dateformat"HH:MM:SS")) $(log.message)"))
    end

ConsoleLogger(stdout, Logging.Debug) |> timestamp_logger |> global_logger

# Experiment
const d = 2

run_experiment(
    seed=42,
    bounds=[(0.0, 1.0) for _ in 1:d],
    output_directory="data/experiment_1_mvr_updated",
    n_iterations=256,
    n_repeats=16,
    acquisition_function=mvr,
    acquisition_function_label="MVR",
    reporting_function=maximum_observed_posterior_mean,
    reporting_function_label="Maximum observed posterior mean",
    gp_builders=Dict([
        ("Standard", build_matern52_gp),
        ("Permutation invariant", θ -> build_perminvariantmatern52_gp(θ, permutation_group(d)))
    ]),
    target_gp_builder=θ -> build_perminvariantmatern52_gp(θ, permutation_group(d)),
    target_function_seed=43,
    target_function_n_points=128,
    θ=(
        l=0.12,
        σ_f=1.0,
        σ_n=0.1,
    ),
)