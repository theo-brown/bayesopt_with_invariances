using HDF5
using StatsPlots, LaTeXStrings

fit_gp_times = Dict{String,Vector{Float64}}()
maximise_acqf_times = Dict{String,Vector{Float64}}()

h5open("/home/tab53/rds-home/bayesopt_with_invariances/experiments/data/experiment_3_benchmark/results.h5") do file
    for group in keys(file)
        fit_gp_times[group] = file[group]["fit_gp"]["times"][:]
        maximise_acqf_times[group] = file[group]["maximise_acqf"]["times"][:]
    end
end