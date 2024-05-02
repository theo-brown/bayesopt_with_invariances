using HDF5

include("regret_plot.jl")

output_file = "data/experiment_3_ucb_mpi/results.h5"
figure = Plots.plot()

h5open(output_file, "r") do file
    n_iterations = attrs(file)["n_iterations"]
    n_repeats = attrs(file)["n_repeats"]

    reported_f = Dict([
        kernel => zeros(n_repeats, n_iterations)
        for kernel in keys(file)
    ])

    for kernel in keys(file)
        for repeat in 1:n_repeats
            reported_f[kernel][repeat, :] .= file["$kernel/$repeat/reported_f"]
        end
    end

    f_max = maximum(map(maximum, values(reported_f)))

    for (kernel, f) in reported_f
        regret = cumulative_regret(f_max, f)
        plot_with_ribbon!(
            figure,
            regret,
            kernel,
            "Cumulative regret",
        )
    end
end

savefig(figure, "data/experiment_3_ucb_mpi/regret_plot.pdf")
savefig(figure, "data/experiment_3_ucb_mpi/regret_plot.png")