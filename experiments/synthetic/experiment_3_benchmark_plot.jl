using HDF5
using DataFrames
using StatsPlots
using ColorSchemes
using LatexStrings

# Load the data from the HDF5 file
columns = ["times"]
tasks = ["fit_gp"]
df = DataFrame()

for task in tasks
    h5open("data/experiment_3_benchmark/results.h5") do file
        for kernel in keys(file)
            nrows = length(file["$(kernel)/$(task)/times"][:])
            kernel_df = DataFrame()
            for column in columns
                kernel_df[!, column] = file["$(kernel)/$(task)/$(column)"][:]
            end
            kernel_df[!, "kernel"] = [kernel for _ in 1:nrows]
            kernel_df[!, "task"] = [task for _ in 1:nrows]
            append!(df, kernel_df)
        end
    end
end

# Plot the results for fit_gp
labels = Dict(
    "Standard" => "1",
    "3-block permutation invariant" => "2",
    "2-block permutation invariant" => "6",
    "Fully permutation invariant" => "720",
)

figure = plot()
for kernel in ["Standard", "3-block permutation invariant", "2-block permutation invariant", "Fully permutation invariant"]
    data = filter(row -> row.kernel == kernel, df)
    n = length(data.times)
    group_sizes = [labels[kernel] for _ in 1:n]
    boxplot!(group_sizes, data.times, label=kernel, color_palette=:tol_bright)
end
yaxis!(figure, :log10)
xlabel!(figure, "Group size, " * L"|G|$")
ylabel!(figure, "Time to fit (s)")
plot!(legend=:topleft)

savefig(figure, "data/experiment_3_benchmark/fit_gp_times.png")
savefig(figure, "data/experiment_3_benchmark/fit_gp_times.pdf")
savefig(figure, "data/experiment_3_benchmark/fit_gp_times.svg")
figure
