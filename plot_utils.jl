using Plots
using AbstractGPs

function plot_2d_function_with_observations(func, observation_locations, bounds)
    xi_range = range(bounds[1][1], bounds[1][2], length=100)
    contourf(xi_range, xi_range, (x, y) -> func([x, y]), levels=10, color=:viridis, cbar=true, size=(300, 300))
    scatter!([xi[1] for xi in observation_locations], [xi[2] for xi in observation_locations], label="observed", color=:white)
    scatter!([observation_locations[end][1]], [observation_locations[end][2]], label="latest", color=:red)
    plot!(title="Ground truth")
end

function plot_2d_gp_with_observations(gp, observation_locations, bounds)
    xi_range = range(bounds[1][1], bounds[1][2], length=100)
    x_eval = [[xi, yi] for xi in xi_range for yi in xi_range]
    posterior_gpx = gp(x_eval, 0.0) # Noiseless posterior
    posterior_mean = mean(posterior_gpx)
    posterior_var = var(posterior_gpx)

    # Create subplots
    p1 = contourf(xi_range, xi_range, posterior_mean, levels=10, color=:viridis, cbar=true, size=(300, 300))
    scatter!([xi[1] for xi in observation_locations], [xi[2] for xi in observation_locations], label="observed", color=:white)
    scatter!([observation_locations[end][1]], [observation_locations[end][2]], label="latest", color=:red)
    plot!(title="GP posterior mean")

    p2 = contourf(xi_range, xi_range, posterior_var, levels=10, color=:cividis, cbar=true, size=(300, 300))
    scatter!([xi[1] for xi in observation_locations], [xi[2] for xi in observation_locations], label="observed", color=:white)
    scatter!([observation_locations[end][1]], [observation_locations[end][2]], label="latest", color=:red)
    plot!(title="GP posterior variance")

    plot(p1, p2, layout=(1, 2), size=(600, 300))
end