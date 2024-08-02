from pathlib import Path 
import h5py 
import numpy as np
import matplotlib.pyplot as plt 
import tol_colors 

root_dir = Path("XXXXXXXXXXXXXX")

directories = [
    root_dir/f"ucb_fixedkernel_{kernel}_n=12_seed={i}"
    for kernel in ["standard", "3blockinvariant"]  
]

palette = list(tol_colors.tol_cset('bright'))
params = {'legend.fontsize': 10,
          'figure.figsize': (6, 3),
          'axes.labelsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'axes.titlepad': 5}
titlefontsize = 12
plt.rcParams.update(params)
standard = []
invariant = []

for directory in directories:
    with h5py.File(directory/"results.h5", "r") as f:
        best_value_per_step = []
        for i, group in enumerate(f):
            if "optimisation_step" not in str(group):
                continue
            # Get the best non-nan value from this step
            new_best_value = np.nanmax(f[group]["objective_values"][:])                                      
            # If it's non-empty, append it to the list
            if new_best_value.size > 0:
                best_value_per_step.append(new_best_value.item())
            # If it's empty but we have made previous observations, append 0 to the list
            elif len(best_value_per_step) > 0:
                best_value_per_step.append(0)
            # Otherwise, we have had no successful observations, so we are still in random sampling mode
            else:
                continue             

        
    if "standard" in str(directory):
        standard.append(best_value_per_step)
    if "invariant" in str(directory):
        invariant.append(best_value_per_step)
            
# Pad arrays with nans to be the same length
for l in [standard, invariant]:
    max_len = max([len(r) for r in l])
    for r in l:
        length_difference = max_len - len(r)
        if length_difference > 0:
            r.extend([np.nan]*length_difference)

# Compute the running max - this corresponds to 'best observed' reporting rule
standard_running_max = [np.fmax.accumulate(r) for r in standard]
invariant_running_max = [np.fmax.accumulate(r) for r in invariant]

# Plot the mean +/- sd
standard_mean = np.nanmean(standard_running_max, axis=0)
standard_std = np.nanstd(standard_running_max, axis=0)
invariant_mean = np.nanmean(invariant_running_max, axis=0)
invariant_std = np.nanstd(invariant_running_max, axis=0)

standard_x = range(0, len(standard_mean))
plt.plot(standard_x, standard_mean, color=palette[0], label="Standard")
# plt.fill_between(standard_x, standard_mean + standard_std, standard_mean - standard_std, facecolor=palette[0], alpha=0.3)

invariant_x = range(0, len(invariant_mean))
plt.plot(invariant_x, invariant_mean, color=palette[1], label="Invariant")
# plt.fill_between(invariant_x, invariant_mean + invariant_std, invariant_mean - invariant_std, facecolor=palette[1], alpha=0.3)
plt.xlim(0, 16)
plt.legend(loc="upper left")
plt.xlabel("Optimisation step")
plt.ylabel("Objective value")
plt.title("Safety factor optimisation", fontsize=titlefontsize)
plt.savefig("safety_factor_progress.pdf", bbox_inches="tight")


def gaussian(x: np.ndarray, mean: float, std: float, height: float = 1.0) -> np.ndarray:
    return height * np.exp(-0.5 * ((x - mean) / std) ** 2)


def sum_of_gaussians(
    x: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    heights: np.ndarray,
) -> np.ndarray:
    return np.sum(
        [gaussian(x, mean, std, height) for mean, std, height in zip(means, stds, heights)],
        axis=0,
    )

def sum_of_gaussians_fixed_width_fixed_height_profile(xrho: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    stds = np.full(len(parameters), 0.05)
    heights = np.full(len(parameters), 1.0)
    y = sum_of_gaussians(xrho, parameters, stds, heights)
    return y

parameters = np.array([0., 0.02, 0.1, 0.2, 0.3])
xrho = np.linspace(0, 1, 150)

plt.plot(xrho, sum_of_gaussians_fixed_width_fixed_height_profile(xrho, parameters), color='k', linestyle='--', label="Total")
for i, p in enumerate(parameters):
    g = gaussian(xrho, p, 0.05, 1)
    # plt.plot(xrho, g, color=palette[i], label='_Launcher {i}')
    plt.fill_between(xrho, np.zeros_like(xrho), g, color=palette[i], label=f'Launcher {i+1}', zorder=len(parameters)-i, alpha=0.7)
plt.xlim(0, 0.6)
plt.ylim(0, None)
plt.legend()
plt.xlabel("Normalised radial position")
plt.ylabel("Normalised power deposition")
plt.savefig("ecrh_profile.pdf", bbox_inches="tight")