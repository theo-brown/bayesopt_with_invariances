import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tol_colors

# Use TOL colors
params = {
        'figure.figsize': (7, 2),
        'axes.labelsize': 16, #10,
        'xtick.labelsize': 14, #8,
        'ytick.labelsize': 14, #8,
        'axes.titlesize': 16,
        'axes.titlepad': 5
}
plt.rcParams.update(params)
palette = list(tol_colors.tol_cset('bright'))

kernels = ["standard", "3_block_permutation_invariant", "3_block_permutation_augmented", "2_block_permutation_invariant", "2_block_permutation_augmented", "permutation_invariant"]
# kernel_labels = ["Standard", "3-block\ninvariant", "3-block\naugmented", "2-block\ninvariant", "2-block\naugmented", "Fully\ninvariant"]
kernel_labels = ["Standard", "3-block inv.", "3-block aug.", "2-block inv.", "2-block aug.", "Fully inv."]

colors = [palette[0], palette[2], palette[2], palette[3], palette[3], palette[1]]

plt.figure(figsize=(7, 2))
with h5py.File("experiments/synthetic/data/time_benchmark_results.h5", "r") as h5:
    times = {k: h5[k][:]*1e3 for k in kernels}
    bplot = plt.boxplot([times[k] for k in kernels], labels=kernel_labels, patch_artist=True, showfliers=False)
    for patch, color, kernel in zip(bplot['boxes'], colors, kernels):
        patch.set_facecolor(color)
        if "augmented" in kernel:
            patch.set_facecolor("white")
            patch.set_hatch("////")
            patch._hatch_color = matplotlib.colors.to_rgba(color)
            patch.stale = True
            
    for median in bplot['medians']:
        median.set_color('black')

plt.gca().set_xticklabels(kernel_labels, rotation=30, ha='right', rotation_mode='anchor')
plt.title("Time to fit (ms)")
# plt.tight_layout()
plt.savefig("experiments/synthetic/figures/time_benchmark.pdf", bbox_inches="tight")