import matplotlib.pyplot as plt 
import numpy as np
import h5py 
import tol_colors 

# Use TOL colors
palette = list(tol_colors.tol_cset('bright'))

kernels = ["standard", "3_block_permutation_invariant", "2_block_permutation_invariant", "permutation_invariant"]
kernel_labels = ["Standard", "3-block\ninvariant", "2-block\ninvariant", "Fully\ninvariant"]
colors = [palette[0], palette[2], palette[3], palette[1]]

plt.figure(figsize=(6, 2))
with h5py.File("experiments/synthetic/data/benchmark_results.h5", "r") as h5:
    times = {k: h5[k][:]*1e3 for k in kernels}
    bplot = plt.boxplot([times[k] for k in kernels], labels=kernel_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set_color('black')

plt.xlabel("Kernel", fontsize=14)
plt.ylabel("Time to fit (ms)", fontsize=14)
plt.savefig("benchmark.pdf", bbox_inches="tight")