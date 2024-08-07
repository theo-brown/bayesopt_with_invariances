import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tol_colors

# Use TOL colors
palette = list(tol_colors.tol_cset('bright'))

kernels = ["standard", "3_block_permutation_invariant", "3_block_permutation_augmented", "2_block_permutation_invariant", "2_block_permutation_augmented", "permutation_invariant"]
kernel_labels = ["Standard", "3-block\ninvariant", "3-block\naugmented", "2-block\ninvariant", "2-block\naugmented", "Fully\ninvariant"]
colors = [palette[0], palette[2], palette[2], palette[3], palette[3], palette[1]]

f2 = plt.figure(figsize=(7, 2))
with h5py.File("experiments/synthetic/data/memory_benchmark_results.h5", "r") as h5:
    plt.figure()
    allocated = {k: h5[k]["allocated"][:] / (1024**3) for k in kernels}
    allocated_plot = plt.boxplot([allocated[k] for k in kernels], labels=kernel_labels, patch_artist=True, showfliers=False)
    for patch, color, kernel in zip(allocated_plot['boxes'], colors, kernels):
        patch.set_facecolor(color)
        if "augmented" in kernel:
            patch.set_facecolor("white")
            patch.set_hatch("////")
            patch._hatch_color = matplotlib.colors.to_rgba(color)
            patch.stale = True
            
    for median in allocated_plot['medians']:
        median.set_color('black')
        
    plt.xlabel("Kernel", fontsize=14)
    plt.ylabel("GPU memory\n(GiB)", fontsize=14)
    plt.savefig("experiments/synthetic/figures/allocated_memory_benchmark.pdf", bbox_inches="tight")
    
    plt.figure()
    reserved = {k: h5[k]["reserved"][:] / (1024**3) for k in kernels}
    reserved_plot = plt.boxplot([reserved[k] for k in kernels], labels=kernel_labels, patch_artist=True, showfliers=False)
    for patch, color, kernel in zip(reserved_plot['boxes'], colors, kernels):
        patch.set_facecolor(color)
        if "augmented" in kernel:
            patch.set_facecolor("white")
            patch.set_hatch("////")
            patch._hatch_color = matplotlib.colors.to_rgba(color)
            patch.stale = True
            
    for median in reserved_plot['medians']:
        median.set_color('black')

    plt.xlabel("Kernel", fontsize=14)
    plt.ylabel("GPU memory\n(GiB)", fontsize=14)
    plt.savefig("experiments/synthetic/figures/reserved_memory_benchmark.pdf", bbox_inches="tight")