import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tol_colors

# Set font sizes
params = {'legend.fontsize': 14, #10,
          'figure.figsize': (4, 3),
          'axes.labelsize': 14, #10,
          'xtick.labelsize': 12, #8,
          'ytick.labelsize': 12, #8,
          'axes.titlepad': 5}
titlefontsize = 14 #12
plt.rcParams.update(params)

# Use TOL colors
palette = list(tol_colors.tol_cset('bright'))

if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("objective", type=str, choices=["perminv2d", "cyclinv3d", "perminv6d", "quasiperminv2d_0.01", "quasiperminv2d_0.05", "quasiperminv2d_0.1"])
    parser.add_argument("acqf", type=str, choices=["ucb", "mvr"])
    args = parser.parse_args()
    
    # Set up
    n_legend_cols = 1
    if args.objective == "perminv2d":
        objective = "PermInv-2D"
        kernels = ["standard", "constrained", "permutation_invariant"]
        kernel_labels = ["Standard", "Constrained", "Invariant"]
        colors = [palette[0], palette[5], palette[1]]
        # f_opt = 1.707
        f_opt = 1.20
        n_repeats = 32
        xlim = [0, 128]
        if args.acqf == "ucb":
            ylim = [0, 50]
        elif args.acqf == "mvr":
            ylim = [-0.05, 1.25]
    elif args.objective == "cyclinv3d":
        objective = "CyclInv-3D"
        kernels = ["standard", "cyclic_invariant"]
        kernel_labels = ["Standard", "Invariant"]
        colors = [palette[0], palette[1]]
        f_opt = 1.608108
        n_repeats = 32
        xlim = [0, 256]
        if args.acqf == "ucb":
            ylim = [0, None]
        elif args.acqf == "mvr":
            ylim = [-0.05, 1.25]
            # ylim=[None, None]
    elif args.objective == "perminv6d":
        objective = "PermInv-6D"
        kernels = ["standard", "constrained", "3_block_permutation_invariant", "2_block_permutation_invariant", "permutation_invariant"]
        kernel_labels = ["Standard", "Constrained", "3-block inv.", "2-block inv.", "Fully inv."]
        colors = [palette[0], palette[5], palette[2], palette[3], palette[1]]
        f_opt = 1.1768
        n_repeats = 32
        xlim = [0, 640]
        if args.acqf == "ucb":
            ylim = [0, 800]
        elif args.acqf == "mvr":
            ylim = [-0.05, 1.75]
            n_legend_cols = 2
    elif args.objective == "quasiperminv2d_0.01":
        objective = "QuasiPermInv-2D-0.01"
        kernels = ["standard", "permutation_invariant", "quasi_permutation_invariant"]
        kernel_labels = ["Standard", "Permutation invariant", "Quasi-invariant (truth)"]
        colors = [palette[0], palette[2], palette[1]]
        f_opt = 2.039
        n_repeats = 16
        xlim = [0, 128]
        if args.acqf == "ucb":
            ylim = [0, 70]
        elif args.acqf == "mvr":
            ylim = [-0.05, 2.2]
    elif args.objective == "quasiperminv2d_0.05":
        objective = "QuasiPermInv-2D-0.05"
        kernels = ["standard", "permutation_invariant", "quasi_permutation_invariant"]
        kernel_labels = ["Standard", "Permutation invariant", "Quasi-invariant (truth)"]
        colors = [palette[0], palette[2], palette[1]]
        f_opt = 1.325
        n_repeats = 16
        xlim = [0, 128]
        if args.acqf == "ucb":
            ylim = [0, 70]
        elif args.acqf == "mvr":
            ylim = [-0.05, 1.4]
    elif args.objective == "quasiperminv2d_0.1":
        objective = "QuasiPermInv-2D-0.1"
        kernels = ["standard", "permutation_invariant", "quasi_permutation_invariant"]
        kernel_labels = ["Standard", "Permutation invariant", "Quasi-invariant (truth)"]
        colors = [palette[0], palette[2], palette[1]]
        f_opt = 1.44
        n_repeats = 16
        xlim = [0, 128]
        if args.acqf == "ucb":
            ylim = [0, 70]
        elif args.acqf == "mvr":
            ylim = [-0.05, 1.5]
        
    if args.acqf == "ucb":
        regret_label = "Cumulative regret"
        legend_loc = "upper left"
    elif args.acqf == "mvr":
        regret_label = "Simple regret"
        legend_loc = "upper right"
    
    # Plot
    figure = plt.figure(figsize=(4, 3))
    for kernel, label, color in zip(kernels, kernel_labels, colors):
        reported_f = []
        with h5py.File(f"experiments/synthetic/data/{args.objective}_{args.acqf}.h5", "r") as h5:
            for i in range(n_repeats):
                print(f"Loading {kernel}/{i}")
                try:
                    reported_f.append(np.reshape(h5[f"{kernel}/{i}/reported_f"][:], -1))
                except KeyError:
                    print(f"KeyError: {kernel}/{i}")
                    continue
        reported_f = np.vstack(reported_f)
        
        if args.acqf == "ucb":
            regret = np.cumsum(f_opt - reported_f, axis=1)
        elif args.acqf == "mvr":
            regret = np.minimum.accumulate(f_opt - reported_f, axis=1)
        
        mean_regret = np.mean(regret, axis=0)
        std_regret = np.std(regret, axis=0)
        plt.plot(mean_regret, label=label, color=color)
        plt.fill_between(np.arange(mean_regret.shape[0]), mean_regret - std_regret, mean_regret + std_regret, alpha=0.3, color=color)
        
    plt.xlabel("Number of evaluations")
    plt.xlim(xlim)
    plt.ylabel(regret_label)
    plt.ylim(ylim)
    plt.legend(ncol=n_legend_cols, borderaxespad=0.1, columnspacing=0.6, handlelength=0.7, handletextpad=0.3, loc=legend_loc)  
    
    if "quasi" in objective.lower():
        plt.title(args.acqf.upper(), fontsize=titlefontsize)
    else:
        plt.title(objective, fontsize=titlefontsize)
    
    plt.rcParams.update(params)
    plt.savefig(f"experiments/synthetic/figures/{args.objective}_{args.acqf}_regret.pdf", bbox_inches="tight")