import argparse

import matplotlib.pyplot as plt
import torch
from synthetic_experiments import get_kernel
from synthetic_objective import create_synthetic_objective

# Set font sizes
params = {'legend.fontsize': 10,
          'figure.figsize': (4, 3),
          'axes.labelsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'axes.titlepad': 5}
titlefontsize = 12
plt.rcParams.update(params)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("objective", type=str, choices=["PermInv-2D", "QuasiPermInv-2D-0.01", "QuasiPermInv-2D-0.05",  "QuasiPermInv-2D-0.1"])
args = parser.parse_args()

# Create synthetic objective
if args.objective == "PermInv-2D":
    f = create_synthetic_objective(
        d=2,
        kernel=get_kernel("permutation_invariant", device=torch.device("cpu"), dtype=torch.float64),
        seed=19,
        n_initial_points=64,
    )
    output_filename = "experiments/synthetic/figures/perminv2d.pdf"
    title = "PermInv-2D"
elif args.objective == "QuasiPermInv-2D-0.01":
    f = create_synthetic_objective(
        d=2,
        kernel=get_kernel("quasi_permutation_invariant", device=torch.device("cpu"), dtype=torch.float64, noninvariant_scale=0.01),
        seed=19,
        n_initial_points=64,
    )
    output_filename = "experiments/synthetic/figures/quasiperminv2d_0.01.pdf"
    title = "Objective"
elif args.objective == "QuasiPermInv-2D-0.05":
    f = create_synthetic_objective(
        d=2,
        kernel=get_kernel("quasi_permutation_invariant", device=torch.device("cpu"), dtype=torch.float64, noninvariant_scale=0.05),
        seed=19,
        n_initial_points=64
    )
    output_filename = "experiments/synthetic/figures/quasiperminv2d_0.05.pdf"
    title="Objective"
elif args.objective == "QuasiPermInv-2D-0.1":
    f = create_synthetic_objective(
        d=2,
        kernel=get_kernel("quasi_permutation_invariant", device=torch.device("cpu"), dtype=torch.float64, noninvariant_scale=0.1),
        seed=19,
        n_initial_points=64
    )
    output_filename = "experiments/synthetic/figures/quasiperminv2d_0.1.pdf"
    title = "Objective"
else:
    raise ValueError(f"Unknown objective: {args.objective}")
 
# Compute the objective
x = torch.linspace(0, 1, 100, device=torch.device("cpu"), dtype=torch.float64)
xx, yy = torch.meshgrid(x, x)
z = f(torch.stack([xx.flatten(), yy.flatten()], dim=-1)).reshape(xx.shape)

# Plot 
plt.figure()
plt.title(title)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.contourf(xx, yy, z, levels=20, cmap="viridis")
plt.colorbar()
plt.gca().set_aspect("equal")
plt.savefig(output_filename, bbox_inches="tight")
