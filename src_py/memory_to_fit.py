import argparse

import h5py
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from invariant_kernel import InvariantKernel
from synthetic_objective import create_synthetic_objective
from transformation_groups import block_permutation_group, permutation_group

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()
device = torch.device(args.device)

# Options
n_points_to_fit = 200
n_seeds = 10
n_runs_per_seed = 10
d = 6

# Create synthetic objective
f = create_synthetic_objective(
    d=6,
    kernel=InvariantKernel(MaternKernel(nu=2.5), permutation_group),
    seed=0,
    n_initial_points=512,
    device=device,
) 

# Define kernels
kernels = {
    "standard": ScaleKernel(MaternKernel(nu=2.5)),
    "3_block_permutation_augmented": ScaleKernel(MaternKernel(nu=2.5)),
    "2_block_permutation_augmented": ScaleKernel(MaternKernel(nu=2.5)),
    "3_block_permutation_invariant": ScaleKernel(InvariantKernel(MaternKernel(nu=2.5), lambda x: block_permutation_group(x, 3))),
    "2_block_permutation_invariant": ScaleKernel(InvariantKernel(MaternKernel(nu=2.5), lambda x: block_permutation_group(x, 2))),
    "permutation_invariant": ScaleKernel(InvariantKernel(MaternKernel(nu=2.5), permutation_group)),
}

# Define benchmark 
def benchmark_memory(seed, label, kernel):
    # Seed RNG
    torch.manual_seed(seed)
    # Generate training data
    x = torch.rand(n_points_to_fit, d, device=device, dtype=torch.float64)
    # Augment, if required
    if "augmented" in label:
        x_augmented = block_permutation_group(x, int(label[0])).reshape(-1, d)
        x = torch.cat([x, x_augmented], dim=0)
        del x_augmented
    y = (f(x) + 0.01*torch.randn(x.shape[0], device=device)).to(dtype=torch.float64).unsqueeze(-1)
                
    # Define the model
    model = SingleTaskGP(x, y, covar_module=kernel)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    # Reset memory counter
    torch.cuda.reset_peak_memory_stats(device=device)
    # Fit the model
    fit_gpytorch_mll(mll)
    # Get the peak memory usage
    allocated_memory = torch.cuda.max_memory_allocated(device=device)
    reserved_memory = torch.cuda.max_memory_reserved(device=device)
    
    # Clean up
    del x, y, model, mll
    torch.cuda.empty_cache()
    
    return allocated_memory, reserved_memory


# Run benchmarks
for label, kernel in kernels.items():
    print("Benchmarking kernel", label)
    allocated_memory = torch.empty(n_seeds, dtype=torch.float64)
    reserved_memory = torch.empty(n_seeds, dtype=torch.float64)
    for i, seed in enumerate(range(n_seeds)):
        allocated_memory[i], reserved_memory[i] = benchmark_memory(seed, label, kernel)
    
    with h5py.File("experiments/synthetic/data/memory_benchmark_results.h5", "a") as h5:
        h5[label]["allocated"] = allocated_memory.detach().cpu().numpy()
        h5[label]["reserved"] = reserved_memory.detach().cpu().numpy()

    print("Done.")
