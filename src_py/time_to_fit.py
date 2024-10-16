import argparse

import h5py
import torch
import torch.utils.benchmark as benchmark
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
n_points_to_fit = 64
n_seeds = 100
n_runs_per_seed = 100
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

# Run benchmarks
for label, kernel in kernels.items():
    print("Benchmarking kernel", label)
    times = torch.empty(n_seeds, dtype=torch.float64)
    for i, seed in enumerate(range(n_seeds)):
        result = benchmark.Timer(
            stmt="fit_gpytorch_mll(mll)",
            setup="""
            import torch 
            from botorch.models import SingleTaskGP
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from botorch.fit import fit_gpytorch_mll
            from transformation_groups import block_permutation_group
            
            # Seed RNG
            torch.manual_seed(seed)
            # Generate training data
            x = torch.rand(n_points_to_fit, d, device=device, dtype=torch.float64)
            # Augment, if required
            if "augmented" in label:
                x_augmented = block_permutation_group(x, int(label[0])).reshape(-1, d)
                x = torch.cat([x, x_augmented], dim=0)
            y = (f(x) + 0.01*torch.randn(x.shape[0], device=device)).to(dtype=torch.float64).unsqueeze(-1)
                        
            # Define the model
            model = SingleTaskGP(x, y, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            """,
            globals={'seed': seed, 'f': f, 'n_points_to_fit': n_points_to_fit, 'kernel': kernel, 'device': device, 'label': label, 'd': d},
        ).timeit(n_runs_per_seed)
        
        times[i] = result.times[0]
    
    with h5py.File("experiments/synthetic/data/benchmark_results.h5", "a") as h5:
        h5[label] = times.detach().cpu().numpy()

    print("Done.")