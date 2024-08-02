import torch.utils.benchmark as benchmark
from invariant_kernel import InvariantKernel
from transformation_groups import permutation_group, block_permutation_group
import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from synthetic_objective import create_synthetic_objective
import argparse 
import h5py
    
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()
device = torch.device(args.device)

# Options
n_points_to_fit = 100
n_seeds = 100
n_runs_per_seed = 64

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
            
            # Seed RNG
            torch.manual_seed(seed)
            # Generate training data
            x = torch.rand(n_points_to_fit, 6, device=device, dtype=torch.float64)
            y = (f(x) + 0.01*torch.randn(n_points_to_fit, device=device)).to(dtype=torch.float64).unsqueeze(-1)
                        
            # Define the model
            model = SingleTaskGP(x, y, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            """,
            globals={'seed': seed, 'f': f, 'n_points_to_fit': n_points_to_fit, 'kernel': kernel, 'device': device},
        ).timeit(n_runs_per_seed)
        
        times[i] = result.times[0]
    
    with h5py.File("benchmark_results.h5", "a") as h5:
        h5[label] = times.detach().cpu().numpy()

    print("Done.")