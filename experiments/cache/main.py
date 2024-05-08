from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
import torch
import numpy as np
import argparse
from invariantkernels import InvariantKernel
from itertools import permutations
import h5py

from hrtsim import HRTSim
from mallows import MallowsKernel
from optim import maximise_acqf

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=int, default=2.0)
parser.add_argument("--invariant", type=bool, default=False)
args = parser.parse_args()
label = "invariant" if args.invariant else "standard"

# Random seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Torch settings
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32

# Cache model settings
n_symbols = 50
program_size = 5000
symbols = np.array(range(n_symbols))
program = np.random.choice(symbols, program_size)
tcm_size = 10
cache_model = HRTSim(
    plog=np.random.choice(symbols, program_size),
    tcm_size=tcm_size,
    l1_size=20,
    timescale=1,
    l1_latency=0.1,
    ram_latency=1.5,
)

# Optimization settings
n_restarts = 8
iterations_per_restart = 256
points_per_iteration = 128
n_initial_samples = 1024
objective = (
    lambda x: torch.tensor(-cache_model(x.squeeze().detach().cpu().numpy()))
    .reshape(1, 1)
    .to(device=x.device, dtype=x.dtype)
)

# BayesOpt settings
n_iterations = 256

# GP settings
if args.invariant:
    tcm_indices = torch.arange(tcm_size, device=device, dtype=dtype)
    non_tcm_indices = torch.arange(tcm_size, n_symbols, device=device, dtype=dtype)

    def tcm_permutation(x: torch.Tensor) -> torch.Tensor:
        permuted_indices = [
            torch.cat([torch.tensor(list(permuted_tcm_indices)), non_tcm_indices]).to(device=device, dtype=dtype)
            for permuted_tcm_indices in permutations(tcm_indices)
        ]
        permuted_x = x[..., permuted_indices]
        # Reorder to (..., G, n, d)
        dim_indices = list(range(permuted_x.dim()))
        dim_indices[-2], dim_indices[-3] = dim_indices[-3], dim_indices[-2]
        return permuted_x.permute(*dim_indices)

    kernel = InvariantKernel(
        base_kernel=MallowsKernel(nu=1.0),
        transformations=tcm_permutation,
        is_isotropic=False,
        is_group=True,
    )
else:
    kernel = MallowsKernel(nu=1.0)

# Initialisation
x = torch.randperm(n_symbols).to(device=device, dtype=dtype).unsqueeze(0)
y = objective(x[0].squeeze())
gp = SingleTaskGP(
    x,
    y,
    covar_module=kernel,
)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Main loop
for i in range(n_iterations):
    # Optimize the acquisition function
    candidate, acqf_value = maximise_acqf(
        acqf=UpperConfidenceBound(gp, beta=args.beta),
        # acqf=PosteriorStandardDeviation(gp),
        d=n_symbols,
        n_restarts=n_restarts,
        iterations_per_restart=iterations_per_restart,
        points_per_iteration=points_per_iteration,
        n_initial_samples=n_initial_samples,
        device=device,
        dtype=dtype,
    )

    # Evaluate the objective function
    y_candidate = objective(candidate)

    # Update the data
    x = torch.vstack([x, candidate]).to(device=device, dtype=dtype)
    y = torch.vstack([y, y_candidate]).to(device=device, dtype=dtype)

    # Update the model with the new data
    gp = SingleTaskGP(
        x,
        y,
        covar_module=kernel,
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

# Save the results
with h5py.File(f"{label}_results.h5", "w") as h5:
    h5["x"] = x
    h5["y"] = y
