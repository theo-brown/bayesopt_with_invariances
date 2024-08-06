import argparse
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial

import gpytorch
import h5py
import torch
import torch.multiprocessing.pool
from botorch.acquisition import (
    PosteriorMean,
    PosteriorStandardDeviation,
    UpperConfidenceBound,
)
from botorch.exceptions import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from invariant_kernel import InvariantKernel
from synthetic_objective import create_synthetic_objective
from transformation_groups import (
    block_permutation_group,
    cyclic_group,
    permutation_group,
)


def get_kernel(label, device, dtype, **kwargs):
    base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
    # Fix lengthscale
    base_kernel.lengthscale = torch.tensor([0.12], device=device, dtype=dtype)
    base_kernel.raw_lengthscale.requires_grad = False 
    # Select correct invariance
    if label == "permutation_invariant":
        return InvariantKernel(
            base_kernel=base_kernel,
            transformations=permutation_group,
        )
    elif label == "cyclic_invariant":
        return InvariantKernel(
            base_kernel=base_kernel,
            transformations=cyclic_group,
        )
    elif label == "2_block_permutation_invariant":
        return InvariantKernel(
            base_kernel=base_kernel,
            transformations=lambda x: block_permutation_group(x, 2),
        )
    elif label == "3_block_permutation_invariant":
        return InvariantKernel(
            base_kernel=base_kernel,
            transformations=lambda x: block_permutation_group(x, 3),
        )
    elif label == "quasi_permutation_invariant":
        invariant_base_kernel = gpytorch.kernels.ScaleKernel(
            InvariantKernel(
                base_kernel=base_kernel,
                transformations=permutation_group,
            )
        )
        invariant_base_kernel.raw_outputscale.requires_grad = False
        
        noninvariant_base_kernel = deepcopy(base_kernel)
        noninvariant_base_kernel = gpytorch.kernels.ScaleKernel(noninvariant_base_kernel)
        if kwargs.get("noninvariant_scale", None) is not None:
            noninvariant_base_kernel.outputscale = torch.tensor([kwargs["noninvariant_scale"]], device=device, dtype=dtype)
            noninvariant_base_kernel.raw_outputscale.requires_grad = False
        else:
            noninvariant_base_kernel.outputscale = torch.tensor([0.01], device=device, dtype=dtype)
        invariant_base_kernel.outputscale = 1 - noninvariant_base_kernel.outputscale            
        return invariant_base_kernel + noninvariant_base_kernel
    elif label in ["augmented", "standard"]:
        return base_kernel
    else:
        raise ValueError(f"Unknown kernel {label}")

@dataclass
class RunConfig:
    # Objective settings
    objective_kernel: str   # The kernel of the true function
    objective_n_init: str   # Number of points to generate the true function with
    objective_seed: int     # Seed for generating the true function
    noise_var: float        # Noise variance of the observations
    d: int                  # Dimensionality of the problem
    
    # BO settings
    seed: int               # Seed for the run
    eval_kernel: str        # The kernel to run BO with
    acqf: str               # The acquisition function to use
    n_steps: int            # Number of steps to run BO for
    
    # Output settings
    output_file: str        # The file to save the results to
    output_group: str       # The group in the file to save the results to
    
    # Torch settings
    device: torch.device   
    dtype: torch.dtype = torch.float64

    # Optional settings
    objective_kernel_kwargs: dict = field(default_factory=lambda: {})  # Additional arguments for the objective kernel
    eval_kernel_kwargs: dict  = field(default_factory=lambda: {})  # Additional arguments for the evaluation kernel
    learn_noise: bool = False  # Whether to learn the noise variance
    

def run(lock: torch.multiprocessing.Lock, run_config: RunConfig):
    print(f"Running {run_config.output_group} on {run_config.device}")
    torch.manual_seed(run_config.seed)
    
    # Setup
    bounds = torch.tensor([[0., 1.] for _ in range(run_config.d)], device=run_config.device, dtype=run_config.dtype).T
    kernel = get_kernel(run_config.eval_kernel, run_config.device, run_config.dtype, **run_config.eval_kernel_kwargs)
    
    # Generate objective function
    print("Generating objective...")
    f = create_synthetic_objective(
        d=run_config.d,
        kernel=get_kernel(run_config.objective_kernel, run_config.device, run_config.dtype, **run_config.objective_kernel_kwargs),
        seed=run_config.objective_seed,
        n_initial_points=run_config.objective_n_init,
        device=run_config.device
    )
    print("Done.")
    
    def f_noisy(x):
        return f(x) + run_config.noise_var*torch.randn(1, device=run_config.device, dtype=run_config.dtype)
    
    if run_config.acqf == "ucb":
        acqf = UpperConfidenceBound
        acqf_kwargs = {"beta": 2.0}
        reporting_rule = "latest"
    elif run_config.acqf == "mvr":
        acqf = PosteriorStandardDeviation
        acqf_kwargs = {}
        reporting_rule = "max_posterior_mean"
    else:
        raise ValueError(f"Unknown acqf {run_config.acqf}")
        
    # Initial observation
    train_x = torch.rand(1, run_config.d, device=run_config.device, dtype=run_config.dtype)
    train_y = f_noisy(train_x)
    
    if run_config.eval_kernel == "augmented":
        if run_config.objective_kernel == "permutation_invariant":
            transformation_group = permutation_group
        elif run_config.objective_kernel == "cyclic_invariant":
            transformation_group = cyclic_group
        else:
            raise ValueError(f"Unknown objective kernel {run_config.objective_kernel} for selecting data augmentation.")
        
        def augment(train_x, train_y, new_x, new_y):
            new_x_transformed = transformation_group(new_x).view(-1, run_config.d)
            new_y_transformed = torch.stack([new_y for _ in new_x_transformed])
            return torch.cat([train_x, new_x_transformed]), torch.cat([train_y, new_y_transformed])
        
        train_x, train_y = augment(torch.tensor([]), torch.tensor([]), train_x, train_y)        
    
    # Create arrays to store reported values in
    reported_x = torch.empty((run_config.n_steps, run_config.d), device=run_config.device, dtype=run_config.dtype)
    reported_f = torch.empty((run_config.n_steps,), device=run_config.device, dtype=run_config.dtype)

    for i in range(run_config.n_steps):
        # Update GP with training data
        if run_config.learn_noise:
            model = SingleTaskGP(
                train_x,
                train_y.unsqueeze(-1),
                covar_module=kernel,
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        else:
            model = SingleTaskGP(
                train_x,
                train_y.unsqueeze(-1),
                run_config.noise_var*torch.ones_like(train_y.unsqueeze(-1), device=run_config.device, dtype=run_config.dtype), 
                covar_module=kernel,
            )
            
        # Maximise acqf
        next_x, _ = optimize_acqf(
            acqf(model, **acqf_kwargs),
            bounds,
            q=1,
            num_restarts=8,
            raw_samples=1024,    
        )
        # Make observation
        next_y = f_noisy(next_x)

        # Update training data        
        if run_config.eval_kernel == "augmented":
            train_x, train_y = augment(train_x, train_y, next_x, next_y)
        else:
            train_x = torch.cat([train_x, next_x])
            train_y = torch.cat([train_y, next_y])            
        
        # Report
        if reporting_rule == "latest":
            next_reported_x = next_x 
        elif reporting_rule == "max_posterior_mean":
            next_reported_x, _ = optimize_acqf(
                PosteriorMean(model),
                bounds,
                q=1,
                num_restarts=8,
                raw_samples=1024,
            )
        else:
            raise ValueError(f"Unknown reporting rule {reporting_rule}")
        # Observe true function value
        next_reported_f = f(next_reported_x)
        # Save
        reported_x[i] = next_reported_x.squeeze()
        reported_f[i] = next_reported_f
        
        print(f"{run_config.output_group} [{i+1}/{run_config.n_steps}]: {next_reported_f.item()}")
                
    # Save to file 
    lock.acquire()
    try:
        print(f"Saving output from {run_config.output_group}...")
        with h5py.File(run_config.output_file, 'a') as h5:
            h5[f"{run_config.output_group}/observed_x"] = train_x.detach().cpu().numpy()
            h5[f"{run_config.output_group}/observed_y"] = train_y.detach().cpu().numpy()
            h5[f"{run_config.output_group}/reported_x"] = reported_x.detach().cpu().numpy()
            h5[f"{run_config.output_group}/reported_f"] = reported_f.detach().cpu().numpy()
    finally:
        lock.release()
        

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("objective", type=str, choices=["PermInv-2D", "CyclInv-3D", "PermInv-6D", "QuasiPermInv-2D-0.01", "QuasiPermInv-2D-0.05",  "QuasiPermInv-2D-0.1"])
    parser.add_argument("acqf", type=str, choices=["ucb", "mvr"])
    parser.add_argument("--devices", type=str, nargs="*", default=[None])
    parser.add_argument("--n_processes", type=int, default=1)
    args = parser.parse_args()

    # Experiment setup
    if args.objective == "PermInv-2D":
        objective_kernel = "permutation_invariant"
        objective_n_init = 64
        objective_seed = 19
        noise_var = 0.01
        d = 2
        repeats = 32
        eval_kernels = ["standard", "permutation_invariant", "augmented"]
        acqf = args.acqf
        n_steps = [128, 128, 128]
        output_file = f"experiments/synthetic/data/perminv2d_{acqf}.h5"
        objective_kernel_kwargs = {}
        eval_kernel_kwargs = {}
    elif args.objective == "CyclInv-3D":
        objective_kernel = "cyclic_invariant"
        objective_n_init = 256
        objective_seed = 2
        noise_var = 0.01
        d = 3
        repeats = 32
        eval_kernels = ["standard", "cyclic_invariant", "augmented"]
        acqf = args.acqf
        n_steps = [256, 256]
        output_file = f"experiments/synthetic/data/cyclinv3d_{acqf}.h5"
        objective_kernel_kwargs = {}
        eval_kernel_kwargs = {}
    elif args.objective == "PermInv-6D":
        objective_kernel = "permutation_invariant"
        objective_n_init = 512
        objective_seed = 0
        noise_var = 0.01
        d = 6
        repeats = 32
        eval_kernels = ["standard", "3_block_permutation_invariant", "2_block_permutation_invariant", "permutation_invariant", "augmented"]
        acqf = args.acqf
        n_steps = [640, 640, 640, 200, 200, 640]
        output_file = f"experiments/synthetic/data/perminv6d_{acqf}.h5"
        objective_kernel_kwargs = {}
        eval_kernel_kwargs = {}
    elif args.objective == "QuasiPermInv-2D-0.01":
        objective_kernel = "quasi_permutation_invariant"
        objective_n_init = 64
        objective_seed = 6
        noise_var = 0.01
        learn_noise = False
        d = 2
        repeats = 16
        eval_kernels = ["standard", "permutation_invariant", "quasi_permutation_invariant"]
        acqf = args.acqf
        n_steps = [128, 128, 128]
        output_file = f"experiments/synthetic/data/quasiperminv2d_0.01_{acqf}.h5"
        objective_kernel_kwargs = {"noninvariant_scale": 0.01}
        eval_kernel_kwargs = {"noninvariant_scale": 0.01}
        if acqf == "ucb":
            acqf_kwargs = {"beta": 10.0}
    elif args.objective == "QuasiPermInv-2D-0.05":
        objective_kernel = "quasi_permutation_invariant"
        objective_n_init = 64
        objective_seed = 19 # 6 for 3d
        noise_var = 0.01
        learn_noise = False
        d = 2
        repeats = 16
        eval_kernels = ["standard", "permutation_invariant", "quasi_permutation_invariant"]
        acqf = args.acqf
        n_steps = [128, 128, 128]
        output_file = f"experiments/synthetic/data/quasiperminv2d_0.05_{acqf}.h5"
        objective_kernel_kwargs = {"noninvariant_scale": 0.05}
        eval_kernel_kwargs = {"noninvariant_scale": 0.05}
        if acqf == "ucb":
            acqf_kwargs = {"beta": 10.0}
    elif args.objective == "QuasiPermInv-2D-0.1":
        objective_kernel = "quasi_permutation_invariant"
        objective_n_init = 64
        objective_seed = 19
        noise_var = 0.01
        learn_noise = False
        d = 2
        repeats = 16
        eval_kernels = ["standard", "permutation_invariant", "quasi_permutation_invariant"]
        acqf = args.acqf
        n_steps = [128, 128, 128]
        output_file = f"experiments/synthetic/data/quasiperminv2d_0.1_{acqf}.h5"
        objective_kernel_kwargs = {"noninvariant_scale": 0.1}
        eval_kernel_kwargs = {"noninvariant_scale": 0.1}
        if acqf == "ucb":
            acqf_kwargs = {"beta": 10.0}
        
    # Torch setup
    warnings.filterwarnings("ignore", category=InputDataWarning)
    torch.multiprocessing.set_start_method('spawn')
    devices = [
        torch.device(device)
        if device is not None and device != "cpu"
        else None 
        for device in args.devices
    ]
    # Distribute tasks across devices
    # This is a round-robin method
    devices = [devices[i % len(devices)] for i in range(repeats)]
    
    # Initialise the file
    with h5py.File(output_file, 'w') as h5:
        h5.attrs["objective"] = args.objective
        h5.attrs["acqf"] = args.acqf
        h5.attrs["devices"] = [d if d is not None else '' for d in args.devices]
        h5.attrs["repeats"] = repeats
        h5.attrs["objective_kernel"] = objective_kernel
        h5.attrs["objective_n_init"] = objective_n_init
        h5.attrs["objective_seed"] = objective_seed
        h5.attrs["noise_var"] = noise_var
        h5.attrs["d"] = d
        h5.attrs["eval_kernels"] = eval_kernels
        h5.attrs["acqf"] = acqf
        h5.attrs["n_steps"] = n_steps
        h5.attrs["learn_noise"] = learn_noise
        if acqf == "ucb":
            h5.attrs["ucb_beta"] = acqf_kwargs["beta"]
        for eval_kernel in eval_kernels:
            h5.create_group(eval_kernel)
    
    # Create experiment configs
    run_configs = [
        RunConfig(
            objective_kernel=objective_kernel,
            objective_n_init=objective_n_init,
            objective_seed=objective_seed,
            noise_var=noise_var,
            d=d,
            seed=repeat,
            eval_kernel=eval_kernel,
            acqf=acqf,
            n_steps=n_steps_i,
            output_file=output_file,
            output_group=f"{eval_kernel}/{repeat}",
            device=device, 
            objective_kernel_kwargs=objective_kernel_kwargs,
            eval_kernel_kwargs=eval_kernel_kwargs,           
        )
        for eval_kernel, n_steps_i, device in zip(eval_kernels, n_steps, devices)
        for repeat in range(repeats)
    ]
    
    # Run experiments
    manager = torch.multiprocessing.Manager()
    lock = manager.Lock()
    eval_fn = partial(run, lock)
    with torch.multiprocessing.Pool(processes=args.n_processes) as pool:
        pool.map(eval_fn, run_configs, chunksize=len(run_configs) // args.n_processes)
