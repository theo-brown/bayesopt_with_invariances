from ecrh import sum_of_gaussians_fixed_width_fixed_height_profile
import numpy as np
from jetto_mobo.acquisition import generate_initial_candidates
import torch
import h5py
from pathlib import Path 
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from os import mkdir
from invariant_kernel import InvariantKernel
from transformation_groups import block_permutation_group
import argparse
from botorch.optim import optimize_acqf
from botorch.acquisition import qUpperConfidenceBound
from jetto_mobo import simulation
import jetto_tools
from typing import *
import asyncio
import numpy as np


def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum()


def gaussian(x: np.ndarray, mean: float, std: float, height: float = 1.0) -> np.ndarray:
    return height * np.exp(-0.5 * ((x - mean) / std) ** 2)


def sum_of_gaussians(
    x: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    heights: np.ndarray,
) -> np.ndarray:
    return np.sum(
        [gaussian(x, mean, std, height) for mean, std, height in zip(means, stds, heights)],
        axis=0,
    )


def sum_of_gaussians_fixed_width_fixed_height_profile(xrho: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    stds = np.full(len(parameters), 0.05)
    heights = np.full(len(parameters), 1.0)
    y = sum_of_gaussians(xrho, parameters, stds, heights)
    return y


def soft_hat(
    x: Union[float, np.ndarray],
    x_lower: float = 0,
    y_lower: float = 1e-3,
    x_plateau_start: float = 0,
    x_plateau_end: float = 0,
    x_upper: float = 0,
    y_upper: float = 1e-3,
) -> np.ndarray:
    """
    Smooth top-hat function.

    Passes through (x_lower, y_lower), (x_plateau_start, 1), (x_plateau_end, 1), (x_upper, y_upper).
    Squared exponential decay from 0 to x_plateau_start and from x_plateau_end to infinity, with rate of decay such that y=y_lower at x=x_lower and y=y_upper at x=x_upper.

    Parameters
    ----------
    x : Union[float, np.ndarray]
        Input value
    x_lower : float, optional
        x-value at which y=y_lower (default: 0)
    y_lower : float, optional
        y-value at x=x_lower (default: 1e-3)
    x_plateau_start : float, optional
        x-value at which the plateau starts (default: 0)
    x_plateau_end : float, optional
        x-value at which the plateau ends (default: 0)
    x_upper : float, optional
        x-value at which y=y_upper (default: 0)
    y_upper : float, optional
        y-value at x=x_upper (default: 1e-3)

    Returns
    -------
    np.ndarray
        Smooth objective value
    """
    k_lower = -np.log(y_lower) / np.power(x_lower - x_plateau_start, 2)
    k_upper = -np.log(y_upper) / np.power(x_upper - x_plateau_end, 2)
    return np.piecewise(
        x,
        [
            x < x_plateau_start,
            (x >= x_plateau_start) & (x <= x_plateau_end),
            x > x_plateau_end,
        ],
        [
            lambda x: np.exp(-k_lower * np.power(x - x_plateau_start, 2)),
            1,
            lambda x: np.exp(-k_upper * np.power(x - x_plateau_end, 2)),
        ],
    )


def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return exp / np.sum(exp)


def objective(q: np.ndarray) -> float:
    if np.all(np.isnan(q)):
        return np.nan
    
    q0 = q[0]
    qmin = np.min(q)
    rho = np.linspace(0, 1, 150)
    rho_of_qmin = rho[np.argmin(q)]
    
    # Objective 1: q0 close to qmin
    distance = np.abs(q0 - qmin)
    objective_1 = soft_hat(
        distance,
        x_lower=-1,  # Not used, as 0 < x < 1
        y_lower=1e-3,  # Not used, as 0 < x < 1
        x_plateau_start=0,
        x_plateau_end=0,
        x_upper=2,
        y_upper=0.5,
    )

        
    # Objective 2: qmin close to centre
    objective_2 = soft_hat(
        rho_of_qmin,
        x_lower=-1,  # Not used, as 0 < x < 1
        y_lower=1e-3,  # Not used, as 0 < x < 1
        x_plateau_start=0,
        x_plateau_end=0,
        x_upper=1,
        y_upper=1e-3,
    )
    
    # Objective 3:  qmin in safe region
    objective_3 = soft_hat(
        qmin,
        x_lower=2.2,
        y_lower=0.5,
        x_plateau_start=2.2,
        x_plateau_end=2.5,
        x_upper=3,
        y_upper=0.5,
    )
    
    # Objective 4: q increasing at every radial point
    is_increasing = np.gradient(q) > 0
    objective_4 = soft_hat(
        np.mean(is_increasing),  # Fraction of curve where q is increasing
        x_lower=0,
        y_lower=1e-3,
        x_plateau_start=1,
        x_plateau_end=1,
        x_upper=2,  # Not used, as 0 < x < 1
        y_upper=1e-3,  # Not used, as 0 < x < 1
    )
    
    # Objective 5: q=3 towards edge
    indices = np.nonzero((q >= 3) & (rho >= rho_of_qmin))[0]
    if len(indices) == 0:
        objective_5 = 0
    else:
        radius_of_q_is_3 = rho[indices[0]]
        objective_5 = soft_hat(
            radius_of_q_is_3,
            x_lower=0.5,
            y_lower=0.5,
            x_plateau_start=0.8,
            x_plateau_end=1,
            x_upper=2,  # Not used, as 0 < x < 1
            y_upper=2,  # Not used, as 0 < x < 1
        )
    
    # Objective 6: q=4 towards edge
    indices = np.nonzero((q >= 4) & (rho >= rho_of_qmin))[0]
    if len(indices) == 0:
        objective_6 = 0
    else:
        radius_of_q_is_4 = rho[indices[0]]
        objective_6 = soft_hat(
            radius_of_q_is_4,
            x_lower=0.5,
            y_lower=0.5,
            x_plateau_start=0.8,
            x_plateau_end=1,
            x_upper=2,  # Not used, as 0 < x < 1
            y_upper=2,  # Not used, as 0 < x < 1
        )

    # Weighted sum
    weights = np.array([1., 0.75, 1., 1., 0.5, 0.5])
    normalised_weights = softmax(weights)
    objectives = np.array([objective_1, objective_2, objective_3, objective_4, objective_5, objective_6])
    return objectives @ normalised_weights

def evaluate_q_batch(
    ecrh_parameters_batch: np.ndarray,
    ecrh_function: Callable,
    directory: Path,
    jetto_template: Path,
    jetto_image: Path,
):
    configs = {}

    for i, ecrh_parameters in enumerate(ecrh_parameters_batch):
        # Initialise config object
        config_directory = Path(f"{directory}/candidate_{i}")
        config = simulation.create_config(
            template=jetto_template, directory=config_directory
        )

        # Set the ECRH function
        exfile = jetto_tools.binary.read_binary_file(config.exfile)
        exfile["QECE"][0] = ecrh_function(
            xrho=exfile["XRHO"][0], parameters=ecrh_parameters
        )
        jetto_tools.binary.write_binary_exfile(exfile, config.exfile)

        # Store in dict
        # Currently this is necessary as the JettoTools RunConfig does not store the directory path
        configs[config] = config_directory

    # Run asynchronously in parallel
    batch_output = asyncio.run(
        simulation.run_many(
            jetto_image=jetto_image,
            run_configs=configs,
            timelimit=10800,
        )
    )

    # Parse outputs
    converged_ecrh = []
    converged_q = []
    for results in batch_output:
        if results is not None:
            try:
                profiles = results.load_profiles()
            except:
                print("JETTO output file corrupted.")
                converged_ecrh.append(np.full(150, np.nan))
                converged_q.append(np.full(150, np.nan))
            else:
                converged_ecrh.append(profiles["QECE"][-1])
                converged_q.append(profiles["Q"][-1])
        else:
            print("JETTO failed to converge.")
            converged_ecrh.append(np.full(150, np.nan))
            converged_q.append(np.full(150, np.nan))

    # Compress outputs
    for _, config_directory in configs.items():
        simulation.compress_jetto_dir(config_directory, delete=True)

    return (
        np.array(converged_q),
        np.array(converged_ecrh),
    )

if __name__=="__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--invariant", action='store_true', default=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--beta", type=float, default=2.0)
    args = parser.parse_args()
    invariant = args.invariant
    label = "3blockinvariant" if invariant else "standard"

    # Torch config
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ECRH config
    n_gaussians = 12
    min_value = 0.0
    max_value = 0.5
    bounds = torch.tensor([[min_value, max_value]]*n_gaussians, device=device, dtype=dtype).T
    xrho = np.linspace(0, 1, 150)

    # Kernel config
    if invariant:
        kernel = ScaleKernel(
            InvariantKernel(
                base_kernel=MaternKernel(nu=2.5),
                transformations=lambda x: block_permutation_group(x, 3),
            )
        )
    else:
        kernel = ScaleKernel(MaternKernel(nu=2.5))
        
    def set_model_hyperparameters(model):
        # Likelihood
        # model.likelihood.noise = torch.tensor(1e-4) # Noiseless observations
        # model.likelihood.noise_covar.requires_grad = False
        
        # Kernel
        model.covar_module.outputscale = 0.2
        if isinstance(model.covar_module.base_kernel, InvariantKernel):
            model.covar_module.base_kernel.base_kernel.lengthscale = 0.2
            model.covar_module.base_kernel.base_kernel.raw_lengthscale.requires_grad = False
        else:
            model.covar_module.base_kernel.lengthscale = 0.2
            model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        
        # Mean
        model.mean_module.constant = 0.5 
        model.mean_module.raw_constant.requires_grad = False  

    # Optim config
    batch_size = 10
    n_steps = 24

    # Logging config
    output_dir = Path(f"XXXXXXXXXXXXXXXXXXXXXX")
    output_file = output_dir/"results.h5"

    def save(label, ecrh_parameters, preconverged_ecrh, converged_ecrh, converged_q, objective_values):
        with h5py.File(output_file, 'a') as h5:
            h5[f"{label}/ecrh_parameters"] = ecrh_parameters
            h5[f"{label}/preconverged_ecrh"] = preconverged_ecrh
            h5[f"{label}/converged_ecrh"] = converged_ecrh
            h5[f"{label}/converged_q"] = converged_q
            h5[f"{label}/objective_values"] = objective_values
        
    if output_dir.exists():
        with h5py.File(output_file, 'r') as h5:
            n_previous_steps = len(h5.keys())-1
            print(f"Loading {n_previous_steps} previous steps")
            all_ecrh_parameters = torch.tensor([], device=device, dtype=dtype)
            all_objective_values = torch.tensor([], device=device, dtype=dtype)
            for i in range(0, n_previous_steps+1):
                print(f"Loading optimisation_step_{i}/ecrh_parameters")
                ecrh_parameters = h5[f"optimisation_step_{i}/ecrh_parameters"][:]
                print(f"Loading optimisation_step_{i}/objective_values")
                objective_values = h5[f"optimisation_step_{i}/objective_values"][:]
                all_ecrh_parameters = torch.cat(
                    [
                        all_ecrh_parameters,
                        torch.tensor(ecrh_parameters, device=device, dtype=dtype)
                    ]
                )
                all_objective_values = torch.cat(
                    [
                        all_objective_values,
                        torch.tensor(objective_values, device=device, dtype=dtype)
                    ]
                )
    else:
        mkdir(output_dir)
        n_previous_steps = 0
        
        # Generate initial candidates via Sobol sampling 
        ecrh_parameters = generate_initial_candidates(
                bounds=bounds,
                n=batch_size,
                device=device,
                dtype=dtype,
            )
        converged_q, converged_ecrh = evaluate_q_batch(
            ecrh_parameters_batch=ecrh_parameters.detach().cpu().numpy(),
            ecrh_function=sum_of_gaussians_fixed_width_fixed_height_profile,
            directory=output_dir/"0",
            jetto_template="XXXXXXXXXXXXXXXXXXXXXX",
            jetto_image="XXXXXXXXXXXXXXXXXXXXXX",
        )

        # Compute objective
        objective_values = np.array([
            objective(converged_q[i])
            for i in range(len(converged_q))
        ])

        save(
            label=f"optimisation_step_0",
            ecrh_parameters=ecrh_parameters.detach().cpu().numpy(),
            preconverged_ecrh=np.array([sum_of_gaussians_fixed_width_fixed_height_profile(xrho, p) for p in ecrh_parameters.detach().cpu().numpy()]),
            converged_ecrh=converged_ecrh,
            converged_q=converged_q,
            objective_values=objective_values,
        )

        # Update history
        all_ecrh_parameters = ecrh_parameters
        all_objective_values = torch.tensor(objective_values, device=device, dtype=dtype)

    # BAYESOPT
    for optimisation_step in range(n_previous_steps + 1, n_previous_steps + n_steps + 1):
        print(f"Optimisation step {optimisation_step}")
        
        # Handle failures
        failed_mask = torch.isnan(all_objective_values)
        modified_objective_values = all_objective_values.clone()
        modified_objective_values[failed_mask] = 0.2
        
        # Fit model
        print("Fitting GP...")
        model = SingleTaskGP(
            all_ecrh_parameters,
            modified_objective_values.unsqueeze(-1),
            covar_module=kernel,
        )
        set_model_hyperparameters(model)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        print("Done.")

        # Generate new candidates
        print("Selecting next candidates...")
        new_ecrh_parameters_batch, _ = optimize_acqf(
            qUpperConfidenceBound(model, beta=args.beta),
            bounds=bounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=1024,
            sequential=True,
            options={'max_iter': 256}
        )
        print("Done.")
        
        # Run JETTO
        print("Making observations...")
        converged_q, converged_ecrh = evaluate_q_batch(
            ecrh_parameters_batch=new_ecrh_parameters_batch.detach().cpu().numpy(),
            ecrh_function=sum_of_gaussians_fixed_width_fixed_height_profile,
            directory=output_dir/str(optimisation_step),
            jetto_template="XXXXXXXXXXXXXXXXXXXXXX",
            jetto_image="XXXXXXXXXXXXXXXXXXXXXX",
        )
        print("Done.")
        
        # Compute objective
        objective_values = np.array([
            objective(converged_q[i])
            for i in range(len(converged_q))
        ])

        # Save
        save(
            label=f"optimisation_step_{optimisation_step}",
            ecrh_parameters=new_ecrh_parameters_batch.detach().cpu().numpy(),
            preconverged_ecrh=np.array([sum_of_gaussians_fixed_width_fixed_height_profile(xrho, p) for p in new_ecrh_parameters_batch.detach().cpu().numpy()]),
            converged_ecrh=converged_ecrh,
            converged_q=converged_q,
            objective_values=objective_values,
        )

        # Update history
        all_ecrh_parameters = torch.cat(
            [
                all_ecrh_parameters,
                new_ecrh_parameters_batch
            ]
        )
        all_objective_values = torch.cat(
            [
                all_objective_values,
                torch.tensor(objective_values, device=device, dtype=dtype)
            ]
        )
