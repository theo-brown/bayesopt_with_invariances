import torch
import gpytorch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import warnings
from botorch.exceptions import InputDataWarning

warnings.filterwarnings("ignore", category=InputDataWarning)

def create_synthetic_objective(d, kernel, seed, n_initial_points, jitter=1e-6, device=None):
    """Create a synthetic objective function, sampled from a GP with the given kernel.
    
    Parameters
    ----------
    d : int
        The dimension of the input space.
    kernel : gpytorch.kernels.Kernel
        The kernel to use for the GP.
    seed : int
        The seed for the random number generator. Set this to a different value to get a different function.
    n_initial_points : int
        The number of initial points to sample from the GP.
    jitter : float
        The amount of jitter to add to the covariance matrix.
    device : torch.device, optional
        The device to use for the GP. If None, will use the CPU.
    
    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The objective function.
    """
    if device is None:
        device_list = []
    else:
        device_list = [device]
           
    with torch.random.fork_rng(devices=device_list):
        torch.manual_seed(seed)
        
        # Generate samples from a random function
        x = torch.rand(n_initial_points, d)
        mean = gpytorch.means.ZeroMean()
        prior = gpytorch.distributions.MultivariateNormal(mean(x), kernel(x))
        y = prior.sample() # For some reason this breaks if we are not on the CPU

        # Fit a GP to the samples
        true_gp = SingleTaskGP(
            x.to(device=device, dtype=torch.float64),
            y.unsqueeze(-1).to(device=device, dtype=torch.float64),
            jitter*torch.ones_like(
                y.unsqueeze(-1),
                device=device,
                dtype=torch.float64
            ),
            covar_module=kernel
        )
        mll = ExactMarginalLogLikelihood(true_gp.likelihood, true_gp)
        fit_gpytorch_mll(mll)

        # Use the mean of that GP as our objective function
        def f(x):
            return true_gp(x).mean.detach()
            
        return f
