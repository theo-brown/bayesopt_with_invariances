import torch
from tqdm import tqdm

def swap_ij(x: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Return a version of x with the i-th and j-th elements of the last dimension swapped."""
    x_new = x.clone()
    x_new[..., i], x_new[..., j] = x[..., j], x[..., i]
    return x_new


def pairwise_swaps(x: torch.Tensor) -> torch.Tensor:
    return torch.vstack(
        [swap_ij(x, i, j) for i, j in torch.combinations(torch.arange(x.shape[-1]), 2)]
    )


def random_pairwise_swaps(x: torch.Tensor, n: int) -> torch.Tensor:
    """Return a tensor of shape (n, *x.shape) with n random pairwise swaps of x."""
    if n > x.shape[-1] * (x.shape[-1] - 1) // 2:
        raise ValueError(
            f"n ({n}) must be less than or equal to the number of possible pairwise swaps for a vector of length {x.shape[-1]} ({x.shape[-1] * (x.shape[-1] - 1) // 2})"
        )
    swap_indices = []
    while len(swap_indices) < n:
        i, j = torch.multinomial(torch.ones(x.shape[-1], dtype=torch.float), 2, replacement=False)
        i = i.item()
        j = j.item()
        if (i, j) not in swap_indices and (j, i) not in swap_indices:
            swap_indices.append((i, j))
    return torch.stack([swap_ij(x, i, j) for i, j in swap_indices])


def maximise_acqf(
    acqf, d, n_restarts, iterations_per_restart, points_per_iteration, n_initial_samples, device, dtype, eta=2.0
):
    # INITIALISATION HEURISTIC
    # Generate random initial conditions
    # Ideally this would be done with a space-covering design (e.g. Sobol sequence or Latin hypercube sampling)
    x0 = torch.vstack([torch.randperm(d, device=device) for _ in range(n_initial_samples)]).to(dtype=dtype)
    # Compute the acquisition function at the initial conditions
    acqf_values = acqf(x0.view(-1, 1, d)).view(n_initial_samples)
        
    # Select n_restarts of them according to a multinomial distribution
    zero_mask = acqf_values == 0
    mean = acqf_values.mean()
    std = acqf_values.std()
    if not torch.isclose(std, torch.zeros_like(std)).any():
        normalised_acqf_values = (acqf_values - mean) / std
        weights = torch.exp(eta * normalised_acqf_values)
        weights[zero_mask] = 0
    else:
        # Select at random if all values are the same
        weights = torch.ones_like(acqf_values)
    
    restart_indices = torch.multinomial(weights, n_restarts, replacement=False)

    # MULTI-START OPTIMISATION
    # Start with the initial condition
    candidate_x = x0[restart_indices] # shape: (n_restarts, d)
    # Unsqueeze at 1 to get the shape (n_restarts, 1, d) - this allows us to use non-q-batch acqf implementations
    candidate_acqf = acqf(candidate_x.view(-1, 1, d)) # shape: (n_restarts,)

    for j in tqdm(range(iterations_per_restart)):
        # Generate a random set of points to evaluate
        new_candidate_xs = random_pairwise_swaps(candidate_x, points_per_iteration) # shape: (points_per_iteration, n_restarts, d)
        # Evaluate the acquisition function
        # Reshape input to (points_per_iteration*n_restarts, 1, d) to allow for non-q-batch acqf implementations
        # Reshape output to (points_per_iteration, n_restarts, 1)
        new_candidate_acqfs = acqf(new_candidate_xs.view(-1, 1, d)).view(points_per_iteration, n_restarts)

        # Find the best new candidates
        best_new_candidate_idx = torch.argmax(new_candidate_acqfs, dim=0) # shape: (n_restarts,)
        best_new_candidate_x = new_candidate_xs[best_new_candidate_idx, torch.arange(n_restarts), :] # shape: (n_restarts, d)
        best_new_candidate_acqf = new_candidate_acqfs[best_new_candidate_idx, torch.arange(n_restarts)]
        
        # Update if the best new candidate beats the current candidate
        update_mask = best_new_candidate_acqf > candidate_acqf
        candidate_x[update_mask] = best_new_candidate_x[update_mask]
        candidate_acqf[update_mask] = best_new_candidate_acqf[update_mask]

    # Return the best candidate from all restarts
    best_idx = torch.argmax(candidate_acqf)
    return candidate_x[best_idx], candidate_acqf[best_idx]
