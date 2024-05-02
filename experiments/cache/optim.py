import torch
from tqdm import tqdm


def swap_ij(x: torch.Tensor, i: int, j: int) -> torch.Tensor:
    x_new = x.clone()
    x_new[..., i], x_new[..., j] = x[..., j], x[..., i]
    return x_new


def pairwise_swaps(x: torch.Tensor) -> torch.Tensor:
    return torch.vstack(
        [swap_ij(x, i, j) for i, j in torch.combinations(torch.arange(x.shape[-1]), 2)]
    )


def random_pairwise_swaps(x: torch.Tensor, n: int) -> torch.Tensor:
    indices = torch.arange(
        x.shape[-1], dtype=torch.float
    )  # multinomial requires a float here
    swap_indices = []
    for i in range(n):
        i, j = torch.multinomial(indices, 2, replacement=False)
        i = i.item()
        j = j.item()
        if (i, j) not in swap_indices:
            swap_indices.append((i, j))
    return torch.vstack([swap_ij(x, i, j) for i, j in swap_indices])


def maximise_acqf(
    acqf, d, n_restarts, iterations_per_restart, points_per_iteration, device, dtype
):
    # Generate random initial conditions
    x0 = torch.vstack([torch.randperm(d) for _ in range(n_restarts)]).to(
        device=device, dtype=dtype
    )

    # Create the tensor we'll store the results in
    final_x = torch.empty(n_restarts, d, device=device, dtype=dtype)
    final_acqf = torch.empty(n_restarts, device=device, dtype=dtype)

    # Multi-start optimization
    for i in range(n_restarts):
        print(f"Restart {i+1}/{n_restarts}")
        # Start with the initial condition
        candidate_x = x0[i]
        candidate_acqf = acqf(candidate_x.reshape(-1, d))

        for j in tqdm(range(iterations_per_restart)):
            # Generate a random set of points to evaluate
            new_candidate_xs = random_pairwise_swaps(candidate_x, points_per_iteration)
            new_candidate_acqfs = acqf(new_candidate_xs.reshape(-1, 1, d))

            # Find the best new candidate
            best_candidate_idx = torch.argmax(new_candidate_acqfs)

            # Update if the best new candidate beats the current candidate
            if new_candidate_acqfs[best_candidate_idx] > candidate_acqf:
                candidate_x = new_candidate_xs[best_candidate_idx]
                candidate_acqf = new_candidate_acqfs[best_candidate_idx]

        # Store the best candidate from this restart
        final_x[i] = candidate_x
        final_acqf[i] = candidate_acqf

    # Return the best candidate from all restarts
    best_idx = torch.argmax(final_acqf)
    return final_x[best_idx], final_acqf[best_idx]
