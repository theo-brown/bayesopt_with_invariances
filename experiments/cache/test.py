import torch
from torch import Tensor


# Reference implementation (non-batch) from https://github.com/Lightning-AI/torchmetrics/blob/v1.3.2/src/torchmetrics/functional/regression/kendall.py#L83
def _discordant_element_sum(x: Tensor, y: Tensor, i: int) -> Tensor:
    """Count a total number of discordant pairs in a single sequences."""
    return (
        torch.logical_or(
            torch.logical_and(x[i] > x[(i + 1) :], y[i] < y[(i + 1) :]),
            torch.logical_and(x[i] < x[(i + 1) :], y[i] > y[(i + 1) :]),
        )
        .sum(0)
        .unsqueeze(0)
    )

def _count_discordant_pairs(preds: Tensor, target: Tensor) -> Tensor:
    """Count a total number of discordant pairs in given sequences."""
    return torch.cat([_discordant_element_sum(preds, target, i) for i in range(preds.shape[0])]).sum(0)


# Our implementation
def kendall_tau_distance_t(t1, t2):
    """Compute the Kendall tau distance between two tensors of shape (..., n, d).
    
    Note: The tensors must be 0-indexed ranking tensors; ie, must contain each element in [0, 1, ..., d-1] exactly once."""
    if t1.shape[-1] != t2.shape[-1]:
        raise ValueError(
            f"t1 and t2 must have the same number of dimensions (got t1.shape={t1.shape} and t2.shape={t2.shape})"
        )
    
    # (i, j) are the unique pairs of indices
    i, j = torch.triu_indices(t1.shape[-1], t1.shape[-1])
    # Drop the ones that are equal
    mask = i != j
    i = i[mask]
    j = j[mask]
    
    # Compute discordant conditions
    # TODO: Faster method using sign of differences
    condition1 = (t1[..., :, None, i] > t1[..., :, None, j]) & (t2[..., None, :, i] < t2[..., None, :, j])
    condition2 = (t1[..., :, None, i] < t1[..., :, None,  j]) & (t2[..., None, :, i] > t2[..., None, :, j])
    
    return (condition1 | condition2).sum(-1)

x = torch.tensor(
    [
        [[2, 1, 0, 3],  # x11
         [3, 0, 1, 2],  # x12
         [1, 2, 0, 3]], # x13

        [[2, 1, 0, 3], # x21
         [2, 3, 0, 1], # x22
         [2, 1, 0, 3]] # x23
    ]
)

y = torch.tensor(
    [
        [[2, 1, 3, 0],  # y11
         [2, 3, 1, 0],  # y12
         [2, 1, 3, 0]], # y13

        [[1, 3, 2, 0], # y21
         [2, 3, 1, 0], # y22
         [1, 0, 3, 2]] # y23
    ]
)

b = x.shape[0]
n = x.shape[1]
d = x.shape[2]

n_discordant_pairs_truth = torch.zeros(b, n, n)
for i in range(b):
    for j in range(n):
        for k in range(n):
            n_discordant_pairs_truth[i, j, k] =_count_discordant_pairs(x[i][j], y[i][k])
            
dtype = torch.float64
assert torch.allclose(n_discordant_pairs_truth.to(dtype=dtype), kendall_tau_distance_t(x, y).to(dtype=dtype))