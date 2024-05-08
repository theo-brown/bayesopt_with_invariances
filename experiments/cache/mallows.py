import torch
import gpytorch


class MallowsKernel(gpytorch.kernels.Kernel):
    def __init__(self, nu, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu

    def forward(self, x1, x2, **kwargs):
        # Check tensors are on same device
        if not x1.device == x2.device:
            raise ValueError(f"MallowsKernel received tensors on different devices (x1.device={x1.device}, x2.device={x2.device})")
        return torch.exp(-self.nu * kendall_tau_distance(x1, x2))


def kendall_tau_distance(t1, t2):
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