import torch
import gpytorch


class MallowsKernel(gpytorch.kernels.Kernel):
    def __init__(self, nu, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu

    def forward(self, x1, x2, **kwargs):
        # TODO: Check that each tensor in the batch contains each element in 0:d once
        if x1.ndim == 1 and x2.ndim == 1:
            return torch.exp(-self.nu * count_discordant_pairs(x1, x2))
        elif x1.ndim == 2 and x2.ndim == 2:
            return torch.exp(
                -self.nu
                * count_discordant_pairs_batch(x1.unsqueeze(0), x2.unsqueeze(0))
            ).squeeze(0)
        elif x1.ndim == 3 and x2.ndim == 3:
            return torch.exp(-self.nu * count_discordant_pairs_batch(x1, x2))
        else:
            raise ValueError(
                f"Unsupported number of dimensions for x1 and x2 (got x1.ndim={x1.ndim} and x2.ndim={x2.ndim})"
            )


def count_discordant_pairs(x, y):
    d = x.shape[-1]
    return torch.stack(
        [
            torch.logical_or(
                (x[..., i] < x[..., j]) & (y[..., i] > y[..., j]),
                (x[..., i] > x[..., j]) & (y[..., i] < y[..., j]),
            )
            for j in range(0, d)
            for i in range(0, j)
        ],
        dim=-1,
    ).sum(-1)


def count_discordant_pairs_batch(x_batch, y_batch):
    if x_batch.ndim != y_batch.ndim:
        raise ValueError(
            f"x_batch and y_batch must have the same number of dimensions (got x_batch.shape={x_batch.shape} and y_batch.shape={y_batch.shape})"
        )
    if x_batch.shape[-1] != y_batch.shape[-1]:
        raise ValueError(
            f"Shapes of x_batch and y_batch must match in the last dimension (got x_batch.shape={x_batch.shape} and y_batch.shape={y_batch.shape})"
        )
    if x_batch.shape[0] != y_batch.shape[0]:
        raise ValueError(
            f"Shapes of x_batch and y_batch must match in the first dimension (got x_batch.shape={x_batch.shape} and y_batch.shape={y_batch.shape})"
        )
    n = x_batch.shape[1]
    m = y_batch.shape[1]
    b = x_batch.shape[0]
    output = torch.empty(b, n, m)
    for i in range(b):
        output[i] = count_discordant_pairs(x_batch[i, ...], y_batch[i, ...])
    return output
