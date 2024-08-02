from typing import Callable

import gpytorch
import torch


class InvariantKernel(gpytorch.kernels.Kernel):
    r"""A kernel that is invariant to a collection of transformations.

    Currently only supports group invariance and isotropic simultaneously invariant kernels.
    
    The invariant kernel is given by
    .. math::
        k_G(x, y) = \frac{1}{|G|} \sum_{g \in G} k(g(x), y)

    where :math:`G` is the group of transformations, :math:`k` is the base kernel, and :math:`x` and :math:`y` are the inputs.
    """

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        transformations: Callable[[torch.tensor], torch.tensor],
        is_isotropic: bool = True,
        is_group: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.base_kernel = base_kernel
        self.transformations = transformations
        self.is_isotropic = is_isotropic
        self.is_group = is_group

        if not self.is_group:
            raise NotImplementedError("InvariantKernel currently only supports group invariance.")
        if not self.is_isotropic:
            raise NotImplementedError("InvariantKernel currently only supports isotropic kernels.")


    def forward(
        self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **kwargs
    ) -> torch.tensor:
        if last_dim_is_batch:
            raise NotImplementedError(
                "last_dim_is_batch=True not supported for GroupInvariantKernel."
            )

        x1_orbits = self.transformations(x1)  # Shape is ... x G x N x d
        G = x1_orbits.shape[-3]
        # Sum is over a single set of orbits
        # Construct x2_orbits by tiling x2 along the -3 axis
        # TODO: probably a smarter way of doing this with broadcasting
        dims = [-1] * (x2.dim() + 1)
        dims[-3] = G
        x2_orbits = x2.unsqueeze(-3).expand(dims)
        K_orbits = self.base_kernel.forward(x1_orbits, x2_orbits)
        K = torch.mean(K_orbits, dim=-3)

        if diag:
            return K.diag()
        else:
            return K
        