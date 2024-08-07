import torch
from botorch.optim import gen_batch_initial_conditions


def make_constraints(d):
    return [
        lambda x: x[..., i+1] - x[..., i]
        for i in range(d-1)
    ]
    
    
def get_violated_constraints(x, constraints):
    return torch.stack(
        [
            torch.stack(
                [
                    constraint(x) < 0
                    for constraint in constraints
                ]
            )
            for x in x
        ]
    )

def ic_generator(acq_function, bounds, num_restarts, raw_samples, q, **kwargs):
    # Generate initial conditions using the default strategy
    x_init = gen_batch_initial_conditions(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    constraints = make_constraints(bounds.shape[-1])

    while get_violated_constraints(x_init, constraints).any():
        # Get the indices of the initial conditions that violate the constraints
        first_bad_index = get_violated_constraints(x_init, constraints).nonzero()[0][0]
        
        # Replace the violated initial conditions with new ones
        new_x = gen_batch_initial_conditions(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=1,
            raw_samples=1,
        )
        x_init[first_bad_index] = new_x.squeeze()
        
    return x_init
