# Experiments

This folder contains the scripts for running the experiments and generating the plots.

## Synthetic

### Experiment 1: 2D permutation group

| Target function                                                                                                  | MVR [[code](synthetic/experiment_1_mvr.jl)]                                                       | UCB [[code](synthetic/experiment_1_ucb.jl)]                                                           |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| <img src="synthetic/data/experiment_1_ucb/latent_function.png" alt="Permutation invariant function" width=800px> | <img src="synthetic/data/experiment_1_mvr/regret_plot.png" alt="Simple regret (MVR)" width=800px> | <img src="synthetic/data/experiment_1_ucb/regret_plot.png" alt="Cumulative regret (UCB)" width=800px> |


### Experiment 2: 3D cyclic group

| Target function                                                                                             | MVR [[code](synthetic/experiment_2_mvr.jl)]                                                       | UCB [[code](synthetic/experiment_2_ucb.jl)]                                                           |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| <img src="synthetic/data/experiment_2_ucb/latent_function.png" alt="Cyclic invariant function" width=800px> | <img src="synthetic/data/experiment_2_mvr/regret_plot.png" alt="Simple regret (MVR)" width=800px> | <img src="synthetic/data/experiment_2_ucb/regret_plot.png" alt="Cumulative regret (UCB)" width=800px> |



### Experiment 3: Effect of increasing symmetry

| Target function          | MVR [[code](synthetic/experiment_3_mvr_mpi.jl)]                                                       | UCB [[code](synthetic/experiment_3_ucb_mpi.jl)]                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 6D permutation invariant | <img src="synthetic/data/experiment_3_mvr_mpi/regret_plot.png" alt="Simple regret (MVR)" width=800px> | <img src="synthetic/data/experiment_3_ucb_mpi/regret_plot.png" alt="Cumulative regret (UCB)" width=800px> |

#### Computational cost

<img src="synthetic/data/experiment_3_benchmark/fit_gp_times.png" alt="Simple regret (MVR)" width=800px>

## Cache model

Coming soon

## Nuclear fusion scenario design

| Input visualisation                                                             | Results     |
| ------------------------------------------------------------------------------- | ----------- |
| <img src="fusion/fusion_profiles.png" alt="ECRH launcher profiles" width=800px> | Coming soon |
