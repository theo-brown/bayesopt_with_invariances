# Experiments

This folder contains the Julia scripts for running the experiments and generating the plots.

### Experiment 1: 2D permutation group

| Target function                                                                                        | MVR [[code](experiment_1_mvr.jl)]                                                       | UCB [[code](experiment_1_ucb.jl)]                                                           |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| <img src="data/experiment_1_ucb/latent_function.png" alt="Permutation invariant function" width=800px> | <img src="data/experiment_1_mvr/regret_plot.png" alt="Simple regret (MVR)" width=800px> | <img src="data/experiment_1_ucb/regret_plot.png" alt="Cumulative regret (UCB)" width=800px> |


### Experiment 2: 3D cyclic group

| Target function                                                                                   | MVR [[code](experiment_2_mvr.jl)]                                                       | UCB [[code](experiment_2_ucb.jl)]                                                           |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| <img src="data/experiment_2_ucb/latent_function.png" alt="Cyclic invariant function" width=800px> | <img src="data/experiment_2_mvr/regret_plot.png" alt="Simple regret (MVR)" width=800px> | <img src="data/experiment_2_ucb/regret_plot.png" alt="Cumulative regret (UCB)" width=800px> |



### Experiment 3: Effect of increasing symmetry

| Target function          | MVR [[code](experiment_3_mvr.jl)]                                                       | UCB [[code](experiment_3_ucb.jl)]                                                           |
| ------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 6D permutation invariant | <img src="data/experiment_3_mvr/regret_plot.png" alt="Simple regret (MVR)" width=800px> | <img src="data/experiment_3_ucb/regret_plot.png" alt="Cumulative regret (UCB)" width=800px> |
