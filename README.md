# Sample-efficient Bayesian Optimisation Using Known Invariances

## Repository structure
```
.
├── README.md           # This file
├── experiments
│   ├── fusion          # Results from fusion experiment
│   │   ├── figures     
│   │   └── data        
│   └── synthetic       # Results from synthetic experiments
│       ├── figures     
│       └── data        
├── src_jl              # Julia source code
└── src_py              # Python source code
```

**Note**: the Julia code was not used to run the experiments in the paper, as it is only written to use CPUs rather than GPUs. We used it for rapid prototyping and the creation of a few illustrative figures only. The code in `src_py` should be viewed as the canonical code for our paper.

## Reproducibility

### Data generation

- Figure 1a: n/a
- Figure 1b: n/a
- Figure 1c: n/a
- Figure 2: ```julia --project src_jl/generate_sample_efficiency.jl```
- Figure 3ai: ```python src_py/synthetic_experiments.py PermInv-2D ucb```
- Figure 3aii: ```python src_py/synthetic_experiments.py CyclInv-3D ucb```
- Figure 3aiii: ```python src_py/synthetic_experiments.py PermInv-6D ucb```
- Figure 3bi: ```python src_py/synthetic_experiments.py PermInv-2D mvr```
- Figure 3bii: ```python src_py/synthetic_experiments.py CyclInv-3D mvr```
- Figure 3biii: ```python src_py/synthetic_experiments.py PermInv-6D mvr```
- Figure 4ai: n/a
- Figure 4b: ```python src_py/fusion_experiment.py --invariant``` and ```python src_py/fusion_experiment.py```, repeated with seeds 1-5*
- Figure 5: ```python src_py/time_to_fit.py```

(n/a) denotes illustrative examples, which generate data during plotting.

(*) **Note**: while we can provide complete code for reproducing the synthetic experiments, we cannot provide complete code for the nuclear fusion experiment as it contains proprietary information.
We provide the script used to run the experiment, but without access to JINTRAC and the appropriate STEP SPR45 configuration files the experiments will fail to run.


### Figure generation

- [Figure 1a](experiments/synthetic/figures/permutation_group.pdf): ```julia --project src_jl/plot_permutation_group.jl```
- [Figure 1b](experiments/synthetic/figures/cyclic_group.png): ```julia --project src_jl/plot_cyclic_group.jl```
- [Figure 1c](experiments/synthetic/figures/dihedral_group.pdf): ```julia --project src_jl/plot_dihedral_group.jl```
- [Figure 2](experiments/synthetic/figures/sample_efficiency.pdf): Jupyter notebook, ```src_jl/plot_sample_efficiency.ipynb```
- [Figure 3ai](experiments/synthetic/figures/perminv2d_ucb_regret.pdf): ```python src_py/plot_regret.py perminv2d ucb```
- [Figure 3aii](experiments/synthetic/figures/cyclinv3d_ucb_regret.pdf): ```python src_py/plot_regret.py cyclinv3d ucb```
- [Figure 3aiii](experiments/synthetic/figures/perminv6d_ucb_regret.pdf): ```python src_py/plot_regret.py perminv6d ucb```
- [Figure 3bi](experiments/synthetic/figures/perminv2d_mvr_regret.pdf): ```python src_py/plot_regret.py perminv2d ucb```
- [Figure 3bii](experiments/synthetic/figures/cyclinv3d_mvr_regret.pdf): ```python src_py/plot_regret.py cyclinv3d ucb```
- [Figure 3biii](experiments/synthetic/figures/perminv6d_mvr_regret.pdf): ```python src_py/plot_regret.py perminv6d ucb```
- [Figure 4ai](): ```python src_py/plot_fusion.py```
- [Figure 4b](): ```python src_py/plot_fusion.py```
- [Figure 5](experiments/synthetic/figures/benchmark.pdf): ```python src_py/plot_benchmark.py```
