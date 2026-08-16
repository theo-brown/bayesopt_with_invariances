"""Maximum Variance Reduction (MVR) with a fixed candidate set.

The domain is discretised by a large candidate set; each iteration queries the
candidate with the highest posterior variance, and the incumbent is the
candidate maximising the posterior mean. Simple regret is measured against the
best target value on the candidate set. Kernel hyperparameters are known to
the learner (as in Brown et al. 2024, Appendix B.1); no renormalisation or
other tricks are applied.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def run_mvr(kernel, f_candidates: np.ndarray, candidates: np.ndarray,
            n_init: int, n_iter: int, noise_sd: float,
            seed: int) -> np.ndarray:
    """Run MVR and return simple regret after each of the n_iter iterations.

    Parameters
    ----------
    kernel : object with .pair(X, Y) and .elementwise(X, Y)
    f_candidates : exact target values on the candidate set
    candidates : (n_candidates, d) discretisation of the domain
    """
    rng = np.random.default_rng(seed)
    n_cand = candidates.shape[0]
    f_max = np.max(f_candidates)

    prior_var = kernel.elementwise(candidates, candidates)

    train_idx = list(rng.choice(n_cand, size=n_init, replace=False))
    # k(x_train, x_cand), grown one row per observation
    k_tc = kernel.pair(candidates[train_idx], candidates)
    y = (f_candidates[train_idx]
         + noise_sd * rng.standard_normal(n_init)).tolist()

    regret = np.empty(n_iter)
    for t in range(n_iter):
        k_tt = k_tc[:, train_idx]
        k_tt = 0.5 * (k_tt + k_tt.T)  # symmetrise fp noise
        chol, low = cho_factor(k_tt + noise_sd**2 * np.eye(len(train_idx)),
                               lower=True)
        alpha = cho_solve((chol, low), np.asarray(y))
        mean = k_tc.T @ alpha
        v = solve_triangular(chol, k_tc, lower=True)
        var = np.maximum(prior_var - np.sum(v**2, axis=0), 0.0)

        # Incumbent and regret for this iteration
        regret[t] = f_max - f_candidates[int(np.argmax(mean))]

        # MVR query: maximum posterior variance
        nxt = int(np.argmax(var))
        train_idx.append(nxt)
        k_tc = np.vstack([k_tc, kernel.pair(candidates[nxt][None, :],
                                            candidates)])
        y.append(f_candidates[nxt] + noise_sd * rng.standard_normal())

    return regret
