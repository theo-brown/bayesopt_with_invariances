"""Beta sweep for GP-UCB on the needle benchmarks.

Prediction (see README / discussion): UCB's band mu + beta*sigma keeps the
hidden needle basin alive iff beta >= beta* = H / sigma(x0), where H is the
needle's height above the local smooth field and sigma(x0) is the prior sd at
the (generic, off-locus) needle location:

  vanilla kernel     sigma(x0) = 1          -> beta* ~ H
  plain orbit avg.   sigma(x0) ~ 1/sqrt|G|  -> beta* ~ H * sqrt|G|
  normalised avg.    sigma(x0) = 1          -> beta* ~ H

So the plain invariant arm's failure at beta = 2 should disappear once beta
exceeds roughly H*sqrt(|G|), while the flat-variance arms transition near H.

Usage: python beta_sweep.py
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kernels import NormalizedKernel, OrbitAveragedKernel
from mvr import run_bo
import run_demo as rd

BETAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
N_REPEATS_SWEEP = 2  # per world; 5 worlds x 2 = 10 runs per (beta, arm)
EXPERIMENTS = ["torus3_C3_needle", "sphere_C5_needle"]
ARMS = ["vanilla", "invariant", "normalized"]


def needle_height(name, world):
    """H_w: calibrated needle peak height above the local smooth field."""
    base_name = name[: -len("_needle")]
    smooth = rd.build_experiment(base_name, world=world)
    ndl = rd.build_experiment(name, world=world)
    x0 = ndl["needle_orbit"][:1]
    smooth_max = float(np.max(smooth["target"](smooth["candidates"])))
    return (smooth_max + rd.NEEDLE_MARGIN) - float(smooth["target"](x0)[0])


def sweep(name):
    print(f"[{name}]", flush=True)
    heights = [needle_height(name, w) for w in range(rd.N_WORLDS)]
    final = np.zeros((len(BETAS), len(ARMS),
                      rd.N_WORLDS * N_REPEATS_SWEEP))
    g_order = None
    for w in range(rd.N_WORLDS):
        exp = rd.build_experiment(name, world=w)
        g_order = len(exp["group"])
        action = "permutation" if exp["manifold"] == "torus" else "rotation"
        k_inv = OrbitAveragedKernel(exp["base"], exp["group"], action)
        kernels = {"vanilla": exp["base"], "invariant": k_inv,
                   "normalized": NormalizedKernel(k_inv)}
        f_cand = exp["target"](exp["candidates"])
        t0 = time.time()
        for bi, beta in enumerate(BETAS):
            for ai, arm in enumerate(ARMS):
                for rep in range(N_REPEATS_SWEEP):
                    r = run_bo(kernels[arm], f_cand, exp["candidates"],
                               rd.N_INIT, exp["n_iter"], rd.NOISE_SD,
                               seed=1000 + 100 * w + rep, algo="ucb",
                               beta=beta)
                    final[bi, ai, w * N_REPEATS_SWEEP + rep] = r["regret"][-1]
        print(f"  world {w}: H = {heights[w]:.3f}, {time.time()-t0:.0f}s",
              flush=True)
    np.savez(f"results/beta_sweep_{name}.npz", betas=np.array(BETAS),
             final=final, heights=np.array(heights), g_order=g_order)
    return np.array(BETAS), final, np.array(heights), g_order


def plot_sweep(name, betas, final, heights, g_order):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = {"vanilla": "Vanilla kernel",
              "invariant": "Orbit-averaged kernel",
              "normalized": "Normalised orbit-avg."}
    for ai, arm in enumerate(ARMS):
        mean = np.mean(final[:, ai, :], axis=1)
        se = (np.std(final[:, ai, :], axis=1, ddof=1)
              / np.sqrt(final.shape[2]))
        c = rd.SERIES[arm]
        ax.plot(betas, np.maximum(mean, rd.REGRET_FLOOR), color=c,
                linewidth=2, marker="o", markersize=4, label=labels[arm])
        ax.fill_between(betas, np.maximum(mean - se, rd.REGRET_FLOOR),
                        np.maximum(mean + se, rd.REGRET_FLOOR),
                        color=c, alpha=0.18, linewidth=0)
    H = float(np.mean(heights))
    ax.axvline(H, color=rd.SERIES["vanilla"], linestyle=(0, (4, 3)),
               linewidth=1.2, alpha=0.6)
    ax.annotate("$\\bar H$", (H, ax.get_ylim()[1]), xytext=(3, -12),
                textcoords="offset points", color=rd.SERIES["vanilla"],
                fontsize=9)
    ax.axvline(H * np.sqrt(g_order), color=rd.SERIES["invariant"],
               linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.6)
    ax.annotate("$\\bar H\\sqrt{|G|}$",
                (H * np.sqrt(g_order), ax.get_ylim()[1]),
                xytext=(3, -12), textcoords="offset points",
                color=rd.SERIES["invariant"], fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("UCB exploration parameter $\\beta$")
    ax.set_ylabel("Final simple regret")
    exp = rd.build_experiment(name)
    ax.set_title(f"{exp['label']} — GP-UCB $\\beta$ sweep, mean $\\pm$ s.e. "
                 f"over {rd.N_WORLDS} worlds $\\times$ {N_REPEATS_SWEEP} "
                 "repeats", fontsize=10.5, color=rd.TEXT)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    rd.style_axes(ax)
    fig.tight_layout()
    path = f"plots/beta_sweep_{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    for name in EXPERIMENTS:
        betas, final, heights, g_order = sweep(name)
        plot_sweep(name, betas, final, heights, g_order)
