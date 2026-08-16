"""Run the manifold RKHS benchmark demos and produce plots.

Five experiments, each comparing MVR with the vanilla kernel against MVR with
the plain orbit-averaged (invariant) kernel:

  torus2_S2   : T^2, permutation group S_2            (|G| = 2)
  torus3_C3   : T^3, cyclic coordinate shifts C_3     (|G| = 3)
  torus3_S3   : T^3, full permutation group S_3       (|G| = 6)
  sphere_C5   : S^2, rotations by 72 deg about z, C_5 (|G| = 5)
  sphere_oct  : S^2, octahedral rotation group        (|G| = 24)

Targets are G-invariant, exactly in the RKHS, with exactly known norm, and
have far more active modes than the BO budget. Usage:

    python run_demo.py            # run everything (~10-20 min)
    python run_demo.py torus2_S2  # run a single experiment
"""

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import qmc

from groups import (apply_permutation, cyclic_permutation_group,
                    cyclic_rotation_group, octahedral_rotation_group,
                    permutation_group)
from kernels import NU, OrbitAveragedKernel, SphereMatern, WrappedMatern52
from mvr import run_mvr
from targets import (SphereTarget, TorusTarget, make_sphere_target,
                     make_torus_target, sphere_needle_coeffs,
                     torus_needle_coeffs)

# ---------------------------------------------------------------------------
# Style (light mode; palette from the validated reference instance)
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e8e7e4"
SERIES = {"vanilla": "#2a78d6", "invariant": "#eb6834"}

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948",
     "#8c2726"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "axes.edgecolor": GRID,
    "font.size": 11,
})


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Candidate sets
# ---------------------------------------------------------------------------

def sobol_candidates(d: int, n: int, seed: int) -> np.ndarray:
    return qmc.Sobol(d, scramble=True, seed=seed).random(n)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azimuth = np.pi * (1.0 + np.sqrt(5.0)) * i
    return np.stack([np.sin(polar) * np.cos(azimuth),
                     np.sin(polar) * np.sin(azimuth),
                     np.cos(polar)], axis=-1)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

N_REPEATS = 10
N_INIT = 5
NOISE_SD = 0.05


def build_experiment(name: str) -> dict:
    if name.endswith("_needle"):
        return with_needle(build_experiment(name[:-len("_needle")]))
    if name == "torus2_S2":
        d, l, max_freq = 2, 0.12, 12
        group = permutation_group(d)
        label = "$T^2$, permutation group $S_2$"
        n_cand, n_iter = 4096, 80
    elif name == "torus3_C3":
        d, l, max_freq = 3, 0.12, 6
        group = cyclic_permutation_group(d)
        label = "$T^3$, cyclic group $C_3$"
        n_cand, n_iter = 8192, 150
    elif name == "torus3_S3":
        d, l, max_freq = 3, 0.12, 6
        group = permutation_group(d)
        label = "$T^3$, permutation group $S_3$"
        n_cand, n_iter = 8192, 150
    elif name in ("sphere_C5", "sphere_oct"):
        max_degree, kappa = 25, 0.25
        group = (cyclic_rotation_group(5) if name == "sphere_C5"
                 else octahedral_rotation_group())
        label = ("$S^2$, cyclic rotations $C_5$" if name == "sphere_C5"
                 else "$S^2$, octahedral rotation group $O$")
        base = SphereMatern(max_degree=max_degree, kappa=kappa)
        target = make_sphere_target(max_degree, base.coeffs, group, seed=100)
        cands = fibonacci_sphere(4096)
        return dict(name=name, manifold="sphere", label=label, group=group,
                    base=base, target=target, candidates=cands, n_iter=150,
                    n_modes=(max_degree + 1) ** 2)
    else:
        raise ValueError(name)

    base = WrappedMatern52(d=d, lengthscale=l)
    target = make_torus_target(d, l, max_freq, group, seed=100)
    cands = sobol_candidates(d, n_cand, seed=200)
    return dict(name=name, manifold="torus", label=label, group=group,
                base=base, target=target, candidates=cands, n_iter=n_iter,
                n_modes=(2 * max_freq + 1) ** d)


# ---------------------------------------------------------------------------
# Planted needle: a symmetrised truncated-spectrum kernel atom at a hidden
# generic orbit, scaled so the needle peak sits `margin` above the smooth
# field's maximum. Exact RKHS membership and closed-form norms carry over
# because the needle lives in the same truncated Fourier/harmonic frame.
# ---------------------------------------------------------------------------

NEEDLE_MARGIN = 0.5


def _wrap(a):
    return a - np.round(a)


def generic_torus_point(d: int, rng, min_sep: float = 0.15) -> np.ndarray:
    """Rejection-sample a point at least min_sep from every fixed-point set
    (planes x_i = x_j) of coordinate-permutation groups."""
    while True:
        x = rng.uniform(size=d)
        seps = [abs(_wrap(x[i] - x[j])) for i in range(d)
                for j in range(i + 1, d)]
        if min(seps) >= min_sep:
            return x


def generic_sphere_point(axes: list[np.ndarray], rng,
                         min_angle: float = 0.35) -> np.ndarray:
    """Rejection-sample a unit vector at least min_angle (radians) from
    every rotation axis of the group."""
    while True:
        x = rng.standard_normal(3)
        x /= np.linalg.norm(x)
        if all(np.arccos(np.clip(abs(x @ ax), -1.0, 1.0)) >= min_angle
               for ax in axes):
            return x


def with_needle(exp: dict, margin: float = NEEDLE_MARGIN) -> dict:
    rng = np.random.default_rng(300)
    target, cands, group = exp["target"], exp["candidates"], exp["group"]
    smooth_max = float(np.max(target(cands)))

    if exp["manifold"] == "torus":
        d = cands.shape[1]
        x0 = generic_torus_point(d, rng)
        c_n = torus_needle_coeffs(target.modes, exp["base"].lengthscale, x0,
                                  group)
        needle = TorusTarget(target.modes, c_n, target.lambdas)
        beta = (smooth_max + margin - target(x0[None])[0]) / needle(x0[None])[0]
        new_target = TorusTarget(target.modes, target.coeffs + beta * c_n,
                                 target.lambdas)
        orbit = np.stack([apply_permutation(x0[None], g)[0] for g in group])
    else:
        if "C_5" in exp["label"]:
            axes = [np.array([0.0, 0.0, 1.0])]
        else:
            axes = [np.eye(3)[i] for i in range(3)]
            axes += [np.array(s) / np.sqrt(3.0) for s in
                     [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]]
            axes += [np.array(a) / np.sqrt(2.0) for a in
                     [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
                      (0, 1, 1), (0, 1, -1)]]
        x0 = generic_sphere_point(axes, rng)
        c_n = sphere_needle_coeffs(target.max_degree, exp["base"].kappa, x0,
                                   group)
        needle = SphereTarget(target.max_degree, c_n, target.per_degree)
        beta = (smooth_max + margin - target(x0[None])[0]) / needle(x0[None])[0]
        new_target = SphereTarget(target.max_degree,
                                  target.coeffs + beta * c_n,
                                  target.per_degree)
        orbit = np.stack([g @ x0 for g in group])

    return dict(exp, name=exp["name"] + "_needle", target=new_target,
                needle_orbit=orbit, label=exp["label"] + " $+$ needle")


def run_experiment(exp: dict) -> dict:
    action = "permutation" if exp["manifold"] == "torus" else "rotation"
    kernels = {
        "vanilla": exp["base"],
        "invariant": OrbitAveragedKernel(exp["base"], exp["group"], action),
    }
    f_cand = exp["target"](exp["candidates"])
    results = {}
    for kname, kern in kernels.items():
        t0 = time.time()
        regrets = np.stack([
            run_mvr(kern, f_cand, exp["candidates"], N_INIT, exp["n_iter"],
                    NOISE_SD, seed=1000 + rep)
            for rep in range(N_REPEATS)
        ])
        results[kname] = regrets
        print(f"  {exp['name']} / {kname}: {time.time() - t0:.1f}s, "
              f"final mean regret {np.mean(regrets[:, -1]):.4f}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_target_torus(ax, exp):
    """Heatmap of the target on T^2, or the slice through the global maximum
    for T^3."""
    target, cands = exp["target"], exp["candidates"]
    d = cands.shape[1]
    n = 220
    g1, g2 = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n),
                         indexing="xy")
    if d == 2:
        pts = np.stack([g1.ravel(), g2.ravel()], axis=-1)
        note = ""
    else:
        x_star = cands[np.argmax(target(cands))]
        pts = np.column_stack([g1.ravel(), g2.ravel(),
                               np.full(g1.size, x_star[2])])
        note = f" (slice $x_3 = {x_star[2]:.2f}$)"
    vals = target(pts).reshape(n, n)
    vmax = np.max(np.abs(vals))
    im = ax.pcolormesh(g1, g2, vals, cmap=DIVERGING, shading="auto",
                       norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                       rasterized=True)
    orbit = exp.get("needle_orbit")
    if orbit is not None:
        if d == 3:  # only orbit points near the plotted slice (within half
            # a lengthscale, since the needle has finite width)
            dz = orbit[:, 2] - x_star[2]
            in_slice = np.abs(dz - np.round(dz)) < 0.06
            orbit = orbit[in_slice]
        ax.scatter(orbit[:, 0] % 1.0, orbit[:, 1] % 1.0, marker="x", s=45,
                   color=TEXT, linewidths=1.4, label="needle orbit")
        ax.legend(frameon=False, fontsize=8, loc="upper right",
                  labelcolor=TEXT_2)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Target{note}", fontsize=11, color=TEXT)
    ax.set_aspect("equal")
    return im


def plot_target_sphere(ax, exp):
    """Mollweide heatmap of the target on S^2."""
    n_lon, n_lat = 400, 200
    lon = np.linspace(-np.pi, np.pi, n_lon)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON),
                    np.sin(LAT)], axis=-1).reshape(-1, 3)
    vals = exp["target"](pts).reshape(n_lat, n_lon)
    vmax = np.max(np.abs(vals))
    im = ax.pcolormesh(LON, LAT, vals, cmap=DIVERGING, shading="auto",
                       norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                       rasterized=True)
    orbit = exp.get("needle_orbit")
    if orbit is not None:
        ax.scatter(np.arctan2(orbit[:, 1], orbit[:, 0]),
                   np.arcsin(np.clip(orbit[:, 2], -1.0, 1.0)),
                   marker="x", s=45, color=TEXT, linewidths=1.4,
                   label="needle orbit")
        ax.legend(frameon=False, fontsize=8, loc="lower right",
                  labelcolor=TEXT_2)
    ax.set_title("Target (Mollweide)", fontsize=11, color=TEXT)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=7, colors=TEXT_2)
    return im


REGRET_FLOOR = 2e-4  # display floor; regret 0 = exact optimum on candidates


def plot_theory_rate(ax, exp, results, n_obs):
    """Overlay the theoretical MVR rate from Brown et al. (2024), Thm 1:
    gamma_T^G = O~(T^{m/(2nu+m)} / |G|), hence simple regret
    r_T = O~(B |G|^{-1/2} T^{-nu/(2nu+m)}), with m the manifold dimension.
    Constants and polylog factors are unknown, so the vanilla guide is
    anchored to the vanilla curve one-third of the way in; the invariant
    guide is then fixed by the theoretical |G|^{-1/2} offset."""
    m = exp["candidates"].shape[1] if exp["manifold"] == "torus" else 2
    alpha = NU / (2.0 * NU + m)
    i0 = len(n_obs) // 3
    anchor = np.mean(results["vanilla"], axis=0)[i0]
    if anchor <= REGRET_FLOOR:
        return
    span = slice(i0, None)
    scale = anchor * n_obs[i0] ** alpha
    guide = scale * n_obs[span] ** (-alpha)
    offset = 1.0 / np.sqrt(len(exp["group"]))
    ax.plot(n_obs[span], guide, color=SERIES["vanilla"], linestyle=(0, (4, 3)),
            linewidth=1.4, alpha=0.65,
            label=f"$\\propto T^{{-{alpha:.2f}}}$ (theory)")
    ax.plot(n_obs[span], np.maximum(offset * guide, REGRET_FLOOR),
            color=SERIES["invariant"], linestyle=(0, (4, 3)),
            linewidth=1.4, alpha=0.65,
            label="$\\times\\, |G|^{-1/2}$ (theory)")


def plot_regret(ax, exp, results):
    n_obs = N_INIT + np.arange(exp["n_iter"])
    clipped = False
    for kname, label in [("vanilla", "Vanilla kernel"),
                         ("invariant", "Orbit-averaged kernel")]:
        r = results[kname]
        mean = np.mean(r, axis=0)
        stderr = np.std(r, axis=0, ddof=1) / np.sqrt(r.shape[0])
        clipped = clipped or bool(np.any(mean < REGRET_FLOOR))
        color = SERIES[kname]
        ax.plot(n_obs, np.maximum(mean, REGRET_FLOOR), color=color,
                linewidth=2, label=label)
        ax.fill_between(n_obs, np.maximum(mean - stderr, REGRET_FLOOR),
                        np.maximum(mean + stderr, REGRET_FLOOR),
                        color=color, alpha=0.18, linewidth=0)
        ax.annotate(label, (n_obs[-1], max(mean[-1], REGRET_FLOOR)),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center")
    plot_theory_rate(ax, exp, results, n_obs)
    ax.set_yscale("log")
    ax.set_xlabel("Observations")
    ax.set_ylabel("Simple regret")
    ax.set_title(f"MVR, mean $\\pm$ s.e. over {N_REPEATS} runs",
                 fontsize=11, color=TEXT)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style_axes(ax)
    ax.margins(x=0.12)
    return clipped


def make_figure(exp, results, path):
    fig = plt.figure(figsize=(11.5, 4.4))
    if exp["manifold"] == "sphere":
        ax1 = fig.add_subplot(1, 2, 1, projection="mollweide")
        im = plot_target_sphere(ax1, exp)
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        im = plot_target_torus(ax1, exp)
    fig.colorbar(im, ax=ax1, shrink=0.85, label="$f(x)$")
    ax2 = fig.add_subplot(1, 2, 2)
    clipped = plot_regret(ax2, exp, results)
    if clipped:
        fig.text(0.985, 0.012,
                 "curves at the axis floor reached regret 0 "
                 "(exact optimum on the candidate set)",
                 ha="right", fontsize=8, color=TEXT_2)
    fig.suptitle(
        f"{exp['label']}  —  |G| = {len(exp['group'])},  "
        f"{exp['n_modes']} active modes,  "
        f"$\\|f\\|_{{H_k}}$ = {exp['target'].rkhs_norm:.2f}",
        fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = ["torus2_S2", "torus3_C3", "torus3_S3", "sphere_C5",
                   "sphere_oct",
                   "torus2_S2_needle", "torus3_C3_needle", "torus3_S3_needle",
                   "sphere_C5_needle", "sphere_oct_needle"]


def main():
    names = sys.argv[1:] or ALL_EXPERIMENTS
    for name in names:
        print(f"[{name}]", flush=True)
        exp = build_experiment(name)
        print(f"  target: {exp['n_modes']} modes, "
              f"||f||_H = {exp['target'].rkhs_norm:.3f}", flush=True)
        results = run_experiment(exp)
        np.savez(f"results/{name}.npz", **results)
        make_figure(exp, results, f"plots/{name}.png")


if __name__ == "__main__":
    main()
