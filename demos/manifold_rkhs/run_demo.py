"""Run the manifold RKHS benchmark demos and produce plots.

Five experiments, each comparing MVR with the vanilla kernel against MVR with
the plain orbit-averaged (invariant) kernel:

  torus2_S2   : T^2, permutation group S_2            (|G| = 2)
  torus3_C3   : T^3, cyclic coordinate shifts C_3     (|G| = 3)
  torus3_S3   : T^3, full permutation group S_3       (|G| = 6)
  sphere_C5   : S^2, rotations by 72 deg about z, C_5 (|G| = 5)
  sphere_oct  : S^2, octahedral rotation group        (|G| = 24)

Targets are G-invariant, exactly in the RKHS, with exactly known norm, and
have far more active modes than the BO budget. Regret curves are averaged
over N_WORLDS independently drawn targets ("worlds") x N_REPEATS BO repeats;
each world's smooth target is rescaled to the experiment's fixed reference
norm B_REF. Usage:

    python run_demo.py            # run everything (~30-60 min)
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
from kernels import (NormalizedKernel, OrbitAveragedKernel, SphereMatern,
                     WrappedMatern52)
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
SERIES = {"vanilla": "#2a78d6", "invariant": "#eb6834",
          "normalized": "#1baf7a"}

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

N_WORLDS = 5    # independently drawn objective functions per experiment
N_REPEATS = 4   # BO repeats per world (N_WORLDS * N_REPEATS runs per kernel)
N_INIT = 5
NOISE_SD = 0.05

# Fixed per-experiment reference norm: every world's smooth target is rescaled
# to exactly this RKHS norm, so worlds differ only in the random draw (and, for
# needle variants, the needle location). Round numbers near the seed-100 norms.
B_REF = {
    "torus2_S2": 12.0,
    "torus3_C3": 19.0,
    "torus3_S3": 15.0,
    "sphere_C5": 12.0,
    "sphere_oct": 6.0,
}


def build_experiment(name: str, world: int = 0) -> dict:
    """Build one world of an experiment: the smooth target seed and the needle
    location vary with `world`; candidates, kernels, budgets and groups are
    identical across worlds, and the smooth component's RKHS norm is rescaled
    to the fixed reference B_REF[name]."""
    if name.endswith("_needle"):
        return with_needle(build_experiment(name[:-len("_needle")], world),
                           world=world)
    seed = 100 + 17 * world
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
        target = make_sphere_target(max_degree, base.coeffs, group, seed=seed,
                                    norm=B_REF[name])
        cands = fibonacci_sphere(4096)
        return dict(name=name, manifold="sphere", label=label, group=group,
                    base=base, target=target, candidates=cands, n_iter=150,
                    n_modes=(max_degree + 1) ** 2, b_ref=B_REF[name])
    else:
        raise ValueError(name)

    base = WrappedMatern52(d=d, lengthscale=l)
    target = make_torus_target(d, l, max_freq, group, seed=seed,
                               norm=B_REF[name])
    cands = sobol_candidates(d, n_cand, seed=200)
    return dict(name=name, manifold="torus", label=label, group=group,
                base=base, target=target, candidates=cands, n_iter=n_iter,
                n_modes=(2 * max_freq + 1) ** d, b_ref=B_REF[name])


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


def with_needle(exp: dict, world: int = 0,
                margin: float = NEEDLE_MARGIN) -> dict:
    rng = np.random.default_rng(300 + world)
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


def run_experiment(name: str) -> tuple[dict, dict]:
    """Run all worlds x repeats for one experiment.

    Every world shares the group, candidates, kernels and budget, and its
    smooth component is rescaled to the same reference norm B_REF; what varies
    per world is the target draw (and, for needle variants, the needle
    location). All N_WORLDS * N_REPEATS regret traces per kernel are pooled.
    The regret certificate 2 B sup_x sigma_t is stored per run, computed with
    that run's world's *exact* norm B (which varies slightly across worlds for
    needle targets).

    Returns (exp, results) where exp is the world-0 experiment dict (used for
    the target plot), augmented with the mean total norm across worlds.
    """
    exp0 = None
    traces = {"vanilla": {"regret": [], "cert": []},
              "invariant": {"regret": [], "cert": []},
              "normalized": {"regret": []}}
    norms = []
    for w in range(N_WORLDS):
        exp = build_experiment(name, world=w)
        if w == 0:
            exp0 = exp
        action = "permutation" if exp["manifold"] == "torus" else "rotation"
        k_inv = OrbitAveragedKernel(exp["base"], exp["group"], action)
        kernels = {
            "vanilla": exp["base"],
            "invariant": k_inv,
            # Diagonal renormalisation of the orbit average: flattens the
            # prior variance, removing the fixed-point inflation. No exact-
            # norm certificate exists for this arm (B is known in H_{k_G},
            # not in the normalised kernel's RKHS).
            "normalized": NormalizedKernel(k_inv),
        }
        B_world = exp["target"].rkhs_norm
        norms.append(B_world)
        f_cand = exp["target"](exp["candidates"])
        for kname, kern in kernels.items():
            t0 = time.time()
            runs = [
                run_mvr(kern, f_cand, exp["candidates"], N_INIT,
                        exp["n_iter"], NOISE_SD, seed=1000 + 100 * w + rep)
                for rep in range(N_REPEATS)
            ]
            traces[kname]["regret"].extend(r["regret"] for r in runs)
            if "cert" in traces[kname]:
                traces[kname]["cert"].extend(2.0 * B_world * r["max_sd"]
                                             for r in runs)
            print(f"  {name} / world {w} / {kname}: "
                  f"{time.time() - t0:.1f}s, final mean regret "
                  f"{np.mean([r['regret'][-1] for r in runs]):.4f}",
                  flush=True)
    results = {kname: {key: np.stack(arrs) for key, arrs in tr.items()}
               for kname, tr in traces.items()}
    exp0["mean_norm"] = float(np.mean(norms))
    return exp0, results


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
    ax.set_title(f"Target, world 0{note}", fontsize=11, color=TEXT)
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
    ax.set_title("Target, world 0 (Mollweide)", fontsize=11, color=TEXT)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=7, colors=TEXT_2)
    return im


REGRET_FLOOR = 2e-4  # display floor; regret 0 = exact optimum on candidates


def plot_regret(ax, exp, results):
    n_obs = N_INIT + np.arange(exp["n_iter"])
    clipped = False
    max_mean_regret = 0.0
    arms = [("vanilla", "Vanilla kernel"),
            ("invariant", "Orbit-averaged kernel"),
            ("normalized", "Normalised orbit-avg.")]
    for kname, label in arms:
        if kname not in results:
            continue
        r = results[kname]["regret"]
        mean = np.mean(r, axis=0)
        stderr = np.std(r, axis=0, ddof=1) / np.sqrt(r.shape[0])
        clipped = clipped or bool(np.any(mean < REGRET_FLOOR))
        max_mean_regret = max(max_mean_regret, float(np.max(mean)))
        color = SERIES[kname]
        ax.plot(n_obs, np.maximum(mean, REGRET_FLOOR), color=color,
                linewidth=2, label=label)
        ax.fill_between(n_obs, np.maximum(mean - stderr, REGRET_FLOOR),
                        np.maximum(mean + stderr, REGRET_FLOOR),
                        color=color, alpha=0.18, linewidth=0)
        ax.annotate(label, (n_obs[-1], max(mean[-1], REGRET_FLOOR)),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center")
        # Non-asymptotic certificate: r_t <= 2 B sup_x sigma_t(x), valid
        # because each world's RKHS norm B is exactly known. Absent for the
        # normalised arm (B is known in H_{k_G}, not in that kernel's RKHS).
        if "cert" in results[kname]:
            cert = np.mean(results[kname]["cert"], axis=0)
            ax.plot(n_obs, cert, color=color, linestyle=":", linewidth=1.4,
                    alpha=0.65, label="$2B\\sup_x \\sigma_t$ (bound)")
    ax.set_yscale("log")
    # The certificate starts near 2B, far above the empirical regret; cap
    # the axis so it enters the frame as it decays instead of stretching it.
    ax.set_ylim(top=4.0 * max_mean_regret)
    ax.set_xlabel("Observations")
    ax.set_ylabel("Simple regret")
    ax.set_title(f"MVR, mean $\\pm$ s.e. over {N_WORLDS} worlds "
                 f"$\\times$ {N_REPEATS} repeats",
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
    if exp["name"].endswith("_needle"):
        # Total norm varies slightly across worlds (the smooth component is
        # fixed at B_REF; needle scale and cross terms depend on the world).
        norm_txt = (f"mean $\\|f\\|_{{H_k}}$ = "
                    f"{exp.get('mean_norm', exp['target'].rkhs_norm):.2f}")
    else:
        norm_txt = f"$\\|f\\|_{{H_k}}$ = {exp['b_ref']:.2f}"
    fig.suptitle(
        f"{exp['label']}  —  |G| = {len(exp['group'])},  "
        f"{exp['n_modes']} active modes,  {norm_txt}",
        fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = ["torus2_S2_needle", "torus3_C3_needle", "torus3_S3_needle",
                   "sphere_C5_needle", "sphere_oct_needle"]

# Short label, |G|, and series color (categorical palette order) per
# experiment, for the ratio summary figure.
RATIO_SPEC = {
    "torus2_S2_needle": ("$T^2$, $S_2$", 2, "#2a78d6"),
    "torus3_C3_needle": ("$T^3$, $C_3$", 3, "#eb6834"),
    "torus3_S3_needle": ("$T^3$, $S_3$", 6, "#1baf7a"),
    "sphere_C5_needle": ("$S^2$, $C_5$", 5, "#eda100"),
    "sphere_oct_needle": ("$S^2$, $O$", 24, "#e87ba4"),
}


def make_ratio_figure(path):
    """Constant-free comparison with the theory: the unknown constants and
    polylog factors in the Theorem 1 rate are shared between the two kernels,
    so the ratio of mean simple regrets r_vanilla / r_invariant is directly
    comparable with the theoretical |G|^{1/2} offset (dashed levels)."""
    import os
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    plotted = False
    for name, (label, g_order, color) in RATIO_SPEC.items():
        f = f"results/{name}.npz"
        if not os.path.exists(f):
            continue
        data = np.load(f)
        mean_v = np.maximum(np.mean(data["vanilla_regret"], axis=0),
                            REGRET_FLOOR)
        mean_i = np.maximum(np.mean(data["invariant_regret"], axis=0),
                            REGRET_FLOOR)
        ratio = mean_v / mean_i
        n_obs = N_INIT + np.arange(ratio.size)
        ax.plot(n_obs, ratio, color=color, linewidth=2)
        if "normalized_regret" in data.files:
            mean_n = np.maximum(np.mean(data["normalized_regret"], axis=0),
                                REGRET_FLOOR)
            ax.plot(n_obs, mean_v / mean_n, color=color, linewidth=1.6,
                    linestyle=(0, (5, 1.5, 1, 1.5)), alpha=0.8)
        ax.axhline(np.sqrt(g_order), color=color, linestyle=(0, (4, 3)),
                   linewidth=1.2, alpha=0.55)
        ax.annotate(label, (n_obs[-1], ratio[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("Observations")
    ax.set_ylabel("Simple-regret ratio, vanilla / invariant")
    ax.set_title("Empirical regret ratio (solid) vs theoretical "
                 "$\\sqrt{|G|}$ offset (dashed)", fontsize=11, color=TEXT)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=TEXT_2, linewidth=2,
               label="vanilla / orbit-averaged"),
        Line2D([], [], color=TEXT_2, linewidth=1.6,
               linestyle=(0, (5, 1.5, 1, 1.5)),
               label="vanilla / normalised orbit-avg."),
        Line2D([], [], color=TEXT_2, linestyle=(0, (4, 3)), linewidth=1.2,
               label="$\\sqrt{|G|}$ (theory, Thm 1)"),
    ], frameon=False, fontsize=9, loc="upper left")
    style_axes(ax)
    ax.margins(x=0.14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def main():
    names = sys.argv[1:] or ALL_EXPERIMENTS
    for name in names:
        print(f"[{name}]", flush=True)
        exp, results = run_experiment(name)
        print(f"  target: {exp['n_modes']} modes, "
              f"mean ||f||_H over worlds = {exp['mean_norm']:.3f}", flush=True)
        np.savez(f"results/{name}.npz",
                 **{f"{kname}_{key}": arr
                    for kname, traces in results.items()
                    for key, arr in traces.items()})
        make_figure(exp, results, f"plots/{name}.png")
    make_ratio_figure("plots/ratio_summary.png")


if __name__ == "__main__":
    main()
