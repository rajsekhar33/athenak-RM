#!/usr/bin/env python3
"""Recreate the five report figures from bundled data, without Athena or archives.

Requires Python 3, NumPy and Matplotlib. Run from any directory:
    python docs/feature-validation/conduction/plot_results.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {"vanleer": "#0072B2", "mc": "#D55E00"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((HERE / "results.json").read_text())
    for row in data["ring"]:
        for key, difference in row["error_relative_difference_percent"].items():
            measured = 100 * (row["metrics"][key] / row["reference"][key] - 1)
            assert abs(measured - difference) < 1e-10
    ref_temperature = np.asarray(data["sawtooth"]["profiles"][
        "saw_N400_explicit_cfl0p2"]["final"]["temperature"])
    for ratio in (40, 1000):
        name = f"saw_N400_rkl{ratio}"
        temperature = np.asarray(data["sawtooth"]["profiles"][name]["final"]["temperature"])
        difference = np.abs(temperature - ref_temperature) / 20
        reported = data["sawtooth"]["checks"][name + "_vs_fine_explicit"]
        assert abs(difference.mean() - reported["L1"]) < 1e-14
        assert abs(difference.max() - reported["Linf"]) < 1e-14
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 140})

    def save(fig, name):
        for ax in fig.axes:
            ax.grid(alpha=.2)
        fig.savefig(args.output / (name + ".png"), dpi=150)
        plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
    for ax, key, title in zip(axs, ("paper_l1", "paper_l2", "paper_linf"),
                              ("L1", "L2", "L infinity")):
        for limiter in ("vanleer", "mc"):
            rows = sorted((r for r in data["ring"] if r["configuration"]["limiter"] == limiter),
                          key=lambda r: r["configuration"]["n"])
            n = [r["configuration"]["n"] for r in rows]
            ax.plot(n, [r["metrics"][key] for r in rows], "o-", color=COLORS[limiter],
                    label=limiter + " measured")
            ax.plot(n, [r["reference"][key] for r in rows], "x--", color=COLORS[limiter],
                    label=limiter + " published")
        ax.set(xscale="log", xticks=n, xlabel="Cells per direction", ylabel=title)
        ax.set_xticklabels(n)
        ax.minorticks_off()
    axs[0].legend(fontsize=8)
    fig.suptitle("Ring diffusion at t = 200: Sharma–Hammett comparison\n"
                 "N50: historical CPU; N100–400: GPU; explicit constant coefficient", fontsize=12)
    save(fig, "ring_errors")

    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for i, color in zip((1, 2, 3), ("#0072B2", "#D55E00", "#009E73")):
        m = data["explicit_temporal"]["checks"][f"rk{i}_time_refinement"]["metrics"]
        h = np.asarray(m["mean_actual_dt"])
        error = np.asarray(m["reference_errors_over_amplitude"])
        order = m["observed_self_convergence_order"]
        ax.loglog(h, error, "o-", color=color, label=f"RK{i}: self-order {order:.3f}")
        ax.loglog(h, error[-1] * (h / h[-1])**i, ":", color=color, alpha=.7)
    ax.set(xlabel="Mean actual timestep", ylabel="Max double-observable error / amplitude")
    fig.suptitle("Nonlinear saturated pulse: explicit temporal refinement\n"
                 "N8, t = 0.2; numerical RK3 reference; dotted orders 1, 2, 3", fontsize=11)
    ax.legend()
    save(fig, "explicit_temporal")

    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    m = data["coupled_temporal"]["convergence"]["metrics"]
    h = np.asarray(m["H"])
    for key, label, color in (("aggregate", "Largest scaled U/CT component L1", "#0072B2"),
                              ("temperature", "Temperature L1 / amplitude", "#D55E00")):
        error = np.asarray([e[key] for e in m["errors"]])
        ax.loglog(h, error, "o-", color=color, label=label)
        ax.loglog(h, error[-1] * (h/h[-1])**2, ":", color=color, alpha=.7)
    ax.axhline(.005, color=".4", linestyle="--", label="Declared component/temperature gate")
    ax.set(xlabel=r"Effective timestep $H=[\sum(\Delta t)^3/t]^{1/2}$",
           ylabel="Normalized difference from fine explicit control",
           title="Coupled MHD + harmonic Spitzer/saturation: RKL2")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(.97, .04, "N16, t = 0.2; RKL ratio cap 64\n"
            "Dotted lines: second order\n"
            "Same-grid adaptive temporal study, not spatial convergence",
            transform=ax.transAxes, fontsize=8, ha="right")
    save(fig, "coupled_rkl_temporal")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    profiles = data["sawtooth"]["profiles"]
    reference = profiles["saw_N400_explicit_cfl0p2"]
    x = np.asarray(reference["final"]["x"])
    ref = np.asarray(reference["final"]["temperature"])
    axs[0].plot(x, reference["initial"]["temperature"], color=".7", label="Initial averages")
    axs[0].plot(x, ref, color="black", label="Explicit RK2 CFL 0.2")
    for ratio, color, style in ((40, "#0072B2", "--"), (1000, "#D55E00", ":")):
        t = np.asarray(profiles[f"saw_N400_rkl{ratio}"]["final"]["temperature"])
        axs[0].plot(x, t, linestyle=style, color=color, label=f"RKL2 ratio cap {ratio}")
        axs[1].plot(x, (t-ref)/20, color=color, label=f"RKL2 ratio cap {ratio}")
    other = np.asarray(profiles["saw_N400_explicit_cfl0p4"]["final"]["temperature"])
    axs[1].plot(x, (other-ref)/20, ":", color=".4", label="Explicit timestep-halving gap")
    axs[0].set(xlabel="x", ylabel="Temperature", title="N400 profiles at t = 0.1")
    axs[1].set(xlabel="x", ylabel="Temperature difference / 20",
               title="Difference from numerical explicit reference", yscale="symlog")
    axs[1].set_yscale("symlog", linthresh=1e-8)
    for ax in axs:
        ax.legend(fontsize=8)
    axs[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.16))
    fig.suptitle("Saturated sawtooth: Vaidya et al. setup\n"
                 "Own explicit/RKL2 comparison, not digitized paper curves", fontsize=12)
    save(fig, "saturated_sawtooth")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for limiter in ("vanleer", "mc"):
        rows = [data["coarse_sovinec"]["cases"][f"sovinec_r100_n{n}_{limiter}"] for n in (8, 16)]
        axs[0].plot([8, 16], [r["independent_l1"] for r in rows], "o-",
                    color=COLORS[limiter], label=limiter)
        axs[1].plot([8, 16], [r["fraction_of_parallel"] for r in rows], "o-",
                    color=COLORS[limiter], label=limiter)
    axs[0].set(ylabel="Measured L1 temperature error", title="Stationary does not mean accurate")
    axs[1].set(ylabel="Numerical perpendicular / parallel diffusivity",
               title="Cross-field leakage decreases with refinement")
    for ax in axs:
        ax.set(xlabel="Cells per direction", xticks=[8, 16])
        ax.legend()
    fig.suptitle("Retained coarse Sovinec ratio-100 accuracy FAILs\n"
                 "No relaxed gate and no extrapolated high-resolution acceptance", fontsize=12)
    save(fig, "retained_coarse_limits")
    print("Created five figures from bundled results.json only; no native tests run.")


if __name__ == "__main__":
    main()
