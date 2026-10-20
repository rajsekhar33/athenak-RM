#!/usr/bin/env python3
"""Reproduce the physical-viscosity and Braginskii figures from bundled JSON.

Requires only NumPy and Matplotlib. No simulation or private-data access occurs.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


def read_data(root, name):
    return json.loads((root / "data" / name).read_text())


def style_axes(axes):
    for ax in axes:
        ax.grid(True, which="both", alpha=0.22)
        ax.set_axisbelow(True)


def physical_figure(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    ratios = np.array([1, 4, 16])
    for backend, marker in [("cpu", "o"), ("gpu", "x")]:
        for thermal, color in [("warm", "tab:blue"), ("cold", "tab:orange")]:
            values = [data["scalar_sts"][backend]["cases"]
                      [f"{thermal}_rkl2_ratio{r}"]
                      ["velocity_semidiscrete_time_error"] for r in ratios]
            axes[0].loglog(ratios, values, marker=marker, color=color,
                           linestyle="-" if backend == "cpu" else "none",
                           markersize=7, label=f"{thermal}, {backend.upper()}")
    gate = data["scalar_sts"]["cpu"]["gates"][
        "sts_velocity_semidiscrete_initial_amplitude_scaled"]
    axes[0].axhline(gate, color="black", linestyle="--", label="0.05% accuracy gate")
    axes[0].set(xlabel="Full-cycle / parabolic time-step ratio",
                ylabel="Velocity time error / initial amplitude",
                title="Uniform-grid RKL2 time accuracy")
    axes[0].set_xticks(ratios, [str(r) for r in ratios])
    axes[0].text(0.03, 0.96, "Warm/cold curves overlap\nRatio 16 misses the accuracy gate",
                 transform=axes[0].transAxes, va="top", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    end = data["amr_end_to_end"]["comparisons"]
    fixed = data["amr_fixed_remap"]["comparisons"]
    axes[1].loglog(ratios, [end[f"amr_r{r}_vs025"]["energy"] for r in ratios],
                   "o-", label="End-to-end: refinement timing varies")
    axes[1].loglog(ratios, [fixed[f"r{r}_vs025"]["energy"] for r in ratios],
                   "s-", label="Common post-refinement checkpoint")
    axes[1].axhline(data["amr_fixed_remap"]["gate"], color="black",
                    linestyle="--", label="0.5% profile-error gate")
    axes[1].set(xlabel="Full-cycle / parabolic time-step ratio",
                ylabel="Energy-profile L1 / initial total energy",
                title="Cold active-AMR accuracy qualification")
    axes[1].set_xticks(ratios, [str(r) for r in ratios])
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Standard isotropic viscosity: time-step accuracy is not guaranteed by stability\n"
                 "Standard-operator follow-up preceding the final 31556aa8 source selection",
                 fontsize=12)
    style_axes(axes)
    fig.savefig(output / "physical_viscosity.png", dpi=170)
    plt.close(fig)


def braginskii_figure(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    families = [("fast_oblique", "Fast", "tab:blue"),
                ("slow_oblique", "Slow", "tab:orange"),
                ("alfven", "Alfvén", "tab:green")]
    for kind, label, color in families:
        rows = sorted((c for c in data["rotated_waves"]["cases"] if c["kind"] == kind),
                      key=lambda c: c["n"])
        axes[0].loglog([c["n"] for c in rows], [c["field_relative_l2"] for c in rows],
                       "o-", color=color, label=label)
    axes[0].loglog([32, 64], [0.01, 0.0025], "k--", label=r"$N^{-2}$ guide")
    axes[0].set(title="Rotated 2D coupled MHD waves (GPU)", xlabel="Cells per direction",
                ylabel="Field-relative L2 error")
    axes[0].set_xticks([32, 64], ["32", "64"])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].text(0.03, 0.03, "Earlier centered-operator spatial study;\npre-final STS guards, explicit evolution",
                 transform=axes[0].transAxes, fontsize=8, va="bottom")
    axes[0].legend(fontsize=8, loc="upper right")
    for kind, label, color in [("sheet", "Sheet", "tab:blue"),
                               ("oblique_damped", "Oblique damped", "tab:orange"),
                               ("oblique_null", "Oblique null", "tab:green")]:
        for backend, marker in [("cpu", "o"), ("gpu", "x")]:
            rows = sorted((c for c in data["heating"][backend]["cases"]
                           if c["kind"] == kind), key=lambda c: c["n"])
            axes[1].loglog([c["n"] for c in rows],
                           [c["thermal_gain_l1_relative"] for c in rows],
                           color=color, marker=marker, markersize=7,
                           linestyle="-" if backend == "cpu" else "none",
                           label=f"{label}, {backend.upper()}")
    axes[1].set(title="Local thermal-gain convergence", xlabel="Cells per direction",
                ylabel="Normalized thermal-gain L1 error")
    axes[1].set_xticks([32, 64, 128], ["32", "64", "128"])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].text(0.03, 0.03, "Isolated production kernel, native RK2; not MHD/STS\n"
                 "Null case normalized by initial kinetic energy",
                 transform=axes[1].transAxes, fontsize=8, va="bottom")
    axes[1].legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Braginskii viscosity: centered CT-aware stress\n"
                 "Thermal-gain kernel source matches feature commit 02b50eca",
                 fontsize=12)
    style_axes(axes)
    fig.savefig(output / "braginskii_validation.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Destination for the two PNGs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output or root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    physical_figure(read_data(root, "viscosity_sts_results.json"), output)
    braginskii_figure(read_data(root, "braginskii_results.json"), output)


if __name__ == "__main__":
    main()
