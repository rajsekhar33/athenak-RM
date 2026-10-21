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
    axes[0].axhline(gate, color="black", linestyle="--", label="0.05% tolerance")
    axes[0].set(xlabel="Full-cycle / parabolic time-step ratio",
                ylabel="Velocity time error / initial amplitude",
                title="Uniform-grid RKL2 time accuracy")
    axes[0].set_xticks(ratios, [str(r) for r in ratios])
    axes[0].text(0.03, 0.96, "Warm/cold curves overlap",
                 transform=axes[0].transAxes, va="top", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    end = data["amr_end_to_end"]["comparisons"]
    fixed = data["amr_fixed_remap"]["comparisons"]
    axes[1].loglog(ratios, [end[f"amr_r{r}_vs025"]["energy"] for r in ratios],
                   "o-", label="End-to-end: refinement timing varies")
    axes[1].loglog(ratios, [fixed[f"r{r}_vs025"]["energy"] for r in ratios],
                   "s-", label="Common post-refinement checkpoint")
    axes[1].axhline(data["amr_fixed_remap"]["gate"], color="black",
                    linestyle="--", label="0.5% tolerance")
    axes[1].set(xlabel="Full-cycle / parabolic time-step ratio",
                ylabel="Energy-profile L1 / initial total energy",
                title="Cold gas with adaptive mesh refinement")
    axes[1].set_xticks(ratios, [str(r) for r in ratios])
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Isotropic viscosity: RKL2 temporal and AMR accuracy", fontsize=12)
    style_axes(axes)
    fig.savefig(output / "physical_convergence.png", dpi=170)
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
    axes[0].text(0.03, 0.03, "Centered spatial operator, explicit integration",
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
    axes[1].text(0.03, 0.03, "Isolated viscous evolution with RK2\n"
                 "Null case: error / initial kinetic energy",
                 transform=axes[1].transAxes, fontsize=8, va="bottom")
    axes[1].legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Braginskii viscosity: wave and heating convergence", fontsize=12)
    style_axes(axes)
    fig.savefig(output / "braginskii_validation.png", dpi=170)
    plt.close(fig)


def measured_profiles(data, output):
    """Plot archived native arrays; analytic references are labeled separately."""
    physical = data['physical_shear']
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), layout='constrained')
    x = np.asarray(physical['x'])
    axes[0].plot(x, .3*np.cos(2*np.pi*x), ':', color='0.5', label='Initial velocity')
    axes[0].plot(x, physical['continuum_velocity'], 'k-', label='Continuum diffusion')
    axes[0].plot(x, physical['final_velocity'], 'o', ms=4, mfc='none', label='AthenaK, N=32')
    axes[0].set(xlabel=r'$x/L$', ylabel=r'$v_z$', title=f"Shear decay at t={physical['time']:.5f}")
    t = np.asarray(physical['times'])
    kinetic = np.asarray(physical['kinetic'])
    axes[1].plot(t, kinetic/kinetic[0], color='C0', label='AthenaK kinetic energy / initial kinetic')
    axes[1].plot(t, np.asarray(physical['thermal_gain'])/kinetic[0], color='C1', label='AthenaK generated heat / initial kinetic')
    axes[1].plot(t, np.asarray(physical['continuum_kinetic'])/kinetic[0], 'k--', label='Continuum kinetic decay')
    axes[1].set(xlabel=r'$t$ (code time units)', ylabel='Volume-averaged energy fraction', title='Kinetic energy becomes heat')
    for ax in axes:
        ax.legend(fontsize=8)
    style_axes(axes)
    fig.suptitle(r'Isotropic viscosity: $v_z(x,0)=0.3\cos(2\pi x)$, $\nu=0.1$')
    fig.savefig(output/'physical_viscosity.png', dpi=170)
    plt.close(fig)

    sheet = data['braginskii_sheet']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout='constrained')
    finest = sheet['profiles'][-1]
    for ax, field, label in zip(axes, ['vx', 'vy', 'heat'], [r'$v_x/c$', r'$v_y/c$', r'$\Delta u$']):
        ax.plot(finest['x'], finest['exact_'+field], 'k-', label='Fourier-series reference')
        for row, marker, color in zip(sheet['profiles'][:2], ['s', 'o'], ['C1', 'C0']):
            ax.plot(row['x'], row[field], marker, ms=3.5, mfc='none', color=color,
                    label=f"AthenaK, N={row['n']}")
        if field in ('vx', 'vy'):
            ax.plot(finest['x'], finest['initial_'+field], ':', color='0.5', label='Initial profile')
        ax.set(xlabel=r'$x/L$', ylabel=label)
    axes[0].set_title('Oblique stress generates transverse motion')
    axes[1].set_title('The initial shear broadens')
    axes[2].set_title('Heat is generated near velocity gradients')
    axes[1].legend(fontsize=8)
    style_axes(axes)
    fig.suptitle(r'Braginskii shear layer: $\hat b=(\hat x+\hat y)/\sqrt{2}$, $\nu_\parallel=10^{-3}$, $t=25$')
    fig.savefig(output/'braginskii_profiles.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Destination for the four PNGs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output or root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    physical_figure(read_data(root, "viscosity_sts_results.json"), output)
    braginskii_figure(read_data(root, "braginskii_results.json"), output)
    measured_profiles(read_data(root, "physical_profiles.json"), output)


if __name__ == "__main__":
    main()
