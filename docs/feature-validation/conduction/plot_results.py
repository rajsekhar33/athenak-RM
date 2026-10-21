#!/usr/bin/env python3
"""Recreate the report figures from bundled numerical data.

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
LABELS = {"vanleer": "van Leer", "mc": "MC"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((HERE / "results.json").read_text())
    fields = np.load(HERE / "physical_profiles.npz", allow_pickle=False)
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

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    x = fields["ring_mc_x"]
    for ax, values, title, vmax in (
            (axs[0, 0], fields["ring_mc_initial"], "Initial hot angular patch", 12),
            (axs[0, 1], fields["ring_mc_temperature"], "MC: t = 200", 10.18),
            (axs[1, 0], fields["ring_vanleer_temperature"], "van Leer: t = 200", 10.18)):
        im = ax.imshow(values.T, origin="lower", extent=(-1, 1, -1, 1),
                       vmin=10, vmax=vmax, cmap="inferno", interpolation="nearest")
        for radius in (.5, .7):
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor="cyan",
                                   linewidth=.7, linestyle="--"))
        ax.set(xlabel="x", ylabel="y", title=title)
        fig.colorbar(im, ax=ax, label=r"Temperature $T=p/\rho$", shrink=.8)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    radius = np.hypot(xx, yy)
    edges = np.linspace(.3, .9, 121)
    count = np.histogram(radius, bins=edges)[0]
    for limiter in ("vanleer", "mc"):
        mean = np.histogram(radius, bins=edges,
                            weights=fields["ring_" + limiter + "_temperature"]-10)[0]/count
        axs[1, 1].plot((edges[1:]+edges[:-1])/2, mean, color=COLORS[limiter], label=LABELS[limiter])
    axs[1, 1].plot([.3, .5, .5, .7, .7, .9], [0, 0, 1/6, 1/6, 0, 0], "k--",
                   label="Continuum angular mean")
    axs[1, 1].set(xlabel="Radius r", ylabel=r"Angular-mean excess $\langle T-10\rangle$",
                   title="Radial profile: cross-field leakage")
    axs[1, 1].legend(fontsize=8)
    fig.suptitle(r"Circular-field diffusion: $400^2$ cells, $D_\parallel=0.01$, $D_\perp=0$", fontsize=13)
    save(fig, "ring_solution")

    fig, axs = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
    for row, dimension in enumerate((2, 3)):
        prefix = f"gaussian{dimension}"
        x = fields[prefix + "_x"]
        n = len(x)
        initial = fields[prefix + "_initial"]
        observed = fields[prefix + "_temperature"]
        reference = fields[prefix + "_reference"]
        if dimension == 3:
            initial, observed, reference = (a[:, :, n//2-1] for a in
                                             (initial, observed, reference))
        for col, values, title in ((0, initial, "Initial Gaussian"),
                                    (1, observed, "Temperature at t = 0.02")):
            im = axs[row, col].imshow((values-1).T, origin="lower", extent=(0, 1, 0, 1),
                                     vmin=0, vmax=.1, cmap="inferno", interpolation="nearest")
            axs[row, col].set(xlabel="x", ylabel="y", title=f"{dimension}D N{n}: {title}")
            fig.colorbar(im, ax=axs[row, col], label=r"$T-1$", shrink=.8)
        axs[row, 1].contour(x, x, (reference-1).T, levels=(.01, .03, .06),
                            colors="cyan", linewidths=.7)
        index = n//2-1
        axs[row, 2].plot(x, initial[:, index]-1, color=".6", label="Initial")
        axs[row, 2].plot(x, reference[:, index]-1, "k--", label="Analytic cell average")
        axs[row, 2].plot(x, observed[:, index]-1, "o", color=COLORS["mc"],
                        markersize=3, markerfacecolor="none", label="MC measured")
        axs[row, 2].set(xlabel="x", ylabel=r"Temperature excess $T-1$",
                        title="Cut through central cell row")
        axs[row, 2].legend(fontsize=8)
    fig.suptitle(r"Oblique Gaussian diffusion: $\mathbf{B}=(1,2,3)$; cyan contours: analytic solution", fontsize=13)
    save(fig, "gaussian_solution")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    x = fields["sovinec_mc_x"]
    target = fields["sovinec_mc_reference"]
    im = axs[0].imshow((target-1).T, origin="lower", extent=(-.5, .5, -.5, .5),
                       vmin=0, vmax=1, cmap="inferno", interpolation="nearest")
    axs[0].contour(x, x, target.T-1, levels=(.2, .4, .6, .8), colors="cyan", linewidths=.7)
    axs[0].set(xlabel="x", ylabel="y", title="Continuum equilibrium; field follows contours")
    fig.colorbar(im, ax=axs[0], label=r"$T_*-1$")
    axs[1].plot(x, target[:, 7:9].mean(axis=1)-1, "k--", label="Continuum equilibrium")
    axs[1].plot(x, fields["sovinec_isotropic_temperature"][:, 7:9].mean(axis=1)-1,
                "k:", label="Isotropic discrete control")
    for limiter in ("vanleer", "mc"):
        values = fields["sovinec_" + limiter + "_temperature"][:, 7:9].mean(axis=1)-1
        axs[1].plot(x, values, "o-", color=COLORS[limiter], label=LABELS[limiter])
    axs[1].set(xlabel="x", ylabel=r"Temperature excess $T-1$",
                title=r"N16 central-row average at $t=0.5$")
    axs[1].legend(fontsize=8)
    fig.suptitle(r"Source-balanced diffusion: $D_\parallel/D_\perp=100$", fontsize=13)
    save(fig, "sovinec_solution")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
    for ax, key, title in zip(axs, ("paper_l1", "paper_l2", "paper_linf"),
                              ("L1", "L2", "L infinity")):
        for limiter in ("vanleer", "mc"):
            rows = sorted((r for r in data["ring"] if r["configuration"]["limiter"] == limiter),
                          key=lambda r: r["configuration"]["n"])
            n = [r["configuration"]["n"] for r in rows]
            ax.plot(n, [r["metrics"][key] for r in rows], "o-", color=COLORS[limiter],
                    label=LABELS[limiter] + " measured")
            ax.plot(n, [r["reference"][key] for r in rows], "x--", color=COLORS[limiter],
                    label=LABELS[limiter] + " published")
        ax.set(xscale="log", xticks=n, xlabel="Cells per direction", ylabel=title)
        ax.set_xticklabels(n)
        ax.minorticks_off()
    axs[0].legend(fontsize=8)
    fig.suptitle("Ring diffusion at t = 200: Sharma–Hammett comparison\n"
                 "N50: historical CPU; N100–400: GPU; explicit constant coefficient", fontsize=12)
    save(fig, "ring_errors")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    axs[0].plot(fields["pulse_x"], fields["pulse_initial"], "o--", color=".6", label="Initial")
    axs[0].plot(fields["pulse_x"], fields["pulse_final"], "o-", color="black", label="RK3 reference, t = 0.2")
    axs[0].set(xlabel="x", ylabel=r"Temperature $T=p/\rho$", title="Nonlinear pulse on the fixed N8 grid")
    axs[0].legend(fontsize=8)
    ax = axs[1]
    for i, color in zip((1, 2, 3), ("#0072B2", "#D55E00", "#009E73")):
        m = data["explicit_temporal"]["checks"][f"rk{i}_time_refinement"]["metrics"]
        h = np.asarray(m["mean_actual_dt"])
        error = np.asarray(m["reference_errors_over_amplitude"])
        order = m["observed_self_convergence_order"]
        ax.loglog(h, error, "o-", color=color, label=f"RK{i}: self-order {order:.3f}")
        ax.loglog(h, error[-1] * (h / h[-1])**i, ":", color=color, alpha=.7)
    ax.set(xlabel="Mean timestep", ylabel="Max temperature-diagnostic error / amplitude")
    fig.suptitle("Nonlinear saturated pulse: explicit temporal refinement\n"
                 "N8, t = 0.2; numerical RK3 reference; dotted orders 1, 2, 3", fontsize=11)
    ax.legend()
    save(fig, "explicit_temporal")

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    x = fields["coupled_rkl_x"]
    axs[0].plot(x, fields["coupled_initial"][:, 7], color=".6", label="Initial")
    axs[0].plot(x, fields["coupled_explicit_temperature"][:, 7], "k-", label="Explicit reference, t = 0.2")
    axs[0].plot(x, fields["coupled_rkl_temperature"][:, 7], "o", markerfacecolor="none",
                color=COLORS["mc"], label="RKL2, t = 0.2")
    axs[0].set(xlabel="x", ylabel=r"Temperature $T=p/\rho$",
                title=r"Coupled MHD: central-row temperature")
    axs[0].legend(fontsize=8)
    ax = axs[1]
    m = data["coupled_temporal"]["convergence"]["metrics"]
    h = np.asarray(m["H"])
    for key, label, color in (("aggregate", "Largest scaled U/CT component L1", "#0072B2"),
                              ("temperature", "Temperature L1 / amplitude", "#D55E00")):
        error = np.asarray([e[key] for e in m["errors"]])
        ax.loglog(h, error, "o-", color=color, label=label)
        ax.loglog(h, error[-1] * (h/h[-1])**2, ":", color=color, alpha=.7)
    ax.axhline(.005, color=".4", linestyle="--", label="Component/temperature tolerance (0.005)")
    ax.set(xlabel=r"Effective timestep $H=[\sum(\Delta t)^3/t]^{1/2}$",
           ylabel="Normalized difference from explicit reference",
           title="Coupled MHD: harmonic Spitzer + saturation (RKL2)")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(.97, .04, "N16, t = 0.2; RKL ratio cap 64\n"
            "Dotted lines: second order\n"
            "Fixed-grid temporal convergence",
            transform=ax.transAxes, fontsize=8, ha="right")
    save(fig, "coupled_rkl_temporal")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    profiles = data["sawtooth"]["profiles"]
    reference = profiles["saw_N400_explicit_cfl0p2"]
    x = np.asarray(reference["final"]["x"])
    ref = np.asarray(reference["final"]["temperature"])
    axs[0].plot(x, reference["initial"]["temperature"], color=".7", label="Initial cell averages")
    axs[0].plot(x, ref, color="black", label="Explicit RK2 CFL 0.2")
    for ratio, color, style in ((40, "#0072B2", "--"), (1000, "#D55E00", ":")):
        t = np.asarray(profiles[f"saw_N400_rkl{ratio}"]["final"]["temperature"])
        axs[0].plot(x, t, linestyle=style, color=color, label=f"RKL2 ratio cap {ratio}")
        axs[1].plot(x, (t-ref)/20, color=color, label=f"RKL2 ratio cap {ratio}")
    other = np.asarray(profiles["saw_N400_explicit_cfl0p4"]["final"]["temperature"])
    axs[1].plot(x, (other-ref)/20, ":", color=".4", label="Explicit timestep-halving difference")
    axs[0].set(xlabel="x", ylabel="Temperature", title="N400 profiles at t = 0.1")
    axs[1].set(xlabel="x", ylabel="Temperature difference / 20",
               title="Difference from explicit RK2 reference", yscale="symlog")
    axs[1].set_yscale("symlog", linthresh=1e-8)
    for ax in axs:
        ax.legend(fontsize=8)
    axs[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.16))
    fig.suptitle("Saturated sawtooth: Vaidya et al. (2017)\n"
                 "Explicit RK2 and RKL2 temperature profiles", fontsize=12)
    save(fig, "saturated_sawtooth")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for limiter in ("vanleer", "mc"):
        rows = [data["coarse_sovinec"]["cases"][f"sovinec_r100_n{n}_{limiter}"] for n in (8, 16)]
        axs[0].plot([8, 16], [r["independent_l1"] for r in rows], "o-",
                    color=COLORS[limiter], label=LABELS[limiter])
        axs[1].plot([8, 16], [r["fraction_of_parallel"] for r in rows], "o-",
                    color=COLORS[limiter], label=LABELS[limiter])
    axs[0].set(ylabel="Measured L1 temperature error", title="Steady-state temperature error")
    axs[0].axhline(.05, color=".4", linestyle="--", label="Accuracy criterion: 0.05")
    axs[1].set(ylabel="Numerical perpendicular / parallel diffusivity",
               title="Cross-field leakage decreases with refinement")
    for ax in axs:
        ax.set(xlabel="Cells per direction", xticks=[8, 16])
        ax.legend()
    fig.suptitle("Coarse Sovinec equilibrium: diffusivity ratio 100\n"
                 "N8 and N16 exceed the accuracy tolerance", fontsize=12)
    save(fig, "retained_coarse_limits")
    print("Created eight figures from bundled results.json and physical_profiles.npz.")


if __name__ == "__main__":
    main()
