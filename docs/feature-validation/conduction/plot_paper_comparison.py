#!/usr/bin/env python3
"""Plot thermal-mode benchmarks from the bundled measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=here / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((here / "paper_comparison.json").read_text())
    path = here / "paper_comparison_series.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == data["series_sha256"]
    values = np.load(path, allow_pickle=False)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    colors = {16: "#d1495b", 32: "#4477aa", 64: "#228833"}

    def save(fig, name):
        for ax in fig.axes:
            ax.grid(alpha=0.2)
        fig.savefig(args.output / (name + ".png"), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for col, label in enumerate(("C1", "C2")):
        for rec in data["mode_series"]:
            c = rec["case"]
            if c["label"] != label or (label == "C2" and c["limiter"] != "mc"):
                continue
            name = rec["name"]
            tau = values[name + "_time"] * c["clock_rate"]
            amplitude = c["amplitude"] * c["fv_factor"]
            error = values[name + "_mode1_l2"] / rec["initial_rms"]
            style = "-" if c["method"] == "rk1" else "--"
            line = dict(color=colors[c["n"]], linestyle=style,
                        label=f"N={c['n']}, {c['method'].upper()}")
            if c["method"] == "rkl2" and len(tau) < 40:
                line.update(marker="o", markersize=3, markerfacecolor="none")
            axes[0, col].plot(tau, values[name + "_mode1_amp"] / amplitude, **line)
            positive = (error > 0) & (tau > 0)
            axes[1, col].semilogy(tau[positive], error[positive], **line)
        tau = np.linspace(0, 8, 300)
        axes[0, col].plot(tau, np.exp(-tau), "k:", linewidth=2,
                          label=r"Analytic $\exp(-\Gamma t)$")
        axes[0, col].set(title=("Isotropic conduction" if col == 0 else
                                "Parallel conduction: MC limiter"),
                         ylabel=r"Temperature-mode amplitude $A(t)/A(0)$")
        axes[1, col].set(xlabel=r"Time $\Gamma t$",
                         ylabel=r"$L_2$ error / initial perturbation RMS")
        axes[0, col].legend(fontsize=8, ncol=2)
    save(fig, "paper_thermal_modes")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for label, color in (("C1", "#0072B2"), ("C2", "#D55E00")):
        for rec in data["spatial"]:
            if rec["label"] != label or (label == "C2" and rec["limiter"] != "mc"):
                continue
            style = "o-" if rec["method"] == "rk1" else "s--"
            axes[0].loglog(rec["n"], rec["maximum_l2"], style, color=color,
                            label=f"{'Isotropic' if label == 'C1' else 'Parallel MC'}, "
                                  + rec["method"].upper())
    n = np.array([16., 32., 64.])
    axes[0].loglog(n, 1e-9 * (32 / n) ** 2, "k:", label=r"$N^{-2}$")
    axes[0].set(xticks=n, xticklabels=["16", "32", "64"],
                 xlabel="Cells per direction N", ylabel=r"Maximum $L_2$ temperature error",
                 title="Spatial refinement")
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    t = data["temporal"]
    h = np.array([t["actual_dt_max"][key] for key in ("1", "2", "4", "16")])
    e = np.array([t["errors_to_finest"][key] for key in ("1", "2", "4", "16")])
    axes[1].loglog(h, e, "o-", color="#D55E00", label="Parallel conduction, RKL2")
    axes[1].loglog(h, e[0] * (h / h[0]) ** 2, "k:", label=r"$H^2$")
    axes[1].set(xlabel="Maximum completed timestep H",
                 ylabel="Final temperature RMS difference from H/32",
                 title=r"Temporal refinement at $32^3$")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, "paper_thermal_convergence")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    clock = (5 / 3 - 1) * 11000 * (2 * np.pi / 20) ** 2
    for rec in data["oblique_null"]["cases"]:
        n, limiter, name = rec["n"], rec["limiter"], rec["name"]
        if limiter != "vanleer" or n not in (32, 64):
            continue
        color = {32: "#0072B2", 64: "#D55E00"}[n]
        axes[0].plot(values[name + "_time"] * clock,
                      values[name + "_amplitude"] / rec["initial_amplitude"],
                      color=color, label=f"N={n}, reduced stencil")
        for native in rec["native"]:
            key = native["name"]
            stride = max(1, len(values[key + "_time"]) // 16)
            axes[0].plot(values[key + "_time"][::stride] * clock,
                          values[key + "_amplitude"][::stride] / rec["initial_amplitude"],
                          "o" if native["sts_ratio"] == 50 else "x", color=color,
                          markerfacecolor="none", markersize=5,
                          label=f"N={n}, AthenaK, cap {int(native['sts_ratio'])}")
    axes[0].axhline(1, color="k", linestyle=":", label="Stationary continuum mode")
    axes[0].set(xlabel=r"Time $\tau=(\gamma-1)\alpha_\parallel k^2t$",
                 ylabel=r"Fundamental amplitude $A(t)/A(0)$", title="van Leer thermal mode")
    for limiter, color, marker, label in (("vanleer", "#0072B2", "o", "van Leer"),
                                           ("mc", "#D55E00", "s", "MC")):
        rows = [r for r in data["oblique_null"]["cases"] if r["limiter"] == limiter]
        axes[1].loglog([r["n"] for r in rows],
                       [r["final_l2_initial_rms_scaled"] for r in rows],
                       marker + "-", color=color, label=label + ": reduced stencil")
        for r in rows:
            for native in r["native"]:
                if native["sts_ratio"] == 50:
                    axes[1].plot(r["n"], native["final_error_initial_rms_scaled"],
                                  "kx", markersize=8, label="AthenaK, cap 50" if
                                  limiter == "vanleer" and r["n"] == 32 else None)
    axes[1].set(xticks=[32, 64, 128, 256], xticklabels=["32", "64", "128", "256"],
                 xlabel="Cells per direction N",
                 ylabel=r"$L_2$ error / initial perturbation RMS", title=r"Cross-field leakage at $\tau=8$")
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    for ax in axes:
        ax.legend(fontsize=7.8)
    fig.suptitle(r"Stationary thermal wave: $\mathbf{B}\parallel(1,-1,0)$, $\mathbf{k}=k(1,1,0)$")
    save(fig, "paper_oblique_null")


if __name__ == "__main__":
    main()
