"""Plot the bundled magnetic diffusion measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "data/paper_diffusion_results.json").read_text())
TITLES = {
    "M1": "Ohmic diffusion",
    "M2": "Ambipolar diffusion",
    "M3": "Ohmic + ambipolar diffusion",
}
COLORS = {"M1": "#0072B2", "M2": "#D55E00", "M3": "#009E73"}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def main():
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True,
                             layout="constrained")
    clock = np.linspace(0, 8, 400)
    for ax, label in zip(axes, TITLES):
        ax.semilogy(clock, np.exp(-clock), "k-", label=r"$e^{-\Gamma t}$")
        for row in DATA["series"]:
            if row["case"] != label:
                continue
            indices = np.unique(np.r_[np.arange(0, len(row["time"]), 18),
                                      len(row["time"]) - 1])
            t = np.asarray(row["time"])[indices] * row["rate"]
            amplitude = (np.asarray(row["magnetic_mode_amplitude"])[indices]
                         / row["initial_cell_average_amplitude"])
            is_rkl = row["method"] == "rkl2"
            ax.semilogy(t, amplitude, "x" if is_rkl else "o", ms=5,
                        markerfacecolor="none", color="#D55E00" if is_rkl
                        else "#0072B2", label="RKL2" if is_rkl else "RK1")
        ax.set(title=TITLES[label], xlabel=r"Time $\Gamma t$", xlim=(0, 8))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_ylabel(r"Magnetic mode amplitude $A(t)/A(0)$")
    fig.suptitle(r"Weak magnetic perturbation: $64^3$ cells")
    fig.savefig(figures / "paper_magnetic_decay.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for row in DATA["spatial"]:
        n = np.asarray(row["n"])
        error = np.asarray(row["maximum_l2_initial_rms_scaled"]) * 100
        style = "o--" if row["method"] == "rkl2" else "o-"
        axes[0].loglog(n, error, style, color=COLORS[row["label"]],
                       label=f"{TITLES[row['label']]}, {row['method'].upper()}")
    n = np.array([16, 32, 64])
    axes[0].loglog(n, 0.08 * (16/n)**2, "k:", label=r"$N^{-2}$")
    axes[0].set(title="Spatial refinement", xlabel="Cells per direction",
                ylabel="Maximum field error (% initial RMS)")
    axes[0].set_xticks(n, [str(x) for x in n])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].legend(fontsize=8)

    row = DATA["temporal"][0]
    factors = ["1", "2", "4"]
    dt = np.array([row["actual_dt_max"][f] for f in factors])
    error = np.array([row["errors_to_finest"][f] for f in factors])
    axes[1].loglog(dt, error, "o-", color=COLORS["M1"], label="Ohmic RKL2")
    axes[1].loglog(dt, error[0]*(dt/dt[0])**2, "k:", label=r"$\Delta t^2$")
    axes[1].set(title=r"Time refinement at $32^3$", xlabel="Maximum timestep",
                ylabel="Final magnetic-field RMS difference")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.savefig(figures / "paper_magnetic_convergence.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
