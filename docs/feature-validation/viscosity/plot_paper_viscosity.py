"""Plot the bundled viscous Fourier-mode measurements and analytic references."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


def main():
    base = Path(__file__).resolve().parent
    data = json.loads((base / "data/paper_viscosity.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.22, "savefig.dpi": 150})
    colors = {16: "#d1495b", 32: "#3973ac", 64: "#298c50"}
    for label, title in [("V1", "Isotropic viscous damping"),
                         ("V2", "Parallel Braginskii viscous damping")]:
        fig, axes = plt.subplots(3, 1, figsize=(8, 9.5),
                                 gridspec_kw={"height_ratios": [1, 1, 0.9]})
        rows = [r for r in data["curves"] if r["case"]["label"] == label]
        for row in rows:
            case, fields = row["case"], row["diagnostics"]
            n, method = case["n"], case["method"]
            time = np.asarray(fields["time"]) * case["clock_rate"]
            a0 = np.asarray(case["amplitudes"]) * case["fv_factor"]
            initial_rms = np.linalg.norm(a0) / np.sqrt(2)
            error = np.sqrt(sum(np.asarray(fields[f"mode{j}_l2"]) ** 2
                                for j in (1, 2, 3))) / initial_rms
            style = "-" if method == "rk1" else "--"
            axes[1].semilogy(time[1:], 100 * error[1:], style,
                             color=colors[n], label=f"N={n}, {method.upper()}")
            if n == 64:
                for j in ([1, 2] if label == "V1" else [1]):
                    amplitude = np.asarray(fields[f"mode{j}_amp"]) / a0[j - 1]
                    mode_name = "longitudinal" if j == 1 else "transverse"
                    indices = np.unique(np.linspace(0, len(time) - 1, 30).astype(int))
                    axes[0].plot(time[indices], amplitude[indices],
                                 "o" if method == "rk1" else "s", ms=4,
                                 mfc="none", color="#3973ac" if j == 1 else "#bf6731",
                                 label=f"{mode_name}, {method.upper()}")
                if method == "rk1":
                    fine_time = np.linspace(0, 8, 500)
                    axes[0].plot(fine_time, np.exp(-fine_time), "k-",
                                 label="analytic longitudinal")
                    if label == "V1":
                        axes[0].plot(fine_time, np.exp(-0.75 * fine_time), "k--",
                                     label="analytic transverse")
        for row in data["spatial"]:
            if row["label"] != label:
                continue
            n = np.asarray(row["n"])
            error = np.asarray(row["maximum_l2_initial_rms_scaled"]) * 100
            axes[2].loglog(n, error, "o-" if row["method"] == "rk1" else "s--",
                           label=row["method"].upper())
            if row["method"] == "rk1":
                axes[2].loglog(n, error[-1] * (n[-1] / n) ** 2, "k:",
                               label=r"$N^{-2}$")
        axes[0].set(title=title + r": $64^3$ modes", ylabel=r"$A_j(t)/A_j(0)$",
                    xlabel=r"Time $\Gamma_{\max}t$")
        axes[0].legend(fontsize=8, ncol=2)
        axes[1].set(ylabel="Velocity L2 error (% initial RMS)",
                    xlabel=r"Time $\Gamma_{\max}t$")
        axes[1].legend(fontsize=8, ncol=2)
        axes[2].set(xlabel="Cells per direction", ylabel="Maximum error (% initial RMS)")
        axes[2].set_xticks([16, 32, 64], ["16", "32", "64"])
        axes[2].xaxis.set_minor_formatter(NullFormatter())
        axes[2].legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(base / f"figures/paper_{label.lower()}_decay.png",
                    bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, row in zip(axes, data["temporal"]):
        divisors = [1, 2, 4, 16]
        x = np.array([row["actual_dt_max"][str(d)] for d in divisors])
        y = np.array([row["errors_to_finest"][str(d)] for d in divisors])
        ax.loglog(x, y, "o-", label="Measured final velocity difference")
        ax.loglog(x, y[0] * (x / x[0]) ** 2, "k--", label=r"$\Delta t^2$")
        ax.set(title="Isotropic viscosity" if row["label"] == "V1"
               else "Parallel Braginskii viscosity", xlabel="Maximum completed step (code time)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Velocity L2 difference from $H/32$")
    fig.suptitle(r"RKL2 temporal refinement on a fixed $32^3$ mesh")
    fig.tight_layout()
    fig.savefig(base / "figures/paper_viscosity_temporal.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
