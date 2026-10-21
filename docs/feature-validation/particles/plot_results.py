#!/usr/bin/env python3
"""Check and plot the bundled historical particle validation data.

Requires Python 3, NumPy, and Matplotlib; no simulation or private package is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PUSHERS = ("classical", "ito_2", "lagrangian_mc")
LABEL = {"classical": "Classical", "ito_2": "Itô-2", "lagrangian_mc": "MC"}
COLOR = {"classical": "#3b6fb6", "ito_2": "#111111", "lagrangian_mc": "#b33c2e"}
MARKER = {"classical": "o", "ito_2": "x", "lagrangian_mc": "+"}


def read_json(name):
    return json.loads((DATA / name).read_text())


def read_csv(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_data():
    """Recheck reduced-data contracts, not rerun native acceptance tests."""
    acceptance = read_json("acceptance.json")
    for suite in acceptance["particle_regressions"]:
        assert len(suite["cases"]) == 1
        assert suite["cases"][0]["status"] == "PASS"
        assert "[particles]" in suite["cases"][0]["name"]
    for suite in acceptance["functional_matrices"]:
        assert len(suite["groups"]) == 20
        for group in suite["groups"]:
            for check in group["checks"]:
                assert check["status"].startswith("PASS"), check
                assert not check.get("failures")
    for suite in acceptance["targeted_integration"]:
        for group in suite["groups"]:
            for check in group["checks"]:
                assert check["status"].startswith("PASS"), check
                assert not check.get("failures")

    boundary = read_json("boundary_snapshots.json")["cases"]
    assert len(boundary) == 12
    for name, case in boundary.items():
        assert case["status"] == "PASS" and not case["failures"]
        assert not case["restart_differences"]
        for row in case["snapshots"]:
            assert row["count"] == 256 and not row["invalid_active_owner_tags"]
            assert row["outside"] == row["frozen"]
            if name.endswith(("periodic", "reflect")):
                assert row["frozen"] == 0

    allocation = read_json("mc_child_allocation.json")["result"]
    assert allocation["status"] == "PASS" and allocation["uniform_model_rejected"]
    assert allocation["parents"] == 16 and allocation["comparisons"] == 64
    assert sum(g["count"] for g in allocation["groups"]) == 33191
    rejected = False
    for group in allocation["groups"]:
        n = group["count"]
        counts = np.asarray(group["counts"])
        probability = np.asarray(group["probabilities"])
        assert counts.sum() == n
        np.testing.assert_allclose(probability.sum(), 1.0, rtol=0, atol=1e-14)
        bound = np.sqrt(0.5 * n * np.log(2 * allocation["comparisons"] / allocation["alpha"]))
        np.testing.assert_allclose(bound, group["bound"], rtol=1e-14)
        assert np.max(np.abs(counts - n * probability)) <= bound
        rejected |= np.max(np.abs(counts - n / 4)) > bound
    assert rejected

    amr = read_json("amr_restart.json")["cases"]
    for pusher in PUSHERS:
        case = amr[pusher]
        assert case["status"] == "PASS" and not case["failures"]
        for series in [case["snapshots"], *case["restart"].values()]:
            for row in series:
                assert row["count"] == 512 and not row["invalid_owner_tags"]
                assert row["frozen"] == 0

    metadata = read_json("mach03_metadata.json")["cases"]
    spectra = read_csv("mach03_spectra.csv")
    for pusher, expected in metadata.items():
        rows = [r for r in spectra if r["pusher"] == pusher]
        k = np.array([float(r["k"]) for r in rows])
        gas = np.array([float(r["gas_power"]) for r in rows])
        tracer = np.array([float(r["tracer_power"]) for r in rows])
        coherence = np.array([float(r["signed_coherence"]) for r in rows])
        mask = (k >= 1) & (k <= 6)
        np.testing.assert_allclose(coherence[mask].mean(), expected["coherence_lowk"], atol=1e-14)
        response = coherence * np.sqrt(tracer / gas)
        np.testing.assert_allclose(response[mask].mean(), expected["response_lowk"], atol=1e-14)
        assert np.all(np.isfinite(response))
        assert expected["nparticles"] == 134217728
    convergence = read_csv("converging_flow_metrics.csv")
    assert len(convergence) == 45
    assert len({(r["resolution"], r["pusher"], r["seed"]) for r in convergence}) == 45
    paper = read_json("paper_comparisons.json")
    assert hashlib.sha256((DATA / "paper_comparisons.npz").read_bytes()).hexdigest() == paper["data_sha256"]
    with np.load(DATA / "paper_comparisons.npz") as values:
        for pusher in PUSHERS:
            valid = values["converging_valid"]
            relative = values[f"converging_{pusher}"] / values["converging_gas"] - 1
            expected = next(r for r in convergence if r["pusher"] == pusher
                            and int(r["resolution"]) == 128 and int(r["seed"]) == 42)
            np.testing.assert_allclose(np.abs(relative[valid]).mean(), float(expected["mean_abs_valid"]), atol=1e-14)
            case = paper["families"]["square_hi"]["cases"][pusher]
            profile = values[f"square_hi_{pusher}_cic"]
            error = profile / values["square_gas"] - 1
            np.testing.assert_allclose(np.sqrt(np.mean(error**2)), case["cic_relative_rms"])
            np.testing.assert_allclose(profile.mean(), values["square_gas"].mean())
            case = paper["families"]["turb64_iso"]["cases"][pusher]
            column = values[f"turb64_iso_{pusher}_column"]
            np.testing.assert_allclose(column.mean(), 1)
            np.testing.assert_allclose(np.log10(column / values["turb_gas_column"]).std(),
                                       case["column_log10_ratio_std"])
            np.testing.assert_allclose(np.sum(values[f"turb64_iso_{pusher}_ratio_pdf"] *
                                               np.diff(values["turb_ratio_edges"])), 1)
            np.testing.assert_allclose(np.sum(values[f"turb64_iso_{pusher}_sampled_gas_pdf"] *
                                               np.diff(values["turb_gas_log_edges"])), 1)
            assert values[f"turb64_iso_{pusher}_joint"].sum() == round(
                (1-case["empty_cic_cell_fraction"]) * 64**3)
        for name in ("turb_mass_pdf", "turb_volume_pdf"):
            np.testing.assert_allclose(np.sum(values[name] * np.diff(values["turb_gas_log_edges"])), 1)
    print("PASS: bundled numerical data and figure summaries")


def save(fig, out, name):
    fig.savefig(out / name, dpi=170)
    plt.close(fig)
    print(name)


def plot_square(out):
    data = np.load(DATA / "paper_comparisons.npz")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    x, gas = data["square_x"], data["square_gas"]
    axes[0].step(x, data["square_initial"], where="mid", color="0.6", ls="--",
                 lw=1.2, label="Translated initial profile")
    axes[0].plot(x, gas, color="#d8a020", lw=3, label="Finite-volume gas")
    for pusher in PUSHERS:
        tracer = data[f"square_hi_{pusher}_cic"]
        axes[0].plot(x, tracer, color=COLOR[pusher], lw=1.5, label=LABEL[pusher])
        axes[1].plot(x, tracer/gas-1, color=COLOR[pusher], lw=1.5)
    axes[0].set(ylabel=r"Transverse-mean density $\rho$", ylim=(0.85, 2.15))
    axes[1].set(xlabel=r"$x/L$", ylabel=r"$\rho_{\rm tr}/\rho_{\rm gas}-1$", xlim=(-0.5, 0.5))
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].axhline(0, color="0.5", lw=0.8)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.suptitle("Periodic square-pulse advection: three crossings\n"
                 r"$128\times128\times4$ cells; 1,310,720 tracers; CIC deposition")
    save(fig, out, "square_pulse.png")


def plot_turbulent_maps(out):
    from matplotlib.colors import LogNorm
    data = np.load(DATA / "paper_comparisons.npz")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.3), sharex=True, sharey=True,
                             layout="constrained")
    gas = data["turb_gas_column"]
    density_norm = LogNorm(vmin=0.15, vmax=6)
    for axis, array, label in zip(axes[0], [gas]+[data[f"turb64_iso_{p}_column"] for p in PUSHERS],
                                   ["Gas"]+[LABEL[p] for p in PUSHERS]):
        image = axis.imshow(array, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5),
                            norm=density_norm, cmap="magma", interpolation="nearest")
        axis.set_title(label)
    axes[1, 0].imshow(gas, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5),
                      norm=density_norm, cmap="magma", interpolation="nearest", alpha=0)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.05, 0.85, "Isothermal turbulence\n\n"
                    r"$64^3$ cells; 16 tracers/cell" "\n"
                    r"$t=0.2\,L/c_s$" "\n\nCIC deposition\nProjection along z",
                    transform=axes[1, 0].transAxes, va="top", fontsize=10)
    for axis, pusher in zip(axes[1, 1:], PUSHERS):
        ratio = data[f"turb64_iso_{pusher}_column"] / gas
        image_ratio = axis.imshow(ratio, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5),
                                   cmap="RdBu_r", norm=LogNorm(vmin=0.5, vmax=2),
                                   interpolation="nearest")
        axis.set_title("Tracer / gas")
        axis.set_xlabel(r"$x/L$")
    axes[0, 0].set_ylabel(r"$y/L$")
    axes[1, 1].set_ylabel(r"$y/L$")
    fig.colorbar(image, ax=axes[0], shrink=0.85, extend="both", label=r"$\Sigma/\langle\Sigma\rangle$")
    bar = fig.colorbar(image_ratio, ax=axes[1, 1:], shrink=0.85, extend="both",
                       label=r"$\Sigma_{\rm tr}/\Sigma_{\rm gas}$")
    bar.set_ticks([0.5, 1, 2], labels=["0.5", "1", "2"])
    bar.minorticks_off()
    fig.suptitle("Gas structures and tracer sampling in compressible turbulence")
    save(fig, out, "turbulence_columns.png")


def plot_turbulent_statistics(out):
    from matplotlib.colors import LogNorm
    data = np.load(DATA / "paper_comparisons.npz")
    metadata = read_json("paper_comparisons.json")["families"]["turb64_iso"]["cases"]
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    pdf = fig.add_subplot(grid[1, :])
    for axis, pusher in zip(axes, PUSHERS):
        count = data[f"turb64_iso_{pusher}_joint"]
        im = axis.pcolormesh(data["turb_gas_log_edges"], data["turb_tracer_log_edges"],
                             count.T, norm=LogNorm(vmin=1, vmax=10000), cmap="viridis",
                             shading="flat", rasterized=True)
        axis.plot([-4, 3], [-4, 3], color="white", ls="--", lw=1)
        axis.set(xlim=(-3, 2), ylim=(-4, 2), xlabel=r"$\log_{10}(\rho_{\rm gas}/\bar\rho)$",
                 ylabel=r"$\log_{10}(\rho_{\rm tr}/\bar\rho_{\rm tr})$",
                 title=f"{LABEL[pusher]}: {100*metadata[pusher]['empty_cic_cell_fraction']:.2f}% empty cells")
        edges = data["turb_ratio_edges"]
        values = data[f"turb64_iso_{pusher}_ratio_pdf"]
        case = metadata[pusher]
        pdf.stairs(values, edges, color=COLOR[pusher], lw=1.5,
                   label=f"{LABEL[pusher]}: mean={case['cell_log10_ratio_mean_positive']:.3f}, "
                         f"std={case['cell_log10_ratio_std_positive']:.3f}")
    fig.colorbar(im, ax=axes, label="Cells per bin", shrink=0.9)
    pdf.set(yscale="log", xlim=(-3, 2), ylim=(1e-5, 10),
            xlabel=r"$\log_{10}(\rho_{\rm tr}/\rho_{\rm gas})$", ylabel="Probability density",
            title="Three-dimensional density ratio in cells with positive tracer density")
    pdf.axvline(0, color="0.5", ls=":")
    pdf.legend(fontsize=9)
    pdf.grid(alpha=0.15)
    fig.suptitle("Cell-by-cell tracer fidelity: joint density and ratio distributions")
    save(fig, out, "turbulence_statistics.png")


def plot_turbulent_mass_spectra(out):
    data = np.load(DATA / "paper_comparisons.npz")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    edges = data["turb_gas_log_edges"]
    axes[0].stairs(data["turb_volume_pdf"], edges, color="0.6", ls="--", label="Gas: volume weighted")
    axes[0].stairs(data["turb_mass_pdf"], edges, color="#d8a020", lw=2.5, label="Gas: mass weighted")
    k = data["turb_shell_k"]
    mask = (k > 0) & (k <= 32)
    gas = data["turb_gas_power"]
    axes[1].loglog(k[mask], gas[mask], color="#d8a020", lw=2.5, label="Gas")
    for pusher in PUSHERS:
        axes[0].stairs(data[f"turb64_iso_{pusher}_sampled_gas_pdf"], edges,
                        color=COLOR[pusher], label=LABEL[pusher])
        power = data[f"turb64_iso_{pusher}_power"]
        axes[1].loglog(k[mask], power[mask], color=COLOR[pusher], label=LABEL[pusher])
        axes[2].semilogx(k[mask], power[mask]/gas[mask], color=COLOR[pusher], label=LABEL[pusher])
    axes[0].set(xlim=(-3, 2), xlabel=r"Gas density $\log_{10}(\rho/\bar\rho)$",
                ylabel="Probability density", title="Gas environments sampled by tracers")
    axes[1].set(xlabel=r"Shell wavenumber $kL/(2\pi)$", ylabel=r"$\langle|\widetilde\delta|^2\rangle_k$",
                title="Three-dimensional density spectra")
    axes[2].set(xlabel=r"Shell wavenumber $kL/(2\pi)$", ylabel=r"$P_{\rm tr}/P_{\rm gas}$",
                title="Scale-dependent tracer power")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[2].axhline(1, color="0.5", ls=":")
    for axis in axes:
        axis.grid(alpha=0.15)
    fig.suptitle("Mass sampling and spatial structure in compressible turbulence")
    save(fig, out, "turbulence_mass_spectra.png")


def plot_mass(out):
    result = read_json("mc_child_allocation.json")["result"]
    groups = result["groups"]
    representative = max(groups, key=lambda g: g["probability_range"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(4)
    n = representative["count"]
    axes[0].bar(x - 0.18, representative["probabilities"], width=0.35,
                color="#d8a020", label="Post-prolongation mass fraction")
    axes[0].bar(x + 0.18, np.asarray(representative["counts"]) / n, width=0.35,
                color="#3b6fb6", label="Observed MC fraction")
    axes[0].axhline(0.25, color="0.3", ls=":", label="Uniform allocation")
    axes[0].set(xticks=x, xlabel="Fine child index", ylabel="Fraction",
                title=f"One nonuniform parent cell ({n} tracers)")
    axes[0].legend(fontsize=8)
    index = np.arange(1, len(groups) + 1)
    axes[1].plot(index, [g["max_count_error"] / g["bound"] for g in groups],
                 "o-", color="#3b6fb6", label="Mass-weighted model")
    axes[1].plot(index, [g["uniform_count_error"] / g["bound"] for g in groups],
                 "x--", color="#b33c2e", label="Uniform model")
    axes[1].axhline(1, color="0.3", ls=":", label="Rejection threshold")
    axes[1].set(xlabel="Parent group", ylabel="Maximum count error / simultaneous bound",
                title="All 16 parents; 33,191 refined tracers", xticks=[1, 4, 8, 12, 16])
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.suptitle(r"MC refinement allocation: family-wise $\alpha=10^{-6}$")
    save(fig, out, "mc_child_mass_allocation.png")


def plot_boundary(out):
    data = read_json("boundary_snapshots.json")["cases"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, layout="constrained")
    for axis, bc in zip(axes.flat, ("periodic", "reflect", "outflow", "diode")):
        for pusher in PUSHERS:
            rows = data[f"{pusher}_{bc}"]["snapshots"]
            axis.plot([r["time"] for r in rows], [r["frozen"] / r["count"] for r in rows],
                      color=COLOR[pusher], marker=MARKER[pusher], markersize=5,
                      label=LABEL[pusher])
        axis.set(title=bc.capitalize(), xlabel="Time", ylabel="Frozen fraction",
                 ylim=(-0.015, 0.27))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Particle boundary tests: 256 tracers per case")
    save(fig, out, "boundary_contracts.png")


def plot_amr(out):
    data = read_json("amr_restart.json")["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    rows = data["classical"]["snapshots"]
    axes[0].step([r["cycle"] for r in rows], [r["blocks"] for r in rows],
                 where="post", color="#3b6fb6")
    axes[0].set(ylabel="MeshBlocks", title="Repeated 3-D refinement/coarsening", yticks=[8, 15])
    for pusher in PUSHERS:
        series = data[pusher]["snapshots"]
        for axis, field in zip(axes[1:], ("count", "dt")):
            axis.plot([r["cycle"] for r in series], [r[field] for r in series],
                      color=COLOR[pusher], marker=MARKER[pusher], label=LABEL[pusher])
    axes[1].set(ylabel="Tracer count", ylim=(510, 514), title="Count conserved")
    axes[2].set(ylabel="Stored timestep", title="Timestep changes with refinement")
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.set_xlabel("Cycle")
        axis.grid(alpha=0.2)
    fig.suptitle("Single-GPU AMR evolution: 512 tracers")
    save(fig, out, "amr_restart_contracts.png")


def plot_spectra(out):
    data = read_csv("mach03_spectra.csv")
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), layout="constrained")
    for pusher in PUSHERS:
        key = "ito" if pusher == "ito_2" else pusher
        rows = [r for r in data if r["pusher"] == key]
        k = np.array([float(r["k"]) for r in rows])
        gas = np.array([float(r["gas_power"]) for r in rows])
        tracer = np.array([float(r["tracer_power"]) for r in rows])
        coherence = np.array([float(r["signed_coherence"]) for r in rows])
        if pusher == "classical":
            axes[0].loglog(k, gas, color="#d8a020", lw=2.3, label="Gas")
        axes[0].loglog(k, tracer, color=COLOR[pusher], label=LABEL[pusher])
        axes[1].semilogx(k, coherence, color=COLOR[pusher], label=LABEL[pusher])
        lowk = (k >= 1) & (k <= 6)
        axes[2].plot(k[lowk], (coherence * np.sqrt(tracer / gas))[lowk],
                     color=COLOR[pusher], marker=MARKER[pusher], label=LABEL[pusher])
    axes[0].set_ylabel("Shell-mean density power")
    axes[1].set(ylabel="Signed spectral correlation", ylim=(-0.1, 1.06))
    axes[2].set(ylabel=r"Cross response $P_{gt}/P_{gg}$", xlabel="Integer shell k",
                title=r"Low-wavenumber response: $1 \leq k \leq 6$",
                xlim=(0.8, 6.2), xticks=np.arange(1, 7))
    axes[0].legend(ncol=2, fontsize=9)
    for axis in axes[:2]:
        axis.axvline(6.5, color="0.5", ls=":", lw=1)
        axis.set_xlim(1, 64)
        axis.grid(alpha=0.2, which="both")
        axis.set_xlabel("Integer shell k")
    axes[2].grid(alpha=0.2)
    axes[1].axhline(1, color="0.5", ls=":", lw=1)
    axes[2].axhline(1, color="0.5", ls=":", lw=1)
    fig.suptitle("Gas and tracer density spectra\n"
                 r"$128^3$ cells, 64 tracers/cell, $t=0.1$")
    save(fig, out, "mach03_spectra.png")


def plot_convergence(out):
    data = read_csv("converging_flow_metrics.csv")
    resolutions = (64, 128, 256)
    fig = plt.figure(figsize=(10, 7.5), layout="constrained")
    layout = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    profile = fig.add_subplot(layout[0, :])
    axes = [fig.add_subplot(layout[1, i]) for i in range(2)]
    values = np.load(DATA / "paper_comparisons.npz")
    valid = values["converging_valid"]
    x = values["converging_x"][valid]
    profile.plot(x, values["converging_analytic"][valid], color="0.55", ls="--",
                 lw=2.5, label="Characteristic solution")
    profile.plot(x, values["converging_gas"][valid], color="#d8a020", lw=2, label="Gas")
    for pusher in PUSHERS:
        profile.plot(x, values[f"converging_{pusher}"][valid], color=COLOR[pusher],
                     lw=1, label=LABEL[pusher])
    profile.set(xlabel=r"$x/L$", ylabel="Density / interior mean",
                 title=r"Initially $v_x=-x$: $N_x=128$, seed 42, $t=0.3$")
    profile.legend(ncol=3, fontsize=8)
    profile.grid(alpha=0.2)
    for pusher in PUSHERS:
        for axis, metric in zip(axes, ("mean_abs_valid", "mean_abs_center4")):
            groups = [np.array([float(r[metric]) for r in data
                               if r["pusher"] == pusher and int(r["resolution"]) == n])
                      for n in resolutions]
            axis.errorbar(resolutions, [g.mean() for g in groups],
                          yerr=[g.std(ddof=1) for g in groups],
                          color=COLOR[pusher], marker=MARKER[pusher], capsize=4,
                          label=LABEL[pusher])
    for axis, title in zip(axes, ("Valid interior", "Four stagnation-point cells")):
        axis.set_xscale("log", base=2)
        axis.set_xticks(resolutions, resolutions)
        axis.set(xlabel=r"$N_x$", ylabel="Mean absolute tracer/gas relative error", title=title)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Density compression and tracer fidelity\n"
                 "Resolution study: 45 cases; error bars show seed standard deviation")
    save(fig, out, "converging_flow.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check data without generating figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    check_data()
    if args.check:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.titlesize": 12})
    for plot in (plot_square, plot_turbulent_maps, plot_turbulent_statistics,
                 plot_turbulent_mass_spectra, plot_mass, plot_boundary, plot_amr,
                 plot_spectra, plot_convergence):
        plot(args.output_dir)


if __name__ == "__main__":
    main()
