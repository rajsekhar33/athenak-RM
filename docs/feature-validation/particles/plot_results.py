#!/usr/bin/env python3
"""Check and plot the bundled historical particle validation data.

Requires Python 3, NumPy, and Matplotlib; no simulation or private package is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PUSHERS = ("classical", "ito_2", "lagrangian_mc")
LABEL = {"classical": "Classical", "ito_2": "Ito-2", "lagrangian_mc": "MC"}
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
    print("PASS: bundled-data consistency (not a new native simulation acceptance run)")


def save(fig, out, name):
    fig.savefig(out / name, dpi=170)
    plt.close(fig)
    print(name)


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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
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
    fig.suptitle("Converging flow: 45 cases, five seeds per point\n"
                 "Error bars: seed standard deviation")
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
    for plot in (plot_mass, plot_boundary, plot_amr, plot_spectra, plot_convergence):
        plot(args.output_dir)


if __name__ == "__main__":
    main()
