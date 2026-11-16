#!/usr/bin/env python3
"""Matched driven-turbulence energy budgets and velocity-spectrum comparisons."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time

import numpy as np

from checkpoint import integrals, read_checkpoint, uniform_field


TEMPLATE = Path(__file__).parent / "inputs" / "realistic128.athinput"
POWER = 0.1
GAMMA = 5.0 / 3.0
BUDGET_TOLERANCE = 0.01


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def provenance(binary, manifest):
    """Retain the build-time source pins, rather than hashing a mutable checkout."""
    value = json.loads(manifest.read_text())
    digest = sha(binary)
    artifacts = value.get("build_artifacts", {})
    candidates = [v for k, v in artifacts.items() if k.endswith("/athena")]
    # A compact frozen-experiment manifest may use an explicit executable field.
    candidates += [value.get("binary_sha256")]
    require(digest in candidates, "Executable hash does not match build manifest")
    return {"binary": str(binary), "binary_sha256": digest,
            "build_manifest": str(manifest), "manifest_sha256": sha(manifest),
            "build_record": value}


def read_history(path):
    names = None
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            found = re.findall(r"\[\d+\]=([^\s]+)", line)
            if found:
                names = found
            continue
        if line.strip():
            rows.append([float(item) for item in line.split()])
    require(names and rows, "History has no header or samples")
    array = np.asarray(rows)
    require(array.shape[1] == len(names) and np.isfinite(array).all(),
            "Malformed or nonfinite history")
    keep = [0]
    for index in range(1, len(array)):
        if array[index, 0] == array[keep[-1], 0]:
            # The final output can contain the next dt at an unchanged state.
            require(np.array_equal(array[index, 2:], array[keep[-1], 2:]),
                    "Inconsistent duplicate history time")
        else:
            require(array[index, 0] > array[keep[-1], 0],
                    "History times are not increasing")
            keep.append(index)
    return {name: array[keep, i] for i, name in enumerate(names)}


def spectrum(state, path):
    grid = uniform_field(state, "u")
    velocity = grid[1:4] / grid[0]
    shape = velocity.shape[1:]
    require(len(set(shape)) == 1, "Spectrum currently expects a cubic mesh")
    ncell = int(np.prod(shape))
    transformed = np.fft.fftn(velocity, axes=(1, 2, 3)) / ncell
    z, y, x = np.meshgrid(*[np.fft.fftfreq(n) * n for n in shape], indexing="ij")
    wavevector = np.stack((x, y, z))
    k2 = np.sum(wavevector ** 2, axis=0)
    shell = np.rint(np.sqrt(k2)).astype(int)
    density = 0.5 * np.sum(np.abs(transformed) ** 2, axis=0)
    parallel = np.sum(transformed * wavevector, axis=0)
    compressive = np.zeros_like(density)
    np.divide(0.5 * np.abs(parallel) ** 2, k2, out=compressive, where=k2 > 0)
    total = np.bincount(shell.ravel(), weights=density.ravel())
    comp = np.bincount(shell.ravel(), weights=compressive.ravel())
    realspace = 0.5 * float(np.mean(np.sum(velocity ** 2, axis=0)))
    residual = abs(float(total.sum()) - realspace)
    require(residual <= 2.0e-13 * max(realspace, 1.0), "FFT Parseval identity failed")
    np.savetxt(path, np.column_stack((np.arange(len(total)), total, comp,
                                     total - comp)),
               header="k_shell E_velocity E_compressive E_transverse",
               fmt="%.17e")
    return {"definition": "0.5 sum_k |FFT(v)/Ncells|^2; volume weighted",
            "realspace_half_mean_v2": realspace,
            "fourier_sum": float(total.sum()), "parseval_residual": residual,
            "time": state["time"], "path": str(path)}


def raw_diagnostics(state):
    u = state["u"]
    rho = u[:, 0]
    speed2 = np.sum(u[:, 1:4] ** 2, axis=1) / rho ** 2
    eint = u[:, 4] - 0.5 * rho * speed2
    pressure = (GAMMA - 1.0) * eint
    volume = state["cell_volume"][:, None, None, None]
    mach2 = speed2 * rho / (GAMMA * pressure)
    mass = float(np.sum(rho * volume))
    return {"time": state["time"], "cycle": state["cycle"],
            "mach_rms_volume": float(np.sqrt(np.sum(mach2 * volume)
                                             / state["domain_volume"])),
            "velocity_rms_mass": float(np.sqrt(np.sum(rho * speed2 * volume) / mass)),
            "density_min": float(rho.min()), "density_max": float(rho.max()),
            "pressure_min": float(pressure.min()),
            "pressure_max": float(pressure.max()),
            "integrals": {key: value.tolist() if isinstance(value, np.ndarray) else value
                          for key, value in integrals(state).items()}}


def analyze(directory):
    metadata = json.loads((directory / "run.json").read_text())
    status = json.loads((directory / "execution.json").read_text())
    require(status["returncode"] == 0, "Native run failed")
    paths = list(directory.glob("**/Turb128.hydro.hst"))
    require(len(paths) == 1, "Expected exactly one Hydro history")
    history = read_history(paths[0])
    t = history["time"]
    energy = history["tot-E"]
    require(len(t) >= 3 and t[0] == 0.0, "Need a fresh run with at least two cycles")
    require(abs(energy[1] - energy[0]) < 1.e-11,
            "Initial undriven cycle unexpectedly changed total energy")
    # dt can refer to the completed or next step; use actual elapsed time.
    expected = POWER * np.maximum(t - t[1], 0.0)
    measured = energy - energy[0]
    relative = np.zeros_like(expected)
    np.divide(measured - expected, expected, out=relative, where=expected > 0)
    events = directory / "Turb128.log"
    rows = [list(map(int, line.split())) for line in events.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    require("# Athena event counter data" in events.read_text(),
            "Missing event-counter header")
    require(all(len(row) == 8 and not any(row[1:]) for row in rows),
            "Nonzero floor or repair count")
    checkpoints = sorted(directory.glob("rst/*.rst"))
    require(len(checkpoints) >= 2, "Initial and final raw checkpoints are required")
    raw_results = []
    final = None
    for path in checkpoints:
        state = read_checkpoint(path)
        index = int(np.argmin(abs(t - state["time"])))
        require(abs(t[index] - state["time"]) < 1.e-12,
                "Checkpoint time missing from per-cycle history")
        record = raw_diagnostics(state)
        total = record["integrals"]["total_energy"]
        require(abs(total - energy[index]) < 2.e-11 * max(abs(total), 1.0),
                "History and raw conserved energy disagree")
        record.update(path=str(path), sha256=sha(path))
        raw_results.append(record)
        if final is None or state["cycle"] > final["cycle"]:
            final = state
    require(abs(final["time"] - t[-1]) < 1.e-12, "Final checkpoint/history time mismatch")
    pilot_cycles = metadata["pilot_cycles"]
    if pilot_cycles:
        require(final["cycle"] == pilot_cycles, "Pilot did not reach requested cycle")
    else:
        require(abs(final["time"] - metadata["target_time"]) < 1.e-12,
                "Run did not reach requested physical time")
    require(len(t) == final["cycle"] + 1, "History omitted one or more completed cycles")
    kinetic = sum(history[f"{axis}-KE"] for axis in range(1, 4))
    net = np.stack([history[f"{axis}-mom"] for axis in range(1, 4)], axis=1)
    mass_error = float(np.max(abs(history["mass"] - history["mass"][0])))
    momentum_error = float(np.max(abs(net - net[0])))
    require(mass_error < 1.e-10, "Periodic mass conservation failed")
    window = None
    if t[-1] >= 2.0 - 1.e-12:
        e1 = float(np.interp(1.0, t, energy))
        expected_window = POWER * (t[-1] - 1.0)
        window = {"start": 1.0, "stop": float(t[-1]), "expected": expected_window,
                  "measured": float(energy[-1] - e1),
                  "fractional_error": float((energy[-1] - e1) / expected_window - 1.0)}
    budget_pass = abs(relative[-1]) <= BUDGET_TOLERANCE
    if window:
        budget_pass = budget_pass and abs(window["fractional_error"]) <= BUDGET_TOLERANCE
    summary = {
        "label": metadata["label"], "completed": True,
        "pilot": bool(pilot_cycles), "target_time": metadata["target_time"],
        "time": float(t[-1]), "cycles": final["cycle"],
        "elapsed_seconds": status["elapsed_seconds"],
        "source_identity": metadata["source_identity"],
        "analysis_checker_sha256": sha(__file__),
        "template_sha256": metadata["template_sha256"],
        "first_undriven_dt": float(t[1]), "energy_expected": float(expected[-1]),
        "energy_measured": float(measured[-1]),
        "energy_fractional_error": float(relative[-1]),
        "budget_tolerance": BUDGET_TOLERANCE, "budget_pass": bool(budget_pass),
        "interval_1_to_2": window, "mass_absolute_error": mass_error,
        "momentum_absolute_error": momentum_error, "floor_or_repair_count": 0,
        "momentum_relative_to_rms": momentum_error / max(
            float(np.sqrt(2 * history["mass"][-1] * kinetic[-1])), 1.e-300),
        "field_diagnostics": raw_results,
        "spectrum": spectrum(final, directory / "velocity_spectrum.txt"),
        "projected_t2_seconds_from_pilot": status["elapsed_seconds"] * 2.0 / t[-1]
        if pilot_cycles else None,
    }
    save(directory / "analysis.json", summary)
    np.savetxt(directory / "timeseries.txt",
               np.column_stack((t, measured, expected, relative, kinetic,
                                energy - kinetic, history["mass"], net)),
               header="time energy_gain expected_gain fractional_budget_error kinetic "
                      "internal mass mom_x mom_y mom_z", fmt="%.17e")
    print(json.dumps({key: summary[key] for key in
                      ("label", "time", "cycles", "elapsed_seconds",
                       "energy_fractional_error", "budget_pass",
                       "projected_t2_seconds_from_pilot")}, indent=2))
    return summary


def run(args):
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    binary, manifest = args.binary.resolve(), args.manifest.resolve()
    identity = provenance(binary, manifest)
    text = TEMPLATE.read_text()
    text = text.replace("tlim = 2.0", f"tlim = {args.tlim:.17g}")
    if args.pilot_cycles:
        text = text.replace("nlim = -1", f"nlim = {args.pilot_cycles}")
    inp = directory / "input.athinput"
    inp.write_text(text)
    prefix = shlex.split(args.launcher.format(ranks=args.ranks)) if args.launcher else []
    require(prefix or args.ranks == 1, "Multiple ranks require an MPI launcher")
    command = prefix + [str(binary), "--kokkos-map-device-id-by=mpi_rank",
                        "-i", str(inp), "-t", args.athena_limit]
    metadata = {"label": args.label, "source_identity": identity,
                "hostname": socket.gethostname(), "pid": os.getpid(),
                "ranks": args.ranks, "target_time": args.tlim,
                "pilot_cycles": args.pilot_cycles,
                "template_sha256": sha(TEMPLATE), "input_sha256": sha(inp),
                "checker_sha256": sha(__file__), "command": command,
                "created_unix": time.time(),
                "resource_note": args.resource_note}
    save(directory / "run.json", metadata)
    start = time.monotonic()
    with (directory / "athena.log").open("x") as log:
        child = subprocess.Popen(command, cwd=directory, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=args.timeout)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            save(directory / "execution.json", {"returncode": child.returncode,
                 "elapsed_seconds": time.monotonic() - start, "interrupted": True})
            raise
    save(directory / "execution.json", {"returncode": code,
         "elapsed_seconds": time.monotonic() - start, "interrupted": False})
    require(sha(binary) == identity["binary_sha256"], "Executable changed during run")
    require(sha(__file__) == metadata["checker_sha256"], "Checker changed during run")
    result = analyze(directory)
    if not args.pilot_cycles:
        require(result["budget_pass"],
                "Completed full run missed the 1% energy-budget gate")


def compare(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=False)
    summaries = [json.loads((path / "analysis.json").read_text()) for path in args.runs]
    require(len({item["template_sha256"] for item in summaries}) == 1,
            "Input physics differ between runs")
    require(all(item["completed"] for item in summaries), "Incomplete run")
    require(len({item["pilot"] for item in summaries}) == 1,
            "A pilot and a completed physical-time run cannot be compared")
    require(max(item["time"] for item in summaries)
            - min(item["time"] for item in summaries) < 1.e-12,
            "Comparisons require matching final physical times")
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for path, item in zip(args.runs, summaries):
        data = np.loadtxt(path / "timeseries.txt")
        spec = np.loadtxt(path / "velocity_spectrum.txt")
        label = item["label"]
        axes[0, 0].plot(data[:, 0], data[:, 1], label=label)
        axes[0, 1].plot(data[2:, 0], 100 * data[2:, 3], label=label)
        axes[1, 0].plot(data[:, 0], data[:, 4], label=label)
        axes[1, 1].loglog(spec[1:, 0], spec[1:, 1], label=label)
        rows.append({key: item[key] for key in
                     ("label", "pilot", "time", "cycles", "energy_fractional_error",
                      "budget_pass", "interval_1_to_2", "elapsed_seconds")})
    axes[0, 0].plot(data[:, 0], data[:, 2], "k--", label=r"$P(t-t_1)$")
    axes[0, 0].set(xlabel="Time", ylabel="Total energy gained")
    axes[0, 1].axhline(1.0, color="0.5", linestyle=":")
    axes[0, 1].axhline(-1.0, color="0.5", linestyle=":")
    axes[0, 1].set(xlabel="Time", ylabel="Integrated power error (%)")
    axes[0, 1].set_yscale("symlog", linthresh=1.e-3)
    axes[1, 0].set(xlabel="Time", ylabel="Kinetic energy")
    axes[1, 1].set(xlabel="Wavenumber shell", ylabel=r"$E_v(k)$",
                   title="Final volume-weighted velocity spectrum")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    fig.savefig(args.out / "energy_and_spectrum.png", dpi=170)
    fig.savefig(args.out / "energy_and_spectrum.pdf")
    plt.close(fig)
    save(args.out / "comparison.json", {
        "runs": rows,
        "diagnostics": "Pointwise turbulent fields are not an acceptance criterion.",
        "spectrum_definition": summaries[0]["spectrum"]["definition"]})
    lines = ["# Matched 128³ turbulence comparison", "",
             "A periodic ideal gas receives solenoidal forcing at wavenumbers 1–3. "
             "The unit box starts with density 1, pressure 0.6 and gamma 5/3. "
             "Both runs use RK3, CFL 0.3, total driving power 0.1, correlation "
             "time 0.5, forcing-update interval 0.005 and the same random seed.", "",
             "The measured quantity is the volume integral of conserved total energy. "
             "Its expected gain is power times elapsed driven time, excluding the "
             "initial undriven cycle. The time-1-to-2 interval measures the later "
             "half of the run independently of the initial transient. Each budget "
             "must agree with its expected gain within 1%.", "",
             ("| Run | Time | Cycles | Cumulative error | Time 1–2 error "
              "| 1% budgets | Seconds |"),
             "|---|---:|---:|---:|---:|---|---:|"]
    for row in rows:
        late = row["interval_1_to_2"]
        late_text = f"{100 * late['fractional_error']:.6g}%" if late else "not sampled"
        lines.append(f"| {row['label']} | {row['time']:.6g} | {row['cycles']} | "
                     f"{100 * row['energy_fractional_error']:.6g}% | {late_text} | "
                     f"{'PASS' if row['budget_pass'] else 'FAIL'} | "
                     f"{row['elapsed_seconds']:.1f} |")
    lines += ["", "| Run | Final volume RMS Mach | Kinetic energy | Internal energy | "
              "Maximum mass error | Maximum momentum component change |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in summaries:
        last = max(item["field_diagnostics"], key=lambda entry: entry["cycle"])
        lines.append(f"| {item['label']} | {last['mach_rms_volume']:.7g} | "
                     f"{last['integrals']['kinetic']:.8g} | "
                     f"{last['integrals']['internal']:.8g} | "
                     f"{item['mass_absolute_error']:.3g} | "
                     f"{item['momentum_absolute_error']:.3g} |")
    lines += ["", "The final spectra measure volume-weighted velocity power and satisfy "
              "Parseval's identity against the saved fields. The error panel uses a "
              "symmetric logarithmic scale with a linear interval below 0.001%. "
              "The flow statistics describe its evolution over four forcing-correlation "
              "times; they do not establish statistical stationarity. Timings include "
              "startup and output on a shared visualization node.", "",
              "![Energy and velocity spectra](energy_and_spectrum.png)"]
    (args.out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--binary", type=Path, required=True)
    launch.add_argument("--manifest", type=Path, required=True)
    launch.add_argument("--out", type=Path, required=True)
    launch.add_argument("--label", required=True)
    launch.add_argument("--ranks", type=int, default=1)
    launch.add_argument("--launcher", default="mpirun -np {ranks}")
    launch.add_argument("--tlim", type=float, default=2.0)
    launch.add_argument("--pilot-cycles", type=int, default=0)
    launch.add_argument("--timeout", type=float, default=3300)
    launch.add_argument("--athena-limit", default="00:50:00")
    launch.add_argument("--resource-note", default="")
    inspect = commands.add_parser("analyze")
    inspect.add_argument("directory", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("runs", nargs="+", type=Path)
    comparison.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        run(args)
    elif args.action == "analyze":
        analyze(args.directory)
    else:
        compare(args)


if __name__ == "__main__":
    main()
