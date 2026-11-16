#!/usr/bin/env python3
"""Validate per-stage turbulence forcing and volume-RMS normalization.

Power cases compare the candidate with a specified baseline executable. RMS
cases independently reconstruct each acceleration field from the stored raw OU
field and the preceding checkpoint's density. Generated data stay below --out.
"""

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import traceback

import numpy as np

from checkpoint import compare_states, read_checkpoint, require


TOL = 2.0e-12
GAMMA = 5.0 / 3.0
CASES = [dict(name=f"power-{owner}-{rk}", kind="power", owner=owner, rk=rk)
         for owner in ("hydro", "mhd") for rk in ("rk2", "rk3")]
CASES += [dict(name=f"rms-{owner}-{rk}", kind="rms", owner=owner, rk=rk)
          for owner in ("hydro", "mhd") for rk in ("rk1", "rk2", "rk3", "rk4")]
CASES += [dict(name=f"rms-density-{owner}", kind="density", owner=owner, rk="rk3")
          for owner in ("hydro", "mhd")]
CASES += [dict(name="rms-zero", kind="zero", owner="hydro", rk="rk2"),
          dict(name="rms-zero-field", kind="zero-field", owner="hydro", rk="rk2")]
CASES += [dict(name="reject-"+name, kind="reject", owner="hydro", rk="rk2",
               target=value,
               diagnostic="turb_driving/accel_rms must be finite and nonnegative")
          for name, value in (("negative-rms", "-0.2"), ("nan-rms", "nan"),
                              ("infinite-rms", "inf"))]
CASES += [dict(name="reject-normalization", kind="reject", owner="hydro", rk="rk2",
               normalization="unsupported",
               diagnostic="turb_driving/normalization must be "
               "power or rms_acceleration")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def configuration(case, normalization="rms_acceleration", cycles=8):
    owner = case["owner"]
    cfg = dict(job=dict(basename="PerStage"), mesh=dict(nghost=2),
               meshblock=dict(nx1=8, nx2=8, nx3=8),
               time=dict(evolution="dynamic", integrator=case["rk"], cfl_number=.2,
                         nlim=cycles, tlim=1., ndiag=1),
               problem=dict(pgen_name="turbulence", d_n=1., d_i=1., ifield=2, beta=10.),
               turb_driving=dict(nlow=1, nhigh=3, driving_type=0, random_seed=271828,
                                 dedt=.02, tcorr=.05, dt_turb_update=.01,
                                 dt_turb_thresh=1.e-6, sol_fraction=1., spect_form=2,
                                 expo=1., turb_flag=2, restart_forcing="true"),
               output1=dict(file_type="rst", dcycle=1, single_file_per_rank="false"),
               output2=dict(file_type="log", dcycle=1))
    cfg[owner] = dict(eos="ideal", reconstruct="plm", gamma=GAMMA, fofc="false",
                      rsolver="hllc" if owner == "hydro" else "hlld")
    if owner == "mhd":
        # This generator initializes all CT faces before evaluating magnetic
        # energy. With field_amp=0 and no refinement it provides a uniform state.
        cfg["problem"] = dict(pgen_name="divb_amr", rho0=1., pgas0=.6,
                              vx0=0., vy0=0., vz0=0., guide_b1=0., guide_b2=0.,
                              guide_b3=math.sqrt(.12), field_amp=0.,
                              refine_levels=0, user_hist="false")
    lengths = (2., 1.5, .75) if case["kind"] == "density" else (1., 1., 1.)
    for d, length in enumerate(lengths, 1):
        cfg["mesh"].update({f"nx{d}": 16, f"x{d}min": 0., f"x{d}max": length,
                            f"ix{d}_bc": "periodic", f"ox{d}_bc": "periodic"})
    if normalization is not None:
        cfg["turb_driving"]["normalization"] = normalization
    if normalization == "rms_acceleration":
        cfg["turb_driving"]["accel_rms"] = .3
    if case["kind"] in ("density", "zero-field"):
        cfg["turb_driving"].update(tcorr=10., dt_turb_update=1.)
    return cfg


def input_text(cfg):
    return "\n".join(f"<{name}>\n" + "\n".join(f"{k} = {v}" for k, v in block.items())
                     + "\n" for name, block in cfg.items())


def launch(args, directory, cfg, binary=None, restart=None, reject=None):
    directory.mkdir(parents=True, exist_ok=False)
    binary = args.binary if binary is None else binary
    inp = directory/"input.athinput"
    inp.write_text(input_text(cfg))
    command = ([] if args.launcher.lower() == "none" else
               shlex.split(args.launcher.format(ranks=args.ranks)))
    command += [str(binary)]
    if args.backend == "gpu":
        command += ["--kokkos-map-device-id-by=mpi_rank"]
    command += ["-i", str(inp), "-d", str(directory), "-t", "00:03:00"]
    if restart is not None:
        command += ["-r", str(restart)]
    save(directory/"command.json", dict(
        argv=command, binary_sha256=sha(binary), input_sha256=sha(inp),
        restart_sha256=sha(restart) if restart is not None else None))
    begin = time.monotonic()
    with (directory/"athena.log").open("x") as log:
        process = subprocess.Popen(command, cwd=directory, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            save(directory/"execution.json", dict(status="TIMEOUT",
                                                  elapsed=time.monotonic()-begin))
            raise AssertionError("Native test exceeded its timeout")
    save(directory/"execution.json", dict(returncode=rc, elapsed=time.monotonic()-begin))
    log = (directory/"athena.log").read_text(errors="replace")
    if reject is not None:
        require(rc != 0, "Invalid configuration unexpectedly succeeded")
        require(reject in log, "Run failed for a different reason: missing "+reject)
        return {}, {}
    require(rc == 0, f"Native executable exited with status {rc}")
    require("Terminating on wall clock limit" not in log,
            "Native endpoint was not reached")
    states, files = {}, {}
    for path in sorted((directory/"rst").glob("*.rst")):
        state = read_checkpoint(path)
        cycle = state["cycle"]
        if cycle in states:
            compare_states(states[cycle], state)
        states[cycle], files[cycle] = state, path
    require(states and max(states) == cfg["time"]["nlim"], "Wrong native endpoint")
    start = 0 if restart is None else read_checkpoint(restart)["cycle"]+1
    require(all(c in states for c in range(start, cfg["time"]["nlim"]+1)),
            "Missing scheduled checkpoint")
    event = directory/"PerStage.log"
    require(event.is_file(), "Missing EOS event counter output")
    event_text = event.read_text()
    require("# Athena event counter data" in event_text, "Invalid event counter header")
    rows = [list(map(int, line.split())) for line in event_text.splitlines()
            if line.strip() and not line.startswith("#")]
    # An empty data section means no events occurred.
    require(all(len(row) == 8 and not any(row[1:]) for row in rows),
            "EOS floor or repair counters were nonzero")
    return states, files


def volume_sum(state, field):
    return float(np.sum(field*state["cell_volume"][:, None, None, None],
                        dtype=np.longdouble))


def rms_oracle(before, after, target):
    require(np.array_equal(before["locations"], after["locations"]),
            "RMS oracle requires unchanged topology")
    raw, force = after["raw"], after["force"]
    require(raw is not None and force is not None, "RMS oracle requires stored forcing")
    rho = before["u"][:, 0]
    mass = volume_sum(before, rho)
    volume = volume_sum(before, np.ones_like(rho))
    mean = np.array([volume_sum(before, rho*raw[:, d])/mass for d in range(3)])
    centered = raw-mean[None, :, None, None, None]
    raw_rms = math.sqrt(volume_sum(before, np.sum(centered**2, axis=1))/volume)
    require(raw_rms > 0., "RMS oracle has a degenerate forcing field")
    expected = centered*(target/raw_rms)
    scale = max(target, float(np.max(np.abs(expected))))
    error = float(np.max(np.abs(force-expected)))
    require(error <= TOL*scale, "Acceleration differs from independent RMS normalization")
    measured = math.sqrt(volume_sum(before, np.sum(force**2, axis=1))/volume)
    require(abs(measured-target) <= TOL*target, "Wrong volume-weighted acceleration RMS")
    net = np.array([volume_sum(before, rho*force[:, d]) for d in range(3)])
    require(float(np.max(np.abs(net))) <= TOL*mass*scale, "Nonzero integrated force")
    mass_rms = math.sqrt(volume_sum(before, rho*np.sum(force**2, axis=1))/mass)
    return dict(cycle=after["cycle"], target=target, volume_rms=measured,
                mass_rms=mass_rms, mass_weighted_raw_mean=mean.tolist(),
                maximum_field_error=error, normalized_field_error=error/scale,
                net_integrated_force=net.tolist(), domain_volume=volume,
                minimum_density=float(rho.min()), maximum_density=float(rho.max()))


def edit_fixture(source, destination, zero_field=False):
    """Write a new restart fixture, preserving its mesh, clocks and RNG state."""
    state = read_checkpoint(source)
    require(state["eos"] == "ideal" and state["nvar"] == 5,
            "Fixture requires ideal gas with five fluid variables")
    require(state["updates"] == 1 and state["raw"] is not None,
            "Fixture requires exactly one stored OU update")
    data = bytearray(source.read_bytes())
    shape, active = state["full_shape"], state["active_slices"]
    for m in range(state["nmb"]):
        x = (state["centers"][m, 0, 0, 0, 0]
             + (np.arange(shape[-1])-active[-1].start)*state["widths"][m, 0])
        phase = 2.*math.pi*(x-state["bounds"][0, 0])/(
            state["bounds"][1, 0]-state["bounds"][0, 0])
        sine = np.sin(phase)[None, None, :]
        density = 1.+.25*sine+.25*np.cos(2.*phase)[None, None, :]
        offset = state["payload_offset"]+m*state["block_stride"]
        u = np.ndarray((5,)+shape, dtype="<f8", buffer=data, offset=offset)
        # Set a stationary density fixture while preserving its thermal and
        # magnetic energies, even if the seed has already developed motion.
        u[4] -= .5*np.sum(u[1:4]**2, axis=0)/u[0]
        if not zero_field:
            u[0] = density
        u[1:4] = 0.
        raw = np.ndarray((3,)+shape, dtype="<f8", buffer=data,
                         offset=offset+state["raw_offset"])
        raw[:] = 0.
        if not zero_field:
            raw[1] = sine
        force = np.ndarray((3,)+shape, dtype="<f8", buffer=data,
                           offset=offset+state["force_offset"])
        force[:] = 0.
    destination.write_bytes(data)
    result = read_checkpoint(destination)
    require(result["rng"] == state["rng"] and result["updates"] == state["updates"],
            "Fixture editor changed OU state metadata")
    save(destination.with_suffix(".json"), dict(
        source=str(source), source_sha256=sha(source), fixture_sha256=sha(destination),
        changed_fields=(["density"] if not zero_field else []) +
        ["momentum", "total_energy", "raw", "force"],
        zero_field=zero_field,
        preserved="internal energy, magnetic fields, geometry, clocks, RNG, OU count"))
    return result


def equivalence(left, right):
    require(set(left) == set(right), "Power comparison checkpoint cycles differ")
    results = []
    for cycle in sorted(left):
        a, b = left[cycle], right[cycle]
        errors = compare_states(a, b, TOL)
        exact = all(value == 0. for value in errors.values())
        exact = exact and all(a[key] == b[key]
                              for key in ("time", "dt", "dtold", "last_dt"))
        results.append(dict(cycle=cycle, bitwise_equal=exact,
                            field_relative_errors=errors,
                            time_differences={k: b[k]-a[k] for k in
                                              ("time", "dt", "dtold", "last_dt")}))
    return dict(all_bitwise_equal=all(r["bitwise_equal"] for r in results),
                cycles=results, tolerance=TOL,
                interpretation="Exact physical states, or differences bounded by the "
                "recorded roundoff allowance; parameter dumps may contain the new "
                "normalization default.")


def run_case(args, case):
    directory = args.out/case["name"]
    if case["kind"] == "power":
        directory.mkdir()
        cfg = configuration(case, normalization=None)
        baseline, _ = launch(args, directory/"baseline", cfg, binary=args.baseline_binary)
        default, _ = launch(args, directory/"candidate-default", cfg)
        cfg["turb_driving"]["normalization"] = "power"
        explicit, _ = launch(args, directory/"candidate-explicit", cfg)
        return dict(baseline_vs_default=equivalence(baseline, default),
                    default_vs_explicit=equivalence(default, explicit))
    if case["kind"] == "reject":
        cfg = configuration(case, cycles=0)
        cfg["turb_driving"].update(
            normalization=case.get("normalization", "rms_acceleration"),
            accel_rms=case.get("target", .3))
        launch(args, directory, cfg, reject=case["diagnostic"])
        return dict(expected_rejection=case["diagnostic"])
    cfg = configuration(case)
    if case["kind"] == "zero":
        cfg["turb_driving"]["accel_rms"] = 0.
        states, _ = launch(args, directory, cfg)
        require(all(not np.any(s["force"]) for s in states.values()),
                "Zero RMS produced force")
        return dict(completed_cycles=max(states), maximum_acceleration=0.)
    if case["kind"] in ("density", "zero-field"):
        directory.mkdir()
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["time"]["nlim"] = 1
        _, files = launch(args, directory/"seed", seed_cfg)
        fixture = directory/"fixture.rst"
        initial = edit_fixture(files[1], fixture, zero_field=case["kind"] == "zero-field")
        reject = ("Positive acceleration RMS requires nonzero finite forcing and finite "
                  "positive volume") if case["kind"] == "zero-field" else None
        states, _ = launch(args, directory/"continuation", cfg, restart=fixture,
                           reject=reject)
        if reject:
            return dict(expected_rejection=reject, fixture_sha256=sha(fixture))
        states[1] = initial
    else:
        states, _ = launch(args, directory, cfg)
        require(not np.any(states[1]["force"]), "Startup cycle-zero skip changed")
    checks = [rms_oracle(states[c-1], states[c], cfg["turb_driving"]["accel_rms"])
              for c in sorted(states) if c > 1]
    require(checks, "No active RMS step tested")
    if case["kind"] == "density":
        first = checks[0]
        require(abs(first["domain_volume"]-1.) > .1, "Density fixture has unit volume")
        require(first["maximum_density"]-first["minimum_density"] > .2,
                "Density fixture is effectively uniform")
        require(abs(first["mass_rms"]-first["volume_rms"]) > .01*first["target"],
                "Fixture does not distinguish volume from mass RMS")
        require(np.linalg.norm(first["mass_weighted_raw_mean"]) > .01,
                "Fixture does not exercise mean-force subtraction")
    return dict(completed_cycles=max(states), normalization_checks=checks)


def build_pin(binary, manifest):
    require(binary.is_file() and manifest.is_file(),
            "Missing executable or build source manifest")
    build_sources = json.loads(manifest.read_text())
    require(isinstance(build_sources, dict) and build_sources,
            "Invalid build source manifest")
    receipt = binary.with_suffix(".sha256")
    if receipt.is_file():
        require(receipt.read_text().strip().split()[0] == sha(binary),
                "Binary SHA256 receipt differs")
    return dict(binary=str(binary), binary_sha256=sha(binary),
                source_manifest=str(manifest), source_manifest_sha256=sha(manifest),
                build_sources=build_sources,
                binary_receipt=str(receipt) if receipt.is_file() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--baseline-binary", type=Path)
    parser.add_argument("--baseline-source-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--launcher", default="mpirun -np {ranks}")
    parser.add_argument("--timeout", type=float, default=210.)
    parser.add_argument("--case", action="append", default=[],
                        help="Repeatable case name or glob")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    selected = [c for c in CASES if not args.case or any(
        fnmatch.fnmatchcase(c["name"], pattern) for pattern in args.case)]
    require(selected, "No matching cases")
    if args.list:
        print("\n".join(c["name"] for c in selected))
        return 0
    require(args.binary is not None and args.source_manifest is not None
            and args.out is not None,
            "Provide --binary, --source-manifest and --out")
    require(args.ranks > 0 and args.timeout > 0., "Invalid execution bounds")
    require(args.launcher.lower() != "none" or args.ranks == 1,
            "Multiple ranks require an MPI launcher")
    require(socket.gethostname().split(".")[0] in ("della-vis1", "della-vis2")
            or bool(os.environ.get("SLURM_JOB_ID")),
            "Native tests require a visualization node or approved allocation")
    args.binary, args.source_manifest, args.out = (
        p.resolve() for p in (args.binary, args.source_manifest, args.out))
    pins = [build_pin(args.binary, args.source_manifest)]
    if any(c["kind"] == "power" for c in selected):
        require(args.baseline_binary is not None
                and args.baseline_source_manifest is not None,
                "Power comparisons require baseline executable and source manifest")
        args.baseline_binary = args.baseline_binary.resolve()
        args.baseline_source_manifest = args.baseline_source_manifest.resolve()
        pins.append(build_pin(args.baseline_binary, args.baseline_source_manifest))
    args.out.mkdir(parents=True, exist_ok=False)
    harnesses = {str(Path(__file__).resolve()): sha(__file__),
                 str(Path(__file__).with_name("checkpoint.py").resolve()):
                 sha(Path(__file__).with_name("checkpoint.py"))}
    report = dict(status="RUNNING", builds=pins, harnesses=harnesses,
                  backend=args.backend, ranks=args.ranks, tolerance=TOL, cases=[],
                  scope="Retained per-RK-stage forcing; RMS normalization and "
                  "baseline compatibility, with no finite-kick energy claim.")
    save(args.out/"summary.json", report)
    for case in selected:
        try:
            require(all(sha(p["binary"]) == p["binary_sha256"] and
                        sha(p["source_manifest"]) == p["source_manifest_sha256"]
                        for p in pins),
                    "Build inputs changed during validation")
            row = dict(name=case["name"], status="PASS", **run_case(args, case))
        except Exception as error:
            row = dict(name=case["name"], status="FAIL", error=str(error),
                       traceback=traceback.format_exc())
        report["cases"].append(row)
        save(args.out/"summary.json", report)
        print(case["name"], row["status"], row.get("error", ""), flush=True)
    report["pins_unchanged"] = all(
        sha(p["binary"]) == p["binary_sha256"] and
        sha(p["source_manifest"]) == p["source_manifest_sha256"] for p in pins)
    report["harnesses_unchanged"] = all(
        sha(path) == digest for path, digest in harnesses.items())
    report["status"] = ("PASS" if report["pins_unchanged"]
                        and report["harnesses_unchanged"]
                        and all(row["status"] == "PASS" for row in report["cases"])
                        else "FAIL")
    save(args.out/"summary.json", report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
