#!/usr/bin/env python3
"""Check turbulence forcing through refinement, coarsening and restart.

Examples:
  python amr.py --binary /path/athena --out /scratch/amr-cpu --ranks 2
  python amr.py --binary /path/athena --out /scratch/amr-gpu --backend gpu

The native runs use existing advection and divb_amr problem generators. Inputs
and outputs are created only below --out. Single-file restart data retain AMR
geometry for conservative integrals and independent Fourier interpolation.
"""

import argparse
import copy
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
import traceback

import numpy as np

from checkpoint import compare_states, integrals, read_checkpoint, require, uniform_field


TOL = 2e-12
CASES = ("hydro-stored", "hydro-regenerated", "mhd-stored", "mhd-regenerated")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def configuration(case, cycles, normalization="power", accel_rms=.2,
                  density_amplitude=0.):
    mhd = case.startswith("mhd")
    owner = "mhd" if mhd else "hydro"
    cfg = dict(job=dict(basename="TurbAMR"),
               mesh=dict(nghost=2), meshblock=dict(nx1=8, nx2=8, nx3=8),
               mesh_refinement=dict(refinement="adaptive", num_levels=2,
                                    ncycle_check=3, refinement_interval=1,
                                    max_nmb_per_rank=512),
               amr_criterion1=dict(method="user"),
               # The short forcing cadence is below the fluid CFL bound. A unit
               # multiplier lets the run cross OU windows within sixteen cycles.
               time=dict(evolution="dynamic", integrator="rk2", cfl_number=1.,
                         nlim=cycles, tlim=1., ndiag=1),
               turb_driving=dict(nlow=1, nhigh=3, driving_type=0, random_seed=271828,
                                 dedt=.01, tcorr=.005, dt_turb_update=.001,
                                 dt_turb_thresh=1e-7, sol_fraction=.5, spect_form=2,
                                 expo=1, turb_flag=2,
                                 restart_forcing="true" if case.endswith("stored")
                                 else "false"),
               output1=dict(file_type="rst", dcycle=1, single_file_per_rank="false"),
               output2=dict(file_type="hst", dcycle=1, data_format="%.17e"),
               output3=dict(file_type="log", dcycle=1))
    for d in (1, 2, 3):
        cfg["mesh"].update({f"nx{d}": 32, f"x{d}min": 0., f"x{d}max": 1.,
                            f"ix{d}_bc": "periodic", f"ox{d}_bc": "periodic"})
    cfg[owner] = dict(eos="ideal" if mhd else "isothermal", reconstruct="plm",
                      rsolver="llf")
    cfg[owner].update(dict(gamma=5./3.) if mhd else dict(iso_sound_speed=1.))
    if mhd:
        require(density_amplitude == 0.,
                "Density perturbation fixture supports Hydro only")
        cfg["problem"] = dict(pgen_name="divb_amr", rho0=1., pgas0=10.,
                              vx0=.08, vy0=.05, vz0=.03, guide_b1=.7,
                              guide_b2=.2, guide_b3=-.15, field_amp=0.,
                              refine_levels=1, user_hist="true")
    else:
        cfg["problem"] = dict(pgen_name="advection", flow_dir=1, iproblem=1,
                              velocity=.1, amplitude=density_amplitude,
                              advect_dens="true" if density_amplitude else "false",
                              particle_amr_period=3)
    if normalization == "rms_acceleration":
        cfg["turb_driving"].update(normalization=normalization, accel_rms=accel_rms)
    return cfg


def input_text(cfg):
    return "\n".join(f"<{name}>\n"+"\n".join(f"{k} = {v}" for k, v in block.items())
                     + "\n" for name, block in cfg.items())


def load_states(directory):
    states = {}
    files = sorted((directory/"rst").glob("*.rst"))
    require(files, "No restart outputs")
    for path in files:
        state = read_checkpoint(path)
        cycle = state["cycle"]
        if cycle in states:
            compare_states(states[cycle], state)
        states[cycle] = state
    return states


def run_native(args, directory, cfg, restart=None):
    if args.analyze_only:
        execution = json.loads((directory/"execution.json").read_text())
        command_record = json.loads((directory/"command.json").read_text())
        require(execution.get("returncode") == 0, "Retained native run did not complete")
        require(sha(directory/"input.athinput") == command_record["input_sha256"],
                "Retained test input changed")
        states = load_states(directory)
        require(max(states) == cfg["time"]["nlim"], "Retained endpoint not reached")
        require(all(c in states for c in range(min(states), max(states)+1)),
                "Retained checkpoints have a missing cycle")
        return states
    directory.mkdir(parents=True, exist_ok=False)
    inp = directory/"input.athinput"
    inp.write_text(input_text(cfg))
    command = []
    if args.launcher.lower() != "none":
        command += shlex.split(args.launcher.format(ranks=args.ranks))
    else:
        require(args.ranks == 1, "Direct launcher requires one rank")
    command += [str(args.binary)]
    if args.backend == "gpu":
        command += ["--kokkos-map-device-id-by=mpi_rank"]
    command += ["-i", str(inp), "-d", str(directory), "-t", "00:03:00"]
    if restart is not None:
        command += ["-r", str(Path(restart).resolve())]
    save(directory/"command.json",
         dict(argv=command, input_sha256=sha(inp),
              restart_sha256=None if restart is None else sha(restart)))
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
            save(directory/"execution.json",
                 dict(status="TIMEOUT", elapsed=time.monotonic()-begin))
            raise AssertionError("Native test exceeded its timeout")
    save(directory/"execution.json", dict(returncode=rc, elapsed=time.monotonic()-begin))
    require(rc == 0, "Native exit "+str(rc))
    text = (directory/"athena.log").read_text()
    require("Terminating on wall clock limit" not in text and "### FATAL" not in text,
            "Incomplete or failed native run")
    states = load_states(directory)
    require(max(states) == cfg["time"]["nlim"], "Native endpoint not reached")
    require(all(c in states for c in range(min(states), max(states)+1)),
            "Missing checkpoint cycle")
    logpath = directory/"TurbAMR.log"
    require(logpath.is_file(), "Missing EOS-event output")
    rows = [list(map(int, line.split())) for line in logpath.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    require("# Athena event counter data" in logpath.read_text(),
            "Invalid EOS-event header")
    # EventLogOutput emits a row only when at least one counter is nonzero.
    require(all(len(row) == 8 and not any(row[1:]) for row in rows),
            "EOS, floor or FOFC repair occurred")
    return states


def topology_events(states):
    result = []
    keys = sorted(states)
    for old_cycle, cycle in zip(keys, keys[1:]):
        old, new = states[old_cycle], states[cycle]
        changed = not np.array_equal(old["locations"], new["locations"])
        if changed:
            require(new["topology"] > old["topology"],
                    "Topology changed without sequence update")
            reset = None
            if new["raw"] is not None:
                reset = bool(new["updates"] == 0 and not np.any(new["raw"])
                             and not np.any(new["force"]))
            result.append(dict(cycle=cycle, old_blocks=old["nmb"], new_blocks=new["nmb"],
                               delta_blocks=new["nmb"]-old["nmb"],
                               same_count=new["nmb"] == old["nmb"],
                               reset_checkpoint=reset))
    return result


def divergence_check(state):
    """Check CT divergence against a field- and step-scaled roundoff budget."""
    if not state["active_faces"]:
        return None
    divb = np.zeros_like(state["u"][:, 0])
    field_scale = np.zeros(state["nmb"])
    for d, face in enumerate(state["active_faces"]):
        divb += np.diff(face, axis=3-d)/state["widths"][:, d, None, None, None]
        field_scale += np.max(np.abs(face), axis=(1, 2, 3))/state["widths"][:, d]
    budget = (256*np.finfo(float).eps*(state["cycle"]+state["topology"]+1)*field_scale)
    measured = np.max(np.abs(divb), axis=(1, 2, 3))
    require(np.all(measured <= budget), "CT divergence exceeded scaled roundoff budget")
    return dict(max_abs_divb=float(np.max(measured)),
                max_roundoff_budget=float(np.max(budget)))


def remapping_conservation(directory, states):
    """Compare pre-AMR history integrals with post-AMR checkpoint integrals."""
    owner = next(iter(states.values()))["owner"]
    path = directory/f"TurbAMR.{owner}.hst"
    names, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            fields = re.findall(r"\[\d+\]=([^\s]+)", line)
            if fields:
                names = fields
        elif line.strip():
            rows.append([float(v) for v in line.split()])
    require(names and rows, "Missing pre-AMR conservation history")
    history = np.asarray(rows)
    require(history.shape[1] == len(names) and np.isfinite(history).all(),
            "Malformed conservation history")
    columns = {name: index for index, name in enumerate(names)}
    result = []
    for event in topology_events(states):
        state = states[event["cycle"]]
        indices = np.flatnonzero(
            np.abs(history[:, columns["time"]]-state["time"]) < 1.e-14)
        require(len(indices) > 0, "Missing history at a topology change")
        # Finalization can append a post-AMR row at the same time. The first
        # row is the completed-step history written before redistribution.
        before = history[indices[0]]
        after = integrals(state)
        errors = {"mass": abs(after["mass"]-before[columns["mass"]])}
        for d in range(3):
            errors[f"momentum_{d+1}"] = abs(after["momentum"][d]
                                            - before[columns[f"{d+1}-mom"]])
        if after["total_energy"] is not None:
            errors["total_energy"] = abs(after["total_energy"]-before[columns["tot-E"]])
        for key, error in errors.items():
            scale = (max(1., abs(after["total_energy"])) if key == "total_energy"
                     else max(1., after["mass"]))
            require(error < 5.e-11*scale, f"AMR remapping changed {key}: {error:.17g}")
        result.append(dict(cycle=state["cycle"], errors=errors))
    return result


def rk2_momentum_budget(args, directory, cfg, states, splits):
    """Recover RK2's first stage with one RK1 step from the same checkpoint.

    AddForcing uses the primitive density preceding each stage's conservative
    update. Thus the integrated source is dt/2 * sum V a (rho_n+rho_stage1).
    The first-stage probe keeps the same AMR mesh while obtaining rho_stage1;
    it changes only integrator, stopping cycle and refinement-check cadence.
    """
    result = []
    for split in splits:
        probe = copy.deepcopy(cfg)
        probe["time"].update(integrator="rk1", nlim=split+1)
        probe["mesh_refinement"]["ncycle_check"] = 2147483647
        replay = run_native(
            args, directory/f"stage1-{split}", probe, states[split]["path"])
        old, stage1, end = states[split], replay[split+1], states[split+1]
        require(np.array_equal(old["locations"], stage1["locations"]),
                "Stage-1 momentum probe changed topology")
        dt = stage1["time"]-old["time"]
        require(abs(dt-(end["time"]-old["time"])) < 1.e-14,
                "Stage-1 probe and RK2 step have different dt")
        require(np.array_equal(stage1["locations"], end["locations"]),
                "Momentum probe endpoint crosses a topology change")
        acceleration = stage1["force"]
        difference = float(np.max(np.abs(acceleration-end["force"])))
        require(difference < TOL*max(1., float(np.max(np.abs(acceleration)))),
                "Stage-1 probe changed the applied acceleration")
        volume = old["cell_volume"][:, None, None, None, None]
        expected = .5*dt*np.sum(volume*(old["u"][:, 0:1]+stage1["u"][:, 0:1])
                                * acceleration, axis=(0, 2, 3, 4))
        measured = np.asarray(integrals(end)["momentum"])-integrals(old)["momentum"]
        residual = measured-expected
        require(float(np.max(np.abs(residual))) < 5.e-11*max(1., integrals(old)["mass"]),
                "RK2 momentum change disagrees with the stage-integrated source")
        result.append(dict(split_cycle=split, dt=dt, expected=expected.tolist(),
                           measured=measured.tolist(), residual=residual.tolist(),
                           force_max_difference=difference))
    return result


def interpolate_fourier(reference, state):
    """Evaluate a resolved uniform-grid Fourier series at AMR cell centers.

    Center phases are shifted before zero-padding the coefficient array at each
    refinement level. No interpolation of the evolving fluid state is used.
    """
    field = uniform_field(reference, "raw")
    n = reference["mesh_xyz"][0]
    require(reference["mesh_xyz"] == (n, n, n), "Fourier oracle requires cubic reference")
    coefficients = np.fft.fftn(field, axes=(1, 2, 3))/n**3
    kval = np.rint(np.fft.fftfreq(n)*n).astype(int)
    zz, yy, xx = np.meshgrid(kval, kval, kval, indexing="ij")
    radius2 = xx*xx+yy*yy+zz*zz
    allowed = (radius2 >= 1) & (radius2 <= 9)
    amplitude = float(np.linalg.norm(coefficients.ravel()))
    require(amplitude > 0, "Zero Fourier reference forcing")
    leakage = float(np.linalg.norm(coefficients[:, ~allowed].ravel())/amplitude)
    require(leakage < TOL, "Uniform reference has out-of-band Fourier power")
    iz, iy, ix = np.where(allowed)
    frequencies = (kval[iz], kval[iy], kval[ix])
    result = np.empty_like(state["raw"])
    for level in np.unique(state["locations"][:, 3]):
        cells = n*2**(int(level)-state["root_level"])
        padded = np.zeros((3, cells, cells, cells), dtype=complex)
        phase = np.exp(2j*np.pi*(frequencies[0]+frequencies[1]+frequencies[2])
                       * (.5/cells-.5/n))
        padded[:, frequencies[0] % cells, frequencies[1] % cells,
               frequencies[2] % cells] = coefficients[:, iz, iy, ix]*phase
        grid = np.fft.ifftn(padded*cells**3, axes=(1, 2, 3)).real
        for m in np.flatnonzero(state["locations"][:, 3] == level):
            loc = state["locations"][m]
            starts = [int(loc[d])*state["shape_xyz"][d] for d in (2, 1, 0)]
            dest = tuple(slice(s, s+b) for s, b in zip(starts, state["shape_xyz"][::-1]))
            result[m] = grid[(slice(None),)+dest]
    error = float(np.linalg.norm((state["raw"]-result).ravel()) /
                  max(np.linalg.norm(result.ravel()), 1e-100))
    require(error < TOL, "Raw OU field disagrees with Fourier evaluation on AMR cells")
    require(state["rng"] == reference["rng"], "AMR replay changed seeded RNG history")
    return dict(relative_l2=error, reference_band_leakage=leakage,
                updates=state["updates"], cycle=state["cycle"])


def evaluate_case(args, case):
    cfg = configuration(case, args.cycles, args.normalization, args.accel_rms,
                        args.density_amplitude)
    directory = args.out/case
    full = run_native(args, directory/"full", cfg)
    events = topology_events(full)
    growth = [e for e in events if e["delta_blocks"] > 0]
    shrink = [e for e in events if e["delta_blocks"] < 0]
    require(growth, "No actual refinement growth exercised")
    require(shrink, "No actual derefinement exercised")
    require(max(full) > growth[0]["cycle"],
            "No step completed on enlarged forcing storage")
    measures = {cycle: integrals(state) for cycle, state in full.items()}
    divergence = {cycle: divergence_check(state) for cycle, state in full.items()
                  if state["owner"] == "mhd"}
    first = measures[min(measures)]
    mass_error = max(abs(v["mass"]-first["mass"])/first["mass"]
                     for v in measures.values())
    require(mass_error < 5e-12, "Periodic AMR mass conservation failed")
    momentum_error = max(
        float(np.max(np.abs(np.asarray(v["momentum"])-first["momentum"])))
        for v in measures.values())
    # The mass-weighted mean is removed before the RK stages. Density then
    # changes between stages, so the applied source can change total momentum.
    # Check redistribution conservation and the discrete source budget below.
    row = dict(name=case, status="PASS", events=events, restarts=[], fourier=[],
               mass_relative_error=mass_error, momentum_absolute_error=momentum_error,
               integrals=measures, divergence=divergence, coverage_gaps=[],
               checkpoint_hashes={str(c): s["sha256"] for c, s in full.items()})
    row["remapping_conservation"] = remapping_conservation(directory/"full", full)
    if args.normalization == "rms_acceleration":
        row["rms_acceleration"] = []
        mixed_samples = 0
        mixed_mean_samples = 0
        for cycle, state in full.items():
            if state["force"] is None:
                continue
            squared = np.sum(state["force"]**2, axis=1)
            if not np.any(squared):
                require(cycle <= 1
                        or (state["updates"] == 0 and not np.any(state["raw"])),
                        "Unexpected zero RMS forcing outside startup or AMR reset")
                row["rms_acceleration"].append(
                    dict(cycle=cycle, state="startup_or_amr_reset"))
                continue
            weights = state["cell_volume"][:, None, None, None]
            measured = float(np.sqrt(np.sum(weights*squared)/state["domain_volume"]))
            mass_rms = float(np.sqrt(np.sum(weights*state["u"][:, 0]*squared)
                                     / integrals(state)["mass"]))
            relative = abs(measured-args.accel_rms)/args.accel_rms
            require(relative < TOL,
                    f"Volume-weighted acceleration RMS {measured:.17g} differs from "
                    f"requested {args.accel_rms:.17g}")
            mixed = len(set(state["locations"][:, 3])) > 1
            mixed_samples += int(mixed)
            sample = dict(cycle=cycle, state="active", mixed_levels=mixed,
                          volume_rms=measured, mass_rms=mass_rms, relative_error=relative)
            previous = full.get(cycle-1)
            if (previous is not None
                    and np.array_equal(previous["locations"], state["locations"])):
                # Per-stage forcing is normalized before the RK update. Its
                # density-weighted mean must use that pre-step density.
                rho = previous["u"][:, 0]
                mass = float(np.sum(weights*rho))
                mass_rms_before = float(np.sqrt(np.sum(weights*rho*squared)/mass))
                net_force = np.sum(weights[:, None]*rho[:, None]*state["force"],
                                   axis=(0, 2, 3, 4))
                fraction = float(np.linalg.norm(net_force)/(mass*mass_rms_before))
                require(fraction < TOL,
                        "Pre-step mass-weighted net acceleration is nonzero")
                sample["prestep_net_force_fraction"] = fraction
                mixed_mean_samples += int(mixed)
            else:
                sample["prestep_net_force_check"] = "Unavailable across a topology change"
            row["rms_acceleration"].append(sample)
        if case.endswith("stored"):
            require(mixed_samples > 0,
                    "RMS oracle did not sample a mixed-volume AMR state")
            require(mixed_mean_samples > 0,
                    "Net-force oracle did not sample mixed-volume cells")
            if args.density_amplitude:
                active = [v for v in row["rms_acceleration"] if v["state"] == "active"]
                require(any(abs(v["mass_rms"]-v["volume_rms"]) > 1e-10
                            for v in active),
                        "Density fixture did not distinguish volume- and "
                        "mass-weighted RMS")
        else:
            row["coverage_gaps"].append(
                "Acceleration RMS is absent from regenerate-mode checkpoints; "
                "checked in stored companion")
    if not any(e["same_count"] for e in events):
        row["coverage_gaps"].append("Same-count topology change not encountered")
    # Distinct checkpoints immediately following actual growth and coarsening.
    splits = sorted({growth[0]["cycle"], shrink[0]["cycle"]})
    require(all(split < args.cycles for split in splits), "Restart split is terminal")
    if case.endswith("stored"):
        row["rk2_momentum_budget"] = rk2_momentum_budget(
            args, directory, cfg, full, splits)
    else:
        row["coverage_gaps"].append(
            "Stage-source budget needs stored acceleration; checked in stored companion")
    for split in splits:
        resumed = run_native(args, directory/f"restart-{split}", cfg, full[split]["path"])
        require(max(resumed) == args.cycles, "Restart endpoint missing")
        differences = {}
        for cycle, state in resumed.items():
            require(cycle in full, "Unexpected restart cycle")
            differences[str(cycle)] = compare_states(full[cycle], state)
        row["restarts"].append(dict(split_cycle=split, field_differences=differences,
                                    checkpoint_hashes={str(c): s["sha256"]
                                                       for c, s in resumed.items()}))
    if case.endswith("stored"):
        static = copy.deepcopy(cfg)
        static["mesh_refinement"]["refinement"] = "none"
        static.pop("amr_criterion1")
        reference = run_native(args, directory/"uniform-reference", static)
        reference_momentum = np.asarray(integrals(reference[min(reference)])["momentum"])
        row["uniform_momentum_absolute_error"] = max(
            float(np.max(np.abs(np.asarray(integrals(s)["momentum"])-reference_momentum)))
            for s in reference.values())
        by_updates = {s["updates"]: s for s in reference.values() if s["updates"] > 0}
        post_event_cycles = set()
        for event in events:
            # A reset checkpoint is a documented deferred-replay state. Inspect
            # a completed subsequent cycle before the next topology change.
            for cycle in sorted(full):
                if cycle > event["cycle"] and full[cycle]["updates"] > 0:
                    post_event_cycles.add(cycle)
                    break
        require(post_event_cycles, "No completed OU replay captured after AMR")
        for cycle in sorted(post_event_cycles):
            state = full[cycle]
            require(state["updates"] in by_updates, "Missing matching OU reference state")
            row["fourier"].append(
                interpolate_fourier(by_updates[state["updates"]], state))
        require(any(item["updates"] >= 3 for item in row["fourier"]),
                "AMR replay did not exercise multiple OU updates")
    else:
        row["coverage_gaps"].append(
            "Raw OU field is absent from regenerate-mode checkpoints; "
            "checked in stored-mode companion")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--launcher", default="mpirun -np {ranks}",
                        help="Tokenized launcher, with optional {ranks}; "
                        "'none' runs directly")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--case", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--normalization", choices=("power", "rms_acceleration"),
                        default="power")
    parser.add_argument("--accel-rms", type=float, default=.2)
    parser.add_argument("--density-amplitude", type=float, default=0.)
    parser.add_argument("--source-manifest", type=Path,
                        help="Build-time source SHA256 manifest; "
                        "retained independently of the live tree")
    parser.add_argument("--label", default="candidate",
                        help="Identify comparison or final implementation")
    parser.add_argument("--timeout", type=float, default=210.)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.out = args.out.resolve()
    require(args.ranks > 0 and args.cycles >= 12 and args.timeout > 0,
            "Invalid run bounds")
    require(args.accel_rms > 0 and 0 <= args.density_amplitude < 1,
            "Invalid forcing/density fixture")
    if not args.analyze_only:
        require(socket.gethostname().split(".")[0] in ("della-vis1", "della-vis2")
                or bool(os.environ.get("SLURM_JOB_ID")),
                "Native tests require a visualization node or approved allocation")
    args.out.mkdir(parents=True, exist_ok=args.analyze_only)
    name = (("recheck-"+str(time.time_ns())+".json") if args.analyze_only
            else "summary.json")
    source = Path(__file__).resolve().parents[2]
    pins = {str(p.relative_to(source)): sha(p) for p in sorted((source/"src").rglob("*"))
            if p.suffix in (".cpp", ".hpp")}
    if args.source_manifest is None:
        build = args.binary.parent.parent
        guess = build.parent/(build.name.removesuffix("-release")+"-source.json")
        if (args.binary.parent/"source.json").is_file():
            args.source_manifest = args.binary.parent/"source.json"
        elif guess.is_file():
            args.source_manifest = guess
    build_pins = None
    manifest_sha = None
    if args.source_manifest is not None:
        args.source_manifest = args.source_manifest.resolve()
        build_pins = json.loads(args.source_manifest.read_text())
        manifest_sha = sha(args.source_manifest)
        expected_binary = args.binary.with_suffix(".sha256")
        require(expected_binary.is_file()
                and expected_binary.read_text().strip() == sha(args.binary),
                "Compiled binary differs from completed-build SHA256 record")
    report = dict(status="RUNNING", binary=str(args.binary),
                  binary_sha256=sha(args.binary), harness_sha256=sha(__file__),
                  reader_sha256=sha(Path(__file__).with_name("checkpoint.py")),
                  ranks=args.ranks, backend=args.backend, label=args.label,
                  normalization=args.normalization, accel_rms=args.accel_rms,
                  source_hashes=pins, build_source_hashes=build_pins,
                  build_source_manifest=(None if args.source_manifest is None
                                         else str(args.source_manifest)),
                  build_source_manifest_sha256=manifest_sha, cases=[],
                  gate=TOL,
                  interpretation="PASS covers explicit assertions; "
                  "coverage_gaps remain untested")
    if not args.analyze_only:
        (args.out/"retained-amr.py").write_bytes(Path(__file__).read_bytes())
        (args.out/"retained-checkpoint.py").write_bytes(
            Path(__file__).with_name("checkpoint.py").read_bytes())
    save(args.out/name, report)
    for case in args.case:
        try:
            row = evaluate_case(args, case)
        except Exception as error:
            row = dict(name=case, status="FAIL", error=str(error),
                       traceback=traceback.format_exc())
        report["cases"].append(row)
        save(args.out/name, report)
        print(case, row["status"], row.get("error", ""), flush=True)
    unchanged = all(sha(source/p) == digest for p, digest in pins.items())
    report["source_unchanged"] = unchanged
    report["binary_unchanged"] = sha(args.binary) == report["binary_sha256"]
    report["build_manifest_unchanged"] = (manifest_sha == sha(args.source_manifest)
                                          if args.source_manifest is not None else None)
    provenance_ok = (report["build_manifest_unchanged"] if build_pins is not None
                     else unchanged)
    report["status"] = ("PASS" if provenance_ok and report["binary_unchanged"]
                        and all(r["status"] == "PASS" for r in report["cases"])
                        else "FAIL")
    save(args.out/name, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
