#!/usr/bin/env python3
"""Check the turbulence driver's timestep cap, energy injection and power root.

Lanes (select with --case, glob patterns allowed):

  timestep-*   Every step length equals min(2*dt_prev, CFL*hydro dt, cap), with
               cap = min(dt_turb_update, next boundary - t), computed here from
               each saved state. A time within dt_turb_thresh below a boundary
               counts as reaching it. The OU update count follows the same rule.
  short-*      A restart placed just below a boundary must take the same step and
               update the forcing exactly as a restart placed on the boundary.
  kick-*       Energy injected in the first driven step, from rest, for RK1-RK4.
               RK1 and RK2 integrate the work of a constant acceleration
               exactly. RK3 and RK4 differ from dedt*dt by the work done on the
               flow's own evolution within the step.
  work-*       From a stationary shear running with or against the forcing
               pattern, the applied acceleration must equal the smallest-magnitude
               root of m0*s^2 + m1*s = dedt, solved here from the saved state.
  reject-*     Invalid power inputs and zero forcing variance must abort.

Run with an MPI-enabled CPU or GPU binary. Generated inputs, checkpoints and logs
go below --out.
"""

import argparse
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import struct
import subprocess
import time
import traceback

import numpy as np

from checkpoint import read_checkpoint


EPS = np.finfo(np.float64).eps
GAMMA = 5.0 / 3.0
ROUNDOFF_ULPS = 512.0
# Relative agreement of a step length with the independently computed limit.
# The hydro limit is recomputed from conserved variables, so it differs from the
# code's primitive-variable value by a few roundoff units.
DT_RTOL = 1.0e-12
EXACT_RTOL = 1.0e-12      # RK1/RK2 first-kick energy, relative to dedt*dt
TRUNCATION_RTOL = 2.0e-3  # RK3/RK4 first-kick energy, relative to dedt*dt
WORK_KINETIC_RTOL = 1.0e-8
IMPULSE_RTOL = 1.0e-4     # work cases: pointwise momentum change vs impulse


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def volume_sum(state, field):
    weights = state["cell_volume"][:, None, None, None]
    return float(np.sum(field * weights, dtype=np.longdouble))


def thermal(state):
    u = state["u"]
    kinetic = 0.5 * np.sum(u[:, 1:4] ** 2, axis=1) / u[:, 0]
    magnetic = 0.0
    if state.get("bcc") is not None:
        magnetic = 0.5 * np.sum(state["bcc"] ** 2, axis=1)
    return u[:, 4] - kinetic - magnetic


def totals(state):
    u = state["u"]
    require(u.shape[1] == 5, "Checks require an ideal gas without scalars")
    eint = thermal(state)
    require(np.isfinite(eint).all() and np.all(eint > 0.0),
            "Nonpositive internal energy")
    return {"mass": volume_sum(state, u[:, 0]),
            "energy": volume_sum(state, u[:, 4]),
            "kinetic": volume_sum(state, 0.5 * np.sum(u[:, 1:4] ** 2, axis=1) / u[:, 0]),
            "internal": volume_sum(state, eint)}


def check_budget(measured, expected, rtol, arithmetic_scale, label):
    """|measured - expected| <= rtol*|expected| + roundoff of the energies subtracted."""
    residual = abs(measured - expected)
    allowance = rtol * abs(expected) + ROUNDOFF_ULPS * EPS * abs(arithmetic_scale)
    require(residual <= allowance,
            f"{label}: residual {residual:.6g} exceeds {allowance:.6g} "
            f"(expected {expected:.6g})")
    return {"measured": measured, "expected": expected, "residual": residual,
            "relative_residual": residual / abs(expected) if expected else None,
            "allowance": allowance}


# --------------------------------------------------------------------------------------
# Inputs and native execution

def configuration(case, *, cycles=None, power=None, update=None):
    owner = case.get("owner", "hydro")
    lengths = case.get("lengths", (1.0, 1.0, 1.0))
    groups = {
        "job": {"basename": "TurbCheck"},
        "mesh": {"nghost": 2},
        "meshblock": {"nx1": 8, "nx2": 8, "nx3": 8},
        "time": {"evolution": "dynamic", "integrator": case.get("rk", "rk2"),
                 "cfl_number": case.get("cfl", 0.2),
                 "nlim": case.get("cycles", 2) if cycles is None else cycles,
                 "tlim": 10.0, "ndiag": 1},
        owner: {"eos": "ideal", "reconstruct": "plm",
                "rsolver": "hllc" if owner == "hydro" else "hlld",
                "gamma": GAMMA, "fofc": "false"},
        "problem": {"pgen_name": "turbulence", "d_n": 1.0, "d_i": 1.0,
                    "ifield": 2, "beta": 10.0},
        "turb_driving": {"nlow": 1, "nhigh": 3, "driving_type": 0,
                         "random_seed": 271828,
                         "dedt": case.get("power", 0.02) if power is None else power,
                         "tcorr": case.get("tcorr", 0.05),
                         "dt_turb_update": case.get("update", 0.01)
                         if update is None else update,
                         "dt_turb_thresh": case.get("thresh", 1.0e-6),
                         "sol_fraction": 1.0, "spect_form": 2, "expo": 1.0,
                         "turb_flag": 2, "restart_forcing": "true"},
        "output1": {"file_type": "rst", "dcycle": 1},
        "output2": {"file_type": "log", "dcycle": 1},
    }
    for d, length in enumerate(lengths, 1):
        groups["mesh"].update({f"nx{d}": 16, f"x{d}min": 0.0, f"x{d}max": length,
                               f"ix{d}_bc": "periodic", f"ox{d}_bc": "periodic"})
    return "\n".join(f"<{name}>\n" + "\n".join(f"{k} = {v}" for k, v in block.items())
                     + "\n" for name, block in groups.items())


def launch(args, directory, text, *, restart=None, reject=None):
    directory.mkdir(parents=True, exist_ok=False)
    input_path = directory / "input.athinput"
    input_path.write_text(text)
    command = []
    if args.launcher:
        command += shlex.split(args.launcher.format(ranks=args.ranks))
    else:
        require(args.ranks == 1, "An MPI launcher is required for multiple ranks")
    command += [str(args.binary), "--kokkos-map-device-id-by=mpi_rank"]
    if restart is not None:
        command += ["-r", str(Path(restart).resolve())]
    command += ["-i", str(input_path.resolve()), "-t", "00:02:00"]
    save(directory / "command.json", command)
    started = time.monotonic()
    with (directory / "athena.log").open("x") as log:
        process = subprocess.Popen(command, cwd=directory, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    save(directory / "execution.json",
         {"returncode": code, "elapsed_seconds": time.monotonic() - started,
          "input_sha256": sha(input_path),
          "restart_sha256": sha(restart) if restart else None})
    log = (directory / "athena.log").read_text(errors="replace")
    if reject is not None:
        require(code != 0, "Invalid forcing configuration was accepted")
        require(reject in log,
                f"Failed for a different reason; missing diagnostic: {reject}")
        return {}, {}
    require(code == 0, f"Athena exited with status {code}")
    require("Terminating on wall clock limit" not in log,
            "Athena reached its wall-clock limit")
    states, files = {}, {}
    for path in sorted((directory / "rst").glob("*.rst")):
        state = read_checkpoint(path)
        states[state["cycle"]], files[state["cycle"]] = state, path
    require(states, "No checkpoint files were written")
    diagnostic = directory / "TurbCheck.log"
    require(diagnostic.is_file(), "Missing floor/repair diagnostic")
    rows = [list(map(int, line.split())) for line in diagnostic.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    require(all(len(row) == 8 and not any(row[1:]) for row in rows),
            "EOS floor or repair counters were nonzero")
    return states, files


# --------------------------------------------------------------------------------------
# T1: timestep cap

def hydro_dt(state, cfl):
    """CFL-limited ideal-hydro step of the saved state, as Hydro::NewTimeStep."""
    u = state["u"]
    rho = u[:, 0]
    velocity = u[:, 1:4] / rho[:, None]
    pressure = (GAMMA - 1.0) * thermal(state)
    sound = np.sqrt(GAMMA * pressure / rho)
    widths = state["widths"]          # (block, 3) cell widths x1, x2, x3
    limit = math.inf
    for d in range(3):
        local = widths[:, d][:, None, None, None] / (np.abs(velocity[:, d]) + sound)
        limit = min(limit, float(local.min()))
    return cfl * limit


def boundaries_reached(t, update, thresh):
    return math.floor((t + thresh) / update)


def turbulence_cap(t, update, thresh):
    return min(update, update * (boundaries_reached(t, update, thresh) + 1) - t)


def check_timesteps(case, states):
    update, thresh, cfl = case["update"], case.get("thresh", 1.0e-6), case.get("cfl", 0.2)
    cycles = case["cycles"]
    require(set(states) == set(range(cycles + 1)), "Missing cycle checkpoints")
    counts = {"cap": 0, "hydro": 0, "growth": 0}
    boundary_landings, short_landings, worst = 0, 0, 0.0
    for cycle in range(1, cycles + 1):
        before, after = states[cycle - 1], states[cycle]
        dt = after["last_dt"]
        require(abs(after["time"] - before["time"] - dt)
                <= 32.0 * EPS * max(after["time"], dt),
                f"Cycle {cycle}: elapsed time disagrees with the step length")
        limits = {"cap": turbulence_cap(before["time"], update, thresh),
                  "hydro": hydro_dt(before, cfl)}
        if cycle > 1:
            limits["growth"] = 2.0 * before["last_dt"]
        binding = min(limits, key=limits.get)
        expected = limits[binding]
        error = abs(dt - expected) / expected
        worst = max(worst, error)
        require(error <= DT_RTOL,
                f"Cycle {cycle}: dt {dt:.17g} differs from {binding} limit "
                f"{expected:.17g} (limits {limits})")
        counts[binding] += 1
        reached_before = boundaries_reached(before["time"], update, thresh)
        reached_after = boundaries_reached(after["time"], update, thresh)
        require(reached_after - reached_before <= 1, f"Cycle {cycle} skipped a boundary")
        if reached_after > reached_before:
            boundary_landings += 1
            target = update * reached_after
            require(abs(after["time"] - target) <= thresh,
                    f"Cycle {cycle} crossed boundary {target} without landing on it")
            short_landings += after["time"] < target
        # The update count after a step equals the boundaries reached at its start + 1.
        require(after["updates"] == reached_before + 1,
                f"Cycle {cycle}: {after['updates']} OU updates, expected "
                f"{reached_before + 1}")
    final = boundaries_reached(states[cycles]["time"], update, thresh)
    require(boundary_landings == final,
            "Number of boundary landings differs from boundaries passed")
    for name in case.get("require_binding", ()):
        require(counts[name] > 0, f"No step was limited by {name}; case lacks coverage")
    return {"binding_counts": counts, "boundary_landings": boundary_landings,
            "landings_below_boundary_by_rounding": short_landings,
            "maximum_relative_dt_error": worst,
            "final_time": states[cycles]["time"]}


def set_restart_time(source, destination, new_time):
    """Copy a checkpoint and replace only its time."""
    data = bytearray(source.read_bytes())
    end = data.find(b"<par_end>\n") + len(b"<par_end>\n")
    offset = end + struct.calcsize("<ii9d19i19i")
    old = struct.unpack_from("<d", data, offset)[0]
    require(old == read_checkpoint(source)["time"], "Time field offset mismatch")
    struct.pack_into("<d", data, offset, new_time)
    destination.write_bytes(data)
    require(read_checkpoint(destination)["time"] == new_time, "Edited time not read back")


def check_short_restart(args, case, directory):
    """Restart on a boundary versus one rounding error short of it."""
    seed, files = launch(args, directory / "seed", configuration(case))
    cycle = case["cycles"]
    on_boundary = seed[cycle]
    reached = boundaries_reached(on_boundary["time"], case["update"], case["thresh"])
    target = case["update"] * reached
    require(abs(on_boundary["time"] - target) <= 4.0 * EPS * target,
            "Seed did not end on an update boundary")
    exact, short = directory / "exact.rst", directory / "short.rst"
    set_restart_time(files[cycle], exact, target)
    set_restart_time(files[cycle], short, np.nextafter(target, 0.0))
    continuation = dict(case, cycles=cycle + 1)
    reference, _ = launch(args, directory / "from-exact", configuration(continuation),
                          restart=exact)
    test, _ = launch(args, directory / "from-short", configuration(continuation),
                     restart=short)
    a, b = reference[cycle + 1], test[cycle + 1]
    require(b["updates"] == a["updates"] == reached + 1,
            "A restart just below a boundary did not count it as reached")
    require(abs(b["last_dt"] - a["last_dt"]) <= 4.0 * EPS * target,
            f"Short restart took step {b['last_dt']:.17g}, reference {a['last_dt']:.17g}")
    require(b["last_dt"] > 0.5 * case["update"], "Short restart took a sliver step")
    require(np.array_equal(a["raw"], b["raw"]),
            "Short restart produced a different OU field")
    return {"boundary": target, "short_time": float(np.nextafter(target, 0.0)),
            "reference_dt": a["last_dt"], "short_dt": b["last_dt"],
            "updates": b["updates"]}


# --------------------------------------------------------------------------------------
# T2: first-kick energy

def check_first_kick(case, states):
    require(set(states) == {0, 1, 2}, "Missing first-kick checkpoints")
    initial = totals(states[0])
    startup = totals(states[1])
    require(abs(startup["energy"] - initial["energy"])
            <= ROUNDOFF_ULPS * EPS * abs(initial["energy"]),
            "The undriven first cycle changed the total energy")
    # The initial state is uniform and at rest, so the undriven first cycle must
    # leave it unchanged: no momentum and a uniform internal energy.
    require(not np.any(states[1]["u"][:, 1:4]),
            "The undriven first cycle developed a flow; the initial state is not uniform")
    eint = thermal(states[0])
    require(np.ptp(eint) <= ROUNDOFF_ULPS * EPS * float(np.max(eint)),
            "Initial internal energy is not uniform")
    first, second = startup, totals(states[2])
    dt = states[2]["last_dt"]
    expected = case.get("power", 0.02) * dt
    exact = case["rk"] in ("rk1", "rk2")
    energy = check_budget(second["energy"] - first["energy"], expected,
                          EXACT_RTOL if exact else TRUNCATION_RTOL,
                          abs(first["energy"]) + abs(second["energy"]),
                          "first driven step total energy")
    result = {"dt": dt, "energy": energy,
              "kinetic_change": second["kinetic"] - first["kinetic"],
              "internal_change": second["internal"] - first["internal"]}
    if case["rk"] == "rk1":
        # A single stage from a uniform state at rest has no fluxes, only the kick.
        result["kinetic"] = check_budget(second["kinetic"] - first["kinetic"], expected,
                                         EXACT_RTOL, abs(first["energy"]),
                                         "RK1 kinetic energy")
        require(abs(result["internal_change"])
                <= ROUNDOFF_ULPS * EPS * abs(first["energy"]),
                "RK1 kick changed the internal energy")
    return result


# --------------------------------------------------------------------------------------
# T3: power root with positive and negative work

def shear_checkpoint(source, destination, velocity, zero_force=False):
    """Stationary transverse shear v_y = velocity*sin(2 pi x) with raw force along it."""
    state = read_checkpoint(source)
    require(state["u"].shape[1] == 5 and state.get("bcc") is None,
            "Shear fixture requires ideal hydro")
    require(state["raw"] is not None and state["updates"] == 1,
            "Shear fixture needs one stored OU update")
    data = bytearray(source.read_bytes())
    full = tuple(state["full_shape"])
    active = state["active_slices"]
    xmin = float(state["params"]["mesh"]["x1min"])
    width = float(state["params"]["mesh"]["x1max"]) - xmin
    internal = thermal(state)
    for m in range(len(state["locations"])):
        x = state["centers"][m, 0, 0, 0]
        dx = x[1] - x[0]
        xfull = x[0] + (np.arange(full[-1]) - active[-1].start) * dx
        sine = np.sin(2.0 * math.pi * (xfull - xmin) / width)[None, None, :]
        offset = state["payload_offset"] + m * state["block_stride"]
        u = np.ndarray((5,) + full, dtype="<f8", buffer=data,
                       offset=offset + state["u_offset"])
        density = float(state["u"][m, 0, 0, 0, 0])
        u[0] = density
        u[1:4] = 0.0
        u[2] = density * velocity * sine
        u[4] = float(internal[m, 0, 0, 0]) + 0.5 * u[2] ** 2 / density
        raw = np.ndarray((3,) + full, dtype="<f8", buffer=data,
                         offset=offset + state["raw_offset"])
        raw[:] = 0.0
        if not zero_force:
            raw[1] = sine
        force = np.ndarray((3,) + full, dtype="<f8", buffer=data,
                           offset=offset + state["force_offset"])
        force[:] = 0.0
    destination.write_bytes(data)
    totals(read_checkpoint(destination))


def smallest_root(m0, m1, power):
    """Root of m0*s^2 + m1*s = power nearest zero, by direct quadratic formula."""
    disc = math.sqrt(m1 * m1 + 4.0 * m0 * power)
    roots = [(-m1 + disc) / (2.0 * m0), (-m1 - disc) / (2.0 * m0)]
    return min(roots, key=abs)


def check_work(args, case, directory):
    seed, files = launch(args, directory / "seed", configuration(case, cycles=1))
    require(set(seed) == {0, 1}, "Missing shear-seed endpoint")
    fixture = directory / "shear.rst"
    shear_checkpoint(files[1], fixture, case["velocity"])
    initial = read_checkpoint(fixture)
    control, _ = launch(args, directory / "zero-power", configuration(case, power=0.0),
                        restart=fixture)
    driven, _ = launch(args, directory / "positive-power", configuration(case),
                       restart=fixture)
    require(max(control) == max(driven) == 2, "Shear continuations did not reach cycle 2")
    before, after = control[2], driven[2]
    dt = after["last_dt"]
    require(before["last_dt"] == dt, "Paired continuations took different timesteps")
    scale = max(float(np.max(np.abs(initial["u"]))), 1.0)
    require(float(np.max(np.abs(before["u"] - initial["u"])))
            <= ROUNDOFF_ULPS * EPS * scale,
            "Stationary-shear control evolved; it does not isolate the forcing")
    raw = initial["raw"]
    require(np.array_equal(after["raw"], raw), "OU process replaced the prescribed mode")

    # Independent normalization from the state at the start of the driven step.
    rho, momentum = initial["u"][:, 0], initial["u"][:, 1:4]
    mass = volume_sum(initial, rho)
    mean = [volume_sum(initial, rho * raw[:, d]) / mass for d in range(3)]
    pattern = raw - np.array(mean)[None, :, None, None, None]
    m0 = 0.5 * dt * volume_sum(initial, rho * np.sum(pattern ** 2, axis=1))
    m1 = volume_sum(initial, np.sum(momentum * pattern, axis=1))
    power = case.get("power", 0.02)
    s = smallest_root(m0, m1, power)
    require(m1 * case["velocity"] > 0.0,
            "Fixture does not exercise the requested work sign")
    require((s > 0.0) == (case["velocity"] > 0.0),
            "Smallest-magnitude root has an unexpected sign")
    expected_force = s * pattern
    force_error = float(np.max(np.abs(after["force"] - expected_force)))
    force_scale = float(np.max(np.abs(expected_force)))
    require(force_error <= ROUNDOFF_ULPS * EPS * force_scale,
            f"Applied force differs from the smallest-root oracle by {force_error:.3g}")
    other = -m1 / m0 - s
    first, last = totals(before), totals(after)
    energy = check_budget(last["energy"] - first["energy"], power * dt, EXACT_RTOL,
                          abs(first["energy"]) + abs(last["energy"]),
                          "work-case total energy")
    kinetic = check_budget(last["kinetic"] - first["kinetic"], power * dt,
                           WORK_KINETIC_RTOL, abs(first["energy"]) + abs(last["energy"]),
                           "work-case kinetic energy")
    # Within an RK2 step the first stage adds the kinetic energy of the kick before
    # the matching total energy, so the stage pressure dips by rho*|a|^2*dt^2/2 where
    # the force is strong. The resulting flow is O(dt^3), far below the impulse.
    impulse = rho[:, None] * expected_force * dt
    momentum_error = float(np.max(np.abs(after["u"][:, 1:4] - momentum - impulse)))
    impulse_scale = float(np.max(np.abs(impulse)))
    require(momentum_error <= IMPULSE_RTOL * impulse_scale,
            f"Momentum change differs from the oracle impulse by {momentum_error:.3g}")
    net = [volume_sum(after, after["u"][:, d] - initial["u"][:, d]) for d in (1, 2, 3)]
    require(max(map(abs, net)) <= ROUNDOFF_ULPS * EPS * max(mass, 1.0),
            "Forcing changed the net momentum")
    return {"dt": dt, "m0": m0, "m1": m1, "root": s, "other_root": other,
            "maximum_force_error": force_error, "energy": energy, "kinetic": kinetic,
            "momentum_error_relative_to_impulse": momentum_error / impulse_scale,
            "net_momentum_change": net}


# --------------------------------------------------------------------------------------

POWER_REJECTION = "Turbulence forcing requires finite, nonnegative turb_driving/dedt"
VARIANCE_REJECTION = "Positive turbulence power requires nonzero finite forcing variance"

CASES = (
    [{"name": "timestep-cap-every-step", "kind": "timestep", "update": 0.004,
      "cycles": 40, "require_binding": ("cap",)},
     {"name": "timestep-mixed", "kind": "timestep", "update": 0.03, "cycles": 40,
      "require_binding": ("cap", "hydro")},
     {"name": "timestep-mixed-mhd", "kind": "timestep-mhd", "owner": "mhd",
      "update": 0.004, "cycles": 24, "require_binding": ()},
     {"name": "short-restart", "kind": "short", "update": 0.004, "thresh": 1.0e-6,
      "cycles": 6}]
    + [{"name": f"kick-{owner}-{rk}", "kind": "kick", "owner": owner, "rk": rk}
       for owner in ("hydro", "mhd") for rk in ("rk1", "rk2", "rk3", "rk4")]
    + [{"name": "work-positive", "kind": "work", "velocity": 0.5,
        "update": 1.0, "tcorr": 10.0},
       {"name": "work-negative", "kind": "work", "velocity": -0.5,
        "update": 1.0, "tcorr": 10.0},
       {"name": "reject-negative-power", "kind": "reject", "power": -0.02},
       {"name": "reject-nan-power", "kind": "reject", "power": "nan"},
       {"name": "reject-infinite-power", "kind": "reject", "power": "inf"},
       {"name": "reject-zero-variance", "kind": "zero-field", "update": 1.0,
        "tcorr": 10.0}])


def check_mhd_timestep_cap(case, states):
    """MHD: steps never cross a boundary, land on each one, and update counts match."""
    update, thresh = case["update"], case.get("thresh", 1.0e-6)
    landings = 0
    for cycle in range(1, case["cycles"] + 1):
        before, after = states[cycle - 1], states[cycle]
        cap = turbulence_cap(before["time"], update, thresh)
        require(after["last_dt"] <= cap * (1.0 + DT_RTOL),
                f"Cycle {cycle} exceeded the cap")
        reached_before = boundaries_reached(before["time"], update, thresh)
        reached_after = boundaries_reached(after["time"], update, thresh)
        if reached_after > reached_before:
            landings += 1
            require(abs(after["time"] - update * reached_after) <= thresh,
                    f"Cycle {cycle} crossed a boundary without landing on it")
        require(after["updates"] == reached_before + 1, f"Cycle {cycle}: OU update count")
    require(landings == boundaries_reached(states[case["cycles"]]["time"], update,
                                           thresh),
            "Boundary landings differ from boundaries passed")
    require(landings >= case["cycles"] // 2, "MHD case barely reached any boundary")
    return {"boundary_landings": landings}


def run_case(args, case):
    directory = args.out / case["name"]
    kind = case["kind"]
    if kind == "reject":
        launch(args, directory, configuration(case), reject=POWER_REJECTION)
        return {"expected_rejection": True}
    if kind == "zero-field":
        directory.mkdir()
        _, files = launch(args, directory / "seed", configuration(case, cycles=1))
        fixture = directory / "shear.rst"
        shear_checkpoint(files[1], fixture, 0.0, zero_force=True)
        launch(args, directory / "positive-power", configuration(case), restart=fixture,
               reject=VARIANCE_REJECTION)
        return {"expected_rejection": True}
    if kind == "timestep":
        states, _ = launch(args, directory, configuration(case))
        return check_timesteps(case, states)
    if kind == "timestep-mhd":
        states, _ = launch(args, directory, configuration(case))
        return check_mhd_timestep_cap(case, states)
    if kind == "short":
        directory.mkdir()
        return check_short_restart(args, case, directory)
    if kind == "kick":
        states, _ = launch(args, directory, configuration(case))
        return check_first_kick(case, states)
    if kind == "work":
        directory.mkdir()
        return check_work(args, case, directory)
    raise AssertionError(f"Unknown case kind {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--launcher", default="mpirun -np {ranks}")
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--case", action="append", default=[],
                        help="Case name or glob; repeatable")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    selected = [case for case in CASES if not args.case or any(
        fnmatch.fnmatchcase(case["name"], pattern) for pattern in args.case)]
    require(selected, "No cases selected")
    if args.list:
        print("\n".join(case["name"] for case in selected))
        return 0
    require(args.binary is not None and args.out is not None,
            "Provide --binary and --out")
    require(args.ranks > 0 and args.timeout > 0, "Ranks and timeout must be positive")
    args.binary = args.binary.resolve()
    require(args.binary.is_file(), "Executable does not exist")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    scripts = [Path(__file__).resolve(),
               Path(__file__).resolve().with_name("checkpoint.py")]
    report = {"status": "RUNNING", "binary": str(args.binary),
              "binary_sha256": sha(args.binary), "ranks": args.ranks,
              "script_sha256": [sha(p) for p in scripts], "cases": []}
    save(args.out / "summary.json", report)
    try:
        for case in selected:
            try:
                result = {"name": case["name"], "status": "PASS", **run_case(args, case)}
            except Exception as error:
                result = {"name": case["name"], "status": "FAIL", "error": str(error),
                          "traceback": traceback.format_exc()}
            report["cases"].append(result)
            save(args.out / "summary.json", report)
            print(result["name"], result["status"], result.get("error", ""), flush=True)
        report["status"] = ("PASS" if all(c["status"] == "PASS" for c in report["cases"])
                            else "FAIL")
    finally:
        unchanged = (sha(args.binary) == report["binary_sha256"] and
                     [sha(p) for p in scripts] == report["script_sha256"])
        report["inputs_unchanged"] = unchanged
        if not unchanged or report["status"] == "RUNNING":
            report["status"] = "FAIL" if not unchanged else "INCOMPLETE"
        save(args.out / "summary.json", report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
