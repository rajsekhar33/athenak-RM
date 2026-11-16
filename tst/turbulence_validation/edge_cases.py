#!/usr/bin/env python3
"""AMR edge cases for the turbulence driver: driving end and OU replay cost.

duration: ideal MHD with turb_flag=1. Driving ends between two refinement
  events, so the reset happens both while driving and after the end. Before
  the end total energy must grow. Afterwards each step may change energy and
  momentum by less than 1e-5 of a driven step's energy gain, through every
  later mesh change and restart; an undriven control with the same mesh
  schedule records the scheme's own conservation residual for comparison.
replay: Hydro with one OU update per cycle and rare mesh changes. The log
  gives the wall time of each cycle and the number of OU updates replayed
  after each reset, so the replay cost per update can be compared with the
  cost of an ordinary cycle.

Examples:
  python edge_cases.py duration --binary /path/athena --out /scratch/edge-duration
  python edge_cases.py replay --binary /path/athena --out /scratch/edge-replay --backend gpu
"""  # noqa: E501

import argparse
import copy
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time
import traceback

import numpy as np

from amr import configuration, run_native, save, sha, topology_events, input_text
from checkpoint import compare_states, integrals, require

END_TIME = .0075   # between the AMR events at cycles 6 and 9 (dt = 1e-3)


def duration_case(args, case):
    cfg = configuration(case, args.cycles)
    cfg["turb_driving"].update(turb_flag=1, tdriv_duration=END_TIME)
    directory = args.out/case
    full = run_native(args, directory/"full", cfg)
    events = topology_events(full)
    require(any(full[e["cycle"]-1]["time"] < END_TIME for e in events),
            "No mesh change while driving")
    after = [e for e in events if full[e["cycle"]-1]["time"] >= END_TIME]
    require(len(after) >= 2, "Fewer than two mesh changes after the driving end")
    control_cfg = copy.deepcopy(cfg)
    # The first step is never forced, so this control is never driven.
    control_cfg["turb_driving"]["tdriv_duration"] = .0005
    control = run_native(args, directory/"unforced-control", control_cfg)
    require([e["cycle"] for e in topology_events(control)]
            == [e["cycle"] for e in events],
            "Unforced control has a different mesh-change schedule")
    steps = step_changes(full, events)
    driven = [s["delta_energy"] for s in steps if s["driving"]]
    require(driven and min(driven) > 0, "Driving did not inject energy")
    gate = 1.e-5*float(np.median(driven))
    for s in steps:
        if not s["driving"] and s["start_time"] >= END_TIME:
            require(abs(s["delta_energy"]) < gate,
                    f"Energy changed after driving end at cycle {s['cycle']}")
            require(s["delta_momentum"] < gate,
                    f"Momentum changed after driving end at cycle {s['cycle']}")
    if case.endswith("stored"):
        for c in sorted(full)[1:]:
            if full[c-1]["time"] >= END_TIME and full[c]["force"] is not None:
                prev, cur = full[c-1]["force"], full[c]["force"]
                require(not np.any(cur)
                        or (prev.shape == cur.shape and np.array_equal(prev, cur)),
                        f"Stored acceleration changed after driving end at cycle {c}")
    control_steps = step_changes(control, events)
    restarts = []
    for split in (events[1]["cycle"], after[0]["cycle"]):
        resumed = run_native(args, directory/f"restart-{split}", cfg, full[split]["path"])
        differences = {str(c): compare_states(full[c], s) for c, s in resumed.items()}
        restarts.append(dict(split_cycle=split, field_differences=differences))

    def residual(rows, key):
        return max(abs(r[key]) for r in rows if r["start_time"] >= END_TIME)
    return dict(name=case, status="PASS", end_time=END_TIME, events=events, steps=steps,
                post_end_gate=gate,
                post_end_max_energy_change=residual(steps, "delta_energy"),
                post_end_max_momentum_change=residual(steps, "delta_momentum"),
                control_max_energy_change=residual(control_steps, "delta_energy"),
                control_max_momentum_change=residual(control_steps, "delta_momentum"),
                control_steps=control_steps, restarts=restarts)


def step_changes(states, events):
    measures = {c: integrals(s) for c, s in states.items()}
    steps = []
    for c in sorted(states)[1:]:
        old, new = measures[c-1], measures[c]
        start = states[c-1]["time"]
        steps.append(dict(cycle=c, start_time=start,
                          # UpdateForcing starts at cycle 1: the first step is unforced
                          driving=(c-1 >= 1 and start < END_TIME),
                          delta_energy=new["total_energy"]-old["total_energy"],
                          delta_momentum=float(np.max(np.abs(np.asarray(new["momentum"])
                                                             - old["momentum"]))),
                          mesh_change=any(e["cycle"] == c for e in events)))
    return steps


def replay_case(args):
    cfg = configuration("hydro-stored", args.cycles)
    cfg["mesh"].update({f"nx{d}": args.root for d in (1, 2, 3)})
    cfg["meshblock"].update({f"nx{d}": args.block for d in (1, 2, 3)})
    cfg["problem"]["particle_amr_period"] = args.period
    cfg["turb_driving"].update(tcorr=.5, dt_turb_update=.001)
    cfg["time"].update(ndiag=1, tlim=10.*args.cycles*.001)
    for key in ("output1", "output2", "output3"):
        cfg.pop(key)
    directory = args.out/"replay"
    directory.mkdir(parents=True, exist_ok=False)
    inp = directory/"input.athinput"
    inp.write_text(input_text(cfg))
    command = []
    if args.launcher.lower() != "none":
        command += shlex.split(args.launcher.format(ranks=args.ranks))
    command += [str(args.binary)]
    if args.backend == "gpu":
        command += ["--kokkos-map-device-id-by=mpi_rank"]
    command += ["-i", str(inp), "-d", str(directory)]
    save(directory/"command.json", dict(argv=command, input_sha256=sha(inp)))
    begin = time.monotonic()
    with (directory/"athena.log").open("x") as log:
        rc = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                            timeout=args.timeout).returncode
    save(directory/"execution.json", dict(returncode=rc, elapsed=time.monotonic()-begin))
    require(rc == 0, "Native exit "+str(rc))
    # Assign OU-update lines to the cycle whose step follows the elapsed line.
    cycles, current = [], None
    for line in (directory/"athena.log").read_text().splitlines():
        m = re.match(r"elapsed=(\S+) cycle=(\d+)", line)
        if m:
            current = dict(cycle=int(m.group(2)), elapsed=float(m.group(1)), updates=[])
            cycles.append(current)
        m = re.match(r"i_turb_update = (\d+)", line)
        if m and current is not None:
            current["updates"].append(int(m.group(1)))
    require(len(cycles) == args.cycles+1, "Missing per-cycle diagnostics")
    rows = []
    for a, b in zip(cycles, cycles[1:]):
        rows.append(dict(cycle=a["cycle"], wall=b["elapsed"]-a["elapsed"],
                         updates=len(a["updates"]),
                         replay=(bool(a["updates"]) and a["updates"][0] == 0
                                 and a["cycle"] > 0)))
    ordinary = [r["wall"] for r in rows[5:]
                if r["updates"] == 1 and not r["replay"]]
    replays = [r for r in rows if r["replay"]]
    require(ordinary and len(replays) >= 2, "Too few ordinary cycles or replays")
    base = float(np.median(ordinary))
    n = np.array([r["updates"] for r in replays], dtype=float)
    extra = np.array([r["wall"] for r in replays])-base
    slope, intercept = np.polyfit(n, extra, 1)
    return dict(name="replay", status="PASS", root=args.root, block=args.block,
                period=args.period, ordinary_cycle_median=base, replays=replays,
                seconds_per_replayed_update=float(slope),
                replay_intercept=float(intercept),
                replayed_updates_per_ordinary_cycle=(float(base/slope) if slope > 0
                                                     else None),
                total_wall=sum(r["wall"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lane", choices=("duration", "replay"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--launcher", default="mpirun -np {ranks}")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--root", type=int, default=64,
                        help="replay: root cells per side")
    parser.add_argument("--block", type=int, default=16,
                        help="replay: MeshBlock cells per side")
    parser.add_argument("--period", type=int, default=300,
                        help="replay: cycles between mesh changes")
    parser.add_argument("--timeout", type=float, default=1800.)
    args = parser.parse_args()
    args.analyze_only = False
    args.binary = args.binary.resolve()
    args.out = args.out.resolve()
    if args.cycles is None:
        args.cycles = 16 if args.lane == "duration" else 1500
    require(socket.gethostname().split(".")[0] in ("della-vis1", "della-vis2")
            or bool(os.environ.get("SLURM_JOB_ID")),
            "Native tests require a visualization node or approved allocation")
    args.out.mkdir(parents=True, exist_ok=False)
    stamp = args.binary.with_suffix(".sha256")
    report = dict(status="RUNNING", lane=args.lane, binary=str(args.binary),
                  binary_sha256=sha(args.binary),
                  binary_record_matches=(
                      stamp.is_file()
                      and stamp.read_text().strip() == sha(args.binary)),
                  harness_sha256=sha(__file__), backend=args.backend, ranks=args.ranks,
                  cases=[])
    (args.out/"retained-edge_cases.py").write_bytes(Path(__file__).read_bytes())
    save(args.out/"summary.json", report)
    work = (["mhd-stored", "mhd-regenerated"] if args.lane == "duration" else ["replay"])
    for case in work:
        try:
            row = (duration_case(args, case) if args.lane == "duration"
                   else replay_case(args))
        except Exception as error:
            row = dict(name=case, status="FAIL", error=str(error),
                       traceback=traceback.format_exc())
        report["cases"].append(row)
        save(args.out/"summary.json", report)
        print(case, row["status"], row.get("error", ""), flush=True)
    report["status"] = ("PASS" if report["binary_record_matches"]
                        and all(r["status"] == "PASS" for r in report["cases"])
                        else "FAIL")
    save(args.out/"summary.json", report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
