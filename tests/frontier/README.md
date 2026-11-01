# Frontier: HIP correctness and scaling of the on-the-fly sf/spec outputs

This kit checks HIP correctness and scaling of the sf/spec outputs on Frontier.
Complete Steps 1-2 with the intended binaries before requesting a multi-node
allocation.

**Build.** No build options are specific to this feature. Use the usual AthenaK
Frontier build: Kokkos HIP, `Kokkos_ARCH_AMD_GFX90A`, MPI on. Build twice:
- (a) default, for scaling and production;
- (b) `-DPROBLEM=sf_unit_test`, for the correctness lanes.

**Environment.** `frontier_env.sh` holds the module set and MPICH settings of the
production turbulence job:
- PrgEnv-cray, cce 19.0.0, rocm 6.4.2, cray-mpich 8.1.31;
- `MPICH_GPU_SUPPORT_ENABLED=1` and the IPC cache, MR cache and single-copy settings.

Both batch scripts source it and stop if a module operation fails or a required
module is missing from the resulting list; build with the same modules. Every run
records the resolved modules, `hipcc` version, CMake options and binary hash.

**Build check.** `check_build.sh ATHENA scaling|lanes|any` refuses, before any launch:
- a missing CMake cache, or one newer than the binary;
- MPI off (a serial build under `srun` would run independent copies);
- a backend other than HIP with gfx90a (`REQUIRE_BACKEND=CUDA` on della, `none` to
  skip);
- the wrong problem generator: the default build (`built_in_pgens`) for scaling and
  memory, `-DPROBLEM=sf_unit_test` for the lanes.

The lanes, the scaling job and `memcheck.sh` run it, and also refuse tracked files
that differ from `HEAD` (lanes and scaling job).

**Python.** The checkers need python3 with numpy, scipy and h5py (`cray-python`, or
`PYENV_BIN=<env>/bin`). `spec_reference.py` also imports `spectra.py` from a checkout of
`athena_research`: `export ATHENA_RESEARCH=<checkout>`. Access to this separate
reference checkout is required for the complete lanes and cross-node preflight.
The [test guide](../README.md) lists checks that run from AthenaK alone. The
reference is pinned by content: the lanes and the selection runner refuse a checkout
whose `spectra.py` (or `structure_functions.py`) and helper modules are not
byte-identical to the files della validated; the hashes are in the scripts (numpy
fallback; no CuPy needed).

**Results are never overwritten.** Every script refuses an output directory that
already holds results; a rerun goes to a new directory, and failed results stay.

**Kit self-test.** `kit_selftest.sh OUTDIR SCALEBIN LANESBIN SERIALBIN [MPIRUN]` injects
failures (wrong builds, failed module loads, too little memory, occupied directories, a
failed launch, a failed simulation under `memcheck.sh`, 2 nodes without `TESTBIN`, a
cross-node skip without a matching earlier pass, header-only, truncated and
trailing-data output files, NaN and infinite loop times, and missing, vanishing,
invalid or growing GCD samples) and checks that every guard rejects them: 36 cases. It
needs no allocation; with MPIRUN it adds one small `mpirun` series and one short
`memcheck.sh` run. Its last line must be `KIT SELFTEST: PASS`.

**Launch style.** By default the kit launches as production does: `srun -N <nodes> -n
<8 x nodes>`, with `--kokkos-map-device-id-by=mpi_rank` right after the binary
(`ATHENA_ARGS`). Every rank sees all 8 GCDs and Kokkos picks one by rank.

To use the OLCF-documented binding instead:
- scaling: `LAUNCH_STYLE=closest`;
- lanes: `SRUN_EXTRA="-c 7 --gpus-per-task=1 --gpu-bind=closest" ATHENA_ARGS=`.

**Time limits.** Every launch is bounded:
- `srun_np.sh`: `LAUNCH_TIMEOUT`, default 1800 s;
- `run_variants.sh`: `RUN_TIMEOUT`, default 1200 s;
- `check_binding.sh`: `BIND_TIMEOUT`, default 600 s;
- `memcheck.sh`: `RUN_TIMEOUT`, default 3600 s (the sampler step too).

When the limit is reached, `timeout` sends SIGTERM, then SIGKILL 60 s later, and a Slurm
step `--time` limit ends the step a minute after that.

**Queue policy (OLCF, checked 2026-10-27):**

| Queue | Nodes | Max wall time |
|---|---|---|
| batch | 1-91 | 2 h |
| batch | 92-183 | 6 h |
| batch | 184 and more | 12 h |
| `-p service -q develop` | at most 4 | 2 h |

- batch allows four eligible-to-run jobs per user.
- The develop nodes are isolated from batch; one job runs at a time.

## Device placement: `check_binding.sh NRANK ATHENA OUTDIR`

**How it works.** It runs the real executable with the same launcher and `ATHENA_ARGS`
as the tests, using `srun -l` and `--kokkos-print-configuration`, on a tiny input with
`nlim = 0`. Each task prints its host and device visibility and then execs Athena, so
all three records come from the same process. It maps every rank to a (host, device)
pair:
- production style: the device Kokkos marks "Selected";
- closest style: the visible device (`VISVAR`, default `ROCR_VISIBLE_DEVICES`).

**PASS:**
- exactly one Athena rank banner, with NRANK ranks;
- every rank one Kokkos-selected device of the HIP backend (`REQUIRE_BACKEND`);
- the ranks on each host use distinct devices.

**Where it runs:** first in the lanes, and as the scaling preflight over all ranks.

## 1. HIP correctness (build b, one node, develop QOS)

```
cd tests/frontier
sbatch --export=ALL,BIN=/path/to/build-b/src/athena,ATHENA_RESEARCH=/path/to/athena-research-rm \
  frontier_lanes.sbatch
```

**Preflight.** The job stops before any lane unless:
- the environment loaded (`frontier_env.sh`);
- `check_build.sh <b> lanes` passes;
- the tracked files equal `HEAD`;
- numpy, scipy and h5py import, and `spectra.py` imports and matches the validated file;
- it is inside an allocation, with a new output directory.

The `binding` and `mpi_ranks` lanes are fatal: if either fails, no later lane runs.

**Lanes:**

| Lane | np |
|---|---|
| binding | 8 |
| MPI rank preflight | 2 |
| pencil FFT matrix | up to 8 |
| spectrum reference | 1, 8 |
| SF2/FFT identity | 1, 8 |
| focused checks | 1, 2 |
| sf suite | 1, 2 |
| sparse sampling | 2 |
| selection | 2 |
| changed-rank restart | 1 then 2 |

The job writes `lanes_<jobid>/record.txt`; its last line must be
`FRONTIER LANES: PASS`. Resolve failed lanes and repeat them in a fresh output
directory before increasing the allocation. Keep every failed result.

## 2. One-node checks (build a)

**Smoke scaling:**

```
sbatch -N 1 --export=ALL,BIN=/path/to/build-a/src/athena,NX=512,MB=128 frontier_scaling.sbatch
```

**Repeated-output memory**, at the target workload (256^3 cells per GCD), inside a
one-node allocation (`salloc -N 1 ...`, or a batch script):

```
source tests/frontier/frontier_env.sh
BUDGET=light tests/frontier/memcheck.sh /path/to/build-a/src/athena 8 512 128 100 memcheck_out
```

`memcheck.sh` writes sf and spec outputs every cycle for 100 cycles, while sampling
each GCD's memory (`rocm-smi --showmeminfo vram --csv`, override with `MEMCMD`).
- **The run must be valid:** exit 0, the 8-rank banner, cycle `NLIM`, and every
  output present with its layout's size and finite values. Otherwise
  `MEMCHECK: FAIL`, whatever the memory does.
- **Windows come from completed output events** (the output files' cycle and time):
  early from event max(2, NLIM/10) over NLIM/5 events, late the last NLIM/5 events.
- **Every GCD must be sampled and is judged on its own** (`memcheck_analyze.py`,
  `MEMMODE=device`, the default): all of `MEMEXPECT` (default `card0`..`card7`) need at
  least 3 finite readings in both windows. A GCD never sampled, gone before the late
  window or with invalid readings gives `INCONCLUSIVE`, never PASS. `MEMCHECK: PASS`
  means no GCD grew more than 5%; `GROWTH` otherwise. Each GCD's sampled peak is
  printed (a 2 s sampler does not give the exact high-water mark).
- `MEMMODE=process` is for shared nodes with per-process rows (della `nvidia-smi`): a
  process seen in only one window is another job's and is not judged; at least NRANK
  processes must be.
- Check the parsed "used" column and the card names against `mem_samples.txt` the
  first time; if `rocm-smi` names the GCDs differently, set `MEMEXPECT` to its eight
  names.
- If memory grows, repeat with `TRANSPORT=host` and with `PERSISTENT=1`.

## 3. Scaling (build a)

The run uses driven turbulence: the built-in `pgen_name = turbulence` with
`<turb_driving>`.

**Variants.** Each job runs four variants on the same binary and mesh, interleaved over
`REPEAT` = 2 repeats (base, sf, spec, both, base, ...):

| Variant | Outputs |
|---|---|
| `base` | none |
| `sf` | `scalar` (6 variables including computed `temperature` and `pressure`, weights `ones`, `rho`, histograms, `sparse`) and `long_trans` (velocity); 100 log bins, `max_order = 3` |
| `spec` | `scalar` (`rho`, `vx`, weights `ones`, `rho`) and `helmholtz` (velocity) |
| `both` | `sf` and `spec` together |

Outputs are written every `DCYCLE` cycles over `NLIM` cycles; the defaults are 10 and
40.

**Preflight.** Before any timed run, the job:
- checks the environment, the build (`check_build.sh scaling`) and that the tracked
  files equal `HEAD`;
- with 2 or more nodes, requires `TESTBIN=<build b>` and checks it
  (`check_build.sh lanes`), before any launch;
- checks the placement of all ranks;
- with 2 or more nodes, runs the spectrum reference on 16 ranks over 2 nodes, with
  device and host transport. A pass writes `xnode_accepted.txt` with both binary
  hashes; `XNODE_PREFLIGHT=0` skips the check only with `XNODE_ACCEPTED=<that file>`
  and matching hashes;
- checks the mesh: `NX` divisible by `MB`, and at least one MeshBlock per rank;
- checks a lower bound on the device memory per GCD (hydro and forcing state, plus the
  spectrum arrays): above 90% of `GPU_MEM_GIB` (default 64) the job stops.

**Failures stop the series:** the first launch that fails or times out ends the job;
the partial summary is kept (and INVALID).

```
cd tests/frontier
# weak scaling, 256^3 per GCD (start with 1 and 8 nodes; 64 only after those are understood)
sbatch -N 1  --export=ALL,BIN=<a>,NX=512,MB=128  frontier_scaling.sbatch
sbatch -N 8  --export=ALL,BIN=<a>,TESTBIN=<b>,NX=1024,MB=128 frontier_scaling.sbatch
sbatch -N 64 --export=ALL,BIN=<a>,TESTBIN=<b>,NX=2048,MB=128 frontier_scaling.sbatch
# strong scaling: one MB per curve, at least one MeshBlock per rank, memory headroom:
#   NX=1024 MB=128 on 8..64 nodes; NX=1024 MB=64 on 8..128 nodes
#   (1024^3 on one node needs more than a GCD's 64 GiB and is refused)
```

**sf budgets** (`make_input.sh`; all values are recorded in `run_info.txt`):
- `BUDGET=production` (default): the calibrated order-3 settings, `ndraw_block`
  262144, anchors x samples 262144 x 262144, `nreplicate` 8.
- `BUDGET=light`: 16384, 262144 x 16384, 4. A throughput setting, not a production
  precision.

**Sampling.** E2 pair work per rank is about A·S/(R·P):

| `SAMPLING` | Budgets | E2 pair work per rank |
|---|---|---|
| `global` (default) | A and S fixed | ~ 1/P |
| `sqrtp` | A, S ~ sqrt(P/8) | about constant |
| `stress` | A, S ~ P/8 | ~ P (an increasing-work stress test, not weak scaling) |

E1, computed fields, histogram set-up and selection stay roughly constant per rank, so
the total sf cost does not fall as 1/P.

Measured (Frontier, 2026-10-29, default build at cf011c41, production budgets, median of
2 repeats; seconds per output event of the two hydro sf outputs): `global` 1.59 / 0.41 /
0.28 s at 1 / 8 / 64 nodes; `sqrtp` 1.73 s at 8 nodes and 2.09 s at 64 nodes. `sqrtp`
is kept as an option, but `global` is the recommended production setting.

**Other options:**
- `VARIANTS="base sf"` runs a subset.
- `REPEAT=3` adds repeats.
- `PHYS=mhd` runs MHD turbulence and adds an `aniso_mhd` output.
- `TRANSPORT=host` stages the spectrum transposes through host memory.
- `SELECT=1` adds a temperature band 0.3 <= T <= 3. It measures the selection machinery
  only: the initial state is uniform at T = 0.6, so short runs select nearly every
  cell. `ncell_selected` is printed.
- `NLIM` and `DCYCLE` set the run length and output cadence.
- `RUN_TIMEOUT` and `LAUNCH_STYLE` are described above.

**Output.** `run_N<NX>_n<ranks>_<jobid>/` holds:
- the preflight results;
- one directory per variant and repeat;
- `run_info.txt`;
- `summary.txt` (`collect.py`, rerunnable on any finished directory).

**`summary.txt` validates first.** Every expected run must have exited 0, reported
the expected rank count and reached `NLIM`. Every block must have its output cycles
(0, each `DCYCLE`, and the final write at `NLIM`); every file needs the header keys of
its layout and exactly the payload size that layout implies; float64 arrays must be
finite; the spectrum sum rules must hold; every loop time must be a finite positive
number (if only that fails: `INVALID (timing unavailable ...; outputs valid)`). Otherwise the last line is
`SCALING SUMMARY: INVALID` and the exit status is non-zero.

**Timings:**
- **Loop:** AthenaK's `cpu time used`. It includes the cadence and final outputs, not
  the cycle-0 outputs.
- **Event cost:** (variant loop - base loop, same repeat) / output events. An event is
  one output time at which every configured block writes.
  - It is shown with the base scatter and the min/median/max over repeats.
  - It is marked "unresolved" when not above the base scatter.
- **Components** (maximum over ranks):
  - sf E1 and E2 pair kernels only;
  - spec setup, fft and bin, each the maximum of its own stage;
  - plus pencil shape, per-stage times of the last transform, and the spec-owned
    device memory (not the whole process).

**What to send back:**
- `summary.txt` and `run_info.txt` of every job;
- the `record.txt` of the lanes;
- the memcheck output.
