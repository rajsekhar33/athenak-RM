# HIP validation and scaling on Frontier

This directory contains build checks, GPU-placement checks, diagnostic benchmarks
and repeated-output memory measurements. Establish correctness and memory use on
one node before increasing the allocation.

## Builds and environment

Use MPI-enabled Kokkos HIP builds targeting `Kokkos_ARCH_AMD_GFX90A`:

- the normal `built_in_pgens` build for driven-turbulence benchmarks;
- `-DPROBLEM=sf_unit_test` for deterministic Fourier and structure-function tests.

Build and run with matching compiler, ROCm and MPI modules. The supplied
`frontier_env.sh` loads the configured module set and enables GPU-aware MPI.
The executable should retain its build directory and `CMakeCache.txt`.

`check_build.sh ATHENA scaling|lanes|any` checks the MPI/backend configuration,
GPU architecture, problem generator, and cache/executable timestamps before a
launch. For example:

```bash
tests/frontier/check_build.sh /absolute/path/to/default-build/src/athena scaling
tests/frontier/check_build.sh /absolute/path/to/test-build/src/athena lanes
```

The checking scripts need Python with NumPy and SciPy. Use fresh result
directories for each invocation.

## Placement and one-node correctness

Run the following inside a one-node allocation. Set absolute executable and
result paths. Frontier has eight GPU compute dies per node; assign one MPI rank
to each die.

```bash
source tests/frontier/frontier_env.sh
ATHENA=/absolute/path/to/test-build/src/athena
LAUNCHER="$PWD/tests/frontier/srun_np.sh"
OUT=/absolute/path/to/new-validation-results
export ATHENA_ARGS=--kokkos-map-device-id-by=mpi_rank

tests/frontier/check_binding.sh 8 "$ATHENA" "$OUT/binding"
tests/fft/run_pencil_suite.sh "$ATHENA" "$LAUNCHER" "$OUT/fft" 8
tests/sf/run_suite.sh "$ATHENA" "$LAUNCHER" 2 "$OUT/sf"
tests/sf/run_select.sh "$ATHENA" "$LAUNCHER" 2 "$OUT/selection"
tests/integration_checks.sh "$ATHENA" "$LAUNCHER" "$OUT/restart" 1 2
```

The placement checker records each rank's host and Kokkos-selected device and
requires distinct devices on a host. The launcher adapter bounds each Slurm step
and supplies the device-mapping argument.

For rank-local GPU visibility, use `SRUN_EXTRA='-c 7 --gpus-per-task=1 --gpu-bind=closest'`
and set `ATHENA_ARGS` to the empty string. Keep the same assignment method for
placement checks and subsequent runs.

The Fourier tests compare coefficients and mode inventories. Structure-function
tests compare sampled moments with exact sums on deterministic fields, including
selected populations. Restart tests require consistent state and diagnostic
continuation when the MPI rank count changes. Each harness returns nonzero on
failure.

## Diagnostic overhead

`run_variants.sh` evolves the same driven-turbulence setup with four output
configurations: baseline, structure functions, spectra, and both. Repeats are
interleaved to measure timing variability close to each diagnostic run.

A small one-node example, inside an allocation, is:

```bash
ATHENA=/absolute/path/to/default-build/src/athena
export ATHENA_ARGS=--kokkos-map-device-id-by=mpi_rank
LAUNCH='srun -N 1 -n 8' BUDGET=light \
  tests/frontier/run_variants.sh "$ATHENA" 8 128 32 40 10 \
  /absolute/path/to/new-overhead-results
```

The positional arguments after the executable are MPI ranks, cells per direction,
MeshBlock size, cycle limit, output cadence and result directory. The harness
checks divisibility, available MeshBlocks, build configuration and a lower bound
on device memory before launching. Leave additional memory for MHD fields and
runtime allocations.

`summary.txt` requires completed runs, correct rank counts and output cycles,
valid payload sizes, finite values and spectrum sum rules. It reports elapsed
loop time and diagnostic cost per output event, with baseline scatter and
repeat statistics. An unresolved timing difference remains identified as such.
Component timings and spectrum-buffer memory appear separately.

Useful settings include:

| Setting | Meaning |
|---|---|
| `PHYS=mhd` | MHD turbulence, including a field-aligned structure function |
| `TRANSPORT=host` | Host-staged spectrum communication |
| `VARIANTS='base sf'` | Run the named subset |
| `REPEAT=3` | Three repeats of each variant |
| `SELECT=1` | Apply a temperature band and report its selected population |
| `BUDGET=production` | 262144 same-block draws, 262144 global anchors and samples, eight replicates |
| `BUDGET=light` | 16384 same-block draws, 262144 anchors, 16384 samples, four replicates |

Use independent sampling budgets when assessing statistical precision. The
throughput benchmark alone does not establish the uncertainty of high-order
moments in a production flow.

## Scaling design

With eight ranks per node, a workload of `256³` cells per rank gives global meshes
of `512³`, `1024³` and `2048³` on 1, 8 and 64 nodes, respectively. Keep MeshBlock
size and the physical problem fixed when comparing allocations.

The global anchor and partner budgets also determine the sampled pair work:

| `SAMPLING` | Global budgets | Approximate different-block pair work per rank |
|---|---|---|
| `global` | Fixed | Decreases as `1/P` |
| `sqrtp` | Each grows as `sqrt(P/8)` | Approximately constant |
| `stress` | Each grows as `P/8` | Increases as `P` |

Here `P` is the rank count. Same-block work, selection and histogram setup have
different scaling, so report the complete budget alongside timings. For strong
scaling, fix the global mesh and sampling budgets and retain at least one
MeshBlock per rank. Verify placement and communication on the intended nodes
before a larger benchmark.

## Repeated-output memory

`memcheck.sh` writes SF and spectrum outputs every cycle and samples GPU memory.
For the `256³`-cells-per-rank workload inside a one-node allocation:

```bash
ATHENA=/absolute/path/to/default-build/src/athena
export ATHENA_ARGS=--kokkos-map-device-id-by=mpi_rank
BUDGET=light tests/frontier/memcheck.sh "$ATHENA" 8 512 128 100 \
  /absolute/path/to/new-memory-results
```

The run must reach its cycle limit with every expected output present and valid.
The checker compares early and late windows of completed output events for each
GPU. A pass requires at least three finite samples in both windows for every
expected device and at most 5% growth. Missing samples produce an inconclusive
result. Sampled peaks depend on the sampling interval and can be below the true
peak allocation.

Inspect `mem_samples.txt` to confirm device names and the memory-used column.
Set `MEMEXPECT` if the monitoring command uses different device names. Repeat
with `TRANSPORT=host` or `PERSISTENT=1` to measure communication staging and
retained FFT buffers.

`RUN_TIMEOUT`, `LAUNCH_TIMEOUT` and `BIND_TIMEOUT` bound benchmark, launcher and
placement steps. Set them within the available allocation wall time. Preserve
`run_info.txt`, `summary.txt`, placement records and memory samples together with
the executable hash and generated inputs.
