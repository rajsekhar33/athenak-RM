# Production-like partial-selection cross-check (sf `select1_*` vs Python masks)

Checks that AthenaK's on-the-fly structure functions with a cell selection
(`file_type = sf`, `weights = ones`, `select1_*`, `sparse = true`) agree with
`athena_research/operations/structure_functions.py` run the production way: a 0/1
mask installed as `ad.data_func[name]` and passed as the weight with
`sparse_data = True`. The Python side is the exact all-pairs sum, so the only
uncertainty is AthenaK's sampling error, which its replicates measure.

## The case

- `selection.athinput`: driven ideal-gas turbulence (built-in `pgen_name = turbulence`,
  `<turb_driving>` from `tests/frontier/make_input.sh` with `dedt = 1`,
  `random_seed = 1`), 64^3 periodic mesh, 16^3 MeshBlocks (64 blocks), `nlim = 600`
  (t = 0.145, density 10-90% range 0.958-1.044, v_rms ~ 0.5).
  At cycle 600 it writes:
  - `bin/sel.hydro_w.00001.bin`: the float32 snapshot Python loads;
  - `sf_scalar/sel.00001.sf`: `mode = scalar`, variables vx, vy, vz;
  - `sf_longtrans/sel.00001.sf`: `mode = long_trans`, the same velocity.

  Both sf blocks use:
  - weight `ones`;
  - selection `select1_variable = hydro_w_d`, `select1_min = 0.96875` (31/32),
    `select1_max = 0.9990234375` (1023/1024), `sparse = true`;
  - `max_order = 3`, `nbin = 24`, log bins, `r_min = 0.015625` (dx),
    `r_max = 0.8660254037844386` (half box diagonal). These are the values Python's
    `get_sf_mb` derives for this box, so its edges are the same;
  - `ndraw_block = 262144`, `nanchor_global = 262144`, `nsample_global = 65536`,
    `nreplicate = 32`, seeds 11 and 12.

  Output 00000 is the uniform initial state (no cell in the band). The final output at
  `nlim` writes a duplicate 00002 of the cycle-600 state with independent draws.
  `compare.py` reports it as information only.
- Selected fraction: 81796 / 262144 = 0.312 (header `ncell_selected`).

## Commands

This comparison requires a separate `athena_research` checkout and a Python
environment with CuPy, NumPy and SciPy. Use the normal built-in-problem MPI
build. For two CUDA GPUs on one machine, with a configured CUDA-aware MPI
launcher and a fresh output directory:

```bash
set -e
D=tests/sf/python_crosscheck/selection
export PYTHON="$(command -v python3)"
export ATHENA_RESEARCH=/absolute/path/to/athena-research
ATHENA=/absolute/path/to/default-gpu-build/src/athena
OUT=/absolute/path/to/new-selection-run
REQUIRE_BACKEND=CUDA "$D/run_athena.sh" "$ATHENA" mpirun 2 "$OUT" \
  --kokkos-map-device-id-by=mpi_rank
"$D/run_python.sh" "$OUT"
```

The run splits into two steps that can run on different machines:

```bash
# 1. AthenaK machine (e.g. Frontier). MPIRUN is called as "$MPIRUN -np N athena ..."; for
#    srun set MPIRUN_NP_FLAG=-n, e.g. MPIRUN_NP_FLAG=-n $D/run_athena.sh athena "srun -N2 --gpus-per-task=1" 2 OUT
$D/run_athena.sh ATHENA MPIRUN NRANKS OUTDIR
# copy OUTDIR/athena/bin/sel.hydro_w.00001.bin (5.3 MB) and OUTDIR/athena/sf_*/ (0.8 MB),
# keeping the paths, to the reference-analysis machine.
# 2. Set PYTHON and ATHENA_RESEARCH on that machine, then compute the CuPy reference.
$D/run_python.sh OUTDIR [NAME]       # results in OUTDIR/NAME (default python)
```

**Guards** (as in the Frontier kit):
- `run_athena.sh` runs `tests/frontier/check_build.sh ATHENA scaling` (MPI, backend
  `REQUIRE_BACKEND`, default HIP; `CUDA` on della; the default build), bounds the launch
  (`RUN_TIMEOUT`, default 1200 s, plus a step `--time` with srun) and requires exactly
  one banner with NRANKS ranks. Relative binary and launcher paths are resolved.
- `run_python.sh` refuses an existing `OUTDIR/NAME` (a rerun uses a new NAME), and an
  `athena_research` checkout whose `structure_functions.py` and the modules it and the
  reader use are not byte-identical to the validated files (hashes in the script;
  use a checkout with those file hashes). It writes `reference.txt` (checkout, commit,
  `structure_functions.py` hash, package versions).

`run_python.sh` runs:

```bash
"$PYTHON" "$D/python_sel.py" OUTDIR/athena/bin/sel.hydro_w.00001.bin \
  --pyrepo "$ATHENA_RESEARCH" --out OUTDIR/NAME/python_exact.npz
"$PYTHON" "$D/compare.py" OUTDIR/athena OUTDIR/NAME/python_exact.npz
```

**Checker self-test:** `python3 selftest.py OUTDIR NEWDIR` on a passing run damages
copies (one reference-populated bin emptied, one NaN replicate, one bin count doubled)
and requires each to FAIL while the unmodified copy passes.

`python_sel.py` defaults: `--lo 0.96875 --hi 0.9990234375 --nbins 24 --max_order 3`.

## Python side (`python_sel.py`)

The mask is built like `_install_interface_mask`
(`plot_rdshk_mach_transverse_interface.py`) and `install_interface_mask`
(`all_radshock_analysis.py`):

```python
ad.data_func['sel_bool'] = lambda ad, mbl, mbh: where(lo <= dens <= hi, 1, 0)
```

Both bounds are inclusive, as AthenaK's `select1_min`/`select1_max` are. The script
then calls:
- `get_sf_mb(ad, ['velx','vely','velz'], weights=['sel_bool'], sparse_data=True,
  use_exhaustive=True, max_order=3, nbins=24, log_bin_flag=True)`;
- `get_sf_helmholtz_mb(ad, 'vel', weights=['sel_bool'], sparse_data=True, ...)`.

`npairs = (4096 x 64)^2` makes `n_samples1` equal the block size, so every block pair
takes the exhaustive kernel. The script checks this: the raw pair count must equal
Nsel(Nsel-1)/2, otherwise it exits with an error.

How the two codes normalise:
- **Python.** `sf` is the sum of |w2 q2 - w1 q1|^p over pairs with both weights
  nonzero, which for a 0/1 mask is the sum of |dq|^p over selected pairs.
  `weight_sum_bin` = sum(w1 + w2) = 2C. `num_bin_points` = C. Python's S_p is
  `sf / (weight_sum_bin/2)`.
- **AthenaK.** The selection turns `ones` into the same 0/1 indicator. `T`, `W`, `C`
  estimate the same three sums (W = 2C), and S_p = T/C.

## Acceptance rule (`compare.py`, fixed before the run)

Exact checks (no sampling involved):

| | check |
|---|---|
| X1 | header `select1` = `hydro_w_d,lo,hi`, equal to Python's band |
| X2 | sf header time = snapshot time (same state) |
| X3 | selected fraction `ncell_selected/ncell_region` in [0.10, 0.50] |
| X4 | 0 <= nsel_Python - ncell_selected <= n_on_edge, with the difference resolved (below) |
| X5 | Python raw pair count = Nsel(Nsel-1)/2 (exhaustive) |
| X6 | bin edges identical bits, or max relative difference <= 1e-14 with every lattice separation more than 1e-12 (relative) from every edge |
| X7 | bins empty in Python (bin 1: no lattice separation) are exactly 0 in AthenaK T, W, C; all AthenaK arrays finite |
| X8 | W = 2C in every AthenaK replicate (rel 1e-12) and in Python |
| X9 | a test with zero replicate spread agrees to 1e-12 relative |
| X10 | every bin with reference pairs (Python exact C > 0) has AthenaK pairs |
| X11 | every expected statistic is present: a finite z, or a zero-spread equality (X9) |

Statistical test. The tests are fixed by the exact reference, not by what AthenaK
populated: every mode, variable, order p <= 3 and bin with reference pairs:
- **S_p**: z = (S_A - S_P)/se. The se is the delta-method replicate error,
  sd_r(T_r - S_A C_r) / (C_A sqrt R).
- **T_p**: z = (T_A - T_P)/(sd_r(T_r)/sqrt R).
- **C**: z = (C_A - C_P)/(sd_r(C_r)/sqrt R).

T_r and C_r are the per-replicate E1+E2 sums from the `*_er` arrays, with R = 32.
There are n = 736 tests in total; n is this fixed inventory, so a missing statistic
cannot lower the threshold.

**PASS iff max|z| < t_31^{-1}(1 - alpha/(2n)), with alpha = 1e-3**: a two-sided
Bonferroni test whose family-wise false-alarm rate stays at or below 1e-3 under any
correlation between tests. Here the threshold is 5.964. A test with zero replicate
spread would have to agree to 1e-12 relative instead (none occur).

The last line is `SELECTION CROSSCHECK: PASS` or `FAIL`, and the exit status is 0 or 1.

## Float32 snapshot and band edges

AthenaK selects on the double density. The `.bin` dump is float32, so Python selects on
the float32 density.
- **Edges are float32 numbers.** Rounding is monotone, so every cell AthenaK selects is
  also in the Python mask.
- **Only on-edge cells can differ.** A cell can be in the Python mask but not in
  AthenaK's selection only if its float32 density equals an edge exactly.
  `python_sel.py` counts these cells (`n_on_edge`) and computes their exact pair
  contributions with the formulas of the Python exhaustive kernels.
- **Cross-check of that pair code.** `validate_edge.py` checks it against
  `get_sf_mb`/`get_sf_helmholtz_mb`: identical counts, sums equal to 5e-14.
- **Correction in `compare.py`.** If `ncell_selected = nsel - n_on_edge`, AthenaK
  excluded all of them, and `compare.py` subtracts their pairs from the Python sums,
  which makes the selected sets identical.
- **This run.** One cell had float32 density exactly on an edge, and AthenaK excluded
  it (81796 vs 81797; its double value lies just outside the band). Its 81796 pairs
  were subtracted. Without this correction the change is at most 8e-5 relative per bin,
  about 0.3 se in the most precise bin.
- **Unresolved case.** A partial difference (possible only with two or more on-edge
  cells) is bounded instead (X4b: < 0.2 se_C per bin).
- **Velocities.** The float32 velocities shift the exact sums by about 1e-8 relative,
  which is negligible.

## Result on della-vis2 (2026-10-27, 2x A40)

- Binary: `build-default-gpu/src/athena` (CUDA + MPI).
- Python packages: CuPy 14.0.1 and NumPy 2.4.3.
- `athena_research` commit `4a61a007e1040d03ca79ae6334521b2239651692`.

| ranks | AthenaK | Python exact | tests | max\|z\| (threshold 5.964) | rms z | result |
|---|---|---|---|---|---|---|
| 2 | 88 s | 45 s | 736 | 3.60 (longtrans sum T_1 of the transverse part, bin 19) | 1.21 | PASS |
| 1 | 27 s | 56 s | 736 | 3.60 (longtrans S_1 L, bin 15) | 1.20 | PASS |

Details of the 2-rank run:
- All exact checks passed. The bin edges differ by at most 5.8e-16 relative (not
  bit-identical) and lie at least 1e-9 from any lattice separation; that 1e-9 is the
  widening of the outer edges, which both codes apply.
- Pooling the two independent outputs 00001 and 00002 (64 replicates) gives
  |mean z| <= 0.16 and rms z 0.83-1.28 per quantity, with no sign of bias.
- The Python step runs in about 50 s on an A40: 3.3e9 exhaustive pairs per mode.
- The complete X1-X11 checker passed all 736 statistics in both runs. A fresh
  two-rank run passed with max|z| 3.602, and the checker self-test passed.
- The split workflow also passed when the Python reference was computed from
  copied snapshot and SF files.

Note: the driven-turbulence time step is set by the turbulence driver's update
alignment (`dt_turb_update = tcorr/100`). dt falls to about 1.5e-5, far below the
CFL value. That is why 600 cycles reach only t = 0.145.
