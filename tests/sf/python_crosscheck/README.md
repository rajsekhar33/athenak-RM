# Cross-check of the `sf` output against `athena_research` structure functions

End-to-end check that AthenaK's on-the-fly structure functions (`file_type = sf`,
`docs/structure_function_output.md`) agree with the post-processing tool
`athena_research/operations/structure_functions.py` on the same simulation state.

## Setup

- `crosscheck.athinput`: `PROBLEM=sf_unit_test` fields (deterministic, `noise = 1`),
  64^3 periodic mesh, 16^3 MeshBlocks (64 blocks), `nlim = 0`. Outputs:
  - `bin` of `hydro_w` (the snapshot Python reads; float32, see below);
  - `sf_scalar`: vx, vy, vz; weights `ones`, `hydro_w_d`; `max_order 4`, `nbin 24`, log
    bins, default r_min/r_max; `ndraw_block 2e6`, `nsample_global 65536`,
    `nanchor_global 1e6`, `nreplicate 8`;
  - `sf_longtrans`: same variables and budgets, weight `ones`.
  Each run writes outputs 00000 and 00001 of the same state (initial and final output
  at t = 0); their draws are independent (the output number enters every hash).
- Python (`python_sf.py`, read-only use of the Python repo via `sys.path`):
  - `exact`: every distinct cell pair enumerated (scalar: `get_sf_mb(...,
    use_exhaustive=True)`; long_trans: `get_sf_helmholtz_mb` switches to its
    exhaustive kernel by itself when `n_samples1` = cells per block). The raw pair count
    equals N(N-1)/2 = 34359607296, so these are the exact all-pairs sums.
  - `sampled`: the normal random-pair estimator with `n_samples1 = 1024` per block
    (`npairs = (1024*64)^2`), 16 seeds. Seed `s != 0` XORs a constant into
    `structure_functions._pair_seed` at run time (monkeypatch; seed 0 is the stock
    behaviour).
- `exact_lag.py` (CuPy): an independent exact reference by lag enumeration on the
  periodic grid, split into the populations E1 (same MeshBlock) and E2, on the double
  fields of `sf_reference.build_fields` (or float32-rounded with `--f32`).

## Rerun

This comparison requires access to a separate `athena_research` checkout and
a Python environment with CuPy, NumPy and SciPy. Set `ATHENA_RESEARCH` to its
absolute path. Use an absolute path to a `PROBLEM=sf_unit_test` executable and
an MPI launcher configured for the current machine.

```bash
set -e
export ATHENA_RESEARCH=/absolute/path/to/athena-research
ATHENA=/absolute/path/to/sf-unit-build/src/athena
MPIRUN=mpirun
cd tests/sf/python_crosscheck
test ! -e run
./run_athena.sh "$ATHENA" "$MPIRUN" 1 32  # run/main + run/seeds/s1..s31
mkdir -p run/python
for mode in scalar longtrans; do
  python3 python_sf.py run/main/bin/xchk.hydro_w.00000.bin --mode "$mode" \
    --kind exact --pyrepo "$ATHENA_RESEARCH" --out "run/python/${mode}_exact.npz"
  for seed in $(seq 0 15); do
    python3 python_sf.py run/main/bin/xchk.hydro_w.00000.bin --mode "$mode" \
      --kind sampled --seed "$seed" --ns1 1024 --pyrepo "$ATHENA_RESEARCH" \
      --out "run/python/${mode}_sampled_s${seed}.npz"
  done
done
python3 exact_lag.py run/main/sf_scalar/xchk.00000.sf run/exact_lag_f64.npz         # ~1 min
python3 exact_lag.py run/main/sf_scalar/xchk.00000.sf run/exact_lag_f32.npz --f32
python3 compare.py            # summary; add --full for every per-bin row
```

`compare.py` reports, per mode, weight, variable and order, the z-scores and relative
differences of:
0. Python exact against `exact_lag.py` (float32 fields: same input values);
1. the primary AthenaK file (`run/main/.../xchk.00000.sf`, error = replicate standard
   error) against the Python exact sums;
2. the pooled replicates of all independent AthenaK outputs (2 outputs x 8 replicates
   per run) against the Python exact sums, S_p as a ratio of means with a delta-method
   error: a tighter bias test; 2a/2b the same for E1 and E2 separately against
   `exact_lag.py` (float64 fields);
3. the Python sampled mean against the Python exact sums (error = seed scatter);
4. the primary AthenaK file against the Python sampled mean (combined error).

Compared quantities: `S_p = T/C` vs `sf/(weight_sum_bin/2)` (unit weight), `T` vs `sf`,
`W` vs `weight_sum_bin`, `C` vs the exact pair count (both weights). With the density
weight only the raw sums are compared (the normalisations differ by design).

## Caveats

- Half-box separations: AthenaK (`SFMinImage`) and Python both wrap a displacement only
  when |d| > L/2, so a component of exactly L/2 keeps the sign of x2 - x1. Swapping a
  pair flips both the increment and the displacement, so the longitudinal projection is
  unchanged and endpoint order does not matter. `exact_lag.py` originally used the fixed
  enumeration lag for every pair instead. That made its long_trans sums differ by up to
  6e-5 in the half-box bins, an error in the reference that a review found. It now takes
  the displacement from the wrapped endpoints and classifies E1 by physical block; the
  half-box bins agree with Python's exhaustive sums to 1.4e-13 (`compare_summary_v3.txt`).
- The `.bin` snapshot is float32: each field value equals the float32 rounding of the
  double value AthenaK's SF uses (checked against `sf_reference.build_fields`, max
  difference 6e-8). The effect on the exact sums is at most 2.4e-9 relative
  (`exact_lag.py` float64 vs float32), far below the sampling errors.
