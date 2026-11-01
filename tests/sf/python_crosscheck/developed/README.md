# Developed-turbulence comparison with `athena_research`

Integration check of the production `sf` and `spec` outputs against the Python tools
(`athena_research/operations/structure_functions.py` and `spectra.py`) on one developed
state, before the Python post-processing is retired.

## State and outputs

- Validation state: 256^3 driven isothermal hydro turbulence with 64^3
  MeshBlocks and periodic boundaries, at t = 3 (cycle 16526). The checkpoint
  was generated from `tests/sf/precision/turb.athinput`.
- `make_run.py SOURCE.rst RUNDIR 5` copies that restart with new outputs and prepares 5 more
  cycles. At the final cycle every output is written from the same state:
  - `rst`: the double-precision state used by both references;
  - `bin` of `hydro_w`: float32, used only for the metadata `athena_research` reads
    (`python_sf.py --rst` replaces its fields by the restart's doubles, after checking
    that the float32 values are the rounded restart values);
  - `sf_scalar` (vx, vy, vz, rho; weights ones and rho) and `sf_longtrans`, the
    production settings of `tests/frontier/make_input.sh`: max_order 3, 100 log bins,
    histograms, sparse, ndraw_block 262144, 262144 anchors x 262144 samples, 8
    replicates;
  - `spec_scal` (rho, vx; weights ones and rho; 256 bins) and `spec_helm` (velocity,
    200 bins).

## Acceptance (fixed before the comparison was run)

**Spectra** (`tests/spec/spec_reference.py RUNDIR --rst RUNDIR/rst/dev.NNNNN.rst`): both
sides transform the same double-precision fields, so every array (bin edges, k, shell
counts, spectra, DC power, sum of squares) must agree to 1e-10 relative to the array's
largest value, and Parseval must hold. The final log edge follows the documented
AthenaK convention (inclusive corner |k|).

**Structure functions** (`compare_developed.py`, rules in its header):
- compared quantity `Q = T / W` (both sides estimate the same population ratio for any
  weight; unit weight `Q = S_p / 2`), all modes, weights, increments, orders and bins;
- pair counts of AthenaK and Python each against the exact count of the mesh;
- errors: AthenaK delete-one-replicate jackknife, Python standard error over 16
  independent seeds; combined variance, Welch degrees of freedom, two-sided t-test;
- family-wise: no p-value below 1e-3 / (number of compared values) (Bonferroni);
- inventory: exactly the bins holding an exact pair are populated on both sides;
  nothing nonfinite; unit-weight `W = 2C`; all 16 seeds, orders 1–3, four scalar
  variables with two weights, and longitudinal/transverse components with unit weight;
- precision: median relative combined error at most 1% per mode and order, otherwise
  INCONCLUSIVE.

Selection masks and histograms are not part of this comparison: this isothermal state
has no thermal phases, and both were checked against exact references in
`tests/sf/python_crosscheck/selection/` and `tests/sf/run_suite.sh`.

## Rerun

Provide a developed-state checkpoint with the mesh and equation of state above;
the historical checkpoint is a generated artifact. Set `ATHENA_RESEARCH` to a
separately available reference-code checkout and choose fresh `RUN` and `PYOUT`
directories. The commands below use two GPUs through the configured MPI launcher.
Generation records hashes of the input state, producer scripts and reference
Python sources before and after computing the 32 sampled references. A completed cache
can be reused while these hashes and every output hash match. Earlier caches without a
manifest require regeneration in a new directory.

```bash
set -e
D=tests/sf/python_crosscheck/developed
export ATHENA_RESEARCH=/absolute/path/to/athena-research
ATHENA=/absolute/path/to/sf-unit-build/src/athena
SOURCE_RST=/absolute/path/to/developed-state.rst
RUN=/absolute/path/to/new-developed-run
PYOUT=/absolute/path/to/new-python-reference
test ! -e "$RUN" && test ! -e "$PYOUT"
python3 "$D/make_run.py" "$SOURCE_RST" "$RUN" 5
(cd "$RUN" && mpirun -np 2 "$ATHENA" --kokkos-map-device-id-by=mpi_rank \
  -r start.rst > log 2>&1)
"$D/run_python.sh" "$RUN" "$PYOUT"              # 16 seeds x 2 modes
python3 tests/spec/spec_reference.py "$RUN" --rst "$RUN"/rst/dev.*.rst
python3 "$D/compare_developed.py" "$RUN" "$PYOUT"
```
