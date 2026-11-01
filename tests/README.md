# Structure-function and spectrum validation

The tests check spatial statistics against discrete mathematical references:
exact pair sums for structure functions, and Fourier coefficients and sum rules
for spectra. Restart tests check the saved state, output numbering and sampling
sequence.

## Compact tests

The compact suite uses the normal AthenaK build with built-in problem generators,
NumPy and pytest. From the repository root:

```bash
cd tst
ATHENAK_FEATURE_TEST_BINARY=/absolute/path/to/athena python3 -m pytest \
  test_suite/onthefly_sf_spectra/test_onthefly_sf_spectra_cpu.py
```

The neighboring GPU and MPI modules use the standard test suite's launch
configuration. Their checks cover analytic moments, single Fourier modes,
Parseval normalization, restart output, invalid-input rejection and inclusion
of modes at the final spectral edge.

## Extended tests

Build AthenaK with MPI and `-DPROBLEM=sf_unit_test`. The following harnesses use
files included in this repository and need NumPy and SciPy:

| Harness | Physical or numerical check |
|---|---|
| `fft/run_pencil_suite.sh` | Every distributed Fourier coefficient; non-cubic, 1D and 2D meshes; host/device transport and excess ranks |
| `sf/run_suite.sh` | Scalar and vector increments, moments and histograms against exact discrete pair sums |
| `sf/run_sparse.sh` | Sparse pair populations and their sampling normalization |
| `sf/run_select.sh` | Restricted cell populations, correlations, and temperature/density selections |
| `sf/run_guards.sh` | Input validation, lattice geometry and bin limits |
| `integration_checks.sh` | Fluid-state consistency and diagnostic continuation across a restart with changed MPI rank count |

For example, from the repository root, with an MPI launcher configured for the
chosen executable:

```bash
ATHENA=/absolute/path/to/sf-unit-build/src/athena
tests/sf/run_suite.sh "$ATHENA" mpirun 2 /absolute/path/to/new-sf-results
tests/fft/run_pencil_suite.sh "$ATHENA" mpirun /absolute/path/to/new-fft-results 2
tests/integration_checks.sh "$ATHENA" mpirun /absolute/path/to/new-restart-results 1 2
```

Use a fresh output directory for each invocation. GPU launches require the
appropriate device assignment and GPU-aware MPI or host-staged transport.
The [Frontier guide](frontier/README.md) gives HIP launch, memory and scaling
examples.

## Statistical benchmarks

The extended benchmark notes describe the tested populations and acceptance rules:

- [Deterministic periodic fields](sf/python_crosscheck/README.md): exact pair
  inventories, scalar moments and longitudinal/transverse increments.
- [Correlations](sf/python_crosscheck/correlation/README.md): ordered-pair
  correlations and pair-product weighting.
- [Partial selections](sf/python_crosscheck/selection/README.md): density-band
  statistics and selection-boundary precision.
- [Developed turbulence](sf/python_crosscheck/developed/README.md): multivariate
  structure functions and spectra from one common evolved state.

Sampled estimates are assessed using replicate or independent-seed uncertainty.
The tests require complete bin and statistic inventories before comparing errors;
an undersampled comparison is reported as inconclusive.
