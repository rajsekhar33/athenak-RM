# Structure-function and spectrum tests

The compact tests use the normal AthenaK build with built-in problem generators,
NumPy and pytest. They check analytic moments and spectra, restart output,
unsupported-input rejection and Fourier-edge mode counts. From the repository root,
run the CPU module with an absolute executable path:

```bash
cd tst
ATHENAK_FEATURE_TEST_BINARY=/absolute/path/to/athena python3 -m pytest \
  test_suite/onthefly_sf_spectra/test_onthefly_sf_spectra_cpu.py
```

The neighboring GPU and MPI modules use the same checks and the standard suite's
launcher settings. These compact tests use only files in this repository.

The extended harnesses use an MPI build with `-DPROBLEM=sf_unit_test`:

| Harness | Reference and requirements |
|---|---|
| `fft/run_pencil_suite.sh` | Fourier coefficients against NumPy FFTs. |
| `sf/run_suite.sh`, `sf/run_sparse.sh`, `sf/run_select.sh` | Exact discrete pair sums and sampling checks; NumPy and SciPy. |
| `sf/run_guards.sh` | Input validation and lattice geometry; NumPy and SciPy. |
| `integration_checks.sh` | State and diagnostic restart comparisons; NumPy and SciPy. |
| `spec/spec_reference.py`, `focused_checks.sh`, `spec/edge_regression.py` | Spectrum comparisons using a separate `athena_research` checkout. |

The optional external comparisons require access to that reference-code checkout.
Set `ATHENA_RESEARCH` or pass the documented `--pyrepo`/`--athena-research` argument.
Some comparisons also require CuPy. Their README files describe the reference
versions, input generation and acceptance criteria. Generated snapshots, checkpoints
and comparison logs are produced by those workflows.

The [Frontier kit](frontier/README.md) provides HIP correctness and scaling jobs.
Its complete correctness lanes include the external spectrum comparison and require
the matching reference checkout. The standalone analytic tests above and the
compact suite provide validation from the AthenaK checkout alone.
