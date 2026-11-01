# `mode = correlation` against the production Python `get_cf`

The production correlation is `athena_research.operations.correlation_functions.get_cf`
(`run_radshock_sf_checkpoint.py --method correlation`). This directory keeps the
evidence that the exact reference (`tests/sf/sf_reference.py`, correlation branch of
`run_exhaustive`) is Python's definition, and that the AthenaK output agrees with it.

## Python source

- Reference package: a separately available `athena_research` checkout.
  Set `ATHENA_RESEARCH` to its absolute path.
- Commit `c4cc9d7e5b6fa5662e002c77d7cd2cba133806c1` (2026-10-25). The files used
  had no local modifications.
- SHA-256 of the files used: `python_source.sha256`.

## Files

The input and checker are included here. The `run1/`, `scan/` and result-text
paths below describe generated outputs from the validation workflow.

- `corr.athinput`: `inputs/tests/sf_select_test.athinput` turned into correlation (the
  same transformation as the `correlation` case of `tests/sf/run_select.sh`):
  - output1: `hydro_w_vx` x `hydro_w_d`, weights `ones, hydro_w_d`, `sparse = true`,
    temperature selection;
  - output2: temperature autocorrelation, with temperature and density selections.
- `run1/`: its AthenaK output (dev CUDA build of the correlation source, GPU np 1) and
  log.
- `py_getcf_check.py SFFILE REFSCRIPT`: the wrapper.
  - It rebuilds the `sf_unit_test` fields and selection weights of an output.
  - It feeds them to the real `get_cf` through a minimal AthenaData stand-in: block
    arrays, `mb_geometry`, a periodic `_header`.
  - It uses exhaustive enumeration: `nsamples_block_min` = cells per block and
    `npairs` = (cells)^2, so every block pair is enumerated.
  - It compares Python's `cf`, `weight_sum_bin`, `num_bin_points/2` and edges with
    the exact reference.
  - It compares AthenaK `T/W` with Python `cf/weight_sum_bin`: the ratio of summed
    numerator and denominator, with a delete-one-replicate jackknife and absolute
    errors.
  - The Python repository is taken from `$ATHENA_RESEARCH`.
- `getcf_results.txt`: the output for the four files of `run1/`.
  - The first run used a per-replicate-ratio spread instead of the jackknife. Its
    exact-sum agreement was the same: cf <= 1.4e-15, weight_sum_bin <= 1.1e-15,
    counts exact.

- `scan/seed1NN/`: 12 independent runs of `corr.athinput`, with
  `output1/seed=1NN` and `output2/seed=1NN+500`, NN = 01..12 (the same dev build),
  plus `ref12.txt`, the `sf_reference.py` result per run, including the jackknife
  `T/W`.
- `scan_results.txt`: `seed_scan.py`, run per output id (the two outputs have
  different weight counts; `seed_scan.py` needs one output per scan directory), and
  the summary of the per-file jackknife.

## Reproduce

Use a `PROBLEM=sf_unit_test` executable and create a fresh result directory.
From the repository root:

```bash
set -e
export ATHENA_RESEARCH=/absolute/path/to/athena-research
ATHENA=/absolute/path/to/sf-unit-build/src/athena
RUN=/absolute/path/to/new-correlation-run
mkdir "$RUN"
"$ATHENA" -i "$PWD/tests/sf/python_crosscheck/correlation/corr.athinput" -d "$RUN"
python3 tests/sf/python_crosscheck/correlation/py_getcf_check.py \
  "$RUN/sf_scalar/sfsel.00000.sf" tests/sf/sf_reference.py
```

Needs numpy and scipy; it takes about 2-3 minutes per file on a CPU.

## Interpretation

- The exact-sum agreement at round-off shows the reference implements `get_cf`:
  - both orientations;
  - pair weight `w1*w2` entering once;
  - no mean subtraction;
  - the same bin edges.
- The per-file `T/W` z-scores are informational, because the entries of one file share
  samples. The bias test for the estimator is the seed scan in `scan/` (`seed_scan.py`
  per output id).
