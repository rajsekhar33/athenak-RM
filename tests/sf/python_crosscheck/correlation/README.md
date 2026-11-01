# Correlation benchmark

This test measures the spatial correlation of two scalar fields and a temperature
autocorrelation on a deterministic periodic mesh. It checks the ordered-pair
normalization, selection masks and product of endpoint weights.

## Definition

For fields `a` and `b` and a nonnegative endpoint weight `w`, the normalized
correlation in separation bin `r` is

$$
C_{ab}(r)=
\frac{\sum_{i\ne j,\,|\mathbf x_j-\mathbf x_i|\in r}
 w_iw_j a_i b_j}
 {\sum_{i\ne j,\,|\mathbf x_j-\mathbf x_i|\in r}w_iw_j}.
$$

Both pair orientations contribute. The means of `a` and `b` remain in the
correlation. The output's raw numerator and denominator are `T` and `W`, so the
reported correlation is `T/W`.

`corr.athinput` contains two cases:

- velocity–density correlation with unit and density weights, restricted by a
  temperature selection;
- temperature autocorrelation with both temperature and density selections.

An exhaustive calculation on the same grid supplies the expected numerator,
denominator, pair inventory and bin edges. The exact correlation-sum check
agreed within `1.4e-15` relative; the weight sums agreed within `1.1e-15`, and
pair counts agreed exactly.

## Sampling and acceptance

For the on-the-fly estimate, compare the ratio of the summed numerator and
summed denominator. A delete-one-replicate jackknife accounts for their
covariance. Independent seeds test the estimator across repeated realizations;
individual bins from one realization share samples and are correlated.

Acceptance requires the intended variables, weights and selected populations,
finite statistics, the complete reference-populated bin inventory and
agreement within the specified sampling uncertainty.

The self-contained selection suite includes this correlation configuration.
With NumPy, SciPy and an MPI-enabled `PROBLEM=sf_unit_test` executable, run from
the repository root:

```bash
ATHENA=/absolute/path/to/sf-unit-build/src/athena
tests/sf/run_select.sh "$ATHENA" mpirun 2 /absolute/path/to/new-correlation-results
```

See the [output guide](../../../../docs/structure_function_output.md) for the
correlation input keys and binary layout.
