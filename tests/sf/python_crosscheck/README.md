# Exact-pair benchmarks on periodic fields

These benchmarks test structure-function moments and their sampling uncertainties
on a deterministic periodic velocity and density field. The reference is the
complete discrete population of distinct cell pairs.

## Physical quantities

For unit weights, the order-`p` scalar structure function in separation bin `b` is

$$
S_p(b)=\frac{1}{C_b}\sum_{\{i,j\}\in b}|q_j-q_i|^p,
$$

where `C_b` is the number of unordered pairs. The longitudinal velocity increment
is `delta_v · r_hat`; the transverse increment is the magnitude of the component
perpendicular to `r_hat`. The benchmark checks their absolute moments separately.
Density-weighted raw sums are also tested using the definition in the
[structure-function guide](../../../docs/structure_function_output.md).

The pair population is partitioned into pairs within one MeshBlock and pairs in
different MeshBlocks. Comparing both partitions independently tests their
normalization before combining them.

## Setup and expectations

`crosscheck.athinput` uses the `sf_unit_test` problem generator on a `64³` mesh
with `16³` MeshBlocks and zero evolution cycles. It writes velocity scalar moments
through order four with unit and density weights, plus longitudinal/transverse
moments with unit weights. The 262144 cells contain exactly 34359607296 unordered
distinct pairs.

Periodic displacements use the minimum image. A displacement component exactly
half a box length retains the sign of the endpoint difference; reversing the
pair reverses both the displacement and velocity increment. The longitudinal
projection is consequently independent of endpoint order.

`exact_lag.py` enumerates the grid's nonzero displacement lags and accumulates
all pairs, separately for both MeshBlock populations. It reconstructs the
specified field in double precision; `--f32` repeats the calculation after
single-precision rounding. This quantifies the effect of comparing an in-memory
statistic with an ordinary float32 mesh output.

The sampled output should reproduce the exact moments and pair counts within
its replicate uncertainties. Unit weights require `W = 2C`. Empty separation
bins must remain empty, and every populated reference bin must have a finite
estimate. Independent seeds and output numbers provide additional realizations
for pooled bias tests.

## Standalone checks

The smaller exact-reference matrix can be run with an MPI-enabled
`PROBLEM=sf_unit_test` executable, NumPy and SciPy:

```bash
ATHENA=/absolute/path/to/sf-unit-build/src/athena
tests/sf/run_suite.sh "$ATHENA" mpirun 2 /absolute/path/to/new-sf-results
```

For a generated SF file from `crosscheck.athinput`, the exact lag calculation
requires CuPy and a compatible GPU:

```bash
python3 tests/sf/python_crosscheck/exact_lag.py \
  /absolute/path/to/output.sf /absolute/path/to/new-exact-pairs.npz
```

The resulting archive contains bin edges, pair counts and raw moments for each
pair population. Use identical fields, bin edges, periodic geometry and weights
when comparing sampled and exhaustive quantities. For a ratio of sampled sums,
retain numerator–denominator covariance when estimating its uncertainty.
