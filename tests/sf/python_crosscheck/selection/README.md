# Structure functions within a density-selected population

This benchmark checks velocity statistics within a subset of a turbulent flow.
It tests whether sparse sampling preserves the selected pair population and
whether selection boundaries are treated consistently at finite precision.

## Physical setup

`selection.athinput` evolves driven ideal-gas turbulence on a periodic `64³`
mesh with `16³` MeshBlocks for 600 cycles. The forcing parameters include
`dedt = 1` and `random_seed = 1`. The sampled state has an rms velocity of about
0.5; its density 10th–90th percentile range is approximately 0.958–1.044.

A cell belongs to the sample when

$$
0.96875\le\rho\le0.9990234375.
$$

The output uses `weights = ones`, this density selection and `sparse = true`.
It measures scalar moments of all three velocity increments and their
longitudinal/transverse moments through order three. The 24 logarithmic
separation bins span one cell width to half the box diagonal. Each output has
32 replicates, 262144 same-block draws, 262144 anchors and 65536 partner samples.

The measured selected population was 81796 of 262144 cells, approximately 31.2%.
The initial uniform state selects no cells and supplies an empty-population
check. Repeated outputs of the evolved state use independent sample draws.

## Expected quantities

The reference enumerates every unordered pair whose two endpoints are selected.
For `N_sel` selected cells, the total pair count must be
`N_sel*(N_sel-1)/2`. In each separation bin:

- `C` is the selected pair count;
- `T_p` is the sum of the absolute increment to order `p`;
- `W = 2C` for unit weights;
- `S_p = T_p/C` is the selected-population structure function.

The same definitions apply to scalar, longitudinal and transverse increments.
Exactly empty reference bins must have zero estimated sums. Every populated
reference bin must have a finite estimate.

## Acceptance

The test first checks variable/selection metadata, identical physical time,
selected-cell counts, bin edges, complete statistic inventory and finite arrays.
The selected fraction must lie between 0.10 and 0.50. Every replicate must satisfy
`W = 2C` within relative `1e-12`. A zero-spread statistic must agree with the exact
reference within relative `1e-12`.

For each populated bin, the tested statistics are `S_p`, `T_p` and `C`. With
`R = 32` replicates, the standard error for the ratio uses

$$
\operatorname{SE}(S_p)=
\frac{\operatorname{sd}_r(T_{p,r}-S_p C_r)}{\bar C\sqrt R}.
$$

Raw sums and counts use their replicate standard errors. All 736 statistics
must satisfy a two-sided Bonferroni-corrected Student test at family-wise
significance `1e-3`. The corresponding maximum absolute standardized residual
is 5.964 for 31 degrees of freedom.

One- and two-rank runs both passed all 736 statistics. Their largest absolute
standardized residual was 3.60, with rms residuals of 1.20 and 1.21. Fault tests
that emptied a reference-populated bin, inserted a nonfinite replicate or doubled
a pair count were all rejected.

## Precision at the selection boundary

Selection uses the in-memory density. Ordinary mesh snapshots store float32
values, which can round a density just outside the interval onto a boundary.
The bounds above are exactly representable in float32, so such discrepancies
can be isolated to cells whose saved density equals an edge.

In the measured case, one additional snapshot cell lay exactly on an edge.
Removing that cell's 81796 pair contributions gave the same selected population
as the in-memory calculation. The largest uncorrected change was `8e-5` relative
per bin. For an ambiguous set of boundary cells, the comparison instead requires
its bounded count error to stay below 0.2 standard errors per bin. Velocity
rounding changed the exact moments by approximately `1e-8` relative.

## Standalone selection tests

The deterministic selection matrix exercises the same selection keys and
population accounting on smaller grids. It needs NumPy, SciPy and an MPI-enabled
`PROBLEM=sf_unit_test` executable:

```bash
ATHENA=/absolute/path/to/sf-unit-build/src/athena
tests/sf/run_select.sh "$ATHENA" mpirun 2 /absolute/path/to/new-selection-results
```

For configuration and normalization details, see the
[structure-function guide](../../../../docs/structure_function_output.md).
