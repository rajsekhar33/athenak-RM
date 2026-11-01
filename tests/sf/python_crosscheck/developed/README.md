# Statistics of developed turbulence

This benchmark measures structure functions and spectra from a common evolved
turbulent state. It tests the diagnostics on intermittent, multiscale fields
beyond the deterministic small-grid cases.

## State and measured quantities

The reference configuration is driven isothermal hydrodynamic turbulence on a
periodic `256³` mesh with `64³` MeshBlocks. Its input is
[precision/turb.athinput](../../precision/turb.athinput). The benchmark state was
sampled at `t = 3`, followed by five additional evolution cycles with diagnostic
outputs enabled.

All compared quantities are taken from the same final double-precision state:

- scalar increments of `vx`, `vy`, `vz` and density, with unit and density weights;
- longitudinal and transverse velocity increments with unit weights;
- moments through order three in 100 logarithmic separation bins;
- scalar density and velocity power, including field-weighted power;
- compressive and solenoidal velocity spectra.

The SF configuration uses eight replicates, 262144 same-block pair draws,
262144 global anchors and 262144 partner samples. The raw output and weighting
conventions are defined in the [SF guide](../../../../docs/structure_function_output.md)
and [spectrum guide](../../../../docs/spectrum_output.md).

## Expected behavior and acceptance

The spectrum comparison transforms the same discrete field and checks every
array: bin edges, representative wavenumbers, mode counts, scalar or vector power,
zero-mode power and squared-field sum. The comparison tolerance is `1e-10`
relative to each array's largest magnitude. Parseval's relation and inclusion of
all represented modes provide independent normalization checks.

The SF comparison uses `Q = T/W`, including both weights. For unit weights,
`Q = S_p/2`. The statistical comparison combines the output's delete-one-replicate
jackknife uncertainty with the standard error from 16 independent reference
realizations. A two-sided Welch test accounts for the estimated variances.
The family-wise significance level is `1e-3`, distributed over the complete
inventory of compared bins and statistics with a Bonferroni correction.

Every populated exact-pair bin must be represented. All 16 realizations, four
scalar variables, both scalar weights, both vector components and orders 1–3
must be present and finite. Unit weights must satisfy `W = 2C`. The median
relative combined uncertainty must be at most 1% for each mode and order;
otherwise the comparison is inconclusive.

A developed turbulent state has no prescribed analytic structure-function slope
at every separation. This benchmark therefore tests discrete diagnostic accuracy
and sampling uncertainty. Analytic Fourier modes and exact small-grid pair sums
are covered by the [standalone test matrix](../../../README.md).

## Reusing a state

`make_run.py` prepares diagnostic output from a supplied checkpoint. A comparison
must preserve the checkpoint, mesh geometry, precision, variable and weight
inventory, bin edges, sampling budgets and seed list. Keep hashes of the input
state and generated reference arrays so results remain tied to the same data.
Thermal-phase selections are covered by the
[partial-selection benchmark](../selection/README.md).
