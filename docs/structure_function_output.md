# Structure-function output

`file_type = sf` measures field increments and two-point correlations during a
simulation. It writes their separation-dependent moments, sample counts and
replicate statistics at the chosen output cadence. The calculation supports
uniform Cartesian meshes with periodic or nonperiodic boundaries.

## Basic configuration

This example measures longitudinal and transverse velocity increments:

```ini
<output5>
file_type      = sf
id             = velocity
mode           = long_trans
variables      = hydro_w_vx, hydro_w_vy, hydro_w_vz
weights        = ones
max_order      = 3
nbin           = 64
logscale       = true
r_min          = 0.0
r_max          = 0.0
ndraw_block    = 4096
nanchor_global = 1048576
nsample_global = 65536
nreplicate     = 4
seed           = 0
dt             = 0.1
```

Use the corresponding `mhd_w_vx`, `mhd_w_vy`, `mhd_w_vz` variables for MHD. Each
output block needs a unique `id`; files appear in `sf_<id>/`. Without an explicit
`id`, the directory uses the output block name. `dcycle` can replace `dt` to set
an output interval in cycles.

## Quantities and normalization

For a scalar field, the order-p structure function is

$$
S_p(r)=\left\langle |q(\mathbf{x}+\boldsymbol{\ell})-q(\mathbf{x})|^p\right\rangle,
\qquad r=|\boldsymbol{\ell}|.
$$

Pairs contain distinct cell centres. Both cells must lie within the selected
region and have positive weight. With a non-unit weight, the increment is formed
from the weighted field: `delta = w2*q2 - w1*q1`.

The output estimates these sums over unordered valid pairs in each separation bin:

| Array | Definition |
|---|---|
| `T[w,q,p-1,b]` | Sum of `abs(delta)**p`, for `p = 1..max_order`. |
| `W[w,b]` | Sum of `w1 + w2`. |
| `C[w,b]` | Number of pairs. |

The structure function is `T/C`. For unit weights, `W = 2*C`. Non-unit weights
therefore give moments of `w*q`; their physical units include the units of the
weight. Empty bins have `C = 0` and should be masked when forming ratios.

Separations use integer cell offsets multiplied by the grid spacing. Periodic
axes use the minimum-image separation; nonperiodic axes use the actual offset.
This gives the same bin assignment across MeshBlock and MPI decompositions.
Bins satisfy `edges[b] <= r < edges[b+1]`. Logarithmic bins span `r_min` to
`r_max`; linear bins span zero to `r_max`. Nonpositive `r_min` selects the smallest
active grid spacing, and nonpositive `r_max` selects half the box diagonal. The
outer edge is enlarged by a relative `1e-9`; the first logarithmic edge is reduced
by the same fraction so endpoint pairs are included. The saved edge arrays give
the precise boundaries.

## Measurement modes

### Scalar moments

`mode = scalar` measures each listed variable separately. `nq` equals the number
of variables. `max_order` sets the highest integer moment.

### Longitudinal and transverse velocity

`mode = long_trans` takes three vector components in x1, x2, x3 order. With
`du = w2*u2 - w1*u1` and separation unit vector `lhat`, its two channels are:

- `q = 0`: longitudinal magnitude `abs(du . lhat)`;
- `q = 1`: transverse magnitude `sqrt(|du|^2 - (du . lhat)^2)`.

The p-th powers of these increments supply `T`. Each transverse measurement is
the magnitude of the vector perpendicular to the separation.

### Magnetic-field-aligned moments

`mode = aniso_mhd` takes velocity x1, x2, x3 followed by magnetic field x1, x2, x3:

```ini
mode      = aniso_mhd
variables = mhd_w_vx, mhd_w_vy, mhd_w_vz, mhd_bcc1, mhd_bcc2, mhd_bcc3
```

The local magnetic direction `bhat` is the unit vector along the unweighted pair
mean `(B1 + B2)/2`. Pairs with a mean-field magnitude below `1e-30` in code units
are skipped. The four increment channels are `|dv|`, `|dv_perp|`, `|dB|` and
`|dB_perp|`; perpendicular components are projected relative to `bhat`, and
increments use the requested weights.

Each channel has three separation families, concatenated along its bin axis:

| Family | Separation and selection |
|---|---|
| Total | `|l|`, using all valid pairs. |
| Parallel | `abs(l . bhat)`, with `abs(l . bhat)/|l| > cos(theta_prll_max)`. |
| Perpendicular | `sqrt(|l|^2 - (l . bhat)^2)`, with `abs(l . bhat)/|l| <= cos(theta_perp_min)`. |

Defaults are `nbin_prll = 50`, `nbin_perp = 50`,
`nbin = max(nbin_prll, nbin_perp)`, `theta_prll_max = pi/18` and
`theta_perp_min = pi/2 - pi/18`. Angles are in radians. All three families use the
same radial bounds and spacing rule. Their individual edge arrays and bin counts
are stored in the header and payload.

### Two-point correlations

`mode = correlation` takes one variable for an autocorrelation or two variables
for a cross-correlation. It measures raw correlations of the supplied fields.
For each unordered pair, both orientations contribute:

```text
T[w,0,p-1,b] += w1*w2 * ((A1*B2)**p + (B1*A2)**p)
W[w,b]       += 2*w1*w2
C[w,b]       += 1
```

The normalized correlation is `T/W`; `p = 1` gives the two-point function.
Weights enter through `w1*w2`, and field means are retained. The header's
`products` entry identifies A and B. This mode has one output channel.

## Fields and cell selection

Variables and weights must each name one output component, such as `hydro_w_d`
or `mhd_bcc1`. There can be at most eight distinct loaded fields, eight variables
and eight weights. A passive-scalar selection such as `hydro_w_s` is valid only
when it expands to one component. Variable groups such as `hydro_w` are rejected.

For Newtonian, single-fluid ideal-gas hydro or MHD, two computed variables are
also available as variables or selection fields:

- `pressure = (gamma - 1)*eint`;
- `temperature = (gamma - 1)*eint/rho`.

Both use code units; `temperature` represents pressure divided by density.
These computed fields cannot be used as weights.

Spatial bounds `x1_min`, `x1_max`, and the corresponding x2/x3 keys restrict the
region by cell-centre position. Up to two additional field conditions can be
applied. For example, the following selects a temperature interval:

```ini
select1_variable = temperature
select1_min      = 0.1
select1_max      = 1.0
sparse           = true
```

The numeric bounds are in code units and should match the intended physical
phase. A second condition uses `select2_variable`, `select2_min` and
`select2_max`, together with the first condition. Every condition must hold.
Bounds are inclusive; omitted limits are negative or positive infinity. To
express a strict lower bound, use the next representable value above that bound
in the simulation's precision.

Selection is evaluated at every output. Selected cells retain their requested
weight, while other cells receive zero weight. Thus `weights = ones` measures
increments within the selected phase. With a selection, the total number of
variable and weight entries must be at most eight. The header records the
conditions and the global number of selected cells.

## Sampling, uncertainty and cost

The estimator separates pairs into those within one MeshBlock (E1) and those
across MeshBlocks (E2). E1 samples local lags and anchors. E2 pairs local anchors
with a globally shared sample of cells. Inverse sampling probabilities weight
the contributions to `T`, `W` and `C`. These raw sums estimate the full pair
population; their ratios have finite-sample uncertainty.

| Parameter | Default | Meaning |
|---|---:|---|
| `ndraw_block` | 4096 | E1 draws per MeshBlock, shared across replicates. |
| `nanchor_global` | 1048576 | Global E2 anchor budget, shared across replicates. |
| `nsample_global` | 65536 | Global E2 sample budget, shared across replicates. |
| `nreplicate` | 4 | Independent sampling replicates. |
| `sparse` | false | Concentrate sampling on cells with positive weight. |

`ndraw_block` must be at least `nreplicate`. Each global E2 budget must be at
least `nreplicate` times the number of ranks containing region cells. With
`sparse = true`, this minimum uses ranks containing positive-weight cells.
Sample communication must fit one MPI collective:
`nsample_global*(4 + nfield) <= 2^31 - 1`.

Use `sparse = true` when a weight or selection occupies a small fraction of the
region. It changes the sampling proposals while retaining their probability
corrections. An empty weight support writes zero sums; a spatial region
containing no cells is rejected. Rare selections sampled without this option can
have poorly determined moments and unreliable replicate error estimates.

E1 work scales approximately as `local_MeshBlocks*ndraw_block`. E2 work per rank
scales approximately as
`nanchor_global*nsample_global/(nranks*nreplicate)`. Every rank receives the E2
sample. Its records occupy `8*nsample_global*(5 + nfield)` bytes per copy,
including coordinates, sampling factors, fields and MeshBlock identifiers. Allow
additional memory for host/device copies, local anchors and accumulators. Start
with a short timing run, inspect replicate uncertainty and increase budgets
where needed. Higher-order moments generally require more samples.

Draws are determined by the seed, output number, layout and replicate indices.
The diagnostic has its own counter-based sampling and leaves the simulation's
random-number state unchanged. Restarted outputs continue their file numbering.
`timing = true` reports sampling times. `kernel = auto` chooses a shared-memory
pair accumulator where it fits; `global` uses global atomics, and `team` requires
the shared-memory accumulator to fit. Reduction ordering can change last-bit sums.

## Increment histograms

`histogram = true` adds increment distributions for `scalar` and `long_trans`.
Scalar histograms use the absolute increment; longitudinal histograms retain its
sign, and transverse histograms use its magnitude.

The default magnitude-edge template has 96 logarithmic intervals from `1e-4`
to `1e2`. The signed longitudinal template mirrors positive logarithmic edges
about zero and includes a central interval. Configure it with `hist_nbin`
(an even integer of at least two), `hist_min` and `hist_max`.

With `hist_scale = sigma`, edges are multiplied by the weighted standard
deviation of the underlying field, recomputed over valid cells at each output.
For a vector, the scale is `sqrt((sigma_x^2 + sigma_y^2 + sigma_z^2)/3)`. The
variance uses a centered two-pass reduction. A constant field, empty support or
nonfinite scale uses a scale of one. `hist_scale = absolute` uses the template
in the field's units. Exact edges and scales are saved in the file.

Histograms contain importance-weighted counts, including underflow and overflow.
Their sum over increment bins equals `C` for each separation bin. Divide by `C`
and the finite increment-bin width to obtain a probability density; underflow
and overflow remain probability masses. The payload can include padding columns,
as identified by `hist_nedge` and `hist_ncol`.

## Reading the output

Files are named `sf_<id>/<basename>.NNNNN.sf`. An ASCII header ends with
`header offset=N`; binary arrays begin at byte N. The header specifies byte
order, array shapes, configuration, geometry, sampling budgets and timings.
Floating-point payloads are float64; accepted-draw counts are uint64.

| Arrays | Contents |
|---|---|
| `rbins` | Separation edges; field-aligned mode also has `rbins_prll` and `rbins_perp`. |
| `T`, `W`, `C` | Replicate-averaged moment sums, weight sums and population pair counts. |
| `T_er`, `W_er`, `C_er` | Separate E1/E2 sums for every replicate. |
| `n_acc_er` | Raw accepted draws for each estimator and replicate. |
| `H`, `H_er` | Optional increment histograms, averaged and per replicate. |
| `hist_edges`, `hist_sigma` | Optional histogram edges and scales. |

The leading dimensions of `T_er` are `[estimator, replicate, weight, channel,
order, separation]`; estimator 0 is E1 and estimator 1 is E2. `C_er` omits channel
and order. Population counts include probability corrections and may be
nonintegral; accepted-draw counts are integers.

The bundled [reader](../vis/python/sf_spectra_read.py) loads the header and arrays.
From the repository root:

```python
import sys
import numpy as np
sys.path.insert(0, "vis/python")
from sf_spectra_read import read_sf

meta, a = read_sf("sf_velocity/run.00010.sf")
counts = a["C"][:, None, None, :]
S = np.divide(a["T"], counts, out=np.full_like(a["T"], np.nan), where=counts > 0)
# S[weight, channel, p-1, separation_bin]
```

Estimate uncertainty by a delete-one-replicate jackknife of the ratio of sums,
keeping each replicate's numerator and denominator together. First add the E1
and E2 contributions within each replicate. For `R >= 2`, omit replicate r to
form `S_(r)`; the variance estimate is
`(R-1)/R * sum_r((S_(r) - mean_r(S_(r)))**2)`. Mask bins with zero denominators.
For correlations, use `W` and `W_er` as the denominators.

## Supported configurations

Uniform meshes are supported in one, two and three dimensions. SMR/AMR and
shearing-periodic boundaries are rejected. Sampling uses active cells;
`sf_halo` has an effective value of zero. Generic output options `gid`,
`ghost_zones` and `slice_x1/2/3` are rejected. Histograms are available for scalar
and longitudinal/transverse moments. Input fields should be finite: nonfinite
values propagate into the measured sums. Fused floating-point operations can
leave roundoff-sized increments between equal weighted values.
