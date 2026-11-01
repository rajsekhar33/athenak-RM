# Structure-function output

`file_type=sf` measures spatial increments or two-point correlations at the
output cadence. It samples cell pairs on a uniform grid and writes the sums,
pair counts, and independent replicates needed to estimate uncertainty.

## Physical quantities

For each weight `w`, variable `q`, order `p=1..max_order`, and separation
bin `b`, scalar mode estimates sums over unordered pairs of distinct valid cells:

| Array | Population sum |
|---|---|
| `T[w,q,p-1,b]` | `sum(abs(w2*q2-w1*q1)**p)` |
| `W[w,b]` | `sum(w1+w2)` |
| `C[w,b]` | Number of pairs |

A valid cell lies inside the spatial region, passes the optional selection,
and has positive weight. The structure function is `S_p=T/C`. With
`weights=ones`, this is the usual mean absolute increment to order `p`,
and `W=2*C`. Other weights measure increments of the weighted field `w*q`;
the sums also permit alternative normalization such as `2*T/W`.

Separations use global integer cell offsets times the cell widths, with the
minimum image on periodic axes. An offset exactly half a periodic axis keeps
its sign. Both pair estimators use the same geometry, independent of MeshBlock
decomposition. Bins are half-open: `rbins[b] <= r < rbins[b+1]`.

| `mode` | Variables and increments |
|---|---|
| `scalar` | One increment per variable: `abs(w2*q2-w1*q1)`. |
| `long_trans` | Three vector components in x1/x2/x3 order. Reports `abs(du dot l)/abs(l)` and `sqrt(abs(du)**2-(du dot l/abs(l))**2)`, with `du=w2*u2-w1*u1`. |
| `aniso_mhd` | Six components: velocity x1/x2/x3, then magnetic field x1/x2/x3. Reports `abs(dv), abs(dv_perp), abs(dB), abs(dB_perp)`; perpendicular is relative to the unweighted pair-mean field. |
| `correlation` | One variable for autocorrelation or two for cross-correlation; product moments defined below. |

In `aniso_mhd`, `bhat` is the direction of `(B1+B2)/2`; pairs with
mean-field magnitude below `1e-30` are omitted. The four weighted-field
increments are accumulated in three concatenated bin families:

- `l`: full separation `abs(l)`;
- `l_prll`: `abs(l dot bhat)`, where
  `abs(l dot bhat)/abs(l) > cos(theta_prll_max)`;
- `l_perp`: `sqrt(abs(l)**2-(l dot bhat)**2)`, where
  `abs(l dot bhat)/abs(l) <= cos(theta_perp_min)`.

All families share `r_min/r_max`. Defaults are `nbin_prll=50`,
`nbin_perp=50`, and `nbin=max(nbin_prll,nbin_perp)` for full separation.
The angle defaults are `pi/18` and `pi/2-pi/18`, respectively, in radians,
with allowed range `[0,pi/2]`.

For correlation, set `B=A` when only one variable is supplied. Each
unordered pair contributes:

```text
T[w,0,p-1,b] += w1*w2*((A1*B2)**p + (B1*A2)**p)
W[w,b]       += 2*w1*w2
C[w,b]       += 1
```

The observable is `T/W`; its first-order row is the symmetric raw correlation.
Means are retained in the fields.

## Configuration

Add an output block with one parameter per line:

```ini
<output5>
file_type      = sf
id             = velocity
mode           = scalar
variables      = hydro_w_vx, hydro_w_vy, hydro_w_vz
weights        = ones
max_order      = 3
nbin           = 100
logscale       = true
ndraw_block    = 4096
nsample_global = 65536
nanchor_global = 1048576
nreplicate     = 4
dt             = 0.1
```

| Parameter | Default and meaning |
|---|---|
| `mode`, `weights` | `scalar`, `ones`. |
| `max_order` | `10`; positive integer. |
| `nbin` | `100` for scalar, long/transverse, and correlation; positive integer. |
| `logscale` | `true`; logarithmic or linear separation edges. |
| `r_min`, `r_max` | `0` selects the smallest cell width and half the box diagonal, respectively. |
| `ndraw_block` | `4096` within-block draws per MeshBlock, total over replicates. |
| `nsample_global`, `nanchor_global` | `65536`, `1048576`; global cross-block budgets, total over replicates. |
| `nreplicate`, `seed` | `4`, `0`; independent replicates and seed. |
| `sparse` | `false`; sample positive-weight support when enabled. |
| `kernel` | `auto`; use team scratch accumulation when it fits, otherwise global atomics. Explicit `team` and `global` choices are available. |
| `timing` | `false`; print pair-kernel timing ranges over ranks. |
| `x1_min` ... `x3_max` | Mesh bounds; restrict cell centres to a region. |
| `id` | Output block name; each SF block needs a distinct ID. |

Each variable or weight must select exactly one output component, such as
`hydro_w_vx` or `mhd_bcc1`. At most eight variables, eight weights, and
eight loaded fields are supported; repeated names count toward the variable
and weight limits. Groups such as `hydro_w` are rejected. A passive-scalar
name such as `hydro_w_s` is valid only when it resolves to one component.

Variables and selection fields may also be `pressure=(gamma-1)*e` and
`temperature=pressure/rho`, in code units, where `e` is internal-energy
density. These require Newtonian, single-fluid, ideal-gas hydro or MHD.
Computed fields cannot be weights; their storage scales with local cell count.

Uniform grids and ordinary periodic or physical boundaries are supported.
SMR/AMR, shearing boundaries, generic `slice_x1/2/3`, `gid`, and
`ghost_zones` are rejected. `sf_halo` defaults to zero; its effective value
remains zero if a positive value is requested, with a warning.

### Cell selection

Up to two inclusive conditions restrict the statistic:

```ini
select1_variable = temperature
select1_min      = 0.5
select1_max      = 2.0
select2_variable = hydro_w_d
select2_min      = 1.0
sparse           = true
```

Bounds default to negative/positive infinity. `select2_*` requires
`select1_variable`. Every condition must hold; a NaN selection value fails.
Selection is recomputed within the spatial region at each output. Each
effective weight is its original value for selected cells and zero otherwise,
so `ones` becomes a phase indicator. With selection, variables plus weights
must total at most eight. For a strict lower cut, use the next representable
value above it in the run's precision.

## Sampling and uncertainty

E1 samples pairs within one MeshBlock: draw an integer lag with Chebyshev
radius `c` distributed proportional to `1/c`, then an admissible anchor.
E2 samples different-MeshBlock pairs: draw rank-local anchors and globally
shared sample cells, with quotas proportional to region cell counts. Each
anchor is paired with samples in the same replicate.

Inverse-probability factors make `T`, `W`, and `C` estimates of full population sums.
The reported totals are replicate averages. Ratios such as `T/C` have
finite-sample uncertainty and can have finite-sample bias. Counter-based
draws leave the simulation RNG unchanged.

For rare selections, `sparse=true` draws E2 endpoints from cells where any
weight is positive. E1 mixes support-based and ordinary anchors with equal
probability and applies the probability correction. Empty support gives
zero sums; a geometric region containing no cells is an error.

Budgets must be whole numbers from 1 to `9e15`.
`ndraw_block >= nreplicate >= 1`; both global budgets must cover at least one
endpoint per participating rank per replicate. The all-gather requires
`nsample_global*(4+number_of_fields) <= 2^31-1`.
E1 work scales with local MeshBlocks times `ndraw_block`; nominal E2 work
per rank scales as `nanchor_global*nsample_global/(nranks*nreplicate)`.
The global sample occupies memory on every rank.

Choose budgets from measured replicate errors, especially for high moments
and sparsely populated bins. Retain several replicates and increase sampling
until the quantities of interest converge.

## Increment histograms

`histogram=true` adds distributions for scalar and long/transverse modes.
Defaults are `hist_nbin=96`, `hist_min=1e-4`, `hist_max=1e2`, and
`hist_scale=sigma`. The bin count must be even and at least two, with
`0 < hist_min < hist_max`. `hist_scale=absolute` uses the template directly.

Scalar histograms use absolute weighted-field increments. Long/transverse
histograms use the signed longitudinal increment and transverse magnitude.
Unsigned edges are logarithmic; signed edges mirror half as many positive
intervals about zero, including a central bin.

For sigma scaling, edges multiply the weighted standard deviation of the
raw field over valid cells. Long/transverse modes use
`sqrt((sigma_x**2+sigma_y**2+sigma_z**2)/3)` for both rows. A centred
two-pass variance is recomputed at each output, with the final scale rounded
to 12 significant digits; a zero or nonfinite scale falls back to one.

`H` stores importance-weighted pair counts. With `nedge=hist_nedge[q]`,
column zero is underflow, column `nedge` is overflow, and interior column
`j` covers `edges[j-1] <= increment < edges[j]`. Extra columns are padding.
Summing a row's columns recovers `C`.

## Files and reader

Files are `sf_<id>/<basename>.NNNNN.sf`. The ASCII header ends with
`header offset=N`, the binary payload start. It records configuration,
time/cycle, mesh, region/selection, budgets, seed, estimator version, and
timings. Arrays use C ordering and the header's `byte_order`, with float64
values except uint64 `n_acc_er`:

```text
rbins[nbin+1]
T[nweight,nq,max_order,nbin]       W[nweight,nbin]    C[nweight,nbin]
T_er[2,nreplicate,nweight,nq,max_order,nbin]
W_er[2,nreplicate,nweight,nbin]    C_er[2,nreplicate,nweight,nbin]
n_acc_er[2,nreplicate,nweight,nbin]
# histogram=true adds:
hist_edges[nweight,nq,hist_ncol-1]    hist_sigma[nweight,nq]
H[nweight,nq,nbin,hist_ncol]         H_er[2,nreplicate,nweight,nq,nbin,hist_ncol]
```

For `aniso_mhd`, replace the first array with `rbins[nbin_l+1]`,
`rbins_prll[nbin_prll+1]`, and `rbins_perp[nbin_perp+1]`. Other arrays
concatenate those families; header `nbin` is their total.
The leading `*_er` index is 0 for E1 and 1 for E2.
`n_acc_er` counts accepted draws; `C_er` includes population weights.

Use the NumPy reader [sf_spectra_read.py](../vis/python/sf_spectra_read.py):

```python
import sys
import numpy as np

sys.path.insert(0, "/path/to/athenak/vis/python")
from sf_spectra_read import read_sf

meta, a = read_sf("sf_velocity/run.00010.sf")
den = a["C"][:, None, None, :]
S = np.divide(a["T"], den, out=np.full_like(a["T"], np.nan), where=den > 0)

# Delete-one-replicate jackknife of the ratio.
Tr = a["T_er"].sum(axis=0)
Cr = a["C_er"].sum(axis=0)[:, :, None, None, :]
R = Tr.shape[0]
if R < 2:
    raise ValueError("jackknife requires nreplicate >= 2")
den = Cr.sum(axis=0) - Cr
Sj = np.divide(Tr.sum(axis=0) - Tr, den,
               out=np.full_like(Tr, np.nan), where=den > 0)
err = np.sqrt((R - 1) / R * ((Sj - Sj.mean(axis=0))**2).sum(axis=0))
```

For correlation, replace `C/C_er` with `W/W_er`. Empty denominators remain
NaN in this example.

Extensive validation and sampling studies are retained with the
[feature-branch tests](https://github.com/rajsekhar33/athenak-RM/tree/feature/onthefly-sf-spectra/tests).
