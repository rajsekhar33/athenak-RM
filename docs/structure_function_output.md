# On-the-fly Structure Function Output (`file_type = sf`)

Computes structure functions of one or more fields **during the run**, at their own
output cadence, without snapshots. Pairs within a MeshBlock need no communication;
pairs across MeshBlocks use a global sample of field values that every rank receives
(two host-staged `MPI_Allgatherv` calls per output). The full field is never moved. The
conventions follow `athena_research/operations/structure_functions.py`. The estimator
is described below ("What is computed" and "How the sums are estimated").

Status: uniform grids only, `mode = scalar | long_trans | aniso_mhd | correlation`,
optional increment histograms (scalar, long_trans), no ghost-cell halo (`g = 0`).

## What is computed

For every weight `w` and variable `q`, and every separation bin `b` (|l| between
cell centres, using the minimum image on periodic axes only), the output estimates
sums over **all** unordered pairs of distinct valid cells:

| Array | Definition | Python equivalent |
|---|---|---|
| `T[w,q,p,b]` | sum of abs(w2*q2 - w1*q1)^p, p = 1..max_order | `sf` |
| `W[w,b]` | sum of (w1 + w2) | `weight_sum_bin` |
| `C[w,b]` | number of pairs | (new) |

- A cell is valid when its weight is > 0 and its centre lies inside the optional region.
- **Separation.** `l` is computed from the pair's global cell offset `(di, dj, dk)`:
  `l = (di h1, dj h2, dk h3)` with `h = domain length / cells` on each axis, after the
  minimum image in whole cells (an offset of exactly half a periodic axis keeps its
  sign). Within-block and cross-block pairs use the same arithmetic, so a pair's bin
  does not depend on the MeshBlock layout or the rank count. Bins are half-open,
  `edges[b] <= |l| < edges[b+1]` (`numpy.searchsorted(..., 'right') - 1`). Python
  subtracts floating-point cell centres instead, so a separation lying exactly on a
  bin edge (to rounding) can fall in the neighbouring bin there.
- **Structure function:** `S_p = T / C`.
  - With unit weights this equals Python's `sf / (weight_sum_bin/2)`.
  - With non-unit weights, `T/C` is the pair-averaged moment of the weighted field
    `w*q`, and Python's normalisation differs. Both can be computed from `T`, `W` and `C`.
- **`mode = long_trans`** takes three variables (vector components along x1, x2, x3) and
  reports two increments:
  - `q = 0`: longitudinal |du.l|/|l|
  - `q = 1`: transverse sqrt(|du|^2 - (du.l/|l|)^2)

  Here `du = w2*u2 - w1*u1`, as in Python's "Helmholtz" SF.
- **`mode = aniso_mhd`** (Python `get_sf_aniso_mhd_mb`) takes six variables: velocity
  x1, x2, x3, then magnetic field x1, x2, x3 (e.g. `mhd_w_vx, mhd_w_vy, mhd_w_vz,
  mhd_bcc1, mhd_bcc2, mhd_bcc3`).
  - `b` is the unit vector of the **unweighted** pair mean `(B1 + B2)/2`. Pairs with
    `|B1 + B2|/2 < 1e-30` are skipped.
  - Four increments of the weighted fields: `q = 0..3` = |dv|, |dv_perp|, |dB|,
    |dB_perp|, where perp means perpendicular to `b`.
  - Each pair is binned in three families, stored side by side along the bin axis
    (`nbin = nbin_l + nbin_prll + nbin_perp`):
    - `l`: |l|;
    - `l_prll`: |l.b|, only when cos(theta) = |l.b|/|l| > cos(`theta_prll_max`);
    - `l_perp`: sqrt(|l|^2 - (l.b)^2), only when cos(theta) <= cos(`theta_perp_min`).
  - All families use the same [r_min, r_max] edges. The header gives `nbin_l`,
    `nbin_prll`, `nbin_perp` and the angles; the payload starts with the three edge
    arrays.
  - Keys and defaults (Python's): `nbin_prll = 50`, `nbin_perp = 50`,
    `nbin = max(nbin_prll, nbin_perp)`, `theta_prll_max = pi/18`,
    `theta_perp_min = pi/2 - pi/18` (radians).
- **`mode = correlation`** (Python `correlation_functions.get_cf`, the production
  `--method correlation`) is the raw two-point correlation `<A(x) B(x+r)>`.
  - One variable gives the autocorrelation (`B = A`), two give `<A(x) B(x+r)>`.
  - The fields are not multiplied by the weights. A weight enters once, through the
    pair weight `w1*w2`. There is no mean subtraction, as in the production runs.
  - Both orientations enter every bin. Per unordered pair:
    - `T[w,0,p,b] += w1*w2*((A1*B2)^p + (B1*A2)^p)`;
    - `W[w,b] += 2*w1*w2`;
    - `C[w,b] += 1`.
  - `T` and `W` then estimate Python's `cf` and `weight_sum_bin`, and the correlation
    is `T/W`.
  - `C` estimates the population pair count. For an exhaustive Python run,
    `num_bin_points = 2*C`. A sampled Python run instead records accepted draws,
    without the importance rescaling applied to its sums, so its `num_bin_points` is
    not comparable to `C`.
  - `nq = 1`. `histogram = true` is rejected. The header line `products=` names A and B.

### How the sums are estimated

Pairs fall into two disjoint populations that together cover every pair exactly once:

- **E1, both cells in the same MeshBlock.** Sampled locally, with no communication.
  - A lag `l` is drawn first: Chebyshev radius `c` with probability proportional to 1/c,
    then a point uniform on that integer cube shell.
  - An anchor is then drawn among the cells whose partner `x + l` lies in the block.
  - Lags go up to the full block size.
- **E2, cells in different MeshBlocks.**
  - Every rank draws anchors and sample cells, proportional to its region cells, with at
    least one of each per replicate.
  - The sample cells are all-gathered once per output (only sampled cells, never whole
    blocks).
  - Every local anchor is paired with every sample of the same replicate.
- **`sparse = true`** (Python `sparse_data`) is for weights that are zero almost
  everywhere, e.g. a phase indicator. At every output it builds the list of support
  cells: region cells where some weight is > 0 (all region cells if a weight is
  `ones`).
  - E2 anchors and samples are drawn from support cells only, with rank quotas
    proportional to support counts.
  - Each E1 draw takes its anchor, with probability 1/2 each, either from the block's
    support cells or from the default proposal. The default branch keeps the
    largest in-block lags covered.
  - Importance factors use the exact mixture probability, so the raw sums `T`, `W`, `C`
    stay unbiased by construction (inverse-probability weighting). The ratio
    `S = T/C` is a ratio estimate and need not be exactly unbiased at finite sample
    size.
  - Blocks and ranks without support cells draw nothing. An empty support gives an
    all-zero output instead of an error.
  - The header records `sparse=1`, `ncell_support=` (global support cells),
    `nrank_support=` (ranks holding support) and `time_support=` (seconds to build
    the support list and quotas on rank 0, not included in `time_e1`/`time_e2`).
  - Memory: the support list holds 8 bytes per local support cell plus per-block
    offsets; with a `ones` weight that is 8 bytes per region cell.
  - An all-zero weight field is not an error: the histogram scale falls back to 1 (as
    for a constant field) and every sum is 0. A region containing no cells at all is
    still a fatal error.

Each pair carries an importance factor that makes `T`, `W` and `C` unbiased estimates
of the full sums. The estimate is split into `nreplicate` independent replicates. The
reported arrays are the replicate average, and the per-replicate arrays are written too,
so error bars come from the replicate spread.

All random numbers are counter-based hashes of (seed, output number, stream, block or
rank, replicate, draw):
- there is no RNG state, so the simulation is never affected;
- the same layout always gives the same draws;
- E1 draws do not depend on the number of ranks.

## Input block

One key per line: AthenaK's parser reads only one `=` per line.

```ini
<output5>
file_type      = sf
id             = vel                # directory sf_<id>/ (default: block name, e.g. output5)
mode           = scalar             # scalar | long_trans | aniso_mhd | correlation
variables      = hydro_w_vx, hydro_w_vy, hydro_w_vz
weights        = ones, hydro_w_d    # "ones" = unit weight; others are output variables
max_order      = 10
nbin           = 64
logscale       = true
r_min          = 0.0                # 0 -> min cell size (Python default)
r_max          = 0.0                # 0 -> half the box diagonal (Python default)
ndraw_block    = 4096               # E1 draws per MeshBlock, total over replicates
nsample_global = 65536              # E2 sample cells, total over replicates
nanchor_global = 1048576            # E2 anchors, total over replicates
nreplicate     = 4
seed           = 0
sparse         = false              # true: draw only cells with some weight > 0
select1_variable = temperature     # optional cell selection (see "Cell selection")
select1_min    = 0.5                # inclusive; default -inf
select1_max    = 2.0                # inclusive; default +inf
select2_variable = hydro_w_d        # optional second condition (needs select1_*)
select2_min    = 1.001
timing         = false              # print E1/E2 times (min..max over ranks)
kernel         = auto               # auto | team | global (pair-kernel variant)
histogram      = false              # increment histograms (Python delta_pdf)
hist_nbin      = 96                 # log |delta| bins (even; see below)
hist_min       = 1.0e-4             # template range, in units of sigma ...
hist_max       = 1.0e2
hist_scale     = sigma              # ... or 'absolute' to use the template as is
dt             = 0.1
# optional region: x1_min, x1_max, x2_min, x2_max, x3_min, x3_max
```

**Variables:**
- Variables and weights must be single, non-derived output variables (e.g. `hydro_w_vx`,
  `mhd_bcc1`), at most 8 distinct fields, at most 8 variables and at most 8 weights
  (repeated names count). Every name must give exactly one field component in the
  run (e.g. `hydro_w_s` only with exactly one passive scalar); otherwise a fatal error.
- Real-valued keys must be single finite numbers (selection bounds may also be `inf`
  or `-inf`). `ndraw_block`, `nsample_global` and `nanchor_global` must be whole
  numbers from 1 to 9e15, and `nsample_global x (4 + number of fields)` must fit one
  `MPI_Allgatherv` (at most 2^31 - 1 values). An empty region is a fatal error, also
  with `sparse = true` (whose support may still be empty at an output).
- The generic output keys `slice_x1/2/3`, `gid` and `ghost_zones` are rejected.
- Two computed variables are also accepted (not as weights):
  - `temperature` = `(gamma - 1) e / rho` (Python `temp`, code units);
  - `pressure` = `(gamma - 1) e` (Python `pres`).
  They are computed from the primitives at every output (Newtonian, single-fluid hydro
  or MHD, `eos = ideal`; otherwise a fatal error, including two-fluid runs with both
  `<hydro>` and `<mhd>`, where the fluid would be ambiguous), into one extra array of
  reals per computed
  variable and cell. This covers the production list `velx, vely, velz, dens, pres,
  temp` as `hydro_w_vx, hydro_w_vy, hydro_w_vz, hydro_w_d, pressure, temperature`.
- Each sf output needs its own directory: without `id` it uses the block name
  (`sf_output5/`); two sf blocks with the same explicit `id` are a fatal error.
- NaN/Inf in the input fields are not filtered: they propagate into the sums (as in
  Python), and the validation scripts reject nonfinite outputs.
- Variable groups (`hydro_w`, ...) are rejected.

## Cell selection (Python region masks)

`select1_*` and the optional `select2_*` restrict the output to the cells where every
condition `min <= value <= max` holds (inclusive bounds; a NaN value fails). This is the
on-the-fly form of the Python masks passed as weights, e.g. the RadShock interface
(`temp` in `[T_cold, T_hot]`, weight `interface_bool`) and the downstream cut
(`dens > 1.001 rho_u`).
- `select<n>_variable` is `temperature`, `pressure` or any single output variable
  (e.g. `hydro_w_d`, `hydro_w_s`). It need not appear in `variables` or `weights`.
- `temperature` is Python's `temp` in code units, `(gamma - 1) e / rho` from the
  primitives (Newtonian, single-fluid hydro or MHD, `eos = ideal`; otherwise a fatal
  error).
- Every weight becomes `selected ? weight : 0`: a `ones` weight becomes the 0/1
  indicator (Python `weights='interface_bool'`), and `hydro_w_d` becomes the density
  restricted to the selection. Everything downstream (validity `w > 0`, the `sparse`
  support, histogram scales, `W`) follows from these weights. Combine with
  `sparse = true` when the selection is a small fraction of the region.
- The selection is evaluated at every output (it follows the evolving flow) and is
  applied inside the spatial region (`x1_min`, ...), so both restrictions hold.
- Bounds are inclusive. Python's density cut is strict (`>`), and the two differ for
  every cell whose value equals the cut exactly, which a plateau (e.g. an unshocked
  upstream at exactly `rho_u`) can make a large number of cells. To reproduce a strict
  `value > cut`, set `min` to the next representable number above the cut in the
  run's precision, `nextafter(cut, +inf)` (double: `numpy.nextafter(cut, np.inf)`).
  The production cut `1.001*rho_u` lies above the upstream plateau, so it is not
  affected in practice, but check this for any cut equal to a value the flow holds.
- Header: `nselect=`, `select<n>=name,min,max`, `ncell_selected=` (selected region
  cells, all ranks) and `time_select=` (seconds on rank 0). Without `select*` keys
  none of these are written and the output is bit-identical to the code without the
  feature (checked on the serial backend).
- Memory: one extra array of `nweight` reals per cell (including ghosts) while a
  selection is configured, e.g. 2 weights on 128^3 cells per GPU (plus 2 ghost layers): about 37 MB.
- With a selection, `variables + weights` must be at most 8.

Example: the RadShock production call
`checkpointed_sf(ad, 'interface', ('velx','vely','velz','dens','pres','temp'),
weights='interface_bool')` (`nbins=100`, `max_order=3`, log bins), with
`T_cold`, `T_hot` from `get_shock_temperatures()` written into the input:

```ini
<output6>
file_type      = sf
id             = interface
mode           = scalar
variables      = hydro_w_vx, hydro_w_vy, hydro_w_vz, hydro_w_d, pressure, temperature
weights        = ones                 # becomes interface_bool through the selection
select1_variable = temperature
select1_min    = 0.0123               # T_cold (code units)
select1_max    = 1.23                 # T_hot
sparse         = true
max_order      = 3
nbin           = 100
logscale       = true
histogram      = true                 # Python delta_pdf, edges in units of sigma
dt             = 1.0
```

The Helmholtz variant (`checkpointed_helmholtz_sf`, the longitudinal/transverse
velocity SF) is `mode = long_trans` with the three velocity components and the same
selection. A downstream region adds `select2_variable = hydro_w_d` and
`select2_min = 1.001*rho_u` (inclusive; Python uses `>`). Python's `correlation`
method (`--method correlation --fields A B`, raw `<A(x) B(x+r)>`) is
`mode = correlation` with `variables = A, B` and the same selection.
- Validation: `tests/sf/run_select.sh` (exact reference, empty selection, invalid
  configurations); release lanes `select_cpu_np3`, `select_gpu_np2`, vis2/cross-node
  `select_np2`.

**Budgets and cost:**
- `nsample_global` and `nanchor_global` must each be at least
  `nreplicate x (number of ranks holding region cells)`. With `sparse = true`, count
  ranks holding support cells at that output instead; a violation is a fatal error.
- E1 costs about `nmb x ndraw_block` pair evaluations per rank (nmb = MeshBlocks on
  the rank).
- E2 costs about `(nanchor_global x nsample_global) / (nranks x nreplicate)` nominal
  pair evaluations per rank: global totals split over replicates and ranks. Measured
  wall time also includes the all-gather and the rank quotas.
- Communication per output: one all-gather of the sample (about
  `nsample_global x (4 + nfield) x 8` bytes) plus small reductions.

**Scaling with the number of ranks (measured on Frontier, 2026-10-29):**
- Keep `nanchor_global` and `nsample_global` fixed as the mesh and rank count grow. The
  E2 pair work per rank then falls as 1/P. This is the recommended setting.
- Budgets may instead be raised with the rank count, for example as sqrt(P/8). That
  is an input choice, not an sf parameter; the benchmark input generator
  `tests/frontier/make_input.sh` offers it as `SAMPLING=sqrtp`. The E2 pair work per
  rank then stays about constant, while the gathered sample (held by every rank) grows,
  so the cost per output does not fall. It is retained, but it is expensive at scale.
- Measured per-output cost of the hydro `longtrans` + `scalar` pair of outputs
  (production budgets: `ndraw_block` 262144, `nreplicate` 8; 256^3 cells and 8
  MeshBlocks per GCD; median of 2 repeats):

| Nodes (GCDs) | Mesh | Fixed budgets 262144 x 262144 | sqrt(P/8) budgets |
|---|---|---|---|
| 1 (8) | 512^3 | 1.59 s | 1.59 s (same budgets) |
| 8 (64) | 1024^3 | 0.41 s | 1.73 s (741455 x 741455) |
| 64 (512) | 2048^3 | 0.28 s | 2.09 s (2097152 x 2097152) |

- For comparison, a hydro cycle without outputs took 0.106 s at 1 node and
  0.093-0.094 s at 8 and 64 nodes (0.090 s in a 400-cycle run). That 400-cycle run at
  64 nodes, with sf and spec outputs every 10 cycles, showed no drift in the cost per
  output or in the counted spectrum-buffer sizes (whole-device memory was not
  measured).
- Full records: `tests/frontier/README.md`.

**Choosing budgets (measured, `tests/sf/precision/`):**
- **Setup.** 256^3 driven isothermal hydro turbulence on 2x A40. Scalar velocity, 64
  log bins, `nreplicate = 8`, weights ones and rho. (The constructor defaults are
  `nreplicate = 4`, `ndraw_block = 4096`.)
- **Statistic.** The stored estimate is `sum_r T_r / sum_r C_r`. Its standard error is
  a delete-one-replicate jackknife, which keeps each replicate's (T_r, C_r) pair
  together (`tests/sf/precision/analyze.py`).
- **Table.** Median / 90th-percentile relative standard error over bins, weight ones,
  E1 + E2 combined:

  | ndraw_block | anchors x samples | E2 time / output | p = 1 | p = 2 | p = 3 |
  |---|---|---|---|---|---|
  | 4096 | 65536 x 4096 | 0.12 s | 0.8% / 2.6% | 1.7% / 5.1% | 2.9% / 9.0% |
  | 16384 | 262144 x 16384 | 1.1-1.4 s | 0.4% / 1.1% | 0.9% / 2.4% | 1.4% / 4.5% |
  | 65536 | 1048576 x 65536 (E2 default) | 21-25 s | 0.2% / 0.4% | 0.4% / 1.0% | 0.7% / 2.1% |
  | 262144 | 262144 x 262144 | 22 s | 0.1% / 0.3% | 0.3% / 0.7% | 0.4% / 1.3% |

  For the record, p = 10 at the E2 default is 7% median and 37% at the 90th
  percentile. High orders have heavy tails, and a typical-bin figure is no all-bin
  guarantee.
- **E1 is nearly free.** It took at most 0.04 s per output even at 262144 draws per
  block. Its error falls as 1/sqrt(ndraw_block), and it sets the combined error in
  many small-separation bins. Use `ndraw_block` of 262144 or more.
- **E2.**
  - Both E1 and E2 budgets changed between the first three rows, so their trend is not
    an E2 scaling law.
  - With E1 fixed (last row and its two equal-pair alternatives, 1048576 x 65536 and
    65536 x 1048576), the balanced split gave about 30% lower E2 error at p <= 3. This
    is consistent with variance terms of the form A/n_anchor + B/n_sample +
    C/(n_anchor n_sample), where shared endpoints matter.
  - This holds for this field and these three configurations; it is not a general
    law.
- **Smallest budget.** The 65536 x 4096 row differs more from the largest budget than
  its error bars allow at high orders (paired jackknife: rms z 1.8-2.1 at p = 10). At
  p <= 3 it is consistent (rms z 1.1-1.4). This is a precision warning, not a measured
  bias: the reference is itself sampled.
- **More ranks.** For a fixed global budget, E2 work per rank falls as 1/nranks; the
  relative error does not change.
- **Sparse weights.** On a 32^3 test with 64 blocks and the same budgets, `sparse = true`
  improved the median E2 error:
  - 0.17% of cells selected: 13% -> 0.4%;
  - 2.4% of cells selected: 3.2% -> 0.5%.
- **Rare selections without `sparse`.** Each replicate then holds only a few valid E2
  samples. An 80-file seed scan is consistent with the exact raw sums (no bias is
  resolved at that precision), but the replicate error
  bars underestimate the true error. Use `sparse = true` whenever the weight support
  is a small fraction of the region.

### Pair kernels

- `kernel = auto` (default) accumulates each team's pairs in a shared-memory histogram
  and flushes it to global memory once.
  - When all weights do not fit in 48 KiB, the work runs in several passes over weight
    groups.
  - If not even one weight fits, it falls back to `global`.
- `kernel = global` adds every pair to the global accumulators with atomics. It is the
  reference variant: same pairs, sums equal to round-off.
- The header records the variant in use (`kernel=`).

### Increment histograms (`histogram = true`)

These are Python's `delta_pdf`: per weight, increment and separation bin,
importance-weighted pair counts in columns `searchsorted(edges, v, 'right')`. Column 0
is the underflow, the last column the overflow, so every pair is counted once and each
row sums to `C`.

- **Value histogrammed:**
  - `scalar`: |w2 q2 - w1 q1|;
  - `long_trans`: the **signed** longitudinal increment (as Python) and the transverse
    magnitude.
- **Default edges** are `structure_functions.py`'s `DEFAULT_NORMALIZED_DELTA_EDGES`
  (`logspace(-4, 2, 97)`) times sigma; the signed longitudinal row uses
  `DEFAULT_NORMALIZED_SIGNED_DELTA_EDGES` (mirrored, one central bin).
- **sigma** is recomputed at every output: the weighted standard deviation over the
  valid cells, as `masked_increment_scales` (12 significant digits).
  - It is computed with a centered two-pass variance, so a large mean (e.g. a bulk
    velocity) does not cancel a small dispersion.
  - `long_trans` uses the 1D dispersion sqrt((s_x^2 + s_y^2 + s_z^2)/3) of the raw
    component variances for both rows. A one-component wave of amplitude A gives
    A/sqrt(6).
  - Only a zero or nonfinite **final** scale falls back to 1 (constant field), never
    a single component.
- Histograms are accumulated with global atomics (too large for shared memory). They
  cost one extra atomic per (weight, increment) per pair.

## Output files

`sf_<id>/<basename>.NNNNN.sf`. Each file starts with an ASCII `key=value` header ending
in `header offset=N`. The header records the full configuration: variables, weights,
bins, periodicity, domain lengths, region, mesh and MeshBlock sizes, budgets, seed,
`proposal_version`, requested and effective halo, attempt counts, and timings. The
binary payload starts at byte N, in the byte order given by `byte_order=`, as float64
except `n_acc`
(uint64), in the order given by the header's `payload=` key:

```text
rbins[nbin+1]
T[nweight,nq,max_order,nbin]   W[nweight,nbin]   C[nweight,nbin]   (replicate averages)
T_er[2,nrep,nweight,nq,max_order,nbin]   W_er[2,nrep,nweight,nbin]
C_er[2,nrep,nweight,nbin]   n_acc_er[2,nrep,nweight,nbin]
# only with histogram = 1 (header keys hist_ncol, hist_nedge per q):
hist_edges[nweight,nq,hist_ncol-1]   hist_sigma[nweight,nq]
H[nweight,nq,nbin,hist_ncol]   H_er[2,nrep,nweight,nq,nbin,hist_ncol]
```

Unsigned rows hold `hist_nedge[q]` edges, padded with zeros to `hist_ncol - 1`. Their
last column is unused.

The leading index of the `*_er` arrays is the estimator (0 = E1 within-block,
1 = E2 cross-block), followed by the replicate. `n_acc` counts raw accepted pair
evaluations (Python's `num_bin_points`).

Use `read_sf()` from [sf_spectra_read.py](../vis/python/sf_spectra_read.py):

```python
meta, a = read_sf("sf_vel/run.00010.sf")
S = a["T"] / a["C"][:, None, None, :]           # S[w, q, p-1, bin]
# standard error of S_p: delete-one-replicate jackknife of the ratio of sums, keeping
# each replicate's (T_r, C_r) together (as tests/sf/precision/analyze.py)
Tr = a["T_er"].sum(0)                            # E1 + E2, [R, w, q, p-1, bin]
Cr = a["C_er"].sum(0)[:, :, None, None, :]       # [R, w, 1, 1, bin]
R = Tr.shape[0]
Sj = (Tr.sum(0) - Tr) / (Cr.sum(0) - Cr)         # estimate without replicate r
err = np.sqrt((R - 1) / R * ((Sj - Sj.mean(0))**2).sum(0))
```

## Validation (tests/sf)

The [test guide](../tests/README.md) lists the self-contained checks and the
optional comparisons that require a separate `athena_research` checkout.

- **`sf_reference.py` (single-file smoke test).** Rebuilds the deterministic fields of
  `src/pgen/sf_unit_test.cpp` and computes exact all-pairs sums for each population.
  Each sampled sum is compared with its exact value using the replicate spread. With
  histograms it also checks:
  - `H` against exact per-population histograms (input
    `inputs/tests/sf_histogram_test.athinput`);
  - that the row sums equal `C`;
  - sigma against the Python formula.
- **`compare_sf.py`.** Two files with the same draws, e.g. `kernel = team` vs
  `global`: counts exactly, sums to round-off.
- **`seed_scan.{sh,py}` (bias test).** Many independent seeds; the seed-averaged error
  of every bin must be consistent with zero.
- **`run_suite.sh` (edge cases):**
  - base, region, outflow axis, 2D, one block spanning the periodic box, non-cubic
    blocks;
  - seed scans for the minimal-quota, region and R = 1 and 2 cases;
  - constant field;
  - too-small budget (fatal error).

Run with:

```bash
tests/sf/run_suite.sh BUILD/src/athena MPIRUN NRANKS OUTDIR
```

## Limitations of this release

- Uniform grids only (fatal error on SMR/AMR); shearing-periodic boundaries rejected.
- No ghost-cell halo (`sf_halo` is accepted but the effective halo is 0, recorded in
  the header).
- No derived variables. No histograms for `aniso_mhd` or `correlation` (Python has none
  either). `correlation` has no mean subtraction (Python's optional `subtract_mean`).
- **Floating-point detail:** compilers that contract `w2*q2 - w1*q1` into a fused
  multiply-add (nvcc, and nvhpc on CPU) give increments of exactly equal weighted
  values at the rounding level (about 1e-16 relative) rather than exactly 0. The
  Python CuPy kernels do the same.
