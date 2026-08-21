# On-the-fly Spatial Profiles and (Region-Restricted) PDFs in AthenaK

This note documents two on-the-fly output features added on top of the existing
PDF output (`file_type = pdf`): a new spatial **Profile** output
(`file_type = prof`) and an optional **spatial region restriction** usable by
both `prof` and `pdf`. Both reuse the same Kokkos `ScatterView`-based
binning/reduction machinery, and both write a compact **binary** file format
(not ASCII) so large 2D PDFs stay cheap to write and read.

## Motivation

Computing a spatial profile (e.g. mean density vs. `x1`) or a joint PDF of two
variables has historically meant dumping full 3D snapshots and post-processing
them in Python. That is expensive and limited to whatever snapshot cadence was
saved. These two output types compute the same quantities **during the run**,
at their own output cadence, with no snapshot required.

## Where the Code Lives

- `src/outputs/outputs.hpp`
  - `OutputParameters`: new fields `coord_axis`, `radial_type`, `cyl_axis`,
    `xc`/`yc`/`zc` (profile-only), and `x1_min`/`x1_max`/`x2_min`/`x2_max`/
    `x3_min`/`x3_max` (shared region restriction, used by both `pdf` and `prof`).
  - `ProfileData` / `ProfileOutput`: new struct/class for the profile output,
    modeled directly on the existing `PDFData`/`PDFOutput`.
- `src/outputs/outputs.cpp`
  - `ParseRegionRestriction()`: reads the six optional region bounds, each
    defaulting to the full mesh extent on that axis (so "not set" is a true
    no-op, not a special case in the per-cell kernel).
  - New `prof` branch in the `Outputs` constructor: parses `coord_axis`,
    `radial_type`/`cyl_axis` (only relevant when `coord_axis = r`), and
    `xc`/`yc`/`zc`, then constructs a `ProfileOutput`.
- `src/outputs/profile.cpp` (new file): `ProfileOutput`'s
  `LoadOutputData()`/`WriteOutputFile()`.
- `src/outputs/pdf.cpp`: unchanged except for the region-restriction mask added
  to the existing binning kernel.
- `src/CMakeLists.txt`: registers `outputs/profile.cpp`.

## `prof`: Spatial Profile Output

Bins cells by a coordinate — `x1`, `x2`, `x3`, or radius `r` — and records the
volume- or mass-weighted mean of **one** variable per bin (one variable per
output block, matching the existing `pdf` convention of `variable`/`variable_2`
rather than a variable list).

```ini
<output2>
file_type     = prof
variable      = hydro_w_d      # any single var_choice entry (outputs.hpp)
coord_axis    = x1             # x1 | x2 | x3 | r
bin_min       = 0.0
bin_max       = 3.0
nbin          = 100
logscale      = false          # log-spaced bins if true (bin_min must be > 0)
mass_weighted = false          # false = volume-weighted (default), true = mass-weighted
dt            = 0.05
id            = dens_vs_x1
```

### Radial profiles

`coord_axis = r` additionally requires `radial_type`:

```ini
<output3>                      # spherical: r = distance from (xc,yc,zc)
file_type     = prof
variable      = hydro_w_e
coord_axis    = r
radial_type   = spherical
xc = 0.5   yc = 0.5   zc = 0.5
bin_min = 1e-3   bin_max = 1.0   nbin = 64
logscale      = true
mass_weighted = true
dt            = 0.1
id            = press_vs_r

<output4>                      # cylindrical: R = distance from an axis through (xc,yc,zc)
file_type     = prof
variable      = hydro_w_d
coord_axis    = r
radial_type   = cylindrical
cyl_axis      = x3             # x1 | x2 | x3 -- the symmetry axis
xc = 0.0   yc = 0.0
bin_min = 0.01   bin_max = 5.0   nbin = 50
logscale      = true
dt            = 0.1
id            = dens_vs_cylr
```

`xc`/`yc`/`zc` default to `0.0` (the coordinate origin) if omitted. Unlike
`athena_research`'s Python-side `r`/`R` data functions (fixed at the origin,
see "Cross-checking" below), the C++ side supports an arbitrary center.

### Output files

- `prof_<id>/<basename>.bins.prof` — written once: bin edges.
- `prof_<id>/<basename>.NNNNN.prof` — one per output dump: three arrays of
  `nbin+2` values each (index `0` and `nbin+1` are the underflow/overflow
  bins): `sum_weight`, `sum_weight_times_var`, and `mean` (the last one
  computed host-side after the MPI reduce, with a `0.0` fallback for empty
  bins, so the file is immediately plot-ready).

## Region Restriction (`pdf` and `prof`)

Both output types accept six independent, optional bounds:

```ini
x1_min = -0.5   x1_max = 0.5    # each optional; unset axes default to the
x2_min = ...    x2_max = ...    # full mesh extent on that axis, i.e. no
x3_min = ...    x3_max = ...    # restriction
```

A cell is included only if its cell-center coordinate falls inside all three
ranges. This is useful for, e.g., a PDF restricted to a post-shock slab, or a
profile restricted to a transverse core region.

```ini
<output5>                      # 2D joint PDF, mass-weighted, x1 in [-0.5, 0.5] only
file_type     = pdf
variable      = hydro_w_d
variable_2    = hydro_w_e
bin_min = 1e-2   bin_max = 1e2   nbin = 100   logscale = true
bin2_min = 1e-2  bin2_max = 1e2  nbin2 = 100  logscale2 = true
mass_weighted = true
x1_min = -0.5   x1_max = 0.5
dt            = 0.1
id            = rho_e_pdf_slab
```

## Binary File Format (both `pdf` and `prof`)

Every file (`.bins.pdf`/`.bins.prof` and each numbered dump) starts with a
short ASCII `key=value` header, one line per key, ending in a fixed-width
`header offset=%012ld` line giving the exact byte at which the binary payload
begins — the same "ASCII header, then raw payload" convention already used by
AthenaK's `file_type = bin` mesh output (`src/outputs/binary.cpp`). The
payload is raw, native-byte-order `Real` values (row-major for the 2D PDF
case) — `size of Real` in the header tells you whether that's 4 or 8 bytes
(depends on whether `SINGLE_PRECISION_ENABLED` was set at build time).

**Why binary, not ASCII:** the previous ASCII writer called `fprintf` once per
scalar value. That cost is negligible for a profile/1D-PDF (`O(nbin)` values)
but grows quadratically with resolution for a 2D PDF (`O(nbin*nbin2)` — e.g.
40,000 individual `fprintf` calls at 200x200 bins). A single `fwrite` of the
whole contiguous array replaces all of that, and is also more compact on disk
(~8 raw bytes/value vs. ~13 ASCII bytes/value).

Minimal Python reader:

```python
import struct

def read_prof_or_pdf(fname):
    with open(fname, "rb") as f:
        meta = {}
        while True:
            key, _, val = f.readline().decode("ascii").strip().partition("=")
            meta[key] = val
            if key == "header offset":
                break
        f.seek(int(meta["header offset"]))
        payload = f.read()
    n = len(payload) // 8   # assumes size of Real == 8; check meta if not
    return meta, struct.unpack(f"={n}d", payload)
```

`pdf`'s per-dump payload is the `(nbin2+2, nbin+2)` result array (`(1, nbin+2)`
if 1D) in row-major order; `prof`'s per-dump payload is the three
`sum_weight`/`sum_weight_times_var`/`mean` arrays back-to-back, each
`nbin+2` values.

## Cross-checking

Validated end-to-end against an independent computation from a raw `.bin` mesh
snapshot of the same run, using the `athena_research` package's own histogram
and profile operations (`set_profile`, `set_radial`, `set_radial_cyl`,
`set_dist2d`) rather than a hand-rolled comparison — all four matched to
within the ~1e-7 float32 precision floor of AthenaK's `.bin` mesh format (which
downcasts field data to `float` on disk, unlike this on-the-fly output, which
computes from double-precision in-memory data directly).

`athena_research`'s `r`/`R` radial coordinates are fixed at the origin (not
configurable), so a center other than `(0,0,0)` cannot be cross-checked
directly against that package as-is — the C++ side's `xc`/`yc`/`zc` still work
correctly for such cases, this is purely a limitation of what the comparison
tooling supports.

## Common Gotchas

- `logscale = true` (or `logscale2`) requires the corresponding `bin_min` to be
  `> 0.0` — this is checked at startup and is a fatal error otherwise. For a
  radial profile with `logscale = true`, keep `bin_min` small and positive
  (a bin including the exact center point cannot be log-binned).
- `prof` accepts exactly one `variable` (like `pdf`'s single-variable mode); a
  variable **group** (e.g. `hydro_w`) is rejected the same way `pdf` already
  rejects it — profile multiple variables by adding multiple `<outputN>` blocks.
- Region-restriction bounds are independent per axis; leaving all six unset is
  equivalent to no restriction at all (each defaults to that axis's full mesh
  extent), not to skipping the check.
