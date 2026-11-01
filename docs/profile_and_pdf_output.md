# Spatial profiles and probability distributions

Spatial profiles (`file_type = prof`) and histograms (`file_type = pdf`) accumulate
statistics directly from the simulation state at the requested output cadence.
Each output can be restricted to a rectangular region of the mesh.

## Quantities stored

Let `V_i` be a cell's volume, `rho_i` its density and `q_i` the selected variable.
The cell weight is `w_i = V_i` for volume weighting and `w_i = rho_i V_i` when
`mass_weighted = true`.

A profile bins cells by a spatial coordinate and writes, for each bin,

$$
W_b=\sum_{i\in b}w_i,\qquad
Q_b=\sum_{i\in b}w_iq_i,\qquad
\langle q\rangle_b=Q_b/W_b.
$$

The stored mean is zero when `W_b = 0`; use the weight array to identify empty
bins. A histogram bins by the value of one or two variables and stores the
accumulated weight `H_b = sum(w_i)`.

To obtain a probability per histogram bin, divide `H_b` by the total selected
weight. To obtain a probability density, also divide by the bin width, or by
the product of the two widths for a joint distribution. For a density per unit
logarithmic variable, use the corresponding logarithmic bin widths. Underflow
and overflow weights remain available for accounting for material outside the
chosen bin limits.

## Spatial profiles

Each profile block selects one field component and one coordinate:

```ini
<output2>
file_type = prof
variable = hydro_w_d
coord_axis = x1
bin_min = 0.0
bin_max = 3.0
nbin = 100
logscale = false
mass_weighted = false
dt = 0.05
id = density_vs_x1
```

`coord_axis` accepts `x1`, `x2`, `x3` or `r`. For radial profiles,
`radial_type = spherical` measures distance from `(xc,yc,zc)`;
`radial_type = cylindrical` measures distance from the axis through that centre
specified by `cyl_axis = x1`, `x2` or `x3`. The centre defaults to `(0,0,0)`.

```ini
<output3>
file_type = prof
variable = hydro_w_d
coord_axis = r
radial_type = spherical
xc = 0.5
yc = 0.5
zc = 0.5
bin_min = 1e-3
bin_max = 1.0
nbin = 64
logscale = true
mass_weighted = true
dt = 0.1
id = density_vs_radius
```

For a cylindrical profile around x3, set `radial_type = cylindrical` and
`cyl_axis = x3`; its radius is `sqrt((x1-xc)^2 + (x2-yc)^2)`.

## Histograms and joint distributions

A one-dimensional histogram needs `variable` and its bin specification. Add
`variable_2`, `bin2_min`, `bin2_max`, `nbin2` and `logscale2` for a joint
distribution. This example measures the mass distribution in density and
internal-energy density within an x1 slab:

```ini
<output4>
file_type = pdf
variable = hydro_w_d
variable_2 = hydro_w_e
bin_min = 1e-2
bin_max = 1e2
nbin = 100
logscale = true
bin2_min = 1e-2
bin2_max = 1e2
nbin2 = 100
logscale2 = true
mass_weighted = true
x1_min = -0.5
x1_max = 0.5
dt = 0.1
id = density_energy_slab
```

Both output types accept `x1_min`, `x1_max`, `x2_min`, `x2_max`, `x3_min` and
`x3_max`. Each omitted bound defaults to the corresponding mesh boundary.
A cell contributes when its centre lies inside all three specified intervals.

Each variable must select a single supported field component. Multiple profile
variables require separate output blocks. Logarithmic bins require positive
lower limits. Bin intervals include their lower edge; values at or above the
last edge enter the overflow bin.

## Files and reading

Profiles use `prof_<id>/`. One-dimensional histograms use `pdf_<id>/`;
joint histograms use `pdf_<id>_<variable_2>/`. Each directory contains
`<basename>.bins.prof` or `<basename>.bins.pdf` for the bin edges, followed by
numbered output files.

Every file begins with an ASCII `key=value` header ending in `header offset=N`,
the byte offset of its binary payload. `size of Real` gives the element size:
4 bytes for a single-precision build and 8 bytes for a double-precision build.
The payload uses native byte order.

| File | Payload |
|---|---|
| Profile bin file | `nbin+1` edges |
| Profile snapshot | Three rows of `nbin+2` values: weight, weighted-variable sum, mean |
| 1D histogram bin file | `nbin+1` edges |
| 2D histogram bin file | First-axis edges, then `nbin2+1` second-axis edges |
| 1D histogram snapshot | `nbin+2` accumulated weights |
| 2D histogram snapshot | Row-major array with shape `(nbin2+2, nbin+2)` |

Index zero is underflow; the last index is overflow. The following reader
returns the header and a flat payload array:

```python
import numpy as np

def read_profile_or_pdf(path):
    with open(path, "rb") as stream:
        metadata = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Missing header offset")
            key, separator, value = line.decode("ascii").strip().partition("=")
            if separator:
                metadata[key.strip()] = value.strip()
            if key.strip() == "header offset":
                break
        width = int(metadata["size of Real"])
        if width not in (4, 8):
            raise ValueError("Unsupported element size")
        stream.seek(int(metadata["header offset"]))
        payload = stream.read()
    if len(payload) % width:
        raise ValueError("Truncated binary payload")
    return metadata, np.frombuffer(payload, dtype=f"=f{width}")
```

For a profile snapshot, reshape the payload to `(3, nbin+2)`. Retain the bin
weights when plotting means or combining outputs, so empty bins and unequal
sample weights are handled correctly.
