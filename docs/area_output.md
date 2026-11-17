# Multiscale interface areas

An `area` output measures isosurface areas of a cell-centred scalar on a uniform
three-dimensional mesh. It supports direct sampling at a chosen stride and two
Gaussian-smoothed definitions. All methods use marching cubes and report area
in squared code length.

```ini
<output3>
file_type = area
id = temperature_area
variable = temperature
levels = 1,5,20
strides = 1,3,8,16
methods = sampled,indicator,field
sample_transform = log10
sigma_factor = 1.0
truncate = 4.0
dt = 0.1
```

Choose `levels` in the units of the selected variable. `temperature` means
`P/rho` in code units. Multiply by the units module's `temperature_cgs()` to
convert this value to kelvin. A single output scalar, such as `hydro_w_d` or `mhd_w_d`,
can also be used. A grouped variable requires an explicit zero-based `component`.
Each area output needs a distinct `id`.

## Area definitions

For scalar `q` and contour level `q*`, `sampled` measures the surface `q = q*`
on a fixed-origin lattice. Samples start at the first global cell centre and
advance by `s` cells along each axis. `sample_transform = linear` interpolates
`q`; `log10` interpolates its logarithm and requires positive field values and
levels. The transform changes interpolation between samples.

Periodic axes close onto the first sample. Nonperiodic axes end at the final
interior cell centre. A shortened last interval uses its actual physical width.
Strides need not divide the MeshBlock dimensions or global cell counts. The
maximum stride is half the smallest periodic-axis cell count, rounded down.
For a fully nonperiodic mesh, it is the smallest axis cell count minus one.

Scalar values are normalized consistently across the whole mesh before
triangulation. A vertex within `1e-12*max(abs(q),abs(q*))` of the contour is
assigned to its outside phase. For a zero contour, the tolerance is `1e-12`
times the largest absolute field value; an entirely zero field has zero area.
With logarithmic interpolation, these rules apply to `log10(q)` and
`log10(q*)`. The convention suppresses roundoff contours on uniform plateaus
and is invariant under a positive rescaling of a linearly interpolated scalar.

The `indicator` method classifies cells using

$$
\chi(\mathbf x)=\mathbf 1[q(\mathbf x)\leq q_*]
$$

and measures the half-level surface of the smoothed phase distribution:

$$
A_\chi(\sigma;q_*)=\operatorname{area}\{G_\sigma*\chi=1/2\}.
$$

The `field` method smooths the original scalar and measures

$$
A_q(\sigma;q_*)=\operatorname{area}\{G_\sigma*q=q_*\}.
$$

For temperature, indicator smoothing describes the geometry of the selected
cold phase. Field smoothing also responds to temperature contrast within and
between phases. An isolated planar thermal step illustrates the distinction: its filtered
indicator crosses one half at the original interface, whereas a filtered
temperature contour generally shifts unless its level is the midpoint of the
two temperatures.

Both smoothed definitions use the original fine-grid lattice for area extraction.
The Gaussian standard deviation is `sigma_factor*s` cells along each axis. Its
physical widths are therefore `(sigma*dx1, sigma*dx2, sigma*dx3)`. The finite
one-dimensional kernel has radius `floor(truncate*sigma + 0.5)` and normalized
weights proportional to `exp(-j*j/(2*sigma*sigma))`. Three separable passes apply
the kernel along x3, x2 and x1. Periodic boundaries wrap; nonperiodic boundaries
extend the nearest interior value. Kernel support may exceed a domain length.

## Files, cadence and restarts

Files are written as `area_<id>/<basename>.NNNNN.area`.
Their metadata describes the variable/component, geometry, contour levels,
strides, transform and filter. Data rows contain

```text
method level stride sigma_cells area
```

The directory also contains `configuration.txt`; subsequent events must match
its scalar, geometry and measurement settings. Time and cycle are recorded in
each event. Ordinary output controls `dt` or
`dcycle` set the cadence, and the driver also writes initial and final outputs.
Restart checkpoints retain the output counters. An existing target file is
protected from overwrite; select a new output location for a continuation that
would overlap previously written files.

`sampled` is the default method. Gaussian methods are enabled explicitly through
`methods`. Smoothing can remove an isosurface altogether, producing zero area.
Use separate output IDs when different fields or analysis cadences are needed.

## Interpreting the measurements

For cubic cells, the sampled scale is `s*dx`, and the Gaussian scale is
`sigma_factor*s*dx`. The two definitions measure different filtered geometries.
Hold the scalar, threshold, boundary convention and averaging interval fixed
when comparing runs. Normalize by the transverse box area when measuring the
area enhancement of a nominally planar interface.

A single planar interface normal to x1 has area `L2*L3` when both transverse
axes are periodic. A periodic slab has two interfaces and total area `2*L2*L3`.
For nonperiodic transverse axes, the measured extent runs between their first
and last interior cell centres.

Coarse sampling can miss structures between sampled vertices. Indicator
smoothing near one cell can retain roughness from the binary classification.
At widths comparable to the box, phase fractions and boundary extension can
dominate the result. Infer an area-versus-scale slope from an intermediate
range that is stable under changes in resolution.

## Cost and supported meshes

The output requires a uniform 3D Newtonian Cartesian mesh. Mesh refinement, shearing-periodic
boundaries and partial-mesh selections require separate geometric definitions
and are currently rejected. Existing marching-cubes utilities remain available
for problem-defined calculations.

Each Gaussian width visits all active cells. Indicator smoothing needs a
separate filtered field per level; field smoothing shares one filtered scalar
among its levels. Fine-grid extraction exchanges a diagnostic scalar halo and
reduces local cube areas on the device. Larger sampling strides request their
required vertices from the owning blocks and ranks. Wide Gaussian kernels use
distributed line transforms rather than a correspondingly wide 3D halo.

Measure cost at the intended resolution and cadence before selecting many
levels or smoothing widths. The diagnostic leaves the evolved fluid fields
unchanged.
