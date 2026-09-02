# Marching-Cubes Area Utility

This note documents `marching_cubes`, a device-callable, header-only utility
for computing the area of an isosurface inside a single hexahedral cell
("cube") given the sampled scalar values at its eight corners. It implements
the topologically correct Marching Cubes 33 algorithm of Lewiner et al.
(2003), and is deliberately generic: it has no notion of a mesh, MeshBlock,
boundary, or reduction policy. Callers own their traversal and own combining
per-cell areas into a global quantity (e.g. an isosurface's total area,
reduced with `MPI_Allreduce`).

## Where the Code Lives

- `src/utils/marching_cubes.hpp` -- the `marching_cubes` namespace: `Cube`,
  `CellSpacing`, and `ComputeArea()`. Depends only on `athena.hpp` and
  `mc_luts.hpp`; safe to include from any translation unit and callable from
  a Kokkos device kernel.
- `src/utils/mc_luts.hpp` -- the algorithm's lookup tables (case/config/
  subconfig/tiling/test data), adapted from Thomas Lewiner's reference
  `LookUpTable.h`, imported through scikit-image v0.23.2 (BSD 3-Clause;
  license notice preserved in the file header).
- `CMakeLists.txt` -- optional `Athena_BUILD_MARCHING_CUBES_TEST` (default
  `OFF`) adds a standalone `marching_cubes_test` executable
  (`tst/unit_tests/marching_cubes.cpp`) that does not affect the normal
  `athena` target.
- `src/pgen/unit_tests/marching_cubes.cpp` -- a built-in regression pgen
  (`pgen_name = marching_cubes`) that exercises the utility end to end
  (Hydro primitive state -> corner sampling -> `ComputeArea` -> MPI-reduced
  total), used by `tst/inputs/ut_marching_cubes.athinput` and
  `tst/test_suite/unit_tests/test_marching_cubes_cpu.py`.

## API

```cpp
#include "utils/marching_cubes.hpp"

marching_cubes::CellSpacing spacing(dx1, dx2, dx3);   // defaults to (1,1,1)
marching_cubes::Cube cube(v0, v1, v2, v3, v4, v5, v6, v7, spacing);
Real area = marching_cubes::ComputeArea(cube);         // physical units
```

`v0`..`v7` are the sampled scalar **minus the isovalue** at the cube's eight
corners, in the fixed order `(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1),
(1,0,1), (1,1,1), (0,1,1)`. Values strictly below zero are inside the
surface; exact-zero samples follow the method's strict-sign tie-break
convention (an exactly balanced ambiguous face is not itself well-defined --
see "Common Gotchas"). `CellSpacing` carries the cube's three physical edge
lengths; a default-constructed `Cube`/`CellSpacing` gives a dimensionless
(unit-cube) area.

## Provenance

The tables and case-dispatch logic in `mc_luts.hpp`/`marching_cubes.hpp` are
a direct port of the published Marching Cubes 33 algorithm (Lewiner,
Lopes, Vieira & Tavares, 2003, "Efficient implementation of Marching Cubes'
cases with topological guarantees"), via the BSD-3-Clause-licensed copy in
scikit-image v0.23.2. The required scikit-image copyright/license text is
retained verbatim at the top of `mc_luts.hpp`.

This utility's original implementation in AthenaK is credited to Lachlan
Lancaster; this port brings that implementation into the public fork as a
generic, mesh-agnostic utility.

## Regression Pgen

`pgen_name = marching_cubes` builds a temperature field
`T = 10^(x1)` (so `log10(T) = x1` exactly, by construction) on a mesh with a
planar isosurface at `x1=0`, then reconstructs
`log10[(gamma-1) * w0(IEN) / w0(IDN)]` at each of the eight corners of every
transverse cell, calls `ComputeArea`, and MPI-reduces the total. It requires
an ideal EOS and a 3D mesh (`nx1 >= 2`), and exits with a fatal error
otherwise. Minimal input:

```ini
<mesh>
nx1 = 8      # x1min/x1max chosen so the T=1 (x1=0) plane sits inside the mesh
nx2 = 32
nx3 = 32
...

<hydro>
eos = ideal
rsolver = llf
...

<problem>
pgen_name = marching_cubes
```

The pgen prints `MARCHING_CUBES_TEMPERATURE_AREA PASS: area=... expected=...
abs_error=... tolerance=... ranks=...` (or `FAIL` with a nonzero exit code)
and compares against the analytic area `(x2max-x2min)*(x3max-x3min)`, since
the isosurface is exactly the x1=0 plane by construction. Each active
transverse `(j,k)` cell is owned by exactly one MPI rank; `evolution =
dynamic` with `nlim = 0` builds a valid Hydro state without advancing a
timestep (`evolution = static` is kinematic and rejects the LLF solver this
test uses).

## Validation

- Standalone `marching_cubes_test` (opt-in `Athena_BUILD_MARCHING_CUBES_TEST`
  target): empty/full cubes, axis-aligned planar cuts (both unit and
  physical spacing, cross-checked against the analytic area), a diagonal
  plane, ambiguous-complement invariance, and the same set of cases
  re-executed inside a `Kokkos::parallel_for` device kernel. Passed on
  serial-Kokkos CPU and on CUDA (A100, `AMPERE80`).
- The `marching_cubes` regression pgen: single-CPU-rank, two-node CPU/MPI
  (four ranks), single-GPU, and two-node GPU/MPI (two ranks) all reproduce
  the analytic area `15` (an anisotropic `Lx2=3, Lx3=5` domain) with
  **exact zero absolute error**, confirming both the device execution path
  and the cross-rank MPI reduction.

## Common Gotchas

- An **exactly** balanced ambiguous face (a bilinear saddle where the
  strict-sign tie-break is itself undefined) does not have a single correct
  triangulation -- construct test fixtures with a nondegenerate ambiguous
  face instead, as the standalone test does.
- `Cube`'s corner values are **already isovalue-subtracted**; the utility
  has no isovalue parameter of its own. If your target isovalue is exactly
  the value that makes the subtraction a no-op (as in the regression pgen's
  `Tiso = 1`, where `log10(1) = 0`), it's easy to forget the subtraction
  entirely -- make sure that's actually true for your isovalue before
  omitting it.
- The regression pgen requires an ideal EOS and `evolution = dynamic`
  (kinematic evolution rejects the LLF solver); both are checked at startup
  with a fatal error, not a silent fallback.
