# Cyclic zoom: scope and current limitations

These notes describe the cyclic-zoom implementation imported from upstream
AthenaK commit `c5a0d7f9155a70149931bf0be5a4ffb673f2532a`. They do not restrict
ordinary SMR/AMR, including this fork's particle redistribution.

## Supported scope versus explicit guards

Cyclic zoom requires a three-dimensional mesh and hydro, MHD and/or radiation.
Its constructor rejects particles, turbulence driving, self-gravity, ion-neutral
two-fluid MHD, Z4c, ADM, dynamical GRMHD and shearing-box configurations.
Fixed-background GR is allowed, as exercised by the Bondi and monopole examples.
Particle support in ordinary AMR does not imply particle support in cyclic zoom.
Porting particles to cyclic zoom is outside this merge's scope.

Radiation and passive-scalar infrastructure is present, but the upstream commit
description identifies these paths as untested. The merge validation does not
establish their correctness. See the
[upstream commit](https://github.com/IAS-Astrophysics/athenak/commit/c5a0d7f9155a70149931bf0be5a4ffb673f2532a).

## Restart limitations

Upstream documents that cyclic-zoom restart evolution need not exactly match an
uninterrupted run. In a short hydro test that stores a nonzero zoom zone, active
fields and stored data survived zero-evolution reload exactly, while subsequent
evolution differed by a maximum absolute density of about `1.64e-7`. This number
is fixture-specific, not a general accuracy bound. Zoom masking can modify active
cells after neighbor exchange; a restart refreshes neighboring ghost cells.
An exact-continuation diagnostic is therefore distinct from the stock physical
accuracy tests and should not be mislabeled as a stock regression failure.

**Do not use `single_file_per_rank=true` for cyclic-zoom checkpoints.** A
four-rank test fails while writing stored zoom data and reports a broken restart
file. The zoom read/write routines accept the per-rank flag but do not implement
the required separate-file I/O/metadata handling. This release does not add a
preventive guard or repair; use shared-file mode instead, with the continuation
limitation above. Ordinary per-rank fluid/particle restart support is separate.

Some high-magnetization GRMHD restart differences also occur without zoom. They
must not all be attributed to the zoom mask ordering or treated as a validated
exact-restart mode merely because the stock monopole observable test passes.

## Bondi callback repair and regression

The fork reconstructs the process-local Bondi boundary parameters before the
restart early return in `src/pgen/tests/gr_bondi.cpp`. Fluid initialization
remains fresh-run-only. This fixes a separate pre-existing restart issue; it
does not change the cyclic-zoom algorithm.

`tst/inputs/gr_bondi_restart.athinput` and the four
`tst/test_suite/gr/test_gr_bondi_restart_*.py` wrappers exercise a static-refinement
Bondi problem with user boundaries. They compare cycle-eight fluid fields with
continuation from cycle three, checking finite values, geometry, time and every
conserved field. The fixture deliberately does not enable cyclic zoom.

Full acceptance must identify the source and CPU/GPU executable hashes and
separate standard regressions, functional checks, accepted limitations and
unsupported combinations. Historical passes from another executable do not
constitute a complete final validation of a modified candidate.
