# Legacy marching cubes for problem generators

These headers restore the implementation from GitLab `radiative_shock` revision
`c98d90457c53e410f457de4c445b55263905c445`. The marching-cubes header at that
revision is identical to the version in `df31e686`.

This copy restores the production GPU behavior tested with RadShock on
Frontier. RadShock, Colliding_RadShock, and TRML problem generators can select
this backend with `utils/legacy_marching_cubes/marching_cubes.hpp`, `::Cube`,
and `::process_cube`. Branches without those problem generators carry the
shared utility for subsequent use. The generic `utils/marching_cubes.hpp` API
remains available for spacing-aware callers and its unit tests. Keep the legacy
value arguments, Cube layout, constructors/destructor, tables, and arithmetic
together when reproducing the RadShock results. TRML and Colliding_RadShock
have not been run through the full Frontier crash-reproduction tests.

Compatibility changes from the historical headers are limited to replacing the
unused, removed `athena_tensor.hpp` include with `<cfloat>`, unique include
guards, and preservation of the scikit-image license notice. The computation
matches the tested legacy build. This restoration also restores the historical
unguarded interior-vertex average; it does not add a new degeneracy fix.

## Frontier validation, September 11, 2026

All tests below use 64 nodes, 512 MPI ranks, production optimization settings,
and all outputs enabled, including expensive user history.

- Current implementation, job **5468842**: restarted at cycle 13000; failed after
  the last printed cycle 13740. Rank 380 reported the recurring GPU memory fault
  at `0x7ffec604a000`.
- Legacy implementation, job **5468834**: same cycle-13000 checkpoint and input;
  completed cycle 15000, time 0.1065710.
- Legacy implementation, job **5468846**: fresh start at time zero; completed
  cycle 15000, time 0.1065739, in about 578 seconds.

This comparison supports restoring the legacy RadShock path. It does not yet
identify the faulty GPU instruction or distinguish the refactored source from
its compiler-generated device code as the underlying mechanism.

The repository restoration compiles and links with the production configuration.
Its GPU `process_cube` function and all `UserHistOutput` kernels are byte-for-byte
identical to the successful legacy test build.
