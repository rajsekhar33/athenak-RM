# Lagrangian Tracer Particles in AthenaK

The module requires a 2-D or 3-D mesh. Its three tracer pushers share storage,
migration and restart machinery:

| `pusher` | Update |
| --- | --- |
| `classical` | Explicit displacement from CIC-interpolated fluid velocity |
| `lagrangian_mc` | Stochastic face selection and discrete cell-center jumps |
| `ito_2` | Continuous stochastic displacement from interpolated flux moments |

The MC approach follows [Genel et al. (2013)](https://arxiv.org/abs/1305.2195);
the Ito method is motivated by
[Moseley, Teyssier & Abel (2026)](https://arxiv.org/abs/2604.23041).

## Configuration and Fresh Initialization

```ini
<particles>
particle_type = mass_tracer
pusher = classical             # or lagrangian_mc, ito_2
ppc = 1.0                      # preliminary per-pack allocation
target_count = 100000           # expected global count, not an exact quota
uniform_by_volume = false      # true selects volume instead of density weighting
random_positions = true
pos_init_seed = 280496
random_seed = -1
ito2_enforce_locality_dt = false
```

`particle_type = mass_tracer`, `lagrangian_tracer`, and `lagrangian_mc` are
equivalent storage-type spellings. **The separate `pusher` selects the
algorithm**; the type alias does not force MC. The name `mass_tracer` does
not imply that classical tracers reproduce the fluid mass distribution.

Pusher aliases are `classical_lagrangian`/`lagrangian_tracer` for
`classical`, and `ito2`/`ito` for `ito_2`. `drift` belongs to the
separate `cosmic_ray` type and is rejected for tracers.

The constructor uses `ppc` for preliminary allocation. A problem generator
must initialize the actual fresh-run population, for example by calling
`ProblemGenerator::InitializeLagrangianParticles(pin, u0)` after initializing
the fluid. Advection and turbulence use this helper, which returns immediately
if no particle object exists.

The helper uses cell weights `density * volume` by default, or `volume`
when `uniform_by_volume = true`. It distributes `target_count` in expectation,
using integer counts plus deterministic stochastic rounding in each cell.
The realized count can differ from the target. Use a positive target and
physically valid weights; a nonpositive total weight is rejected.
Uniform-by-volume placement in a nonuniform-density flow is not initially
an equal-weight sample of gas mass.

For classical/Ito tracers, `random_positions = true` places particles within
their assigned cells; false places them at centers. **MC particles always
start at cell centers**, regardless of this flag. Initialization draws use
`pos_init_seed` and cell/particle indexing. Changing the MPI decomposition can
change the fresh population. Default
`assign_tag = index_order` assigns persistent tags after initialization.
Restart loading restores particles instead of rerunning this initializer.

For regression fixtures, the advection pgen accepts opt-in
`problem/transverse_velocity` (default 0) and `problem/particle_amr_period`
(default 0, disabled). A positive period installs an alternating refinement
callback, including on restart; it targets the lower-coordinate half in each
active direction. The advection fixture requires an isothermal EOS. Custom
pgens must likewise enroll any restart-needed callbacks before returning
early from their restart path.

Sources: [constructor and tags](../src/particles/particles.cpp),
[initializer and restart reader](../src/pgen/pgen.cpp).

## Particle State and Physical-Step Updates

Tracers store three real values, `IPX/IPY/IPZ`, and four integers:

- `PGID`: owning global MeshBlock ID.
- `PTAG`: persistent identity and random-draw input.
- `PLASTMOVE`: nonnegative for active particles; MC also packs previous
  face/parity here. `-1` freezes transport; `-2` is reserved for deletion,
  which is not implemented.
- `PLASTLEVEL`: previous-owner refinement-level bookkeeping used by MC's
  coarse/fine correction. Restart reconstructs it from the current owner.

Tracers do not store independent velocities. They advance once in
`after_timeintegrator`, after the fluid stages, followed by owner/rank
exchange. Fluid RK order is not the order of the classical particle update,
which remains `x += dt * v_interp`.

### Classical

`PushLagrangianTracer()` CIC-interpolates cell-centered primitive velocity,
using surrounding centers and available ghost cells near a block edge.
It advances active coordinates explicitly, sets `PLASTMOVE = 0`, and records
the donor level. It does not request saved mass fluxes. Velocity interpolation
is generally not equivalent to the finite-volume fluid mass update.

### Monte Carlo

`PushLagrangianMC()` uses `u0idnsaved` (start-of-step density) and
`uflxidnsaved` (RK-weighted face mass flux multiplied by `dt/dx`).
Dividing these density-transfer contributions by donor density gives
dimensionless probabilities; the common cell volume cancels.

For each active particle:

1. Locate the donor cell; skip the update for an out-of-active-zone index or
   nonpositive donor density.
2. Keep positive outward contributions in face order
   `-x1,+x1,-x2,+x2,-x3,+x3`. Contributions smaller than
   `1e-12 * donor_density` are set to zero.
3. Initialize a remaining-density budget to donor density. For each active,
   nonzero face, compare a face-specific deterministic draw with
   `min(outward_contribution / remaining_budget, 1)` (or 1 if exhausted).
   On rejection, subtract that contribution before trying the next face.
4. Take the first accepted jump by one donor-cell width, or remain in place.
   Save donor level and face/parity encoding.

This sequence implements conditional sampling. When total outward transfer
fits the donor budget, it realizes the face probabilities up to the small-flux
cutoff. When outgoing mass exceeds the donor budget, the probabilities saturate
and the transitions can depart from the fluid flux ratios. Finite particle
populations also have sampling noise.

`PLASTMOVE` encodes face 0 (stay) or 1--6 (the ordered faces), plus
`32*(i%2) + 16*(j%2) + 8*(k%2)` using donor array indices.

### Ito-2

For each surrounding active cell and coordinate, the code derives outward
probabilities from saved fluxes and donor density, then forms:

```text
cminus = p_right - p_left
cplus = p_right + p_left
variance = max(cplus - cminus*cminus, 0)
displacement = dx * (cminus + sqrt(variance) * xi)
xi = sqrt(3) * (2*u - 1)
```

It CIC-interpolates already-computed `cminus`, `cplus`, and `variance`,
not raw probabilities. Unlike the classical stencil, both corners stay
inside the block's active cells: the lower x1 index is clamped to
`[is, ie-1]`, with analogous bounds in other active coordinates and endpoint
weights at edges. Nonpositive-density corners are skipped without
renormalizing remaining weights; no valid corner means no update. This
guard skips the affected corners; the fluid state still requires positive density.

If the largest interpolated `cplus` exceeds 1, the code uses
`nsub = min(ceil(max_cplus), 1000)` draws per direction, dividing original
mean and variance by `nsub` for each draw. **It sums the displacements and
updates position only once.** It does not re-sample moments or migrate between
sub-draws. Moment splitting preserves the prescribed directional mean and variance, while
the summed displacement can still cross beyond neighboring blocks. Directional
Ito moments and the coupled MC face-selection probabilities can differ when
the MC donor budget saturates.

Source for all three: [particle pushers](../src/particles/particles_pushers.cpp).

## RK Fluxes and Optional Ito Timestep Estimate

MC/Ito construction enables `Hydro::SaveFlux()` or `MHD::SaveFlux()`.
Each stage contributes density flux with weights derived from the driver's
`gam0/gam1/beta/delta` register recurrence for the implemented RK1--RK4
schemes. Stage 1 separately snapshots donor density because RK4 subsequently
repurposes `u1` as an accumulator. These are saved advective density-flux
contributions, not a general representation of arbitrary density source
terms or every operator-split process.

MHD's `CopyCons()` also maintains RK4 conserved and face-field accumulators
after stage 1 as part of the fluid integrator.

With `ito2_enforce_locality_dt = true`, the post-integrator particle task
computes `dtnew = dt_current / (4 * max_abs_cminus)` over active cells and
coordinates when the denominator is positive. `Mesh::NewTimeStep()`
includes it in global timestep selection. Other pushers, or the default
false setting, leave the particle constraint inactive.

This lagged accuracy estimate constrains the next step. The completed step
remains accepted, and individual stochastic displacements can exceed the mean
drift estimate. Dynamic refinement
retains the global old-mesh estimate and tightens it by
`2^(-largest_level_increase)`, without interpreting old flux arrays with new
block indices. Changed future fluxes can invalidate the prediction.

Sources: [Hydro flux saving](../src/hydro/hydro_fluxes.cpp),
[MHD flux saving](../src/mhd/mhd_fluxes.cpp),
[MHD registers](../src/mhd/mhd_tasks.cpp),
[particle task ordering](../src/particles/particles_tasks.cpp),
[mesh timestep](../src/mesh/mesh.cpp).

## Boundaries and Static Mesh Refinement (SMR)

After a push, `SetNewPrtclGID()` handles physical faces and searches the
crossed neighbor interface. Remote owners trigger normal particle MPI
exchange. Lookup searches bounded subface/subedge groups, prefers explicit
edge/corner neighbors, and uses coarse face/edge fallbacks where appropriate.
A missing valid crossed-interface neighbor aborts.

| Physical boundary | Tracer behavior |
| --- | --- |
| Periodic | Wrap by one domain length; roundoff corrections preserve half-open ownership `[lo, hi)` |
| Reflecting | Mirror overshoot inside; MC wall crossings should already be suppressed by wall mass flux |
| Outflow, diode, inflow, vacuum, user | Freeze an exiting particle (`PLASTMOVE = -1`); retain its record, without injecting replacements |

These rules govern existing tracers; fluid inflow does not automatically
create particles. Frozen particles skip subsequent physical pushes and
migration. Existing user-face/excision checks can also freeze particles.
Filter inactive records when measuring an in-domain population.

Migration follows neighboring-block interfaces. Wrapping and reflection handle
one domain crossing per update. Shear-periodic handling applies coordinate
wrapping only; the particle shear-offset and velocity remap are unsupported.

For MC only, `AdjustMeshRefinement()` runs after physical migration across
an existing coarse/fine interface. It uses donor face/parity/level information
to restore cell-center placement; fine-side selection uses saved interface
fluxes. Classical/Ito positions remain continuous and skip this correction.
This operation is distinct from changing mesh topology.

Sources: [boundary routing](../src/bvals/bvals_part.cpp),
[MC coarse/fine correction](../src/particles/particles_pushers.cpp).

## Dynamic AMR and Rank Redistribution

`MeshRefinement::RedistAndRefineMeshBlocks()` brackets topology replacement
with `PrepareMeshRedistribution()` and `FinishMeshRedistribution()`:

1. Deep-copy real/integer records before old arrays and ownership become
   invalid. Map an old leaf to the same leaf, its immediate parent, or its
   immediate children, then select destination rank and GID.
2. After new fluid data, coordinates and neighbors exist, exchange full records
   and donor levels with host-staged MPI counts and `Alltoallv`. Zero-particle
   ranks participate. CUDA MC density is packed into a contiguous device array
   before copying to host.
3. Restore arrays and rank/global counts; validate active ownership, supported
   count ranges and unchanged global particle count.

Classical/Ito positions are unchanged by this transfer. Active MC particles
coarsening to a parent are placed at its cell center. On refinement, each MC
particle selects one of the old cell's 4 (2-D) or 8 (3-D) fine children with
probability proportional to **post-prolongation child mass**, then moves to
that child's center. Equal Cartesian child volumes cancel from mass ratios.
Invalid/negative child densities or a nonpositive total are rejected.
Particles are not split or duplicated: stochastic allocation conserves count,
not an exact quota in each child.

Active tracer levels are updated; MC's last-move encoding is cleared because
topology transfer is not a flux crossing. Inactive records retain coordinates
and negative status. A clamped temporary position selects a new storage owner
without moving an escaped record back into the domain.

The transfer handles one-level changes per event on the supported 2-D/3-D
Cartesian hierarchy, using the host-staged exchanges described above.

Sources: [particle AMR transfer](../src/particles/particles_amr.cpp),
[mesh lifecycle hooks](../src/mesh/mesh_refinement.cpp).

## Deterministic Random Draws

Physical pushers use stateless hashes, with `random_seed = -1` on every rank
by default. For a fixed tag, cycle and seed:

- MC face draws additionally use face index 0--5.
- Ito draws use stream `direction * 10000 + subdraw`, for directions 1--3.
- MC's existing-interface correction uses a separate `random_seed + 1` draw.
- Dynamic-AMR child selection uses a dedicated hash of seed, tag, cycle,
  donor level and global parent-cell coordinates.

These stateless draws are independent of rank and array order. Floating-point
reductions, mesh histories and fresh initialization can still change the evolved
solution across decompositions or hardware. Preserve seeds and pusher settings
on restart.

## Outputs and Paired Checkpoints

`file_type = pbin` writes `.prtclbin` diagnostics under `pbin/`: all three
real and four integer tracer arrays (including `PLASTLEVEL`), plus grid
fields selected by `variable`. Grid fields use containing-cell sampling,
with indices clamped to the stored owner's active cells, not the classical
pusher's CIC interpolation. Values attached to escaped/frozen records are
not physical measurements outside the domain.

Continuation with particles requires both `rst` and `prst`:

```bash
./athena -r rst/Problem.00010.rst -p prst/Problem.00010.prtclrst
```

Use the actual filenames generated by your basename/output ID. Missing `-p`
with enabled particles is an error. Keep both outputs on the same schedule.
The driver defers due `rst`/`prst` writes until after AMR, particle
redistribution, next-timestep selection and STS-state refresh. Ordinary
analysis outputs stay before AMR during the cycle; equal time labels do not
guarantee identical topology between an analysis dump and a checkpoint.

The fluid restart carries a versioned lifecycle extension: previous and last
completed timesteps, AMR/load-balance sequence, and adaptive refinement ages.
The base header carries the selected next timestep. Output counters in embedded
parameters are reserved before a checkpoint batch. Restart initialization
reapplies tighter current timestep constraints without exceeding the saved
next timestep, restores `dtold`, and refreshes STS state. This fluid-checkpoint
lifecycle change also applies to runs without particles.

### Particle checkpoint version 1

The header is 64 bytes: six 64-bit integer values followed by two doubles:

```text
magic=44, particle_count, version=1, cycle, global_block_count, topology_hash,
time, selected_next_dt
```

The payload is six arrays of doubles: `PGID`, `PTAG`, `PLASTMOVE`, `x`,
`y`, `z` (integer fields are encoded as doubles). Count is global for shared
files and local for per-rank files. The reader uses `Real`-sized payload reads,
so this format requires a double-precision build with the same byte order.

The reader rejects mismatched version, cycle, time, next timestep, block count
or topology hash. The hash covers geometry, block dimensions and ordered logical locations.
Rank assignment is excluded, permitting redistribution on restart. Checkpoint
pairing uses this topology hash; payload integrity and physics settings require
separate checks.

`PLASTLEVEL` is not serialized in `prst`: it is reconstructed from the current
owner. MC reload restores cell-center placement for in-block particles;
classical/Ito coordinates are retained, and out-of-owner/excised positions
are frozen. Exact continuation therefore need not mean every bookkeeping
integer matches immediately after reload. Preserve type, pusher, seeds,
physical setup and mesh/AMR policy when comparing continuation.

Shared files allow reassignment by GID to a different MPI decomposition of
the same checkpoint mesh. Preserve the checkpoint topology in the input.
Per-rank files require compatible rank ownership and the complete rank-file set.
Changing rank count can change floating-point reduction order during evolution.

Legacy magic-42 particle files remain readable **with legacy fluid files**.
Mixed new/legacy pairs are rejected. Legacy fluid files lack lifecycle state,
so continuation across topology changes can differ from an uninterrupted run.
Version-1 files require an executable that supports that format.

Sources: [diagnostic output](../src/outputs/bin_prtcl.cpp),
[particle writer](../src/outputs/res_prtcl.cpp),
[reader and reconstruction](../src/pgen/pgen.cpp),
[metadata definitions](../src/outputs/checkpoint_metadata.hpp),
[fluid writer](../src/outputs/restart.cpp),
[fluid reader](../src/mesh/build_tree.cpp),
[driver ordering](../src/driver/driver.cpp).

## Verification

The [particle advection input](../tst/inputs/public_main_particles.athinput)
tests classical, Monte Carlo and Ito-2 tracers in a periodic two-dimensional
flow with uniform velocity `vx = 0.1`. The expected x displacement is `0.1*t`.
The [particle checks](../tst/test_suite/imported_features/scenarios.py) verify:

- particle count and tags are preserved, coordinates remain finite and inside
  the periodic x1/x2 domain;
- classical particles follow the expected displacement, while the mean Monte
  Carlo and Ito-2 displacements agree within the test's sampling allowance;
- Monte Carlo particles remain at cell centres along the advected direction;
- paired fluid/particle restarts for Monte Carlo and Ito-2 reproduce the
  uninterrupted integer, position and sampled-grid arrays in this setup.

These cases are included in the
[imported-feature CPU, GPU and MPI test modules](../tst/test_suite/imported_features).
