# Lagrangian Tracer Particle Pushers in AthenaK

This note documents the Lagrangian tracer particle module and its three
interchangeable pusher (update-rule) choices: a flux-based stochastic
cell-jump method, a continuous stochastic-displacement method built from the
same flux moments, and a classical CIC-velocity tracer, plus the RK-integrator
support and optional accuracy constraint added alongside them.

- `pusher = lagrangian_mc`: flux-based stochastic cell-jump tracer, after
  Genel et al. 2013 (arXiv:1305.2195). After each hydro/MHD update, each
  particle uses the mass fluxes through the faces of its current cell to
  choose whether it jumps to a neighboring cell. In expectation, particles
  sample the same mass transport as the finite-volume fluid update.
- `pusher = ito_2` (aliases `ito2`, `ito`): a continuous Ito-2 tracer built
  from the first two moments of the same local flux transition kernel, after
  Moseley, Teyssier & Abel 2026 (arXiv:2604.23041). Rather than a discrete
  cell jump, the particle receives a continuous stochastic displacement whose
  mean and variance match the MC method's first two moments.
- `pusher = classical` (aliases `classical_lagrangian`, `lagrangian_tracer`):
  an ordinary velocity-field tracer, using cloud-in-cell (CIC) interpolation
  of the primitive velocity onto the particle position.

All three share the same particle storage, boundary-exchange, restart, and
output machinery, selected via `particle_type = mass_tracer` (`lagrangian_mc`
and `lagrangian_tracer` are also accepted, but `mass_tracer` is recommended:
`lagrangian_mc` implies the MC-specific pusher even when using
`classical`/`ito_2`, and `lagrangian_tracer` collides with the
`pusher = classical` alias of the same name — `mass_tracer` avoids both, and
describes what the three pushers have in common: mass-conserving tracer
particles, regardless of update rule) so they can be run and compared
side-by-side on identical setups.

## Where the Code Lives

Core particle module:

- `src/particles/particles.hpp` / `src/particles/particles.cpp`
  `Particles`: particle data arrays, particle types, pusher selection, task
  IDs, and the `<particles>` input block parsing.
- `src/particles/particles_tasks.cpp`
  Registers particle tasks in `after_timeintegrator`, after the fluid time
  integrator, so the completed fluid update and its saved mass fluxes (where
  needed) are available.
- `src/particles/particles_pushers.cpp`
  `Particles::PushLagrangianTracer()`, `Particles::PushLagrangianMC()`,
  `Particles::PushIto2()`, `Particles::AdjustMeshRefinement()`, and
  `Particles::NewTimeStep()` (the optional Ito-2 locality-dt constraint).
- `src/bvals/bvals_part.cpp`
  Particle GID updates and MPI exchange when particles cross
  MeshBlock/rank boundaries.

Restart and output support:

- `src/outputs/res_prtcl.cpp` — particle restart files, `file_type = prst`.
- `src/outputs/bin_prtcl.cpp` — particle diagnostic/analysis files,
  `file_type = pbin`.
- `src/outputs/outputs.cpp` — registers `prst`, `pbin`, `pvtk`, and `trk`.
- `src/pgen/pgen.cpp` — reads particle restart files when run with
  `-p <particle_restart_file>`; also `ProblemGenerator::InitializeLagrangianParticles()`,
  a reusable fresh-run initializer.

Example pgens using the initializer:

- `src/pgen/tests/advection.cpp` (square-wave/advection tests)
- `src/pgen/fluids/turb.cpp` (turbulence tests)

RK-integrator support used by `lagrangian_mc`/`ito_2` (see "RK-Integrator
Support" below):

- `src/hydro/hydro.cpp`/`.hpp`, `src/hydro/hydro_fluxes.cpp` and the MHD
  equivalents `src/mhd/mhd.cpp`/`.hpp`, `src/mhd/mhd_fluxes.cpp` — the
  `SaveFlux()` per-stage density-flux quadrature and the `u0idnsaved`
  start-of-step density snapshot.
- `src/mhd/mhd_tasks.cpp` — `MHD::CopyCons()`'s RK4(4)[2S] stage-register
  update, a general fix independent of particles (any nontrivial MHD+RK4
  evolution needs it; see that section for why).

## Particle State

For `particle_type = mass_tracer`, real particle data (indexed by
`src/athena.hpp`'s `ParticlesIndex` enum):

- `IPX`, `IPY`, `IPZ`: particle position.

None of the three tracer pushers use the particle velocity slots
(`IPVX`/`IPVY`/`IPVZ`) — those are only used by the separate `drift` pusher
(`particle_type = cosmic_ray`). The MC and Ito-2 pushers compute displacement
from fluid mass fluxes; the classical pusher interpolates the fluid velocity
field directly and applies it explicitly.

Integer particle data:

- `PGID`: global MeshBlock ID currently owning the particle.
- `PTAG`: persistent particle tag (used as one input to the deterministic
  random draws below).
- `PLASTMOVE`: status and, for the MC pusher, last-move encoding (see below).
  `>= 0` active; `-1` frozen (crossed a user boundary); `-2` reserved for a
  not-yet-implemented deletion marker.
- `PLASTLEVEL`: refinement level of the particle's previous cell, used by
  `AdjustMeshRefinement()`.

## Pusher Choices and Update Methods

```ini
<particles>
particle_type = mass_tracer
pusher        = classical   # or lagrangian_mc, or ito_2
ppc           = 1.0         # initial per-MeshBlockPack array sizing
target_count  = 100000      # actual fresh-run particle count target
```

`particle_type` selects the shared storage/boundary/restart/output layout
used by all three variants (see "Where the Code Lives" above for why
`mass_tracer` is recommended over the two other accepted spellings). `ppc`
(particles-per-cell) sizes the initial per-pack particle array allocation;
`target_count`, read separately by `InitializeLagrangianParticles()`, is what
actually determines the fresh-run global particle count (see "What a Problem
Generator Must Do").

### Classical Velocity-Field Tracer

`pusher = classical` treats particles as passive points advected by the
cell-centered primitive velocity field. `PushLagrangianTracer()` CIC-
interpolates `w0(IVX/IVY/IVZ)` onto the particle position and applies
`x_new = x_old + dt*v_interp`. It does not request saved mass fluxes and sets
`PLASTMOVE = 0` (it doesn't use the MC face/parity encoding below). It is not
expected to exactly reproduce finite-volume density transport — the fluid
density is updated by face-integrated mass fluxes, while these particles move
by an interpolated velocity field. That mismatch is exactly what the Ito-2
and MC methods are designed to avoid.

### Lagrangian Monte Carlo Tracer

`pusher = lagrangian_mc` uses the saved density fluxes from the
just-completed fluid update (see "RK-Integrator Support"). For each active
particle, `PushLagrangianMC()`:

1. Locates the particle's owner MeshBlock and active cell.
2. Reads the start-of-step donor mass from `u0idnsaved(m,k,j,i)`.
3. Reads the saved density fluxes on each face in each active direction,
   keeping only outward flux (inflow is clamped to zero for this cell's
   transition probability) and normalizing by the donor mass.
4. Draws one deterministic uniform deviate from `PTAG`, `ncycle`, and
   `random_seed` (see "Restart-Safe Randomness").
5. Compares the draw against the cumulative outward face probabilities and
   moves the particle by exactly one local cell width in the selected
   direction, or leaves it in place if the draw falls outside the outgoing
   probability sum.
6. Encodes the current cell parity and selected face in `PLASTMOVE`
   (`1`-`6` = left/right `x1`/`x2`/`x3` face; parity bits
   `32*(i%2) + 16*(j%2) + 8*(k%2)` packed into the high bits), and the
   current refinement level in `PLASTLEVEL`.

`AdjustMeshRefinement()` uses that packed `PLASTMOVE` state when a jump
crosses a coarse/fine boundary — this correction is MC-specific, since only
the MC pusher represents motion as a discrete cell-center-to-cell-center jump.

### Ito-2 Tracer

`pusher = ito_2` converts the same left/right outward transition
probabilities into a continuous stochastic displacement per active direction:

```text
dx * (cminus + sqrt(variance) * xi)
cminus   = p_right - p_left
variance = max(p_left + p_right - cminus^2, 0)
```

`cminus` is the mean displacement (in cell-width units) of the corresponding
MC left/stay/right step; the square-root term supplies the matching variance.
`xi` is a stateless uniform deviate rescaled to zero mean, unit variance
(`sqrt(3)*(2*u - 1)`), with independent streams per coordinate.

Per Moseley, Teyssier & Abel 2026 Sec. 3.1, `PushIto2()` **CIC-interpolates**
these moments onto the particle's continuous position, rather than using a
single nearest-grid-point cell (which the paper notes is "much noisier by
nature"). Concretely, for each of the (up to 8) surrounding cells:

1. Compute that cell's `cminus`/`cplus`/`variance` from its own saved fluxes
   and `u0idnsaved` donor mass (skipping any cell whose donor mass is
   `<= 0.0` — unreachable in practice, since AthenaK's density floor keeps
   `u0idnsaved` strictly positive in any valid simulation state).
2. Accumulate the CIC-weighted sum of these **already-derived moments**
   across valid corners — not the raw face probabilities, since interpolating
   before the nonlinear `cminus^2` step would be a different, incorrect
   quantity.

The lower corner is clamped to `[is, ie-1]` (one cell tighter than the
classical pusher's `[is-1, ie]`), since these moments come from
`u0idnsaved`/saved fluxes, which are only guaranteed valid on the active
zone, unlike cell-centered primitives that extend one ghost cell further.

Because a near-vacuum donor cell can leave `p_left`/`p_right` well outside
`[0,1]` (the hydro CFL condition only bounds the net six-face flux
divergence, not any single face's flux/mass ratio in isolation), a single
unbounded displacement could push a particle to an undefined cell index.
`PushIto2()` guards against this by sub-cycling: it splits the total
`cminus`/`variance` (computed once, from the unscaled moments) into `nsub`
equal shares when `Cplus > 1`, rather than rescaling probabilities and
recomputing moments per sub-step (which would inflate the summed variance).
Each active direction draws independent stateless `xi` values.

Ito-2 particles do not run `AdjustMeshRefinement()` — they are continuous
displacements, not discrete cell-center jumps, and don't store a
last-crossed-face state. Ordinary `NewGID()`/boundary-exchange tasks handle
MeshBlock/rank reassignment after the update.

## RK-Integrator Support

`lagrangian_mc`/`ito_2` need the fluid's density flux integrated over the
**full timestep**, not just the flux from whichever RK stage happens to run
last. `Hydro::SaveFlux()`/`MHD::SaveFlux()` (registered in the `stagen` task
list, a no-op unless a mass-conserving pusher calls
`SetSaveUFlxIdn()` on construction) accumulate each stage's density flux with
an **exact quadrature weight**, derived from the driver's actual per-stage
register recurrence (`gam0`/`gam1`/`beta`/`delta`) rather than assuming a
fixed 2-stage (`dt/2`-per-stage) integrator. This makes it correct for
RK1-RK4, not just RK2.

For RK4(4)[2S] specifically, the driver's `u1` register is legitimately
repurposed as a stage accumulator after the first stage (see
`Hydro::CopyCons()`/`MHD::CopyCons()`), so it no longer holds the
start-of-step density the way it does for RK1-RK3. `SaveFlux()` therefore
also snapshots the true stage-1 density into a separate `u0idnsaved` buffer,
and both pushers read their donor mass from that buffer rather than `u1`
directly — this is what makes the transition-probability normalization
correct regardless of integrator.

**`MHD::CopyCons()`'s RK4 register fix is a separate, general prerequisite**,
not specific to particles: `Hydro::CopyCons()` already applied the RK4(4)[2S]
`u1 += delta*u0` accumulator update for `stage > 1`, but `MHD::CopyCons()`
only copied `u0`->`u1`/`b0`->`b1` at stage 1, with no RK4 handling at all — a
pre-existing AthenaK bug (confirmed against Athena++'s reference RK4(4)[2S]
implementation) that made any nontrivial MHD+RK4 evolution unstable
(reproducibly, a timestep collapse from `dt ~ 1.5e-3` to `dt ~ 1e-25` after
the first step), independent of whether particles are even present. Without
this fix, MHD+RK4 saved-flux particle tracking cannot be validated, since the
underlying fluid evolution itself is broken.

## Optional Ito-2 Locality Timestep Constraint

Moseley, Teyssier & Abel 2026 Eq. 63 gives a stronger, *accuracy*-motivated
locality bound on Ito-2's drift, `|u_i|*dt/h < 1/4`, in addition to the
`Cplus > 1` *stability* guard `PushIto2()` already enforces unconditionally.
The paper itself notes enforcing it "makes little difference to the
results," so it is off by default:

```ini
<particles>
ito2_enforce_locality_dt = true   # default: false
```

When enabled (`ito_2` only; no effect for any other pusher),
`Particles::NewTimeStep()` evaluates `dt_new < dt/(4*max|cminus_i|)` from the
same per-cell saved fluxes `PushIto2()` reads, using the un-interpolated
(host-cell) `cminus_i` — deliberately not CIC-smoothed, since this sets one
scalar `dt` for the whole domain.

## Restart-Safe Randomness

The stochastic pushers use a **stateless** random draw: it depends only on
the particle tag, cycle number, input seed, and (for Ito-2) a coordinate
stream index — never on thread scheduling or random-pool consumption order,
so it reproduces exactly across a restart. MC draws one value from
`(PTAG, ncycle, random_seed)`; Ito-2 draws independently per coordinate using
stream IDs `1`/`2`/`3`; `AdjustMeshRefinement()`'s extra MC draw uses
`random_seed + 1` so it never collides with the main pusher draw.

`random_seed` defaults to a fixed constant (`-1`), the same on every rank,
rather than a rank/GID-varying value: each draw already hashes in the
particle's own globally unique `PTAG`, so a rank-varying base seed would add
no real decorrelation, and it would actively break restart-exactness — a
restart's `ParameterInput` block embeds one already-resolved value that every
rank then reuses, so a rank-varying *default* would diverge from what a
continuous run computes fresh per rank. Only set `random_seed` explicitly if
you need a value different from the default.

## Boundary Exchange

After the pusher moves particles, `src/bvals/bvals_part.cpp`:

1. `SetNewPrtclGID()` checks whether each particle crossed a MeshBlock
   boundary.
2. Particles destined for another MPI rank are queued for send.
3. `CountSendsAndRecvs()`/`InitPrtclRecv()`/`PackAndSendPrtcls()`/
   `RecvAndUnpackPrtcls()`/`ClearPrtclRecv()`/`ClearPrtclSend()` exchange
   particle data.
4. Local particle arrays are resized and compacted.

Particles that cross a user boundary are frozen (`PLASTMOVE = -1`). Periodic
boundaries wrap positions across the global mesh extent.

## What a Problem Generator Must Do

The particle module allocates arrays, selects the pusher, and handles
movement; a problem generator still owns initial particle placement for a
fresh run:

```cpp
if (restart) return;   // particles are read from -p <file> on restart, not regenerated
...
InitializeLagrangianParticles(pin, u0_);
```

The reusable initializer computes the domain-integrated mass (or volume, with
`uniform_by_volume = true`) and divides by `target_count` to get an expected
per-cell particle count, using stateless draws (`pos_init_seed`) for
fractional counts and in-cell positions. It then calls
`ReallocateParticles(nparticles_thispack)` and fills `PGID`/`PLASTLEVEL`/
`IPX`/`IPY`/`IPZ`; `Mesh::FinalizeParticleDataStructures()` assigns tags via
`Particles::CreateParticleTags()` afterward. A new pgen not using the shared
initializer should follow the same pattern, explicitly setting
`PLASTMOVE = 0` for fresh active particles (negative values stay reserved
for frozen/deletion-marked particles).

## Particle Output and Exact Restarts

`file_type = pbin` (`src/outputs/bin_prtcl.cpp`, written under `pbin/` as
`.prtclbin`) writes particle positions, integer data (`PGID`/`PTAG`/
`PLASTMOVE`), and grid quantities sampled at particle positions (the
`variable` key selects which fields, falling back to a default set if unset).

Exact restarts need **both** a fluid restart (`file_type = rst`) and a
particle restart (`file_type = prst`, `src/outputs/res_prtcl.cpp`, written
under `prst/` as `.prtclrst` — magic number `42`, particle count, then
`PGID`/`PTAG`/`PLASTMOVE`/`IPX`/`IPY`/`IPZ`; random seeds are not stored, since
they're recomputed deterministically from `PTAG`/`ncycle`/`random_seed`):

```bash
./athena -r rst/Problem.00010.rst -p prst/Problem.00010.prtclrst -i athinput
```

If particles are enabled and a fluid restart is run without `-p`, the
restart constructor exits with an error — a statistically regenerated
particle set is not an exact restart, so this is treated as a hard error
rather than silently falling back to it. Keep `rst`/`prst` at the same output
cadence, and don't change mesh decomposition, AMR settings, particle count,
or pgen placement settings across a restart (the particle restart reader
maps particles back to ranks by `PGID`, which changing the mesh layout can
disturb).

## Validation

Extensively cross-checked across CPU, MPI-CPU, and GPU builds: stock
regression suites; a 12-leg RK1/RK3/RK4 x Hydro/MHD x MC/Ito-2 saved-flux
matrix; single- and two-node GPU particle restart and migration checks
(exact bitwise match); a two-node rank-local (`single_file_per_rank`)
restart check (exact match); square-wave/advection and driven-turbulence
paper-comparison suites (against Genel et al. 2013 and Moseley, Teyssier &
Abel 2026); a Mach-0.315, 64-particles/cell case with CIC-deposited
gas-tracer cross-spectra; and a resolution/seed study on a converging-flow
setup targeting the exact stagnation-point regime the CIC interpolation was
built for (mean error 18.6% -> 1.3% after CIC vs. nearest-grid-point).

## Common Gotchas

- Missing `<particles>` block: no particle object is constructed, so pgen
  particle initialization is silently skipped.
- `particle_type = mass_tracer` with `pusher = drift`: the constructor
  exits — `drift` is reserved for `particle_type = cosmic_ray`.
- Fluid restart without `-p` when particles are enabled: hard error by
  design (see "Particle Output and Exact Restarts").
- Mismatched `rst`/`prst` cadence: the fluid and particle states come from
  different cycles, so the restart is not exact.
- MHD + `integrator = rk4`: needs the `MHD::CopyCons()` register fix (see
  "RK-Integrator Support") to be stable at all, independent of particles —
  without it, a nontrivial MHD+RK4 run diverges within one step regardless
  of whether particles are present.
- `ito2_enforce_locality_dt = true`: off by default; per the paper, expect
  it to make little practical difference, and it only affects `ito_2`.
