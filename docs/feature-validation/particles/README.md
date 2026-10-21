# Lagrangian tracer implementation and validation

Code reference: `886ab25ae0a58e4cbb12e44158987872eafcdf07`.
This report covers classical velocity-field, Monte Carlo (`lagrangian_mc`) and
continuous stochastic (`ito_2`) tracers on 2-D/3-D meshes. Numerical results are
listed by code revision. All five figures can be reproduced from the accompanying
data and plotting script.

## Implementation history and source map

The particle implementation starts from parent `62892d08`:

| Commit | Implementation change |
| --- | --- |
| `96f0e50f` | Storage, pushers, migration, initialization, output and paired restart |
| `69f0ea7d` | Conditional MC face sampling and isothermal particle-output guard |
| `743f3e45` | Shared particle initialization in turbulence |
| `735f7483` | Ito CIC moments, RK-aware saved flux, donor snapshot and timestep estimate |
| `1707b212` | Particle implementation comment cleanup |
| `8445dcc9` | Converging/diverging-flow diagnostic problem |
| `366a172a` | Boundaries, SMR routing, dynamic AMR transfer and checkpoint lifecycle |

[Source manifest](data/source_manifest.json) records full commit IDs, all 34
source/build paths, each changed diff-hunk header and current file SHA256.
Hunk coordinates refer to their individual commits; links below refer to the
documented revision.

| Source | Changed locations and purpose |
| --- | --- |
| [src/CMakeLists.txt](../../../src/CMakeLists.txt#L118) | Registers particle restart/binary output and AMR transfer translation units. |
| [src/athena.hpp](../../../src/athena.hpp#L72) | `ParticlesIndex`: position/velocity slots and last-move/last-level bookkeeping. |
| [src/particles/particles.hpp](../../../src/particles/particles.hpp#L28) | Type/pusher enums, storage/task APIs; `MeshRedistribution` (100) preserves records across topology changes. |
| [src/particles/particles.cpp](../../../src/particles/particles.cpp#L27) | Constructor parses aliases, restrictions, seeds and options; `ReallocateParticles()` (198), `CreateParticleTags()` (209) maintain storage/identity. |
| [src/particles/particles_tasks.cpp](../../../src/particles/particles_tasks.cpp#L28) | `AssembleTasks()` runs particles after fluid stages. `NewGID`, `SendCnt`, `InitRecv`, `SendP`, `RecvP`, `ClearSend`, `ClearRecv` (101–159) order migration. |
| [src/particles/particles_pushers.cpp](../../../src/particles/particles_pushers.cpp#L25) | Stateless RNG/displacement helpers; `Push()` (62), drift compatibility (91), `PushLagrangianTracer()` (133), `PushLagrangianMC()` (251), `PushIto2()` (415), `AdjustMeshRefinement()` (649), `NewTimeStep()` (950). |
| [src/particles/particles_amr.cpp](../../../src/particles/particles_amr.cpp#L30) | `RequireAll()`, `Bounds()` (45), `ChildDraw()` (71), `PrepareMeshRedistribution()` (86), `FinishMeshRedistribution()` (178): complete-record mapping/exchange and mass-weighted MC child selection. |
| [src/bvals/bvals_part.cpp](../../../src/bvals/bvals_part.cpp#L32) | Periodic-coordinate, bounded-neighbor and physical-boundary helpers (32/47/61); `UpdateGID()` (82), dormant deletion helper (105), `SetNewPrtclGID()` (118); count/receive/send/unpack/cleanup methods (441/507/593/683/808/835). |
| [src/hydro/hydro.hpp](../../../src/hydro/hydro.hpp#L110) | Saved density-flux/donor arrays, lazy enabling API and stage task. |
| [src/hydro/hydro.cpp](../../../src/hydro/hydro.cpp#L362) | Constructor state and `SetSaveUFlxIdn()` allocate only for MC/Ito consumers. |
| [src/hydro/hydro_fluxes.cpp](../../../src/hydro/hydro_fluxes.cpp#L247) | `Hydro::SaveFlux()` snapshots initial density and integrates fluxes with actual RK weights. |
| [src/hydro/hydro_tasks.cpp](../../../src/hydro/hydro_tasks.cpp#L61) | Registers the saved-flux task. |
| [src/mhd/mhd.hpp](../../../src/mhd/mhd.hpp#L157) | Corresponding MHD saved arrays/API. |
| [src/mhd/mhd.cpp](../../../src/mhd/mhd.cpp#L489) | Corresponding MHD lazy allocation. |
| [src/mhd/mhd_fluxes.cpp](../../../src/mhd/mhd_fluxes.cpp#L282) | `MHD::SaveFlux()` quadrature and initial donor-density snapshot. |
| [src/mhd/mhd_tasks.cpp](../../../src/mhd/mhd_tasks.cpp#L54) | Registers saved-flux work for particle consumers. |
| [src/pgen/pgen.hpp](../../../src/pgen/pgen.hpp#L41) | Particle restart constructor argument and shared initialization/restoration interfaces. |
| [src/pgen/pgen.cpp](../../../src/pgen/pgen.cpp#L44) | Initialization RNG; paired restart reader (144); `InitializeLagrangianParticles()` (1260), `InitializeParticlesFromRestart()` (1408). |
| [src/pgen/fluids/turb.cpp](../../../src/pgen/fluids/turb.cpp#L234) | Calls shared fresh-run particle initialization. |
| [src/pgen/tests/advection.cpp](../../../src/pgen/tests/advection.cpp#L22) | Opt-in alternating AMR callback; restart-safe enrollment (51), transverse flow and shared initialization (280). |
| [src/pgen/tests/converging_flow.cpp](../../../src/pgen/tests/converging_flow.cpp) | Converging/diverging-flow diagnostic fixture. |
| [src/main.cpp](../../../src/main.cpp#L168) | `-p` companion checkpoint, shared/per-rank naming and reader plumbing (363–393). |
| [src/mesh/mesh.hpp](../../../src/mesh/mesh.hpp#L142) | Particle counts, checkpoint phase, timestep and refinement-age state. |
| [src/mesh/mesh.cpp](../../../src/mesh/mesh.cpp#L727) | `FinalizeParticleDataStructures()` counts/tags; cleanup; `NewTimeStep()` particle constraint (658). |
| [src/mesh/mesh_refinement.cpp](../../../src/mesh/mesh_refinement.cpp#L431) | `RedistAndRefineMeshBlocks()` prepare/finish hooks (516/674); restored refinement ages (87). |
| [src/mesh/build_tree.cpp](../../../src/mesh/build_tree.cpp#L432) | Reads lifecycle metadata and validates restart timestep/refinement state. |
| [src/driver/driver.cpp](../../../src/driver/driver.cpp#L43) | `ReserveOutputCounters()`; initialization (476), execution (558), finalization (695) align paired checkpoints with AMR and next-step state. |
| [src/outputs/outputs.hpp](../../../src/outputs/outputs.hpp#L439) | Particle restart and binary-output classes. |
| [src/outputs/outputs.cpp](../../../src/outputs/outputs.cpp#L164) | Registers/validates `prst`/`pbin`, diagnostic defaults and writer construction (322/327). |
| [src/outputs/basetype_output.cpp](../../../src/outputs/basetype_output.cpp#L167) | Checks particle-variable availability. |
| [src/outputs/bin_prtcl.cpp](../../../src/outputs/bin_prtcl.cpp#L37) | Constructor, variable parsing (51), EOS-safe containing-cell sampling (152), serialization (373). |
| [src/outputs/res_prtcl.cpp](../../../src/outputs/res_prtcl.cpp#L43) | Constructor, data preparation (61), paired shared/per-rank checkpoint writing (99). |
| [src/outputs/checkpoint_metadata.hpp](../../../src/outputs/checkpoint_metadata.hpp#L24) | Version constants and `TopologyHash()` for checkpoint pairing. |
| [src/outputs/restart.cpp](../../../src/outputs/restart.cpp#L162) | Completed-phase guard and lifecycle timestep/refinement-age payload (270). |

### Transport, refinement and restart semantics

Initialization uses density-times-volume weights by default, or volume with
`uniform_by_volume=true`. Counts are stochastically rounded expectations. MC starts
at cell centers; classical/Ito can start within cells. Classical transport uses
CIC velocity and `x += dt*v`, not the finite-volume mass transfer.

MC divides saved face transfer (`flux*dt/dx`) by initial donor density and uses
sequential conditional face draws. The rejected-face transfer is subtracted from
the remaining donor budget. Tiny transfers below `1e-12` times donor density are
suppressed; budget saturation is a safeguard, not exact sampling in that regime.
Ito derives `cminus=p_right-p_left`, `variance=max(p_right+p_left-cminus^2,0)` and
CIC-interpolates these moments. The displacement is `dx*(cminus+sqrt(variance)*xi)`
with independent zero-mean, unit-variance uniform draws. Large-moment subdraws are
summed without intermediate migration. The optional timestep estimate is lagged.

SMR correction handles physical crossings of existing interfaces; dynamic AMR
handles changed topology. Classical/Ito coordinates stay fixed during AMR. MC
coarsens to the parent center and refines to child centers with probabilities
proportional to post-prolongation child mass. Complete records are exchanged;
particles are neither split nor duplicated. GPU density is packed contiguously
before host-staged redistribution.

Periodic positions wrap; reflecting positions mirror. Open exits are retained
as frozen records, not deleted or replaced by inflow injection. Paired checkpoints
are written after topology redistribution and next-step selection. Version/time/
cycle/topology metadata reject mismatches. `PLASTLEVEL` is reconstructed at reload
rather than serialized; MC remains cell-centered, classical/Ito continuous.

## Particle-specific tests

[Test results](data/acceptance.json) cover particle transport, boundaries, AMR
and checkpoints, with code revisions and result hashes.

| Revision/backend | Particle validation |
| --- | --- |
| Patch on `4baec90d`; CPU, single-host/two-node MPI-CPU and GPU | Uniform transport/restart regression and 20 particle-specific functional groups passed on each backend. |
| Patch on `0ad57060`, committed as `366a172a`; two-node MPI-GPU | 57 boundary cases, diode AMR, paired-checkpoint checks and MC mass-allocation tests passed. |

| Functional groups | What is tested |
| --- | --- |
| `regular3d`, `terminal` | Repeated topology changes; regular/terminal split, reload and resume |
| `outflow`, `reflecting`, `diode`, `mixed_x3` | Retained/frozen exits, valid active ownership and continuation |
| `mhd3level` | Three-level MHD hierarchy, including balance-induced refinement |
| `empty`, `cooldown`, `per_rank` | Zero global particles, refinement ages, rank-local files |
| `ito_high_cfl`, `mc_mass` | Larger tested Ito CFL; MC count/tag/centering and child-mass allocation |
| `uniform_x1`, `uniform_negative_x2`, `uniform_negative_x3` | Signed physical-boundary transport for all three pushers |
| `smr_positive_2d`, `smr_positive_3d`, `smr_negative`, `smr_zero` | Oblique, signed and zero transport across existing coarse/fine interfaces |
| `checkpoint_format` | Mismatched/malformed pairs, legacy reads, tighter timestep limits and output counters |

The [portable regression](../../../tst/test_suite/imported_features/scenarios.py#L179)
checks count/tags, finite/in-domain positions, classical translation, stochastic
mean displacement within 0.025, MC center phase within 1e-12 and exact sorted
MC/Ito restart arrays on a uniform mesh; the larger AMR campaign is recorded
separately above. [Checkpoint tests](data/checkpoint_contracts.json) reject
incorrect versions/topology and truncated headers. A tighter CFL reduces the
saved timestep from 0.15625 to 0.0625. Legacy-file tests cover reading; exact
continuation from these files remains untested.

Twelve positive-oblique SMR reload cases reconstruct `PLASTLEVEL` with a
one-level difference. Position, owner/tag/status and subsequent evolution
pass their checks; [the differences](data/reload_bookkeeping.json) are recorded
for each case.

## Numerical comparisons

### MC AMR child-mass allocation

![MC allocation](figures/mc_child_mass_allocation.png)

This test checks whether refinement assigns tracers in proportion to child-cell
mass. The [data](data/mc_child_allocation.json) cover 33,191 refined tracers,
16 parent cells and 64 child comparisons. For `n` tracers in a parent, the
simultaneous count bound is `sqrt(0.5*n*log(2*64/alpha))`, with `alpha=1e-6`.
All mass-weighted residuals lie within this bound; equal child probabilities
fail it. Reference masses are computed independently from child geometry and
post-prolongation density.

### Physical-boundary behavior

![Particle counts and frozen exits at physical boundaries](figures/boundary_contracts.png)

The [12 boundary tests](data/boundary_snapshots.json) check record conservation
and particle ownership. All retain 256 records with valid active owners.
Periodic and reflecting cases have no frozen exits. Final outflow/diode frozen
counts are 32 classical, 41 MC and 58 Ito, reflecting the different trajectories
of the three pushers.

### Dynamic AMR and restart

![AMR lifecycle](figures/amr_restart_contracts.png)

The [data](data/amr_restart.json) compare uninterrupted, split, reloaded and
resumed runs. The plotted uninterrupted evolution preserves 512 tracers through
repeated changes between 8 and 15 MeshBlocks; restart continuation checks also
pass. Comparisons account for whether a snapshot was written before or after
refinement, since the mesh can change at the same simulation time.

### Low-Mach gas/tracer spectra

![Gas and tracer spectra](figures/mach03_spectra.png)

This uniform-grid turbulence test compares tracer and gas density using the
implementation with CIC-interpolated Ito moments. The [spectra](data/mach03_spectra.csv)
and [metadata](data/mach03_metadata.json) describe `128^3` cells, 64 tracers/cell,
time 0.1 and measured RMS Mach about 0.315. The signed correlation is
`P_gt/sqrt(P_gg*P_tt)`. Mean response/correlation over `k=1–6` is
1.005933/0.998854 classical, 1.003406/0.986461 Ito and 0.991450/0.983874 MC.
High-wavenumber spectra become noise dominated. The response panel highlights
the low-wavenumber behavior; all 111 shells are available in the data. These
AthenaK test conditions differ from the published paper setups.

### Converging-flow resolution and seed dependence

![Converging flow](figures/converging_flow.png)

This resolution study measures errors near a stagnation point. The
[metrics](data/converging_flow_metrics.csv) cover 45 cases: `Nx=64,128,256`,
three pushers and five seeds. At time 0.3, the valid region is
`abs(x)<=0.85*0.5*exp(-0.3)`; profiles are separately mean-normalized there.
The central metric uses four cells. Error bars show sample standard deviations
across seeds. Central Ito error is nonmonotonic with resolution, so no fixed
convergence order follows from this study. It is an exploratory comparison with
no prescribed pass/fail tolerance. [Metadata](data/converging_flow_metadata.json)
give the measurement definitions.

## Resolved issues and limitations

Corrected defects include the isothermal particle-output read, MC conditional
sampling, RK-dependent saved-flux/donor normalization, invalid SMR neighbor
selection, topology-invalidated records, noncontiguous GPU AMR density staging
and incomplete checkpoint phase ordering. The tests above exercise these
corrections.

Coverage is limited to the tested 2-D/3-D configurations. Further validation is
needed for 1-D execution, automatic injection/deletion, arbitrary-distance
migration, repeated box wrapping, general shear remapping, single-precision
and cross-endian checkpoints, exact legacy AMR restart, multidimensional
statistics when transition probabilities saturate, arbitrary density-source
operators and bitwise reproducibility across hardware or mesh decompositions.

## Reproduction and references

```bash
python docs/feature-validation/particles/plot_results.py --check
python docs/feature-validation/particles/plot_results.py
```

Requires Python 3, NumPy and Matplotlib. `--check` verifies the recorded test
results, particle counts and owners, MC statistical bounds and spectral summaries.
The second command recreates all five figures from the bundled data.

- [Genel et al. (2013), *Following the flow*](https://arxiv.org/abs/1305.2195): mass-flux MC exchange and velocity-tracer limitations; the published moving mesh differs from this Cartesian mesh.
- [Moseley, Teyssier & Abel (2026), *Itô tracers*](https://arxiv.org/abs/2604.23041): continuous stochastic tracers, CIC moment interpolation and statistical diagnostics.
