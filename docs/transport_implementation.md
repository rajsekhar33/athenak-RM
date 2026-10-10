# Thermal transport: implementation and changes from main

Reviewed 2026-10-10. This is the implementation companion to
[thermal_conduction.md](thermal_conduction.md), which documents configuration,
validation details and supported scope.

## Comparison baseline and scope

The comparison is against public GitHub `main` at
`588d737fac0251b5538c287cf50c3ccdd4cad706`, verified on the review date. The
feature branch is `feature/anisotropic-conduction`. This is **not** a comparison
against an unspecified current upstream revision or a problem branch.

The feature was developed and validated privately at checkpoint
`c27f3caec57fda08a830372c3185ec0a1d0fb62d`, based on
`90b2b62b62b934e722e71e36adfdd67643fd9aaf`. That private baseline and the public
baseline above have identical file trees. Only the reviewed feature delta is
ported, without private parent history. The complete main-relative inventory
is 82 paths, listed below. Publication cleanup changes documentation and the
standalone patch runner/checker, not numerical source or acceptance thresholds.

The work adds slope-limited anisotropic conduction, temperature-dependent
Spitzer conductivity, physical flux saturation, and guarded nonlinear use of
RKL2. It also adds test problems, numerical oracles, backend adapters and
documentation. It does not add hyperviscosity, Braginskii viscosity, Jiang's
conduction method, a general prescribed thermal-flux boundary interface, or a
new particle algorithm. No `conduction_bc` feature is included.

Several important mechanisms already exist in the comparison main: isotropic
conduction, RKL2 coefficient generation, explicit/STS process selection,
pre/post conduction splitting, AMR flux correction and transfer, CT, MPI
boundary exchange, and completed-cycle restart/output scheduling. This branch
uses that infrastructure; it does not claim to introduce it.

## 1. Physics and variable conventions

For the supported Newtonian ideal-gas closure, write

```text
theta = p/rho = (gamma - 1) * e_int/rho
K(theta) = K0 * theta^(5/2)
C = -rho * alpha_iso   * grad(theta)
    -rho * alpha_aniso * b_hat * dot(b_hat, grad(theta))
    -K(theta) * P * grad(theta)
```

Here `e_int` is internal-energy density, and `P` is the identity for isotropic
Spitzer transport or `b_hat b_hat` for field-aligned Spitzer transport. In
primitive arrays `w(IEN)` is internal-energy density, **not pressure**. In the
dynamical conservative update, the conductive increment enters total energy.

`alpha_aniso` is additional parallel transport: the constant-alpha parallel
and perpendicular coefficients are `alpha_iso + alpha_aniso` and `alpha_iso`.
These code inputs are not themselves cgs conductivities. For fixed density,
the temperature diffusivity corresponding to a conductivity `K` is
`(gamma - 1) * K/rho`.

The `theta^(5/2)` dependence is the Spitzer-type collisional law
([Spitzer & Harm, 1953](https://doi.org/10.1103/PhysRev.89.977)). The implemented
coefficient is supplied by the user; this code does not calculate a varying
Coulomb logarithm, ionization dependence or a low-temperature conductivity law.

### Configuration and dispatch

| Location/function | Responsibility |
| --- | --- |
| [Conduction::Conduction](../src/diffusion/conduction.cpp) | Reads coefficients, limiter, geometry, saturation and integrator; rejects incompatible or nonrepresentable inputs. |
| [Hydro::Hydro](../src/hydro/hydro.cpp), [MHD::MHD](../src/mhd/mhd.cpp) | Construct conduction when any relevant key is present, including incomplete settings that must produce a useful configuration error. |
| [Conduction::AddHeatFluxes](../src/diffusion/conduction.cpp) | Dispatches the single combined saturated operator, or separately adds unsaturated isotropic, anisotropic and Spitzer contributions. |
| [Conduction class](../src/diffusion/conduction.hpp) | Owns coefficients, integrator selection, diffusion/transport timestep bounds and the primitive-state retry backup. |

`conductivity_integrator=explicit|sts` selects the process; STS also requires
`time/sts_integrator=rkl2`. There is no experimental opt-in. The obsolete
`experimental_anisotropic_sts` key is not read. A positive active coefficient
is required for STS. Anisotropy requires nonrelativistic MHD; Hydro permits
isotropic transport. Nonlinear configurations require an SSP `rk1`, `rk2` or
`rk3` fluid integrator and finite `0 < cfl_number <= 1`.

Spitzer conduction requires `alpha_spitzer=true` and exactly one finite,
nonnegative `spitzer_kappa0_code` or `spitzer_kappa0_cgs`. MHD must explicitly
select `spitzer_geometry=isotropic|field_aligned`; Hydro defaults to isotropic.
The cgs prefactor multiplies physical temperature to the power 5/2. The helper
`SpitzerCoefficientCode` converts it using

```text
K0_code = K0_cgs * T_unit^(7/2) / (P_unit * V_unit * L_unit).
```

Code coefficients do not require a units block. Only the supplied coefficient
convention is serialized; restart input does not acquire a conflicting second
coefficient key. Zero coefficients disable their contributions. A true zero
conductivity is distinct from underflow of an active positive coefficient.

## 2. Face fluxes: where the heat actually moves

All four flux kernels add to the energy component of existing face-flux
arrays. They do not directly update density, momentum or the induction
equation. Shared face fluxes then enter the existing conservative update and,
on refined grids, its existing flux-correction pipeline.

### Constant isotropic and anisotropic operators

`Conduction::AddHeatFluxIso` in
[conduction.cpp](../src/diffusion/conduction.cpp) retains the isotropic
arithmetic face density and centered normal temperature difference.
`Conduction::AddHeatFluxAniso` implements the limited anisotropic operator:

1. Form the centered face-normal temperature derivative.
2. Form transverse derivatives from the surrounding four temperature slopes,
   using nested `VLL2State`/`VLL4State` by default or `MCL2State` for MC.
3. Use the CT face-normal field and averaged transverse cell-centered fields.
   Normalize with all three components, including out-of-plane B in 2-D.
4. Multiply the projected gradient by `alpha_aniso` times harmonic face density.
   Exactly zero face field produces zero aligned flux; there is no arbitrary
   weak-field cutoff.

For same-sign slopes, van Leer returns their harmonic combination; MC bounds
their centered average by twice the smaller magnitude. Opposite signs return
zero. MC here is *monotonized central*, unrelated to Monte Carlo particles.
Neither limiter removes finite-resolution diffusion across curved field lines.

The harmonic face coefficient and limited transverse **temperature** gradients
follow the limited-asymmetric construction of
[Sharma & Hammett (2007), Eq. 8 and section 4](https://w3.pppl.gov/~hammett/gyrofluid/papers/2007/aniso.pdf).
Harmonic coefficient interpolation and slope limiting are different operations.
The paper's coefficient multiplying the temperature gradient is averaged;
temperature and diffusivity are not independently harmonically averaged.
The CT magnetic reconstruction is our integration with AthenaK's existing
face-field representation, not a replacement with cell-centered divergence
cleaning.

### Temperature-dependent harmonic Spitzer operator

`Conduction::AddHeatFluxSpitzer` in
[conduction_spitzer.cpp](../src/diffusion/conduction_spitzer.cpp) evaluates
endpoint temperatures and conductivities, then uses one common coefficient

```text
K_face = harmonic(K_left, K_right)
       = 2 * K_low / (1 + K_low/K_high).
```

The normal derivative is a temperature difference. For field alignment,
transverse temperature derivatives use `SpitzerLimited4`, followed by the same
CT-based projection and multiplication by `K_face`. The isotropic Spitzer
path uses the same harmonic coefficient with its normal derivative. Existing
constant-alpha contributions retain their separate arithmetic; their sum is
not replaced by one harmonic average of a combined conductivity.

This is **not** the earlier potential-gradient prototype. The production
operator does not difference or limit `Psi = (2/7) K0 theta^(7/2)`.
`SpitzerPotentialGradient` remains a reference helper with unit tests, not a
runtime option. The former transverse potential-limited stencil lost an affine
temperature perpendicular-null control. Its old failures were preserved;
the harmonic-temperature revision has independent face-oracle/null tests.

At a genuinely zero endpoint conductivity, the harmonic coefficient is zero.
That degeneracy is not proof of accurate propagation into an unresolved cold
front. Positive-temperature conductivity underflow instead raises an error.

### Cowie-McKee saturation of the summed vector

`conduction_saturation=cowie_mckee` and `saturation_phi` enable the closure.
`Conduction::AddHeatFluxSaturated` in
[conduction_saturated.cpp](../src/diffusion/conduction_saturated.cpp) calls
`conduction::SaturatedFace` in
[conduction_saturated.hpp](../src/diffusion/conduction_saturated.hpp). At each
face this routine first builds the complete classical vector `C` from all
enabled terms, then applies

```text
F = C * q_sat / (q_sat + |C|)
q_sat = 5 * phi * p_donor^(3/2) / sqrt(rho_face).
```

`rho_face` is arithmetic face density. The sign of the classical normal heat
flux selects the sending cell's pressure, not a reconstructed Riemann pressure.
The isothermal convention `c_iso^2=p/rho` supplies no extra gamma factor.
Only the conductive increment is limited; an existing fluid energy flux is
left intact. Applying the cap once to the summed vector preserves its direction
and avoids separately saturating components or duplicating contributions.
Even isotropic saturation needs transverse slopes/edge ghosts for this vector
magnitude. Zero phi suppresses flux but does not bypass state validation.

The physical saturation scale comes from
[Cowie & McKee (1977)](https://doi.org/10.1086/154911); the smooth vector
interpolation and isothermal convention are also written explicitly in
[Vaidya et al. (2017), Eqs. 2--4](https://arxiv.org/html/1702.05487v2#S2).
Our precise face reconstruction, numerical range checks and retry policy are
implementation choices, not guarantees derived from those papers.

### Shared numerical helpers

[conduction_closure.hpp](../src/diffusion/conduction_closure.hpp) contains
device-callable pure helpers: `HarmonicMeanPositive`, `SpitzerTemperature`,
`SpitzerConductivity`, `SpitzerLimited4`,
`SpitzerProduct3`, `FluxMagnitude`, `SaturationSuppression`,
`SaturateHeatFlux` and `SaturatedDonorSpeed`. It also provides the host-side
unit conversion `SpitzerCoefficientCode`; the face-specific donor-flux helper
`SaturatedDonorFlux` is in `conduction_saturated.hpp`.
Scaled harmonic/norm arithmetic and logarithmic or exponent-split fallbacks
avoid selected intermediate overflows and loss of representable vector
components. Ordinary-range arithmetic is retained where possible.

These helpers do not guarantee every extreme floating-point product. Active
rate paths reject zero/subnormal intermediate contributions even when a
different algebraic ordering might represent the final result. Another positive
transport contribution must not hide such loss. The legacy `q_limit` input
must remain at or above its finite disabled sentinel `3.40282e38`; it is not a
second user-controlled physical saturation cap.

## 3. Explicit timesteps and raw-state checks

| Function | What it does and why |
| --- | --- |
| `Conduction::NewTimeStep` | Computes the CFL-free conduction bound. Constant-alpha transport uses a conservative density-aware row bound with `alpha_iso + 4*ndim*alpha_aniso`. Nonlinear paths use `SpitzerTimeStep`. |
| `Conduction::SpitzerTimeStep` | Combines constant and nonlinear diffusion rates, with a current-temperature cap and consumed halos. Includes saturation restrictions for explicit evolution. |
| `Conduction::SaturationTimeStep` | Uses actual adjacent-face donor response and outgoing conductive thermal-energy drain. Its `donor_only` form excludes the drain term for nonlinear STS. |
| `Conduction::CheckExplicitStage` | Rechecks the committed constituent forward-Euler interval against the current CFL-free bound before the ordinary Hydro/MHD flux task. It does not change dt mid-stage. |
| `Hydro::NewTimeStep`, `MHD::RecomputeTimeStepFromCurrentState` | Add the unaccelerated saturation donor restriction to the owning fluid's timestep. Existing mesh selection applies the configured CFL once. |
| `Conduction::InspectRawState` | Checks active cells or consumed face/edge halos from conserved variables and current CT faces, including local-copy EOS conversion and floor flags. Returns global extrema/invalid status. |
| `Conduction::ValidateRawState`, `FailIfInvalid` | Coordinate fatal invalid-state or constitutive failures across MPI ranks. |

Relevant files are [conduction.cpp](../src/diffusion/conduction.cpp),
[conduction_saturated.cpp](../src/diffusion/conduction_saturated.cpp),
[hydro_newdt.cpp](../src/hydro/hydro_newdt.cpp),
[mhd_newdt.cpp](../src/mhd/mhd_newdt.cpp), and the Hydro/MHD task files.

The explicit check allows 64-machine-epsilon comparison slack. It does **not**
apply the requested CFL twice: that margin was already used to choose the
full step. For example, modest stage heating can reduce the current bound
while a step chosen with CFL 0.4 still satisfies that conduction bound. The
full constituent interval is checked even for RK2's second stage. A bound violation aborts before that
stage advances; explicit SSP does not have automatic rollback.

Raw validation requires positive density and finite nonnegative internal energy
and temperature, also rejecting states needing the configured EOS repairs.
This protects the conduction-consumed state before normal EOS conversion; it
does not audit every direct EOS call in every module. Saturation drain bounds
refer to internal energy, not energy above a pressure floor, and are not a
proof of positivity for the coupled MHD equations.

## 4. RKL2 and the nonlinear trial controller

### What is inherited

[sts_rkl2.cpp](../src/diffusion/sts_rkl2.cpp) already provided
`parabolic::ComputeRKL2StageCount` and `ComputeRKL2Coefficients` in main.
The mathematical recurrence is unchanged here; the main-relative delta in
that file adds finite-input validation. The controller selects sufficient odd
stage counts, at least three, using the RKL2 capacity

```text
dt_sweep / dt_exp <= (s*s + s - 2)/4.
```

The RKL formulation is described by
[Meyer, Balsara & Aslam (2012)](https://academic.oup.com/mnras/article/422/3/2102/1042639)
and [Meyer, Balsara & Aslam (2014)](https://doi.org/10.1016/j.jcp.2013.08.021).
The [2013 erratum](https://academic.oup.com/mnras/article/428/2/1855/1000287)
corrects the earlier article's printed references, polynomial and table entry.
Existing pre/post half-sweep scheduling is also reused, not newly introduced.

### What this branch adds

`Driver::ExecuteSTSSweep` in [driver.cpp](../src/driver/driver.cpp) replaces the
inline sweep loops. The constant-coefficient branch retains their arithmetic.
`Conduction::UsesNonlinearSTS` selects the new checked path when STS combines
with saturation or a positive Spitzer coefficient.

```text
Driver::Execute
  pre-conduction half-sweep: ExecuteSTSSweep
  ordinary RK fluid/source update
  post-conduction half-sweep: ExecuteSTSSweep
  advance time/cycle -> outputs -> AMR -> next dt -> completed-cycle checkpoint

Nonlinear half-sweep
  refresh boundaries/primitives; validate raw state; refresh diffusion bound
  choose a segment no longer than the current unaccelerated donor cap
  record global thermal envelope; BeginSTSTrial snapshots U and W
  select stage count; run existing RKL stage tasklists with new checks
    accept -> advance through the remaining half-sweep duration
    reject -> finish posted communication; restore U/W; halve and retry
```

Each attempt holds recurrence coefficients fixed but recomputes conductivity,
limited gradients and saturation from the evolving stage state. Both halves
refresh primitives, including the post-fluid half. `BeginSTSTrial` and
`RestoreSTSTrial` manage full owning-fluid U/W snapshots; the W backup resizes
after AMR. The controller allows at most 32 halvings per segment and rejects
non-progress. Accepted segments must cover the original half-sweep duration.

### Stage checks and communication

The important stage dependency chain is:

```text
InitRecvParabolic -> ClearSTSFlux -> STSFluxes -> SendFlux/RecvFlux
 -> STSUpdate[U] -> STSCheckState -> RestrictU -> SendU/RecvU
 -> Prolongate -> ApplyPhysicalBCs -> ConToPrim -> STSRefreshTimeStep
 -> ClearSend/ClearRecv
```

The new/enhanced behavior is implemented in
[hydro_sts.cpp](../src/hydro/hydro_sts.cpp),
[mhd_sts.cpp](../src/mhd/mhd_sts.cpp),
[hydro_tasks.cpp](../src/hydro/hydro_tasks.cpp),
[mhd_tasks.cpp](../src/mhd/mhd_tasks.cpp) and their class declarations:

- `Hydro::STSFluxes` / `MHD::STSFluxes` call
  `Conduction::CheckSTSTransport` before a saturated stage flux. The current
  donor cap constrains the whole trial interval, without STS acceleration.
- Stage updates retain the pretrial snapshot, initialize recurrence RHS storage
  to avoid `0*NaN`, and skip updates after a trial is marked failed.
- `Hydro::STSCheckState` / `MHD::STSCheckState` inspect active raw state
  immediately after the recurrence, before restriction or an EOS conversion.
- Enhanced `Hydro::ConToPrim` / `MHD::ConToPrim` inspect consumed halos after
  prolongation and physical BCs, before normal EOS conversion. Trial checks
  enforce the initial global temperature envelope with a
  `512*epsilon*max(abs(Tmin),abs(Tmax))` allowance, not a unit-temperature floor.
- A trial failure is collective, including a failure detected on one MPI rank.
  Stage checks can restore U/W early to carry safe state through remaining
  tasks. Remaining communication completes before an attempt is abandoned or
  retried; the driver restores U/W again after communication before retry.

Invalid raw states found by trial checks, envelope violations and donor-cap
violations can trigger interval halving. Invalid initial/sweep-start states
and constitutive/range failures remain fatal. Rollback restores U/W, **not the
entire solver or the CT field**. Nonlinear conduction is the sole allowed STS
process, so its recurrence changes only energy. The donor restriction remains
in the outer timestep; the explicit thermal-drain restriction is not imposed
on the full accelerated RKL interval.

This is our defensive acceptance/retry policy, not a theorem from the RKL
papers, a universal nonlinear maximum principle, or a temporal-error estimator.
It does not shorten the already committed full fluid step. Temporal refinement
against explicit references remains necessary even if every trial succeeds.

## 5. SMR/AMR, CT, boundaries and restarts

No production `mesh/`, `bvals/` or `outputs/` file differs from main for this
candidate. Stage flux exchange/correction, restriction/prolongation, physical
BC application and CT remain existing infrastructure. The new work supplies
the conduction fluxes, current-state bounds, stage checks and targeted tests
through those paths.

`Driver::ValidateSTSConfiguration` requires nonlinear STS to have one fluid,
conduction as its sole STS process, no custom user boundary callback or cyclic
zoom, and conserved-variable prolongation on refined meshes. Regridding occurs
at the existing completed-cycle point, not within a retry. Subsequent sweeps
refresh their state, bounds and appropriately sized backups.

Although the conductive recurrence updates only energy, the existing
`MHD::Prolongate` still calls face-centered magnetic prolongation during the
boundary pipeline. Thus "the conduction source has no induction term" does
not prove every active face remains bitwise unchanged throughout every refined
task sequence. B-exact stage checks and divergence bounds are measured results
for the tested fixtures, not a new general CT theorem. On composite meshes,
volume-averaged arithmetic Bcc is not generally a magnetic-flux invariant;
the refined diagnostics use native plane-integrated normal magnetic fluxes.

Actual post-refine/post-derefine restarts were exercised. They use existing
completed-cycle checkpoints and reconstruct callbacks/state normally. There is
no new binary restart layout. The optional annular-diagnostic sidecar belongs
to a test fixture only; it does not change the production restart format.

Refined acceptance covers bounded periodic, one-level static/dynamic cases,
including 2-D coupled MHD, density contrasts and strict raw-field/restart
comparisons on serial CPU, two CPU ranks and one GPU. It is not acceptance of
arbitrary refinement patterns, all boundary conditions or inter-node refined
coupled dynamics. Thermal behavior depends on ghost states; a fluid boundary
name alone does not specify an insulating or prescribed-heat-flux boundary.

## 6. Test problems and their entry points

[pgen.cpp](../src/pgen/pgen.cpp), [pgen.hpp](../src/pgen/pgen.hpp) and
[src/CMakeLists.txt](../src/CMakeLists.txt) register/build `transport_mode` and
`transport_refinement`, plus the two nonlinear flux translation units.
Other `pgen.cpp` changes are comment/line wrapping, not particle-physics edits.

| Fixture/functions | Purpose and scope |
| --- | --- |
| `ProblemGenerator::TransportMode`, `ReadMode`, `MeasureMode`, `CheckAdvectionFlux` in [transport_mode.cpp](../src/pgen/tests/transport_mode.cpp) | Fourier-mode initializer/diagnostics, profile dispatch, MPI reductions and device advection-flux controls; restart registers diagnostics without resetting the restored solution. |
| `transport_operator::Check` in [transport_operator.hpp](../src/pgen/tests/transport_operator.hpp) | Calls the production anisotropic spatial kernel on scratch arrays; checks an independent graph-Laplacian oracle, cold-corner bounds and checkerboard damping through a test RK2 loop. This is not a full-driver STS-stage probe. |
| Helpers in [transport_saturated_operator.hpp](../src/pgen/tests/transport_saturated_operator.hpp) | Independent long-double face oracle for harmonic Spitzer and combined vector saturation, both limiters, affine perpendicular nulls and preservation of preexisting energy flux. |
| `Reset`, `Check`, `Write` in [transport_stages.hpp](../src/pgen/tests/transport_stages.hpp) | Fixed-field fixture raw-state observations after ordinary RK updates and before EOS conversion; not independent dumps of every nonlinear RKL stage. |
| `CellFactor`, `Initial`, `Initialize`, `Finalize` in [transport_gaussian.hpp](../src/pgen/tests/transport_gaussian.hpp) | Cell-averaged periodic Gaussian pulse fixtures in 2-D/3-D. |
| `CDF`, `Fraction` in [transport_front.hpp](../src/pgen/tests/transport_front.hpp) | Exact hot-slab cell fractions or Fourier-smoothed front initialization. |
| `InitializePatch`, `MonitorPatch`, `FinalizePatch`, `SampleRing` in [transport_patch.hpp](../src/pgen/tests/transport_patch.hpp) | Square/radial/ring fixtures; discrete-curl magnetic setup, annular heat and leakage diagnostics. |
| `State::Validate`, `Sample`, `Encode`, `Decode` in [transport_ring_state.hpp](../src/pgen/tests/transport_ring_state.hpp) | Checked, transactional annular-history sidecar with identity/time/cycle matching for diagnostic continuation. |
| `Initialize`, `Boundary`, `Source`, `Monitor`, `Finalize` in [transport_equilibrium.hpp](../src/pgen/tests/transport_equilibrium.hpp) | Explicit forced-equilibrium/Sovinec and density-interface resistance problems; source plus boundary-energy accounting. User BCs here are test-specific, not a generic thermal-BC feature. |
| `Quadrature`, `Az`, `AverageXY`, `Initialize`, `Finalize` in [transport_helical.hpp](../src/pgen/tests/transport_helical.hpp) | 3-D solenoidal helical-field fixture with quadrature/cell averaging and all six CT faces in diagnostics. |
| `Initialize`, `RawState`, `Measure`, `Finalize` in [transport_coupled.hpp](../src/pgen/tests/transport_coupled.hpp) | Dynamical HLLD/RK2 eigenmode fixture. Its pre-CT raw observer is not a synchronized-pressure measurement; post-cycle diagnostics reconstruct pressure with CT. |
| `Initial`, `Dump`, `Initialize`, `Finalize` in [transport_sawtooth.hpp](../src/pgen/tests/transport_sawtooth.hpp) | Periodic constant-conductivity saturated scalar benchmark, with no test-imposed thermal evolution. |
| `ProblemGenerator::TransportRefinement`, `Refine`, `ObserveTopology`, `Measure`, `Finalize` in [transport_refinement.cpp](../src/pgen/tests/transport_refinement.cpp) | Cell-average cosine plus optional density step; real refinement/derefinement callback, topology and raw U/six-face CT output, restart continuation; supports kinematic or dynamical MHD. |

One additional production-header change supports the restricted kinematic
fixtures: `mhd::Advect<ivx>` in
[advect_mhd.hpp](../src/mhd/rsolvers/advect_mhd.hpp) now supplies upwind
internal-energy flux for ideal EOS. This is **not a general dynamical MHD
total-energy solver**. Thermal use is restricted to fixed density, zero
velocity and static B; evolving-field tests use dynamical solvers such as HLLD.

### Python runners, oracles and CI

The four `test_transport_functional_*` adapters call `transport_functional`,
`run_pilot.py` and `run_sts.py` for both limiters. They cover disabled/isotropic/
aligned/perpendicular/oblique/zero-B/3-D modes, explicit references, RKL2 and
actual restarts. `run_sts.py::execute/compare` and
`sts_checks.py::compare_fields` separate same-grid temporal agreement from
restart/decomposition comparisons and diagnostic conservation.

`run_gaussian.py::validate` uses independent cell-average references from
`gaussian_oracle.py`; `run_equilibrium.py` checks steady resistance, prescribed
fixture heating and boundary/source budgets. The localized CPU/GPU adapters
run small subsets, not every extended research case. `run_patch.py::validate`
handles additional leakage/stage/budget controls. Its former development-host
restriction is removed for publication; callers remain responsible for the
execution environment and available resources.

The four `test_transport_nonlinear_*` adapters call `transport_nonlinear` and
`run_nonlinear.py`. `operator_cases`, `guard_cases`, `range_guard_cases`,
`inspect_operator`, `inspect_pulse`, `gate_thermal` and `restart_gap` cover
face oracles, intended rejection diagnostics, same-grid explicit/RKL pulse
comparisons and a real interior restart. The CPU lane adds configuration/range
rejections; CPU/GPU add native face/null tests; MPI lanes run the pulse matrix.
This smoke runner explicitly does not claim continuum order, full CT-face
equivalence, independent raw RKL-stage dumps, injected retry recovery or AMR
acceptance. Separate targeted diagnostic campaigns supply that evidence.

The CPU reference adapter discovers `test_oracles`, `test_gaussian_oracle`,
`test_gaussian_checker`, `test_equilibrium_checker`, `test_patch_checker`,
`test_sts_checks` and `test_result_policy`. These include deliberate corruption
of geometry, fields, times, budgets and stage coverage to test the checkers.
`result_policy.py::assess_case/annotate_report` classifies evidence without
silently converting a numerical FAIL to PASS.

`oracles.py::conduction_rate` supplies the independent constant-background
Fourier decay rate `(gamma-1)*(alpha_iso*|k|^2 + alpha_aniso*(b_hat.k)^2)`.
The same module includes reference-only `hyperviscous_rate` and `sound_roots`:
the former evaluates `nu4*|k|^4` or the squared discrete-Laplacian symbol;
the latter solves the damped acoustic quadratic. Their algebra tests are
preparatory references, not implementation or simulation validation of a new
hyperviscosity operator.

The closure adapter discovers `test_conduction_closure_cpu`,
`test_saturated_speed_cpu` and `test_spitzer_potential_cpu`. These use small
C++17 wrappers around pure helpers, including high-precision reference checks;
they are not simulations. The potential tests do not select a potential-based
production flux. The existing RKL2 controller test adds invalid-input checks,
independent Legendre-polynomial agreement at every scalar stage, scalar
second-order convergence and half-sweep stage-selection coverage. Those
controller checks do not establish nonlinear PDE positivity or accuracy.

Athena adapters use an already built `ATHENAK_TEST_BINARY`, plus
`ATHENAK_TEST_LAUNCHER`/`ATHENAK_TEST_RANKS` for MPI; they do not build Athena or
submit jobs. Helper wrappers may compile unless prebuilt overrides are supplied.
See [the usage instructions](thermal_conduction.md#tests-and-interpretation).
The only [CI workflow](../.github/workflows/main.yml) change is adding SciPy to
the existing CPU/MPI-CPU/GPU Python dependencies; it does not add Slurm jobs or
an MPI-GPU workflow.

## Validation and provenance

The final native inventories completed on 2026-10-10:

| Lane | Result |
| --- | --- |
| CPU | 276 PASS, 15 existing SKIP; all 291 frozen IDs accounted for |
| MPI-CPU | 59 PASS, two-node job 15292089 |
| GPU | 65 PASS in disjoint 14 + 40 + 11 shards; final shard job 15292091 |
| MPI-GPU | 12 PASS, two-node job 15292090 |

The original CPU wrapper timed out after 275 passes and 15 skips; only the
unfinished Z4c AMR-boundary test was resumed and passed. The original exit 124
was retained. The first GPU shard's 14 passing tests were independently audited
from complete phase records/JUnit and pinned files after a subsequent scratch
wrapper error (exit 2); that wrapper is not claimed to have finished cleanly.
No tests, solver files or numerical gates were changed to obtain final coverage.
The 15 existing skips are seven self-reference comparisons and eight
unsupported SR MHD/HLLC combinations.

The tested executables and candidate were pinned as follows:

```text
CPU SHA256: 1ab871a92839a907f68454516ec1aeab3b9df2a4ad16eb1ce93db84da8b70dd8
GPU SHA256: c82ca4a721009bc3a7491964b46449fe1bca2c53f82137291fdf1b0c06eceff4
371-entry source manifest SHA256:
  9f316a5f1424cc10fca62a3c960a1e8f657d9892d8d274a3e3ceeca98bda73bc
590-pin final candidate manifest SHA256:
  c6a7b08d1e4ccff320f7bf258b4abdf0196466fec27a5104ac4094ecc8b3ab6d
```

These hashes identify the original private validation campaign, not new public
binary builds. Its numerical source is preserved exactly in the public port.
Documentation is excluded from those runtime pins. The public standalone patch
runner replaces a Python-3.11-only hashing API and removes a development-host
restriction; two added checker tests verify those changes without running
Athena. No numerical assertion or threshold changes. The original 590-pin
manifest remains authoritative for the original checkpoint, not the modified
public helper files.

Maintainers retain the campaign's full logs, original failures and manifests
under the archive identifier `athenak-transport-validation-20260921`, including
`FINAL_VALIDATION_STATUS_20261009.md`, `MERGE_CHECKPOINT_20261009.md` and the
`nonlinear-final-20261009` frozen-candidate record. Those large/site-local
artifacts are not bundled in this repository. This document records their
results and qualifications without requiring inaccessible filesystem links.

### Targeted evidence is broader than the portable smoke matrix

Separate campaigns exercised harmonic/null face controls; nonlinear pulses;
explicit and RKL temporal refinement; actual restarts; 2-D/3-D fronts and
curved/helical fields; static/dynamic refinement and density contrasts;
coupled evolving-field MHD; and deliberate one-rank stage failures/recovery.
Test-only relinks/observers are not production runtime switches. Earlier
candidate results retain their own provenance and are not all claimed to have
been rerun in the final native smoke matrix.

Examples of completed bounded checks include:

- Coupled dynamic-AMR post-refine/post-derefine restarts on CPU, MPI2 and GPU:
  exact within-lane continued fields/topology/timesteps; CPU1/CPU2 raw fields
  exact and maximum CPU/GPU raw difference approximately 1.10e-14 under the
  unchanged 5e-12 gate.
- Post-regrid raw-state, thermal-envelope and donor-bound fault probes:
  rejected intervals restore U/W exactly, and measured CT faces remain
  unchanged in those cases; recovered solutions match clean-rejection controls.
- Coupled density-jump static SMR: explicit timestep-halving difference 1.22e-7;
  RKL energy normalized L1 0.004816 under the unchanged 0.005 gate, close to
  that ceiling, not evidence of arbitrary-contrast accuracy.
- Current explicit-stage guard: a safe coupled stage passes, while an injected
  interval 1.01 times its current CFL-free bound fails before advancement.
- A periodic saturated sawtooth based on
  [Vaidya et al. (2017), section 4.1.2](https://arxiv.org/html/1702.05487v2#S4.SS1.SSS2):
  constant conductivity 40, rho 1, gamma 2, phi 0.3, N200/N400, endpoint 0.1.
  N400 explicit halving differs by 1.38e-10 in normalized L1; requested RKL
  ratios 40/1000 differ from fine explicit by 1.84e-8/2.32e-5. The actual
  ratio-1000 outer diffusion number is 13.269--28.339 because the donor cap
  remains active. This reproduces the scalar problem setup, not a Spitzer
  case, a digitized paper-error measurement or every published STS regime.

### Accepted limitation, not a hidden strict PASS

Six earlier N32 3-D energy comparisons exceed the original CPU/GPU 5e-12
scaled-agreement gate, reaching 5.4542320335866934e-9. Nonenergy fields, CT
faces, time and topology match exactly. The metric is
`abs(a-b)/max(1,abs(a),abs(b))`; energy is 2.02--2.29 in the worst case, so the
reported worst difference is genuinely relative energy error. The associated
temperature difference divided by the initial perturbation amplitude is
7.69e-8, a different normalization that must not be conflated with energy.

The user explicitly accepted this bounded discrepancy before authorizing final
validation. The six strict FAIL records remain FAIL; no tolerance was changed
and no solver correction is claimed. Timestep/common-seed controls show
precision sensitivity but are not a universal fix. This disposition must
remain visible in a merge/publication review.

Other limits remain: Newtonian ideal gas; restrictions on nonlinear STS listed
above; no arbitrary-AMR/all-BC guarantee; no automatic temporal-error control;
no proof of maximum-principle behavior for all RKL stages or positivity of all
coupled physics; and deliberately bounded floating-point range support.

## Complete main-relative file inventory

Each path below differs from the stated main baseline. Together these are 82
paths: 20 production implementation/header files, 3 build/registration files,
13 fixture implementations, 8 fixture inputs, 35 Python files, 1 CI workflow
and 2 documents. A file listed here may already have been committed in the
earlier explicit-conduction checkpoint; it need not be newly modified in the
final nonlinear commit.

### Production implementation: 20 paths

```text
src/diffusion/conduction.cpp
src/diffusion/conduction.hpp
src/diffusion/conduction_closure.hpp
src/diffusion/conduction_saturated.cpp
src/diffusion/conduction_saturated.hpp
src/diffusion/conduction_spitzer.cpp
src/diffusion/sts_rkl2.cpp
src/driver/driver.cpp
src/driver/driver.hpp
src/hydro/hydro.cpp
src/hydro/hydro.hpp
src/hydro/hydro_newdt.cpp
src/hydro/hydro_sts.cpp
src/hydro/hydro_tasks.cpp
src/mhd/mhd.cpp
src/mhd/mhd.hpp
src/mhd/mhd_newdt.cpp
src/mhd/mhd_sts.cpp
src/mhd/mhd_tasks.cpp
src/mhd/rsolvers/advect_mhd.hpp
```

The conduction files own spatial closures, input validation and stage bounds;
driver files own checked nonlinear sweep orchestration. Hydro/MHD constructors,
declarations, timesteps and task files connect those mechanisms to each fluid.
The limited RKL-helper and kinematic advection changes are explained above.

### Build/registration and fixture implementations: 16 paths

```text
src/CMakeLists.txt
src/pgen/pgen.cpp
src/pgen/pgen.hpp
src/pgen/tests/transport_mode.cpp
src/pgen/tests/transport_refinement.cpp
src/pgen/tests/transport_coupled.hpp
src/pgen/tests/transport_equilibrium.hpp
src/pgen/tests/transport_front.hpp
src/pgen/tests/transport_gaussian.hpp
src/pgen/tests/transport_helical.hpp
src/pgen/tests/transport_operator.hpp
src/pgen/tests/transport_patch.hpp
src/pgen/tests/transport_ring_state.hpp
src/pgen/tests/transport_saturated_operator.hpp
src/pgen/tests/transport_sawtooth.hpp
src/pgen/tests/transport_stages.hpp
```

### Fixture inputs: 8 paths

```text
tst/inputs/transport_mode.athinput
tst/inputs/transport_refinement.athinput
tst/inputs/transport_coupled.athinput
tst/inputs/transport_equilibrium.athinput
tst/inputs/transport_front.athinput
tst/inputs/transport_gaussian.athinput
tst/inputs/transport_helical.athinput
tst/inputs/transport_patch.athinput
```

These provide small controlled transport configurations rather than
problem-specific production simulations. Runners override selected parameters
and preserve the executed input/provenance with their results.

### Python discovery/adapters: 15 paths

```text
tst/test_suite/diffusion/test_sts_rkl2_controller_cpu.py
tst/test_suite/diffusion/test_transport_closure_cpu.py
tst/test_suite/diffusion/test_transport_functional_cpu.py
tst/test_suite/diffusion/test_transport_functional_gpu.py
tst/test_suite/diffusion/test_transport_functional_mpicpu.py
tst/test_suite/diffusion/test_transport_functional_mpigpu.py
tst/test_suite/diffusion/test_transport_localized_cpu.py
tst/test_suite/diffusion/test_transport_localized_gpu.py
tst/test_suite/diffusion/test_transport_nonlinear_cpu.py
tst/test_suite/diffusion/test_transport_nonlinear_gpu.py
tst/test_suite/diffusion/test_transport_nonlinear_mpicpu.py
tst/test_suite/diffusion/test_transport_nonlinear_mpigpu.py
tst/test_suite/diffusion/test_transport_reference_cpu.py
tst/test_suite/diffusion/transport_functional.py
tst/test_suite/diffusion/transport_nonlinear.py
```

### Python runners, references and checker tests: 20 paths

```text
tst/transport/gaussian_oracle.py
tst/transport/oracles.py
tst/transport/result_policy.py
tst/transport/run_equilibrium.py
tst/transport/run_gaussian.py
tst/transport/run_nonlinear.py
tst/transport/run_patch.py
tst/transport/run_pilot.py
tst/transport/run_sts.py
tst/transport/sts_checks.py
tst/transport/test_conduction_closure_cpu.py
tst/transport/test_equilibrium_checker.py
tst/transport/test_gaussian_checker.py
tst/transport/test_gaussian_oracle.py
tst/transport/test_oracles.py
tst/transport/test_patch_checker.py
tst/transport/test_result_policy.py
tst/transport/test_saturated_speed_cpu.py
tst/transport/test_spitzer_potential_cpu.py
tst/transport/test_sts_checks.py
```

### CI and documentation: 3 paths

```text
.github/workflows/main.yml
docs/thermal_conduction.md
docs/transport_implementation.md
```

## References and what they support

- **Sharma, P. & Hammett, G. W. (2007)**, *Preserving monotonicity in anisotropic
  diffusion*, JCP 227, 123--142.
  [Author PDF](https://w3.pppl.gov/~hammett/gyrofluid/papers/2007/aniso.pdf),
  [DOI](https://doi.org/10.1016/j.jcp.2007.07.026).
  Basis for harmonic coefficient interpolation and limited-asymmetric
  transverse temperature gradients. This is Sharma--Hammett, not a
  "Sharma and Stone 2008" attribution.
- **Spitzer, L. Jr. & Harm, R. (1953)**, *Transport Phenomena in a Completely
  Ionized Gas*, Physical Review 89, 977--981.
  [DOI](https://doi.org/10.1103/PhysRev.89.977).
  Physical basis of collisional temperature-dependent conductivity; our
  simplified prefactor and unit conversion are explicit in the code.
- **Cowie, L. L. & McKee, C. F. (1977)**, *The evaporation of spherical clouds
  in a hot gas. I. Classical and saturated mass loss rates*, ApJ 211, 135--146.
  [DOI](https://doi.org/10.1086/154911),
  [paper](https://adsabs.harvard.edu/pdf/1977ApJ...211..135C).
  Physical saturated-flux scale, not a numerical retry/AMR theorem.
- **Meyer, C. D., Balsara, D. S. & Aslam, T. D. (2012)**, *A second-order
  accurate Super TimeStepping formulation for anisotropic thermal conduction*,
  MNRAS 422, 2102--2115.
  [Article](https://academic.oup.com/mnras/article/422/3/2102/1042639),
  [2013 erratum](https://academic.oup.com/mnras/article/428/2/1855/1000287).
  Early RKL formulation, with the printed corrections taken into account.
- **Meyer, C. D., Balsara, D. S. & Aslam, T. D. (2014)**, *A stabilized
  Runge--Kutta--Legendre method for explicit super-time-stepping of parabolic
  and mixed equations*, JCP 257, 594--626.
  [DOI](https://doi.org/10.1016/j.jcp.2013.08.021).
  RKL stability-polynomial/recurrence foundation used by the inherited
  controller; not a proof of our state-dependent retry policy.
- **Vaidya, B., Prasad, D., Mignone, A., Sharma, P. & Rickler, L. (2017)**,
  *Scalable explicit implementation of anisotropic diffusion with
  Runge--Kutta--Legendre super-time-stepping*, MNRAS 472, 3147--3160.
  [Full paper](https://arxiv.org/html/1702.05487v2),
  [abstract](https://arxiv.org/abs/1702.05487).
  Saturation convention, RKL conduction context and saturated-sawtooth problem
  setup. Local error measurements are ours, not values digitized from its plots.

The articles motivate methods and benchmarks. They do not establish acceptance
of this implementation: that comes from the source-specific tests and explicit
limitations recorded above. In particular, numerical retry policy, raw-state
inspection, range rejection, exact restart comparisons and the accepted
backend discrepancy are project-specific decisions/evidence.
