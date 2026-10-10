# Thermal conduction

This document describes the anisotropic-conduction feature, including constant-coefficient
explicit conduction, anisotropic RKL2 super-time-stepping (STS), and validated
bounded Spitzer and Cowie-McKee saturation paths. The final four-lane validation
is complete for the source checkpoint described below, subject to its recorded
limitations. See the
[implementation and main-branch comparison](transport_implementation.md) for
the function map, complete change inventory, numerical methods and references.

## Coefficients and temperature convention

For an ideal gas, the code-temperature variable is `T = p/rho`. The implemented
energy flux is

```
q = -rho * alpha_iso * grad(T)
    -rho * alpha_aniso * bhat * dot(bhat, grad(T)).
```

Thus `alpha_aniso` is additional parallel transport, not the total parallel
coefficient. The parallel/perpendicular coefficients are respectively
`alpha_iso + alpha_aniso` and `alpha_iso`. For fixed uniform density, the
temperature diffusion tensor includes a factor `gamma - 1`. Neither input is
automatically a physical conductivity in cgs units; the active constant-alpha
operator uses code units and does not require a `<units>` block.

Example settings for nonrelativistic ideal-gas MHD:

```text
<mhd>
alpha_iso = 0.0
alpha_aniso = 0.1
alpha_spitzer = false
anisotropic_limiter = vanleer
conductivity_integrator = explicit
```

The remaining mesh, MHD, time and problem settings must be supplied normally.
Anisotropic transport requires MHD; isotropic transport can also be used in
hydro. Both coefficients must be finite and nonnegative. Zero coefficients
disable their respective contributions.

## Spatial flux and time integration

Anisotropic flux uses the CT face-normal magnetic component and averaged
transverse cell-centered components. All three components enter normalization,
including the out-of-plane component in a two-dimensional calculation. Exactly
zero face field gives zero anisotropic flux; there is no weak-field cutoff.
The face coefficient uses the harmonic mean of the two densities times
`alpha_aniso`. Isotropic flux retains its arithmetic face-density interpolation.

Transverse temperature slopes use nested van-Leer limiting by default;
`anisotropic_limiter = mc` selects monotonized-central limiting. MC here means
the slope limiter, not Monte Carlo tracer particles. Changing the limiter changes
spatial truncation error and transverse numerical diffusion; neither limiter
eliminates resolution-dependent leakage across curved field lines.

Explicit conduction contributes to the energy flux used by the ordinary RK
update. Its timestep calculation uses a conservative density-aware bound with
`alpha_iso + 4*ndim*alpha_aniso`, where `ndim` counts active spatial directions.
This bound is not a promise of useful accuracy at any chosen resolution.

### Anisotropic STS

For constant-coefficient, nonrelativistic MHD, the following settings select the
existing RKL2 implementation with anisotropic conduction:

```text
<mhd>
alpha_iso = 0.0
alpha_aniso = 0.1
conductivity_integrator = sts
<time>
integrator = rk2
sts_integrator = rkl2
sts_max_dt_ratio = 16
```

`conductivity_integrator=sts` is the only per-process selector. There is no extra
experimental opt-in. At least one conductivity must be positive; anisotropic
conduction requires nonrelativistic MHD. Isotropic and combined isotropic plus
anisotropic conduction use the same selection mechanism. Global STS configuration
checks still apply. The obsolete `experimental_anisotropic_sts` entry in older
inputs/restarts is no longer read and has no effect, whether true or false.
This interface uses the existing STS energy update, spatial flux and RKL
recurrence; it does not replace the numerical algorithm.

Validation includes
2D/3D mode controls, evolving-field MHD, nonlinear flow, actual restarts and
uniform-grid MPI execution. Stability is not an accuracy guarantee, and internal
RKL stages are not guaranteed to obey a maximum principle. Ratios 16 and 64 pass
the bounded smooth-mode fixtures; they are not universal recommended settings.
A separate sharp-front prototype at ratio 256 failed its 1% temporal-accuracy
gate. Resolve the spatial profile and check timestep refinement for a new setup.

## Temperature-dependent Spitzer conduction

Pure Spitzer conductivity can now be selected independently of the existing
constant-alpha terms. In the unsaturated dispatch, the latter retain their
original arithmetic and density interpolation. For example:

```text
<mhd>
alpha_spitzer = true
spitzer_kappa0_code = 0.1
spitzer_geometry = field_aligned
conductivity_integrator = explicit
<time>
integrator = rk2
```

Supply exactly one of `spitzer_kappa0_code` and `spitzer_kappa0_cgs`, finite and
nonnegative. Code coefficients require no units block. The cgs option requires
valid explicit units and converts according to

```
K0_code = K0_cgs * T_unit^(7/2) / (P_unit * V_unit * L_unit)
K(T) = K0_code * T^(5/2),  T = p/rho.
```

The cgs coefficient is the prefactor of the pure `T_kelvin^(5/2)` law, not the
conductivity evaluated at a reference temperature. No low-temperature Parker
branch or implicit conductivity ceiling is included. Conversion overflow or
underflow of a positive coefficient is rejected. Only the original coefficient
convention is serialized, so cgs restarts do not acquire conflicting code keys.

`spitzer_geometry` is `isotropic` or `field_aligned`. MHD must select it
explicitly; Hydro defaults to isotropic and rejects field alignment. A zero
coefficient contributes no flux. Spitzer-specific keys without
`alpha_spitzer=true` are rejected rather than silently ignored.

The production face coefficient is the harmonic mean of the endpoint
conductivities, `K_face = 2*K_left*K_right/(K_left+K_right)`, evaluated with a
scaled formula to avoid intermediate overflow. This averages the coefficient
multiplying the temperature gradient, not temperature or diffusivity separately.
Both isotropic and field-aligned Spitzer use this law. Field-aligned transport
multiplies the CT-based projection of the temperature gradient by this one face
coefficient; transverse **temperature** slopes use nested van-Leer/MC limiting.
This follows the harmonic interpolation and limited-asymmetric construction in
[Sharma & Hammett (2007), Eq. 8 and section 4](https://w3.pppl.gov/~hammett/gyrofluid/papers/2007/aniso.pdf).
The unsaturated constant-alpha terms keep their separate historical arithmetic.

For a true zero endpoint conductivity the harmonic coefficient is zero. This is
a degenerate cold-face limit, not evidence of resolved cold-front propagation.
Positive-temperature conductivity underflow is rejected, not silently treated
as that limit. Large temperature contrasts need spatial-refinement controls.
An earlier potential-gradient implementation and its evidence are superseded:
limiting `Psi(T)=2*K0_code*T^(7/2)/7` lost the affine-temperature perpendicular
null. The Kirchhoff helper remains a reference-only calculation, not a runtime
option. Its valid 1-D literature precedent does not validate the old transverse
potential-limited stencil.

The explicit path supports Newtonian SSP RK1/RK2/RK3; nonlinear RKL2 is described
below.
Its timestep adds the constant-alpha rate to a global-temperature-cap Spitzer
rate, includes consumed ghost cells, and applies the mesh CFL factor once.
The bound is recomputed at every SSP stage using the full constituent
forward-Euler interval, including RK2's second stage. Stage checks compare
against the unscaled current conduction bound, not a second application of the
requested CFL safety margin. This permits small changes within the reserved
margin as the coupled fluid heats or compresses. Exceeding the current bound is
rejected; the committed timestep is never changed mid-step, and explicit SSP
has no automatic rollback/retry. An unrepresentable
conductivity cap is also rejected, not treated as zero diffusion.
Active diffusion also rejects unrepresentable inverse heat capacity or inverse
cell-width-squared instead of silently erasing a finite physical diffusion rate.
The timestep path additionally requires normal-positive active intermediate
density rows and alpha/Spitzer rate contributions, checking each separately.
Zero or subnormal intermediates are rejected even if later rescaling could
produce a representable final rate; arbitrary extreme-range products are not
supported. A positive contribution from another term cannot hide such loss.
Zero coefficients remain valid and do not activate these checks by themselves.

Before the normal EOS conversion, Spitzer checks raw conserved states and the
current CT field over the consumed halo. Local-copy EOS conversion detects
states that would require density, pressure, temperature or entropy floors.
Failure is coordinated across MPI ranks. This covers normal initialization,
restart and the Hydro/MHD conversion task, but is not an audit of unrelated
modules' direct EOS calls or of all intermediate AMR transfer operations.

Earlier
Fourier/pulse/restart passes and six oblique continuum-rate failures belong to
the potential-based candidate. They remain preserved; they are not acceptance
of this harmonic revision. Dedicated harmonic CPU/GPU face tests now pass,
including exact affine perpendicular nulls for both limiters and dispatches.
CPU, two-rank CPU and GPU saturated pulses and real restarts pass. Separate
nonlinear pulse refinement and explicit RK1/2/3 temporal-order checks pass;
the latter use double diagnostics, not float32 output plateaus. These narrow
checks alone do not establish paper-benchmark, coupled/refined-mesh or nonlinear
Spitzer/saturation/STS acceptance. The subsequent targeted evidence and final
candidate validation are recorded below; the potential-based results were not
substituted for harmonic-operator validation.

## Explicit physical saturation

`conduction_saturation = cowie_mckee` enables physical saturation in the
`<hydro>` or `<mhd>` block. `saturation_phi` is finite and nonnegative, with
default 0.3. The default `conduction_saturation = none` retains the unsaturated
dispatch. Supplying `saturation_phi` when saturation is disabled is rejected.
Saturation can accompany constant-alpha terms, Spitzer conduction, or their sum.

At each face the code first reconstructs the complete classical conductive
vector `C`, including all active terms, then applies

```
F = C * q_sat / (q_sat + |C|)
q_sat = 5 * phi * P_donor^(3/2) / sqrt(rho_face).
```

`rho_face` is the arithmetic mean density; `P_donor` is the cell-centered
pressure on the sending side of the physical normal heat flux. It is not a
reconstructed Riemann pressure. This is the isothermal sound-speed convention,
with no extra gamma factor in `q_sat`. Zero phi suppresses conduction, but does
not bypass thermal-state validation.

Only the conductive increment is limited: preexisting fluid energy flux is
left untouched. Saturation is applied to the summed vector, not to individual
components or separate conductive contributions. Out-of-plane magnetic
components therefore matter in 2-D. Even isotropic saturation consumes
transverse temperature slopes and edge ghost cells to reconstruct the norm.
The vector closure uses a scaled norm. If normalizing a nonzero component would
lose its representable final flux, an exponent-split fallback retains that
component; ordinary-range arithmetic is unchanged. Genuine underflow of the
final component may still produce zero.

The explicit SSP RK1/RK2/RK3 implementation retains a
conservative unsuppressed diffusion timestep and adds a separate rate bound
for the donor-pressure response and outgoing thermal-energy drain. These are
recomputed at each SSP stage. The drain check bounds conduction alone against
the cell's internal energy, not its energy above an EOS floor; it is not a
coupled-MHD positivity guarantee. Explicit SSP has no automatic rollback/retry.
Pure helper, native face-oracle, pulse and restart tests alone do not establish
nonlinear STS or general refined-mesh acceptance. The additional bounded
Spitzer/saturated refined-mesh results are described below.

## Nonlinear RKL2 integration

The same `conductivity_integrator=sts` and `time/sts_integrator=rkl2` selectors
now connect the nonlinear residual to checked RKL2 trials. Conductivity,
limiters and saturation are recomputed every stage; only the recurrence
coefficients are fixed within an attempt. Both split halves refresh primitives
from the current conserved state. Raw active and consumed-halo thermal states
must stay inside the trial-start envelope, with a temperature-scaled roundoff
allowance, without invoking EOS floors. Invalid raw states detected by the
active-cell or consumed-halo checks during a trial, envelope violations, or an
exceeded current donor-transport cap reject that trial. Stage checks can restore
conserved and primitive fluid arrays (U/W) early so remaining tasks see safe
state. After communication completes, the driver restores U/W again, followed
by interval halving. This is not a snapshot of the CT field or
the whole solver. At most 32 halvings are allowed per segment. Invalid initial
or sweep-start raw states and constitutive/rate representability errors remain
fatal; not every numerical error is recoverable.

The saturation donor-pressure bound limits the outer timestep and the full
trial interval, without STS acceleration. The explicit energy-drain limit is
not imposed on the full RKL interval. RKL stage counts use the unsuppressed
diffusion row budget and raw stages are checked independently. This is not a
nonlinear stability theorem or an automatic temporal-accuracy controller.

This path currently requires one fluid, conduction as the sole STS process,
no custom user boundary callbacks or cyclic zoom, and conserved-variable prolongation when
using refined meshes. Earlier constant-coefficient combinations are unchanged.
Targeted nonlinear time-refinement, restart, device, MPI, retry and bounded
refined-mesh validation is complete for the cases recorded here. The final
candidate also completed all four native lanes; this does not extend acceptance
to arbitrary configurations or turn the selector into an accuracy guarantee.

The first nonlinear development matrix now passes finite-amplitude temporal
convergence, serial/two-rank CPU/single-GPU pulse and restart comparisons, and
temperature-normalization controls. Separate test-only CPU relinks exercise
one-rank internal-stage faults, exact U/W rollback and bounded retry exhaustion;
they are not production runtime switches. Later post-refinement MHD relinks
also pass on serial CPU, two CPU ranks and one GPU: negative raw energy,
positive thermal-envelope violation and a measured donor-bound violation each
trigger collective rejection and exact U/W restoration with unchanged CT faces.
The recovered fields and accepted segment schedules match corresponding
clean-rejection controls exactly. MPI injects on one rank only. These bounded
checks exercise one rejected interval after a real regrid, not every possible
failure or arbitrary refinement pattern. The original GPU diagnostic FAIL is
preserved; its correction changed only test-only pointer-to-text reason checking.
Periodic one-level 2-D MC/van Leer
refinement checks pass actual coarse/fine heat transfer, refinement/derefinement
and real
post-regrid restarts on those three lanes, with unchanged strict raw U/CT gates.
Separate strongly saturated MC refinement controls also pass CPU/MPI2/GPU,
including same-grid explicit comparisons and post-regrid restarts. A separate
genuine 3-D MC matrix passes those three local lanes with strict raw U/CT
comparisons. A later test-fixture-only extension adds a cell-aligned density
step and permits dynamic MHD; absent-key default/restart fields remain exact.
Static piecewise-density Spitzer and combined-alpha cases, including a
constant-temperature pressure-jump control, pass CPU/MPI2/GPU with strict
read-only lane comparisons. Checker bookkeeping corrections retain the original
FAIL records and do not change numerical gates.
An additional uniform evolving-field HLLD/RK2 CPU pilot passes two explicit
controls and observed/ordinary RKL2 comparisons, with raw positivity, conserved
budgets, full-cycle CT and actual fluid-entry CFL reconstruction. The observer
does not change the ordinary final parsed fields or timestep trace. This small
ratio4 pilot remains diffusion-limited. A second, ratio64 observed/ordinary
extension passes at 99.94% of configured fluid CFL, reusing both explicit
controls and retaining all gates. Matching ordinary RKL4/RKL64 GPU cases pass
strict final raw CPU/GPU comparisons and zero-EOS checks; they do not include
the CPU-only stage observer. These short uniform checks do not establish
general coupled CFL safety, arbitrary refined coupled evolution or inter-node
operation; the separate bounded refined/recovery checks are described here.

A longer uniform CPU matrix at ten times that endpoint also passes: four
adaptive-CFL RKL refinements, two explicit references, an ordinary control and
its real midrun restart. Finest empirical combined-field rates are near2,
with separately checked small reference sensitivity. The ordinary restart
reproduces final parsed U/CT and the continued timestep/STS sequence exactly.
Matching longer single-GPU and two-rank CPU runs also pass the strict serial
CPU-reference comparison; their own genuine midrun restarts reproduce final
parsed fields and continued timestep/STS sequences exactly. These local
lane tests reuse the serial references, without a GPU/MPI stage observer.
This is empirical adaptive-time evidence, not a formal fixed-step order proof
or validation of coupled refined meshes.

A narrow coupled static-SMR boundary preflight also passes its separately
audited native magnetic-flux checks. All observed conduction stages and
field-prolongation calls leave active magnetic faces unchanged, with exact
ordinary/observer final agreement. Subsequent same-grid explicit controls and
their timestep-halving comparison pass the existing volume-weighted U/CT and
temperature gates, reusing that RKL endpoint. Coarse/fine error partitions
and the independent norm re-audit pass. A separate static-SMR HLLD-entry
observer also passes the first-stage configured-CFL bound and near-active
exercise target, using the actual fine-cell spacing. It retains exact
ordinary-result transparency. Stage2 full-dt CFL remains diagnostic and can
slightly exceed the configured first-stage setting; this is not an every-stage
bound or general stability proof. A subsequent six-cycle coupled dynamic-AMR
case and real post-refine/post-derefine restarts pass on serial CPU, two CPU
ranks and one GPU. Each lane passes raw-state and conservative
fluid/native-magnetic-flux budgets, with exact within-lane continued U/CT,
topology-age and timestep/RKL comparisons. Strict read-only cross-lane audits
also pass: CPU1/CPU2 raw fields are exact; the maximum CPU/GPU raw difference
is about 1.10e-14 under the unchanged 5e-12 scaled gate. The only
final-checkpoint parameter exception is the
unused event-log file counter; physics parameters and restart/binary counters
remain strict. Original checker failures are retained separately.

A separate uniform-density static-SMR adaptive-time refinement also passes
on the recorded pre-fix CPU candidate. RKL64 CFL 0.4/0.2/0.1/0.05 gives finest
empirical aggregate-field rates 2.020/2.017 and temperature rates 1.838/1.986;
the explicit-reference sensitivity ratios are 0.00339/0.00946. Every RKL
endpoint agrees with both explicit controls under the unchanged 0.005 normalized
L1 gate, and the explicit halfstep gate is 0.001. This is same-grid empirical
adaptive-time evidence, not a fixed-step or continuum-order theorem. The
coupled rho=1/2 static-SMR RKL endpoint passes raw/budget checks, while both
original explicit density controls aborted at the old stage guard. A test-only observer
measured unchanged initial/stage-1 bounds and a small stage-2 reduction; actual
dt remains only about 0.10/0.05 of the current CFL-free bound. The guard has
been corrected to check that CFL-free bound, retaining configured CFL for
new-step selection. The three post-fix density cases now pass: the scaled
explicit-halving difference is 1.22e-7, and RKL versus fine explicit is
0.004816 against the unchanged 0.005 gate (energy is limiting). Its temperature
difference is 0.000666. This is a pass close to the declared ceiling, not a
claim of high accuracy for arbitrary density contrasts. The old/new RKL raw
endpoint is exact. A separate test-only observer passes both guard controls:
the safe stage completes, while setting stage2's constituent interval to
1.01 times its current bound triggers the unchanged fatal diagnostic before
that stage advances. Original failures and old-candidate uniform-order evidence remain
preserved with their own binary provenance.

On a composite
mesh, volume-weighted arithmetic cell-centred B is not generally an exact CT
invariant; validation uses native plane-integrated normal magnetic flux and
keeps the cell-centred mean as a separate diagnostic. The original mean-B
diagnostic failure is preserved with its independent quadrature explanation
and analytic divergence-free counterexample, not hidden by a larger tolerance.

A separate periodic saturated-sawtooth benchmark also passes six CPU runs:
constant conductivity40, rho1, gamma2, phi0.3, N200/N400 and endpoint0.1.
The N400 explicit timestep-halving normalized L1 difference is1.38e-10;
RKL2 requested ratios40/1000 differ from the fine explicit result by
1.84e-8/2.32e-5 in normalized L1 and6.26e-7/4.81e-4 in Linf. Normalization
uses the fixed20 temperature scale. Every raw-state, conserved-integral and
ordinary/STS stage-accounting check passes without retries. For ratio1000,
the measured outer diffusion number `40*dt/dx^2` is13.269–28.339, not the
nominal50: the nonaccelerated donor bound remains active. This is a scalar
constant-conductivity saturated test, not a Spitzer benchmark, a digitized
paper-error measurement, or multidimensional/AMR acceptance.

## Rejected or unvalidated configurations

- Nonlinear Spitzer/saturated STS has the restrictions and bounded validation
  listed above. `q_limit` must be finite and
  at least the legacy disabled sentinel `3.40282e38`; it is not an operational
  saturation strength. The rounded sentinel is retained for restart compatibility.
- The supported STS interface does not include the research build switch or
  raw-stage observer machinery. Validation does not cover every configuration.
- Refined-mesh validation is bounded, not general SMR/AMR acceptance. Periodic
  one-level 2-D/3-D kinematic tests now pass conservation, analytic Fourier
  accuracy, static coarse/fine interfaces, real refinement/derefinement and
  restarts with both limiters and explicit/RKL2 evolution. The 2-D matrix passes
  on CPU, two CPU ranks and GPU; the 3-D matrix passes on two CPU ranks and GPU.
  However, six N32 3-D CPU/GPU energy comparisons fail the strict 5e-12 agreement
  gate (maximum scaled energy difference 5.45e-9); nonenergy fields, CT faces,
  times and topology match exactly. A representative common-seed N32 van-Leer
  control is exact across one/two CPU ranks, but CPU/GPU differs by4.07e-10.
  Halving CFL reduces that backend gap to5.34e-12 while a preserved tiny CPU
  seed perturbation still amplifies. Its continuum-accuracy gate passes and
  both CPU endpoints exactly reproduce historical constant-alpha outputs.
  This is measured precision sensitivity, not a demonstrated CFL fix or a
  pass of the six original strict comparisons. The user explicitly accepted
  the bounded discrepancy for this candidate before final validation. The
  maximum uses `abs(a-b)/max(1,abs(a),abs(b))`; the worst energy values are
  2.02--2.29, so 5.45e-9 is a relative energy difference. This is an accepted
  release limitation, not a solver correction or a relaxed test gate. The
  original strict failures remain preserved. This earlier
  constant-coefficient evidence alone does not validate all raw substages,
  arbitrary refinement patterns, evolving-field refined MHD, nonlinear
  conductivity or inter-node refined meshes; the newer bounded nonlinear
  refinement checks are described above. There is
  no blanket SMR/AMR runtime rejection.
- Positive anisotropic conductivity in relativistic configurations is rejected:
  this is a Newtonian energy-flux closure. Arbitrary user boundary conditions are
  outside the current validation claim. Thermal boundary fluxes depend on the ghost-state
  prescription; a hydrodynamic boundary label alone does not prove thermal
  insulation or a prescribed heat flux.

These limitations describe evidence and release gates. Do not assume that every
unvalidated configuration already has a runtime rejection guard.

## Final candidate validation, 2026-10-10

All four frozen native inventories are complete without changing the tested
solver, inputs, assertions or numerical gates:

| Lane | Passed | Existing skips | Completion |
| --- | ---: | ---: | --- |
| CPU | 276 | 15 | 290 completed IDs plus the one interrupted test resumed separately |
| MPI-CPU | 59 | 0 | Two-node job 15292089, exit 0 |
| GPU | 65 | 0 | Disjoint 14 + 40 + 11 test shards |
| MPI-GPU | 12 | 0 | Two-node job 15292090, exit 0 |

The CPU wrapper reached its 55-minute deadline after 275 passes and 15 skips;
the final Z4c AMR-boundary test had started but had no terminal result. Only
that test was resumed and passed. The combined, disjoint coverage equals all
291 frozen CPU IDs; the original timeout/exit 124 is retained. The existing
skips comprise seven reference-versus-reference comparisons and eight
unsupported SR MHD/HLLC combinations, not newly disabled acceptance tests.

The first GPU shard passed all 14 tests but its scratch wrapper subsequently
exited 2 after an in-flight wrapper edit. Complete pytest phase records, JUnit
and unchanged candidate hashes independently establish those 14 passes; the
wrapper failure is preserved and is not claimed as a clean job exit. The other
51 GPU tests completed cleanly. All 590 pinned runtime files were reverified.
The public port preserves the tested numerical source and thresholds. Subsequent
changes are documentation and standalone-runner portability/checker tests; the
latter receive focused validation rather than another full numerical round.

This completion includes the explicit acceptance of the earlier bounded 3-D
energy discrepancy above, not a reclassification of its original strict FAIL.
Detailed provenance and evidence locations are in the
[implementation guide](transport_implementation.md#validation-and-provenance).

## Tests and interpretation

The normal CPU discovery adapter
`tst/test_suite/diffusion/test_transport_reference_cpu.py` runs analytic-reference
and synthetic checker-corruption tests from `tst/transport`. It needs Python,
NumPy, SciPy and pytest, but does not launch Athena or require a site-specific
host. For example, from the repository root:

```sh
python -m pytest tst/test_suite/diffusion/test_transport_reference_cpu.py
```

The `test_transport_functional_{cpu,gpu,mpicpu,mpigpu}.py` adapters run the
provided executable without compiling or submitting jobs. Set
`ATHENAK_TEST_BINARY` to its absolute path (default: `tst/build/src/athena`).
For MPI, set `ATHENAK_TEST_LAUNCHER`, for example `mpirun -np 2` (the default) or
a site-appropriate `srun` prefix inside an approved allocation. Set
`ATHENAK_TEST_RANKS` to its rank count (default: 2); the Fourier runner checks
Athena's reported rank count rather than inferring MPI execution from the build.
GPU binding and
execution permissions are the caller's responsibility. The existing suite driver
does not have an MPI-GPU option; invoke that adapter directly with pytest. Select
only the appropriate lane, not all backend adapters against a single executable.

These portable functional tests cover disabled transport, isotropic explicit/STS,
both anisotropic limiters, field orientation/zero-field cases, a 3D mode and
actual restarts. The single-process lanes also check exact input-rejection
diagnostics. Missing executables fail explicitly. Temporary run directories retain
commands, binary/source hashes, logs, checkpoints and numerical reports.

The `test_anisotropic_sts_*` functions in the same four adapters additionally
run `tst/transport/run_sts.py`: for each limiter, 18 executions and 12 comparisons
cover periodic 24-square and 24-cube modes, plus a 32-by-8 zero-normal-heat-flux
outflow configuration. Each geometry runs explicit CFL 0.1/0.2 controls, STS
ratios 16/64 and a real interrupted/restarted calculation. MPI adapters run all
of those executions at the requested rank count. The standalone runner also
accepts `--comparison-launcher` and `--comparison-ranks` to test another
decomposition against the primary one. For example:

```sh
python tst/transport/run_sts.py --binary /path/to/athena \
  --root /path/to/new-results --comparison-launcher 'mpirun -np 2'
```

The explicit-reference gate is 1e-4 and the STS-to-fine-explicit gate is 1e-2,
using the fixture's fixed peak-to-peak temperature scale (0.02). Restart and
decomposition comparisons require maximum stored-field difference at most 1e-6
and double-precision diagnostic differences at most 1e-10. Coordinate, time,
finite-data, analytic mode, conservation and bound checks are separate gates.
Synthetic corruption tests exercise the comparator through the CPU reference
adapter. The 3D fixture uses N24 because N16 misses the pre-existing continuum
amplitude gate even for the fine explicit control; the gate was not relaxed.

The `test_transport_localized_{cpu,gpu}.py` adapters additionally cover small
2D/3D Gaussian pulses and density-interface/boundary heat budgets with actual
restarts, for both limiters. These are single-process fixtures, not MPI or AMR
acceptance. Additional standalone harnesses cover extended ring/leakage matrices;
the caller supplies an appropriate execution environment. Verbose coupled-3D and raw-stage
observer fixtures remain separate research harnesses, not portable suite tests.

The additional `test_transport_nonlinear_{cpu,gpu,mpicpu,mpigpu}.py` adapters
exercise harmonic Spitzer conduction with Cowie-McKee saturation. Every lane
runs the same small finite-amplitude oblique pulse with two explicit timestep
controls, RKL2 and a real interior restart. These are same-grid accuracy and
restart checks, not a continuum or temporal-order measurement. Single-process
lanes additionally check native three-dimensional total-flux calculations and
both limiters' exact perpendicular-null controls; the CPU lane adds eighteen
configuration, raw-state and timestep-range rejection cases. They use the same executable/launcher
environment variables as the other functional adapters and never build Athena.

`test_transport_closure_cpu.py` discovers the constitutive helper tests and
nonlinear checker corruption tests in the normal CPU suite. By default it
compiles only tiny standalone C++17 wrappers in temporary test directories;
prebuilt wrappers can be selected with `ATHENAK_CONDUCTION_CLOSURE_EXE`,
`ATHENAK_SPITZER_POTENTIAL_EXE` and `ATHENAK_SATURATED_SPEED_EXE`. Potential-form
helpers remain reference utilities, not the production Spitzer face law.
The saturated-vector helpers include decimal-reference checks for small
components and same-backend comparisons against the original ordinary-range
arithmetic. These are constitutive unit tests, not full fluid simulations.
The portable checks complement, but do not replace, the separate full raw-CT,
dynamic-AMR and deliberate retry-recovery diagnostics.

Report execution, invariants, temporal accuracy, spatial accuracy and device
agreement separately. A coarse continuum-target miss must remain visible even
when a timestep-halving control and CPU/GPU comparison pass.
