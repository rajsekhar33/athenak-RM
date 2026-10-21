# Thermal-conduction implementation and validation

Numerical methods, source locations and feature-specific validation for
constant isotropic/anisotropic conduction, Spitzer conductivity, saturation,
RKL2 integration, refinement and restart. Configuration is documented in
[Thermal conduction](../../thermal_conduction.md).

## Source and numerical data

- Baseline: `588d737fac0251b5538c287cf50c3ccdd4cad706`.
- Described source: `c3a8fccca928eef8b22d233458c4a4aabd9cb171`.
- [Source manifest](source_manifest.json): file hashes and every changed hunk
  across the 77-path implementation delta.
- [Numerical results](results.json): measured values, comparison criteria,
  figure arrays and source-report hashes.

The studies cover successive implementation versions, identified by binary
hashes. Early constant-coefficient measurements predate the final nonlinear
implementation. Harmonic Spitzer results supersede the potential-flux
prototype. The final file consolidation preserves the formulas and class
interface.

## Methods

The code temperature is `T=p/rho`; primitive `w(IEN)` stores internal-energy
density. Constant conduction adds
`q=-rho*alpha_iso*grad(T)-rho*alpha_aniso*bhat*(bhat dot grad(T))` to the
conserved energy flux. Anisotropic faces combine the CT normal field with
averaged transverse components and normalize using all three components.
Harmonic face coefficients and nested van Leer or monotonized-central
temperature-slope limiters follow the limited-asymmetric construction of
[Sharma & Hammett (2007)](https://arxiv.org/abs/0707.2616).
Unsaturated constant-isotropic conduction retains arithmetic face density.

Spitzer conductivity uses the harmonic mean of endpoint `K=K0*T^(5/2)`,
multiplied by temperature gradients. The Units module converts a supplied
cgs prefactor when requested. Saturation limits the summed
classical vector once:
`F=C*q_sat/(q_sat+|C|)`,
`q_sat=5*phi*P_donor^(3/2)/sqrt(rho_face)`.
The donor convention and smooth interpolation are described in the
[user guide](../../thermal_conduction.md) and
[Vaidya et al. (2017)](https://arxiv.org/html/1702.05487v2).

Explicit SSP stages check their current CFL-free conduction bound. RKL2
half-sweeps surround fluid evolution, refreshing nonlinear fluxes and primitive
states at each stage. If a trial state is invalid, MPI communication completes
before U/W restoration and interval halving. Each segment allows up to 32
halvings. Invalid initial states or unrepresentable closure/rate intermediates
terminate the run. Nonlinear monotonicity still requires problem-specific checks.

## Changed implementation locations

Line numbers refer to the pinned source above; the [hunk inventory](source_manifest.json)
records every changed location. Helpers are consolidated in one implementation
unit; the earlier closure, Spitzer and saturated auxiliary files were removed.

| File | Changed functions/locations and purpose |
|---|---|
| [conduction.cpp](../../../src/diffusion/conduction.cpp), 47–533 | Device helpers: finite/range checks, harmonic conductivity, temperature/slopes, VL/MC limiting, vector saturation, donor response, unit conversion and saturated-face assembly. Potential-gradient helpers remain reference-only. |
| Same, 589 | Constructor validates coefficients, geometry, saturation, integration and units. |
| Same, 736–829 | `FailIfInvalid`, `ValidateRawState`, `InspectRawState`: pre-floor raw thermal checks and collective failure. |
| Same, 830–869 | `UsesNonlinearSTS`, `BeginSTSTrial`, `RestoreSTSTrial`, `CheckSTSTransport`: nonlinear snapshot, rollback and trial restrictions. |
| Same, 870–1010 | `SpitzerTimeStep`, `CheckExplicitStage`: temperature-/density-aware bounds and current SSP-stage validation. |
| Same, 1012–1225 | `AddHeatFluxes`, `AddHeatFluxIso/Aniso`, `NewTimeStep`: dispatch, constant fluxes, CT projection and combined timestep. |
| Same, 1234–1485 | `AddHeatFluxSpitzer`, `AddHeatFluxSaturated`, `SaturationTimeStep`: harmonic nonlinear faces, summed-vector cap and donor/drain bounds. |
| [conduction.hpp](../../../src/diffusion/conduction.hpp), 30–80 | Coefficients/selectors, raw-state/trial/timestep interfaces and primitive backup. |
| [sts_rkl2.cpp](../../../src/diffusion/sts_rkl2.cpp), 41–49 | Adds NaN/Inf rejection to `ComputeRKL2StageCount`. Odd stage counts and capacity/rounding behavior predate this change. |
| [driver.cpp](../../../src/driver/driver.cpp), 63/341/422/498/730/751; [driver.hpp](../../../src/driver/driver.hpp), 35–38/91 | Nonlinear owner selection, controller reset, supported-combination checks, `ExecuteSTSSweep`, pre/post calls and trial state. |
| [hydro.cpp](../../../src/hydro/hydro.cpp), 104–109; [mhd.cpp](../../../src/mhd/mhd.cpp), 154–159 | Register explicit/STS and nonlinear conduction. |
| [hydro.hpp](../../../src/hydro/hydro.hpp), 170; [mhd.hpp](../../../src/mhd/mhd.hpp), 227 | Declare the `STSCheckState` task. |
| [hydro_newdt.cpp](../../../src/hydro/hydro_newdt.cpp), 130–134; [mhd_newdt.cpp](../../../src/mhd/mhd_newdt.cpp), 169–173 | Retain the unaccelerated saturation donor bound in the fluid timestep. |
| [hydro_sts.cpp](../../../src/hydro/hydro_sts.cpp), 70/91/187/214 | `STSFluxes`, `STSUpdate`, `STSCheckState`, refresh: stage fluxes, recurrence, raw/envelope checks and timestep. |
| [mhd_sts.cpp](../../../src/mhd/mhd_sts.cpp), 97/129/224/328/353 | Matching MHD stages with current CT magnetic energy; rejected-stage handling adds no conductive induction term. |
| [hydro_tasks.cpp](../../../src/hydro/hydro_tasks.cpp), 93/203/449; [mhd_tasks.cpp](../../../src/mhd/mhd_tasks.cpp), 110/285/662 | Task dependencies, explicit-stage guards and active/consumed-halo checks before EOS conversion. |
| [advect_mhd.hpp](../../../src/mhd/rsolvers/advect_mhd.hpp), 17/44/56 | Energy/passive-scalar advection correction used by kinematic transport fixtures. |
| [src/CMakeLists.txt](../../../src/CMakeLists.txt), [pgen.cpp](../../../src/pgen/pgen.cpp), [pgen.hpp](../../../src/pgen/pgen.hpp) | Transport fixture registration, dispatch and declarations. |

### Fixtures, oracles and feature tests

The manifest lists every changed file and hunk. Test initializers and
diagnostics exercise the production conduction operator.

| Location | Feature-specific responsibility |
|---|---|
| [transport_mode.cpp](../../../src/pgen/tests/transport_mode.cpp), 44/133/204 | Advection oracle, Fourier measurements and specialized diagnostic dispatch. |
| [transport_refinement.cpp](../../../src/pgen/tests/transport_refinement.cpp), 51/174/182/207/213 | Raw U/CT measurements, topology observations, refine/derefine control, final diagnostics and cell-average initialization. |
| [src/pgen/tests](../../../src/pgen/tests): `transport_coupled.hpp`, `transport_equilibrium.hpp`, `transport_front.hpp`, `transport_gaussian.hpp`, `transport_helical.hpp` | Coupled modes, source-balanced equilibria, fronts, Gaussian tensors and helical geometry. |
| Same: `transport_operator.hpp`, `transport_patch.hpp`, `transport_ring_state.hpp`, `transport_saturated_operator.hpp`, `transport_sawtooth.hpp`, `transport_stages.hpp` | Small-kernel oracles, patches/rings, restartable ring diagnostics, nonlinear face states, sawtooth and raw-stage checks. |
| [tst/inputs](../../../tst/inputs): eight `transport_*.athinput` decks | Coupled, equilibrium, front, Gaussian, helical, mode, patch and refinement fixtures. |
| [tst/transport](../../../tst/transport): `oracles.py`, `gaussian_oracle.py`, `result_policy.py` and their `test_*` modules | Analytic Fourier/Gaussian references, quadrature and status/checker controls. |
| Same: `run_pilot.py`, `run_patch.py`, `run_equilibrium.py`, `run_gaussian.py` and matching checker tests | Constant transport, localized profiles, equilibrium/budget checks and real restarts. |
| Same: `run_sts.py`, `sts_checks.py`, `test_sts_checks.py` | RKL2 mode, boundary, restart/decomposition comparisons and deliberate checker faults. |
| Same: `run_nonlinear.py` | Harmonic/saturated face oracles, pulses, raw-state/range guards and restart comparisons. |
| Same: `test_conduction_closure_cpu.py`, `test_saturated_speed_cpu.py`, `test_spitzer_potential_cpu.py` | Extracted production scalar helpers, donor-response/range checks and reference-only potential arithmetic. |
| [tst/test_suite/diffusion](../../../tst/test_suite/diffusion): `test_transport_*`, `transport_functional.py`, `transport_nonlinear.py` | Conduction test discovery and existing-binary CPU/GPU/MPI adapters. |
| Same: `test_sts_rkl2_controller_cpu.py` | Independent Legendre stage polynomials, scalar second-order convergence, half-sweep ratios and invalid times. |

## Conduction-specific validation

| Test family | Result and scope |
|---|---|
| Fourier/tensor modes | Aligned/oblique, isotropic/mixed, field reversal, out-of-plane/zero fields, density weighting and decomposition; bounded operator checks pass. |
| Positivity/checkerboard kernels | Kernel tests compare both limiters with raw thermal bounds and a discrete damping reference; coupled-MHD positivity requires separate checks. |
| Patches/fronts and Gaussian diffusion | Localized contrasts, conserved heat and raw stages; Gaussian and refinement cases compared independently with cell-average references. |
| Circular ring | Eight N50/100/200/400 cases, both limiters, to t=200. N400 published norm differences <0.23% for van Leer and <0.12% for MC. |
| Sovinec ratio-100 equilibrium | Four N8/N16 cases become stationary but fail their original accuracy criterion; measured leakage decreases with refinement. |
| Helical/curved fields | Three-component projection, perpendicular-null/heat-flow and restart checks; geometry-/resolution-specific accuracy. |
| Constant-anisotropic RKL2 | CPU/GPU cases and field comparisons, including ordinary taskgraph, refined 3D, boundary ghosts, evolving fields and restart. |
| Inter-node RKL2 conduction | MPI-CPU/MPI-GPU evolution, restart and cross-backend field comparisons. |
| Nonlinear face oracles | CPU/GPU checks of both limiters, mixed terms, units/scaling and exact perpendicular-null configurations. |
| Harmonic Spitzer pulses | Spatial-refinement comparisons against a finer-grid solution; CPU/MPI2/GPU pulse and restart checks. |
| Explicit temporal refinement | Ten cases/four comparisons; RK1/RK2/RK3 self-orders 1.01222/2.02432/3.02180 for a fixed-grid nonlinear pulse. |
| Constitutive/raw-state guards | Rejection tests for invalid selectors/units, saturation-only raw faults, one-owner MPI failure and extreme rate/spacing values. |
| Nonlinear RKL2 recovery | Test-instrumented Hydro and post-refinement MHD CPU/MPI2/GPU: raw/envelope/donor faults, exact U/W restore, clean retry equivalence and bounded exhaustion. |
| Coupled nonlinear MHD | Eight CPU cases/eleven checks to t=.2; GPU/MPI2 restart pairs and within-backend restart endpoint agreement. |
| Static SMR | Plane-flux conservation, stage CFL and explicit-reference checks. Density-step largest RKL normalized error 0.00481588 versus 0.005 criterion. |
| Dynamic AMR | CPU/MPI2/GPU: uninterrupted and post-refine/post-derefine restart runs, 4→7→4 leaves; three endpoint and 17 checkpoint cross-lane pairs. Largest GPU raw gap 1.09499620857e-14 versus 5e-12. |
| Saturated sawtooth | Six CPU cases/four comparisons, explicit halving controls and RKL ratio caps 40/1000 at N400, t=.1; isotropic 1D benchmark. |
| Disabled-feature compatibility | Constant-alpha controls, disabled saturation, legacy sentinels and old/new real restarts preserve measured behavior. |
| Consolidated module | Three conduction runs per CPU/GPU; all fields at 149 checkpoints match the prior consolidated reference bitwise; restart endpoint exact; maximum energy drift 3.92e-15. |

### Ring diffusion

![Ring error comparison](figures/ring_errors.png)

Measured norms are compared with the tabulated asymmetric-limiter values in
[Sharma & Hammett](https://arxiv.org/abs/0707.2616). The sequence combines
historical N50 CPU data with N100–400 GPU data from different builds. Van-Leer
L-infinity error increases from N200 to N400. The ambiguous published N100
perpendicular-diffusivity entry is excluded from the comparison criteria.

### Explicit and RKL2 temporal accuracy

![Explicit temporal accuracy](figures/explicit_temporal.png)

The plotted error is the maximum double-precision cosine/minimum/maximum
temperature difference divided by initial amplitude, against a finer same-grid
RK3 solution. Double diagnostics resolve errors below float32 field precision.
The orders are estimated from adjacent numerical solutions.

![Coupled RKL2 temporal accuracy](figures/coupled_rkl_temporal.png)

Eleven scaled U/CT component norms are distinguished from temperature. The
adaptive timestep measure is `H=sqrt(sum(dt^3)/elapsed_time)`. Finest
combined-field empirical orders are 1.99617/2.03556; temperature orders
2.09397/2.14279. Explicit-reference halving differences are below 1% of the
finest RKL error. Spatial resolution is fixed throughout this temporal study.

### Saturated sawtooth

![Saturated sawtooth](figures/saturated_sawtooth.png)

Temperature fields follow the
[Vaidya et al. section 4.1.2 setup](https://arxiv.org/html/1702.05487v2#S4.SS1.SSS2),
with a refined explicit solution as the reference. RKL40/fine-explicit
normalized L1/L-infinity differences are 1.84469e-8/6.26363e-7;
RKL1000 gives 2.31719e-5/4.80778e-4, below 0.002/0.02. Explicit timestep
halving produces a smaller difference. Actual acceleration also depends on
the active donor bounds. Retry recovery is tested separately.

## Numerical limitations

![Coarse accuracy limitations](figures/retained_coarse_limits.png)

- Coarse Sovinec cases reach stationarity and close their heat budgets, but
  fail the accuracy criteria.
- Six static N32 3D refinement CPU/GPU energy comparisons fail the strict
  `5e-12` agreement criterion. Maximum relative discrepancy is
  `5.4542320335866934e-9`; temperature difference/initial amplitude is
  `7.690473591281943e-8`. The numerical dataset includes all 24 case outcomes,
  including these six failures and their tolerance.
- The earlier potential-flux Spitzer tests had accuracy failures. That
  discretization was replaced by the harmonic stencil tested here.
- CPU/GPU agreement, conservation, restart agreement and continuum accuracy
  are separate tests. CT checks use native plane-integrated magnetic flux;
  cell-centered mean field is not a general composite-mesh invariant.
- Nonlinear STS requires one fluid, conduction as the sole STS process and
  conserved prolongation on refined meshes. Cyclic zoom and custom boundary
  callbacks are unsupported. Thermal flux depends on ghost states; a fluid
  boundary label alone does not imply insulation or prescribed heat flux.
- RKL linear stability alone does not ensure nonlinear monotonicity. Sharp
  fronts require spatial and temporal refinement. Unrepresentable
  constitutive values and timestep rates terminate the run.

## Reproduction

```bash
python docs/feature-validation/conduction/plot_results.py
```

NumPy and Matplotlib are required. All five figures use only bundled data;
`--output DIRECTORY` chooses another destination. The optional
[exporter](export_results.py) documents extraction from original numerical
reports. Native conduction checks use the existing
[runners](../../../tst/transport) and
[backend adapters](../../../tst/test_suite/diffusion) with a supplied executable.
The plotting script reads the bundled results; the runners execute simulations.

## References

- [Sharma & Hammett (2007)](https://arxiv.org/abs/0707.2616), JCP 227, 123–142: limited anisotropic diffusion and ring tests.
- [Spitzer & Harm (1953)](https://doi.org/10.1103/PhysRev.89.977): collisional temperature-dependent conductivity.
- [Cowie & McKee (1977)](https://doi.org/10.1086/154911): saturated heat-flux scale.
- [Meyer, Balsara & Aslam (2014)](https://doi.org/10.1016/j.jcp.2013.08.021): stabilized RKL formulation.
- [Vaidya et al. (2017)](https://arxiv.org/html/1702.05487v2): RKL anisotropic diffusion, saturation and benchmarks.
