# Thermal conduction

AthenaK supports isotropic and magnetic-field-aligned conduction, Spitzer
conductivity, Cowie–McKee saturation, and RKL2 super-time-stepping (STS) for
Newtonian ideal-gas Hydro/MHD. This guide covers configuration, numerical
methods, implementation locations and scientific references.

## Constant-coefficient conduction

The code-temperature variable is `T = p/rho`. The conductive energy flux is

```text
q = -rho * alpha_iso * grad(T)
    -rho * alpha_aniso * bhat * dot(bhat, grad(T)).
```

In primitive arrays, `w(IEN)` stores internal-energy density, not pressure:
`p = (gamma - 1) * w(IEN)`. The conductive increment updates conserved total
energy; it does not directly update density, momentum or magnetic field.

`alpha_aniso` adds parallel transport: the parallel coefficient is
`alpha_iso + alpha_aniso`, while the perpendicular coefficient is `alpha_iso`.
At uniform density the temperature diffusion tensor includes a factor
`gamma - 1`. These coefficients are in code units, not cgs conductivity;
they do not require a `<units>` block.

Example for MHD:

```text
<mhd>
alpha_iso = 0.0
alpha_aniso = 0.1
alpha_spitzer = false
anisotropic_limiter = vanleer
conductivity_integrator = explicit
```

Supply the remaining mesh, fluid, time and problem settings normally.
Both coefficients must be finite and nonnegative; zero disables the
corresponding contribution. Isotropic conduction also works in Hydro.
Anisotropic conduction requires nonrelativistic MHD.

### Spatial discretization

The anisotropic flux uses the CT face-normal magnetic component and averaged
transverse cell-centered components. All three components enter normalization,
including the out-of-plane component in 2-D. Exactly zero face field gives zero
anisotropic flux; there is no weak-field cutoff.

The anisotropic face coefficient is the harmonic mean of the neighboring
densities times `alpha_aniso`. Unsaturated isotropic conduction uses arithmetic
face-density interpolation. Transverse temperature slopes use nested van-Leer
limiting by default; `anisotropic_limiter = mc` selects monotonized-central
limiting, not Monte Carlo particles. Both limiters return zero for opposing
slopes; for same-sign slopes, van Leer uses their harmonic combination while
MC bounds their centered average by twice the smaller magnitude.

Explicit conduction contributes to the energy flux in the ordinary RK update.
Its conservative density-aware timestep bound uses
`alpha_iso + 4*ndim*alpha_aniso`, where `ndim` counts active spatial directions.
Neither limiter eliminates resolution-dependent cross-field diffusion.

## Spitzer conductivity and physical units

Spitzer conductivity can be used alone or with the constant-alpha terms:

```text
<mhd>
alpha_spitzer = true
spitzer_kappa0_code = 0.1
spitzer_geometry = field_aligned
conductivity_integrator = explicit
<time>
integrator = rk2
```

Supply exactly one finite, nonnegative coefficient:
`spitzer_kappa0_code` or `spitzer_kappa0_cgs`. The cgs option requires explicit,
valid physical units and is converted using the units module's physical scales:

```text
K0_code = K0_cgs * T_unit^(7/2) / (P_unit * V_unit * L_unit)
K(T) = K0_code * T^(5/2),  T = p/rho.
```

The cgs coefficient multiplies `T_kelvin^(5/2)`; it is not the conductivity
evaluated at a reference temperature. The code uses a supplied prefactor; it
does not calculate a varying Coulomb logarithm or ionization dependence.
No low-temperature Parker branch or implicit conductivity ceiling is applied.
Conversion overflow or underflow of a positive coefficient is rejected.
Restarts retain the original coefficient
convention rather than serializing conflicting code and cgs inputs.

`spitzer_geometry` is `isotropic` or `field_aligned`. MHD must specify it;
Hydro defaults to isotropic and rejects field alignment. Spitzer-specific
parameters without `alpha_spitzer=true` are rejected.

At each face, the conductivity is the harmonic mean

```text
K_face = 2*K_left*K_right / (K_left + K_right).
```

A scaled evaluation avoids intermediate overflow. Field-aligned transport uses
this coefficient with the CT-based projection of the temperature gradient and
limited transverse temperature slopes, following the limited-asymmetric
construction of
[Sharma & Hammett (2007)](https://w3.pppl.gov/~hammett/gyrofluid/papers/2007/aniso.pdf).
The coefficient multiplying the gradient is averaged—not temperature or
diffusivity separately.

A genuinely zero endpoint conductivity produces zero harmonic face
conductivity. Positive-temperature conductivity underflow is rejected rather
than treated as this cold-face limit. Large temperature contrasts require
spatial-refinement checks.

## Physical saturation

`conduction_saturation = cowie_mckee` enables saturation in `<hydro>` or
`<mhd>`. It can limit constant-alpha conduction, Spitzer conduction, or their
sum. `saturation_phi` must be finite and nonnegative and defaults to 0.3.
The default `conduction_saturation = none` leaves the classical flux unchanged;
supplying `saturation_phi` in that mode is rejected.

The code reconstructs the complete classical conductive vector `C` and applies

```text
F = C * q_sat / (q_sat + |C|)
q_sat = 5 * phi * P_donor^(3/2) / sqrt(rho_face).
```

`rho_face` is the arithmetic face density. `P_donor` is the cell-centered
pressure on the sending side of the normal heat flux, not a reconstructed
Riemann pressure. This uses the isothermal sound-speed convention with no
extra gamma factor. The closure is evaluated in code units; `phi` is
dimensionless. Zero phi suppresses the flux but does not bypass state checks.

Only the conductive increment is limited. Fluid energy flux is unchanged.
Saturation acts on the summed vector, not independently on its components or
individual conduction terms. Thus out-of-plane magnetic components matter in
2-D, and even isotropic saturation needs transverse slopes and edge ghosts to
evaluate the vector norm.

## Time integration

### Explicit stepping

Nonlinear explicit conduction supports Newtonian SSP RK1/RK2/RK3 and requires
finite `0 < cfl_number <= 1`. Its timestep combines the constant-alpha rate
with a global-temperature-cap Spitzer rate.
Saturation adds donor-pressure-response and outgoing thermal-energy-drain
bounds. These bounds include the consumed ghost stencil and are recomputed at
every SSP stage.

The mesh CFL factor is applied once when selecting the next timestep. Stage
checks compare the constituent forward-Euler interval against the current
CFL-free conduction bound; they do not change the committed timestep.
Exceeding that bound is fatal: explicit stepping has no automatic retry.

Raw conserved-state checks precede the normal EOS conversion. States requiring
density, pressure, temperature or entropy floors are rejected, with failures
coordinated across MPI ranks. These checks do not replace unrelated modules'
state validation. The conduction-only drain bound is not a coupled-MHD
positivity guarantee.

### RKL2 super-time-stepping

Select STS for the conduction process using

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

The same selectors work for isotropic or combined transport and for nonlinear
Spitzer/saturated conduction. At least one conductivity must be positive.

Conduction takes pre/post half-sweeps around the fluid update. The RKL2
controller selects an odd stage count `s >= 3` satisfying

```text
dt_sweep / dt_exp <= (s*s + s - 2)/4.
```

For nonlinear conduction, conductivity, limiters and saturation are recomputed
at every RKL stage. Both split halves refresh primitives from the current
conserved state. With conduction as the sole STS process, trial stages must
remain inside the trial-start thermal
envelope, allowing temperature-scaled roundoff, without EOS floors. A rejected
trial restores conserved and primitive fluid arrays and halves the interval
after communication completes. At most 32 halvings are allowed per segment.
Rollback covers fluid U/W, not the CT field or arbitrary problem state.
Invalid initial states and constitutive/rate representability errors are fatal.

When conduction shares STS with viscosity or Ohmic resistivity, the driver
uses the full-state diffusion snapshot: conserved and primitive fluid arrays,
cell-centered magnetic fields, and fine/coarse CT face fields. It reconstructs
scratch primitives from each raw stage and sums the current inverse diffusion
bounds of all selected processes. Trials must retain positive thermal energy
without EOS floors and fit the selected RKL2 stage budget. Viscous and Ohmic
heating can raise temperature above its trial-start value. A rejected trial
restores the shared snapshot, completes communication and halves the interval.

Regridding occurs at completed cycles, not within a retry. Subsequent sweeps
refresh the state and timestep bounds and resize the primitive-state backup.
Restarts use standard completed-cycle checkpoints, with no additional
conduction-specific binary layout.

The donor-pressure bound limits the outer timestep and trial interval without
STS acceleration. Stage counts use the unsuppressed diffusion budget.
`sts_max_dt_ratio` is an upper limit, not a promised acceleration factor or an
accuracy controller. Internal RKL stages are not generally maximum-principle
preserving. Resolve sharp fronts and check timestep refinement rather than
choosing the ratio from stability alone.

## Restrictions and numerical limitations

- Nonlinear Spitzer/saturated STS requires one fluid owner and rejects cyclic
  zoom. Refined meshes must use conserved-variable prolongation.
- Standalone nonlinear-conduction STS rejects custom user boundaries. Joint
  full-state STS requires callbacks to enroll `user_bcs_retry_safe` and
  `user_dt_retry_safe` in the problem generator. Boundary callbacks may update
  only snapshotted ghost fields, and timestep callbacks only `dtnew`,
  deterministically and without persistent side effects.
- Thermal boundary fluxes depend on the ghost-state prescription. A fluid
  boundary label alone does not establish insulation or a prescribed heat flux.
- SMR/AMR is supported through the existing flux-correction and mesh-transfer
  machinery, but conservation does not establish accuracy for arbitrary
  refinement layouts, density contrasts or coupled flows. Check spatial and
  temporal convergence for the intended problem.
- On a composite mesh, volume-weighted cell-centered magnetic field is not
  generally an exact CT invariant. Use native plane-integrated normal magnetic
  flux when checking magnetic-flux conservation.
- CPU/GPU results need not be bitwise identical. Device agreement and
  continuum accuracy are separate checks; set comparison tolerances using
  the physical scale and accuracy required for the problem.
- Extreme-range constitutive calculations are not supported without bounds:
  unrepresentable positive conductivity, heat-capacity or timestep-rate
  intermediates are rejected instead of silently disabling transport.
- `q_limit` is not a physical saturation control. It must be finite and at
  least the legacy disabled sentinel `3.40282e38`. Use
  `conduction_saturation` and `saturation_phi` for physical saturation.

## Implementation map

| Source | Functions and responsibilities |
| --- | --- |
| [conduction.hpp](../src/diffusion/conduction.hpp), [conduction.cpp](../src/diffusion/conduction.cpp) | `Conduction::Conduction` reads and validates inputs; `AddHeatFluxes` dispatches the combined saturated flux or separate classical terms; `AddHeatFluxIso/Aniso` implement constant-coefficient fluxes. `NewTimeStep`, `SpitzerTimeStep`, `CheckExplicitStage` and `InspectRawState` supply timestep and state checks. `RawStageTimeStep` reconstructs scratch primitives for the joint current-stage budget. |
| [conduction.cpp](../src/diffusion/conduction.cpp) | File-local `HarmonicMeanPositive`, `SpitzerConductivity`, `SpitzerLimited4` and `SaturateHeatFlux` provide device-callable constitutive arithmetic; `SpitzerCoefficientCode` converts Units scales. `AddHeatFluxSpitzer` constructs harmonic temperature-dependent face fluxes. The private `SaturatedFace` helper builds and caps the summed vector; `AddHeatFluxSaturated` adds its normal component to energy flux; `SaturationTimeStep` computes donor-response and thermal-drain restrictions. |
| [sts_rkl2.cpp](../src/diffusion/sts_rkl2.cpp) | `parabolic::ComputeRKL2StageCount` and `ComputeRKL2Coefficients` select the stage count and recurrence coefficients. |
| [driver.cpp](../src/driver/driver.cpp), [driver.hpp](../src/driver/driver.hpp) | `Driver::ValidateSTSConfiguration` checks supported combinations; `ExecuteSTSSweep` schedules half-sweeps and nonlinear retries. Standalone conduction uses `Conduction::BeginSTSTrial/RestoreSTSTrial` for fluid U/W. Joint sweeps use `DiffusionSnapshot::Save/Restore`, `CheckFullStateSTSState` and `ExecuteGuardedSTSSweep` for shared fields and current-stage bounds. |
| [hydro_newdt.cpp](../src/hydro/hydro_newdt.cpp), [mhd_newdt.cpp](../src/mhd/mhd_newdt.cpp) | `Hydro::NewTimeStep` and `MHD::RecomputeTimeStepFromCurrentState` include the unaccelerated saturation donor bound in the fluid timestep. |
| [hydro_sts.cpp](../src/hydro/hydro_sts.cpp), [mhd_sts.cpp](../src/mhd/mhd_sts.cpp), [hydro_tasks.cpp](../src/hydro/hydro_tasks.cpp), [mhd_tasks.cpp](../src/mhd/mhd_tasks.cpp) | `STSFluxes`, `Hydro::STSUpdate`, `MHD::STSUpdateU`, `STSCheckState` and `ConToPrim` integrate fluxes, recurrence, raw-state checks and boundary exchanges. Active state is checked before restriction/EOS conversion; consumed halos are checked after prolongation and physical BCs. |

Trial failures are coordinated across MPI ranks. Posted communication is
completed before an attempt is abandoned or retried. The conduction recurrence changes
only energy, but the existing refined-mesh boundary pipeline can also prolongate
CT fields; absence of a conductive induction term does not imply bitwise
unchanged faces throughout that pipeline.

## Validation resources

The existing diffusion and STS regression tests remain part of the normal test
suite. Extended conduction benchmarks, analytic oracles, refinement studies,
and CPU/GPU/MPI comparison runners are maintained on the
[`feature/anisotropic-conduction` branch](https://github.com/rajsekhar33/athenak-RM/tree/feature/anisotropic-conduction/tst/transport).
They are development/validation tools, not required to build or use conduction.

## References

- **Sharma & Hammett (2007)**, *Preserving monotonicity in anisotropic diffusion*,
  JCP 227, 123–142. [Paper](https://w3.pppl.gov/~hammett/gyrofluid/papers/2007/aniso.pdf),
  [DOI](https://doi.org/10.1016/j.jcp.2007.07.026).
  Harmonic face coefficients and limited-asymmetric transverse gradients.
- **Spitzer & Harm (1953)**, *Transport Phenomena in a Completely Ionized Gas*,
  Physical Review 89, 977–981. [DOI](https://doi.org/10.1103/PhysRev.89.977).
  Collisional temperature-dependent conductivity.
- **Cowie & McKee (1977)**, *The evaporation of spherical clouds in a hot gas.
  I. Classical and saturated mass loss rates*, ApJ 211, 135–146.
  [DOI](https://doi.org/10.1086/154911). Physical saturated-flux scale.
- **Meyer, Balsara & Aslam (2012)**, *A second-order accurate Super TimeStepping
  formulation for anisotropic thermal conduction*, MNRAS 422, 2102–2115.
  [Article](https://academic.oup.com/mnras/article/422/3/2102/1042639),
  [2013 erratum](https://academic.oup.com/mnras/article/428/2/1855/1000287).
  Early RKL formulation, with the printed corrections taken into account.
- **Meyer, Balsara & Aslam (2014)**, *A stabilized Runge–Kutta–Legendre method
  for explicit super-time-stepping of parabolic and mixed equations*,
  JCP 257, 594–626. [DOI](https://doi.org/10.1016/j.jcp.2013.08.021).
  RKL stability polynomial and recurrence.
- **Vaidya et al. (2017)**, *Scalable explicit implementation of anisotropic
  diffusion with Runge–Kutta–Legendre super-time-stepping*,
  MNRAS 472, 3147–3160. [Paper](https://arxiv.org/html/1702.05487v2).
  RKL conduction and the smooth saturation interpolation/isothermal convention
  used above.

These articles motivate the physical closures and discretization. The
raw-state checks, numerical range rejection and retry policy are implementation
choices, not guarantees established by those papers.
