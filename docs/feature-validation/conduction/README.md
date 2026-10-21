# Thermal conduction: physical verification

These tests measure how heat spreads along magnetic fields, how much leaks
across them, and how accurately explicit and RKL2 time integration reproduce
the temperature evolution. Configuration and the numerical method are described
in the [thermal-conduction guide](../../thermal_conduction.md).

## Heat equation and coefficient convention

Here temperature means the code variable $T=p/\rho$. For an ideal gas,
$e_{\rm th}=\rho T/(\gamma-1)$. With fixed uniform density, a stationary fluid,
a constant magnetic field, and constant coefficients, conduction reduces to

$$
\partial_t T=\nabla\cdot(\mathsf{D}\nabla T),\qquad
\mathsf{D}=(\gamma-1)
\left[\alpha_{\rm iso}\mathsf{I}+\alpha_{\rm aniso}\hat{\boldsymbol b}\hat{\boldsymbol b}^{\mathsf T}\right].
$$

Thus $D_\parallel=(\gamma-1)(\alpha_{\rm iso}+\alpha_{\rm aniso})$ and
$D_\perp=(\gamma-1)\alpha_{\rm iso}$. Input `alpha` and temperature diffusivity
are equal only when $\gamma=2$. Equivalently, a dimensional conductivity
$\kappa$ gives $D_T=\kappa/(\rho c_v)$.
The magnetic unit vector uses all three components, including $B_z$ in 2D.

For a periodic Fourier mode, these assumptions give the analytic damping law

$$
T=T_0+A\exp[-\boldsymbol{k}^{\mathsf T}\mathsf{D}\boldsymbol{k}\,t]
\cos[\boldsymbol{k}\cdot(\boldsymbol{x}-\boldsymbol{x}_0)].
$$

The mode tests vary field orientation, coefficient mixtures, density and mesh
decomposition. The ring and Gaussian tests below additionally expose the
spatial distribution of heat. Full MHD tests evolve density, velocity and
magnetic fields and use a time-refined numerical solution as their reference.

## 1. Heat flowing around a circular magnetic field

This follows the hot-patch benchmark in
[Sharma & Hammett (2007), §7.1, equation (38)](https://w3.pppl.gov/~hammett/gyrofluid/papers/2007/aniso.pdf).
On $[-1,1]^2$, the initial temperature is 12 in
$0.5<r<0.7$, $11\pi/12<\theta<13\pi/12$, and 10 elsewhere.
The fluid and circular magnetic field are fixed, with $\rho=1$, $\gamma=2$,
`alpha_aniso=0.01`, `alpha_iso=0`, and reflecting outer boundaries.
The field and conduction vanish beyond $r=1$.
Both transverse limiters were evolved with explicit RK2 through $t=200$ at
$N=50,100,200,400$ cells per direction.

On each circle, the continuum equation is
$\partial_t T=(0.01/r^2)\partial_\theta^2 T$.
Angular Fourier modes decay as $\exp(-0.01n^2t/r^2)$ while their angular mean
is conserved. Inside the annulus, the asymptotic temperature is therefore
$10+2(\pi/6)/(2\pi)=10+1/6$; outside it remains 10.
At finite time, angular structure still follows the decaying modes. The
angular-mean radial step is exact in the continuum at every time, whereas
the uniform annulus is the long-time limit.

![Initial hot patch, final ring temperatures and angular-mean radial profiles](figures/ring_solution.png)

The maps show the measured $400^2$ fields. Heat travels around the circular
field, and the radial profiles expose numerical cross-field leakage. MC
preserves a narrower annulus than van Leer. Dashed circles mark its original
edges; initial and final panels have different temperature ranges.

![Ring temperature norms compared with published values](figures/ring_errors.png)

The plotted $L_1$, $L_2$ and $L_\infty$ norms use the long-time annular
profile evaluated at cell centers. With $\delta T_i=T_i-T_{\infty,i}$,

$$
L_1=\frac{1}{N^2}\sum_i|\delta T_i|,\qquad
L_2=\left(\frac{1}{N^2}\sum_i\delta T_i^2\right)^{1/2},\qquad
L_\infty=\max_i|\delta T_i|.
$$

The sums include every active cell in $[-1,1]^2$, including the background
outside the ring. Equal cell areas make these arithmetic means equivalent to
area-weighted means divided by the total domain area 4. The norms are absolute
errors in code-temperature units, without division by temperature or initial
amplitude. This matches the published comparison in Tables 1–4. Agreement with those
tabulated *numerical errors* is below 0.23% for van Leer and 0.12% for MC at
N400; these percentages are not the solution's continuum error.
Van-Leer $L_\infty$ increases from N200 to N400.
The ambiguous published N100 perpendicular-diffusivity entry is excluded from
the comparison criteria. N50 was measured on CPU and N100–400 on GPU with
historical constant-coefficient builds, whose hashes are included in the data.

## 2. Gaussian spreading in two and three dimensions

A Gaussian tests the width, orientation, peak temperature and conserved heat
of an oblique diffusion tensor. The analytic scalar-diffusion example in
[Vaidya et al. (2017), §4.1.1, equations (23)–(24)](https://arxiv.org/html/1702.05487v2#S4.SS1.SSS1)
is extended here to a periodic 2D/3D tensor problem.
For an unbounded domain and initial covariance $\mathsf{\Sigma}_0$, its solution is

$$
\mathsf{\Sigma}(t)=\mathsf{\Sigma}_0+2\mathsf{D}t,\qquad
T-T_0=A\sqrt{\frac{\det\mathsf{\Sigma}_0}{\det\mathsf{\Sigma}(t)}}
\exp\left[-\frac12(\boldsymbol{x}-\boldsymbol{x}_0)^{\mathsf T}
\mathsf{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{x}_0)\right].
$$

The actual runs use a periodic unit box, $T_0=1$, $A=0.1$,
$\mathsf{\Sigma}_0=0.1^2\mathsf{I}$, $\boldsymbol{x}_0=(0.5,\ldots)$,
$\rho=1$, $\gamma=2$, and $\boldsymbol B=(1,2,3)$.
The references sum the periodic Fourier solution and integrate it over each
cell, including the cell-average sinc factors. This avoids confusing a point
sample or an unbounded Gaussian with the finite-volume reference.

![Measured Gaussian temperatures and analytic cell-average profiles in 2D and 3D](figures/gaussian_solution.png)

These MC cases use `alpha_iso=0.001`, `alpha_aniso=0.099` and $t=0.02$.
The full 3D diffusivity ratio is 100; the projected 2D ratio is
$1+99(1^2+2^2)/(1^2+2^2+3^2)=36.36$ because $B_z$ enters normalization.
The 3D maps are the cell plane at $z=0.484375$; profiles use the lower central
cell row in each grid. Cyan contours are the analytic temperature excess.

Heat spreads preferentially along the projected field and the peak decreases.
At $64^2$, the relative excess-temperature $L_1$ error is
$5.88537\times10^{-4}$, with peak 1.09252485 versus 1.09257233 analytically.
At $32^3$, the corresponding error is $6.50841\times10^{-3}$, with peak
1.08026544 versus 1.08100309. Here
$L_1=\sum|T-T_{\rm ref}|/\sum|T_{\rm ref}-T_0|$;
the maximum differences divided by $A$ are $7.23304\times10^{-4}$ and
$7.37651\times10^{-3}$. These are short-time, resolution-specific results.
The matrix also covers isotropic and purely parallel transport, both limiters,
N16/N32 and 2D N64 refinement; the longer N16 evolution at $t=0.2$ failed its
spatial-accuracy criteria.

## 3. A source-balanced equilibrium and cross-field leakage

The Sovinec construction in
[Sharma & Hammett, §7.2, equation (40) and figure 9](https://w3.pppl.gov/~hammett/gyrofluid/papers/2007/aniso.pdf)
places the field along temperature contours. Our positive-temperature version
adds a uniform background of 1. On $[-0.5,0.5]^2$,

$$
T_*=1+\cos(\pi x)\cos(\pi y),\qquad
\boldsymbol B\parallel\nabla\times[\cos(\pi x)\cos(\pi y)\,\hat{\boldsymbol z}],
\qquad Q=2\pi^2\cos(\pi x)\cos(\pi y).
$$

With fixed $\rho=1$, $\gamma=2$, $\alpha_{\rm iso}=1$ and
$\alpha_{\rm aniso}=99$, parallel heat flow vanishes analytically and the
isotropic Laplacian balances $Q$. Boundary temperature is 1.
N8/N16 runs start at this continuum equilibrium and relax through $t=0.5$.
A matched isotropic control separates the discrete isotropic error from
additional leakage introduced by the anisotropic operator.

![Continuum source-balanced equilibrium and measured coarse-grid profiles](figures/sovinec_solution.png)

The measured N16 central excess temperatures are 0.4380 (van Leer) and
0.6488 (MC), below the continuum cell-center value 0.9904. The broad, reduced
temperature profiles demonstrate excessive numerical cross-field transport
on these coarse grids, even though the source and boundary heat loss balance.

![Sovinec temperature errors and inferred perpendicular diffusion](figures/retained_coarse_limits.png)

All four cases exceed the original $L_1<0.05$ criterion, where
$L_1=\sum V_i|T_i-T_{*,i}|/\sum V_i$. For N8→N16, van Leer gives
0.22410→0.17354 and MC gives 0.16729→0.08516.
The central-amplitude diagnostic for numerical perpendicular diffusivity,
relative to the parallel coefficient, falls from 0.02439→0.01286 (van Leer)
and 0.01500→0.005445 (MC). Stationarity and heat-budget closure therefore do
not by themselves establish spatial accuracy.

## 4. Saturated heat flow across a sawtooth profile

Strong temperature gradients limit the heat flux. For the classical vector
$\boldsymbol q_{\rm cl}$, the tested closure is

$$
\boldsymbol q=\frac{q_{\rm sat}}{q_{\rm sat}+|\boldsymbol q_{\rm cl}|}
\boldsymbol q_{\rm cl},\qquad
q_{\rm sat}=5\phi\rho c_{\rm iso}^3,\quad c_{\rm iso}^2=p/\rho.
$$

This is the smooth Cowie–McKee interpolation used by
[Vaidya et al., equations (3)–(5)](https://arxiv.org/html/1702.05487v2#S2).
Their §4.1.2, equation (27) supplies the initial sawtooth,
$T=10+20[x/10-\lfloor x/10+1/2\rfloor]$.
The runs use the periodic interval $[-10,10]$, N400, fixed $\rho=1$,
$\gamma=2$, classical conductivity 40, $\phi=0.3$, and $t=0.1$.

![Saturated sawtooth temperature profiles and time-integration differences](figures/saturated_sawtooth.png)

Heat smooths the sharp drop while the flux cap alters the classical-diffusion
profile. There is no closed-form reference for this nonlinear evolution;
the comparison uses explicit RK2 at CFL 0.2 on the same N400 grid, with a
CFL 0.4 timestep-halving control. RKL2 ratio caps 40 and 1000 give normalized
$L_1/L_\infty$ differences of $1.84469\times10^{-8}/6.26363\times10^{-7}$ and
$2.31719\times10^{-5}/4.80778\times10^{-4}$, below 0.002/0.02.
Differences are divided by the initial temperature range 20. The requested
RKL ratio cap differs from the paper's parabolic Courant number and from the
achieved acceleration, which also depends on donor bounds and retries.

## 5. Temperature-dependent conductivity and temporal accuracy

Spitzer conduction uses $K(T)=K_0T^{5/2}$ and the harmonic face conductivity
$K_f=2K_LK_R/(K_L+K_R)$, following the coefficient averaging in
[Sharma & Hammett, §2.1, equation (8)](https://w3.pppl.gov/~hammett/gyrofluid/papers/2007/aniso.pdf).
The tests below are purpose-built nonlinear convergence problems. They combine
this law with saturation and compare time integrators on a fixed mesh.

### Stationary-fluid pulse: explicit RK1, RK2 and RK3

The periodic unit interval has N8 cells, $\rho=1$, $\gamma=5/3$,
$T(x,0)=1+0.4\cos(2\pi x)$ sampled at cell centers, $K_0=0.1$ and
$\phi=0.03$. Density and velocity remain fixed; the run ends at $t=0.2$.
Conductivity depends on temperature and the flux saturates, so the simple
linear Fourier damping law does not apply.

![Nonlinear temperature pulse and explicit time-refinement errors](figures/explicit_temporal.png)

The pulse cools at its maxima and warms at its minima. The same-grid RK3
CFL 0.025 reference has $T_{\min}=0.68303150$, $T_{\max}=1.25794254$ and
mean temperature 1. Refining CFL through 0.4, 0.2 and 0.1 gives RK1/RK2/RK3
self-convergence orders 1.01222/2.02432/3.02180, consistent with their expected
temporal orders. Errors use double-precision cosine/minimum/maximum
diagnostics divided by amplitude 0.4; the displayed physical profiles come
from float32 field outputs. Keeping N8 fixed isolates time integration from
spatial convergence.

### Coupled MHD pulse: RKL2

The periodic $16^2$ grid starts with cell-averaged
$T=1+0.1\cos[2\pi(x+y)]$, $\rho=1$, $\boldsymbol v=0$,
$\boldsymbol B=(1,0.5,0.25)$ and $\gamma=5/3$.
Field-aligned harmonic Spitzer conduction uses $K_0=0.1$, saturation
$\phi=0.03$, MC slopes, and RKL2 ratio cap 64; fluid evolution uses RK2,
PLM and HLLD. All variables evolve through $t=0.2$.

![Coupled MHD temperature profiles and RKL2 temporal convergence](figures/coupled_rkl_temporal.png)

The temperature perturbation decays while pressure forces generate motion
and density structure. In the fine RKL run, the temperature spans
0.9849534–1.0118426 and the maximum density departure from 1 is 0.082724;
conduction, compressional work and MHD evolution all contribute.
The reference is explicit RK2 on the same spatial grid.
For adaptive steps, $H=[\sum_j\Delta t_j^3/t]^{1/2}$ measures temporal resolution.
Eleven scaled conserved-fluid/CT component norms are checked separately from
temperature. Finest combined-field orders are 1.99617/2.03556 and temperature
orders are 2.09397/2.14279, consistent with second order. Explicit-reference
halving differences are below 1% of the finest RKL error.

## 6. Refinement, conservation and numerical limits

Static mesh refinement checks conserved heat flux across level boundaries
against an explicit solution. In the density-step case the largest normalized
RKL error is 0.00481588, below 0.005. Dynamic AMR runs refine/derefine through
4→7→4 leaves and restart on either side of both transitions. Three endpoint
and 17 checkpoint CPU/MPI2/GPU comparisons give a largest GPU raw-state
difference of $1.09499620857\times10^{-14}$ against a $5\times10^{-12}$ criterion.

Six static N32 3D CPU/GPU energy comparisons exceed a stricter
$5\times10^{-12}$ agreement criterion: the largest relative energy difference
is $5.4542320335866934\times10^{-9}$ and the temperature difference divided by
initial amplitude is $7.690473591281943\times10^{-8}$. All 24 case outcomes,
including those six failures, are in the numerical data. Cross-backend
agreement, heat conservation and continuum accuracy measure different things.
CT conservation checks use native plane-integrated magnetic flux.

Nonlinear STS currently requires one fluid, conduction as the sole STS process,
and conserved prolongation on refined meshes. Cyclic zoom and custom boundary
callbacks are unsupported. Thermal boundary flux follows the supplied ghost
states. RKL linear stability does not guarantee nonlinear monotonicity:
sharp fronts need spatial and temporal refinement. Invalid raw thermal
states trigger coordinated restoration and interval halving, with up to
32 halvings per segment; invalid initial states or unrepresentable closure
and rate values terminate the run.

## Reproducing the figures and identifying the source

```bash
python docs/feature-validation/conduction/plot_results.py
```

NumPy and Matplotlib recreate all eight figures from
[results.json](results.json) and [physical_profiles.npz](physical_profiles.npz).
Use `--output DIRECTORY` for a separate destination.
[Physical-profile metadata](physical_profiles.json) gives the archived field
hashes, coefficients, binary identifiers and independently checked metrics.
The optional [scalar exporter](export_results.py) and
[field exporter](export_physical_profiles.py) document extraction from the
original campaign; the latter also uses SciPy for its analytic Gaussian check.
Native tests use the supplied-executable [runners](../../../tst/transport) and
[backend adapters](../../../tst/test_suite/diffusion).

The studies span successive implementations identified by binary hashes.
Early constant-coefficient results predate the nonlinear implementation.
The potential-flux Spitzer prototype had accuracy failures and was superseded
by the harmonic stencil measured here. The later module consolidation
preserves formulas and interfaces: three cases per CPU/GPU, all fields at
149 checkpoints, and restart endpoints agree bitwise with the preceding
version; the largest energy drift is $3.92\times10^{-15}$.

The source baseline is `588d737fac0251b5538c287cf50c3ccdd4cad706`; the implementation
inventory below describes `c3a8fccca928eef8b22d233458c4a4aabd9cb171`.
The [source manifest](source_manifest.json) records hashes and every changed hunk
across the 77-path implementation delta.

## Appendix A: implementation locations

<details>
<summary>Source locations, changed functions, and test fixtures</summary>

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

</details>

## Appendix B: feature-test coverage

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

## References

- [Sharma & Hammett (2007)](https://arxiv.org/abs/0707.2616), JCP 227, 123–142: limited anisotropic diffusion and ring tests.
- [Spitzer & Harm (1953)](https://doi.org/10.1103/PhysRev.89.977): collisional temperature-dependent conductivity.
- [Cowie & McKee (1977)](https://doi.org/10.1086/154911): saturated heat-flux scale.
- [Meyer, Balsara & Aslam (2014)](https://doi.org/10.1016/j.jcp.2013.08.021): stabilized RKL formulation.
- [Vaidya et al. (2017)](https://arxiv.org/html/1702.05487v2): RKL anisotropic diffusion, saturation and benchmarks.
