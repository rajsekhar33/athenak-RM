# Viscosity

Newtonian Hydro and MHD support constant isotropic kinematic shear viscosity.
Uniform-grid Newtonian MHD also supports leading parallel Braginskii viscosity
with ideal or isothermal equations of state. Set `nu_iso` and/or `nu_aniso`
in the fluid block, in code units of length squared per time.
For a dimensional coefficient, use
`nu_code = nu_cgs * time_cgs / length_cgs^2`; the input is not converted
automatically. The corresponding dynamic coefficients are `rho * nu_iso`
and `rho * nu_aniso`.

## Configuration

Explicit integration is the default:

```ini
<hydro>
nu_iso = 0.01
viscosity_integrator = explicit
```

For RKL2 super-time-stepping (STS), use:

```ini
<hydro>
nu_iso = 0.01
viscosity_integrator = sts

<time>
integrator = rk2
sts_integrator = rkl2
sts_max_dt_ratio = 1
```

Use `<mhd>` instead for MHD. For parallel Braginskii viscosity, for example:

```ini
<mhd>
nu_iso = 0
nu_aniso = 0.01
viscosity_integrator = explicit
```

Set `viscosity_integrator = sts` and use the same `<time>` settings above for
RKL2. Both coefficients may be positive and share `viscosity_integrator`;
STS requires at least one positive coefficient. Malformed, negative or
nonfinite coefficients are rejected.
Nonzero `nu_aniso` requires MHD without SMR/AMR, in either integration mode.
There is no hyperviscosity, temperature-dependent viscosity, gyroviscosity,
or mirror/firehose pressure-anisotropy clipping.

The positive `sts_max_dt_ratio` caps the full cycle relative to the combined
CFL-scaled STS diffusion bound; `-1` removes that cap. Start conservatively and
check time-step convergence before increasing it. Stability is not accuracy,
particularly when the cycle length changes the timing of adaptive refinement.

## Spatial operator and time integration

`Viscosity::AddViscousFluxIso` in `src/diffusion/viscosity.cpp` evaluates
the Newtonian shear stress
`tau_ij = rho*nu_iso*(dv_i/dx_j + dv_j/dx_i - 2/3*delta_ij*div(v))`.
It uses arithmetic face density and centered gradients. The conservative
momentum flux receives `-tau_ij`; ideal-gas total energy receives the stress
work using the arithmetic face velocity. Isothermal fluids have no evolved
energy equation. The same spatial flux is used in explicit and STS modes.

`Viscosity::AddViscousFluxAniso` evaluates the parallel Braginskii stress:

```text
b = B / |B|
S = b_i*b_j*dv_i/dx_j - div(v)/3
Delta_p = 3*rho*nu_aniso*S
tau_ij = Delta_p*(b_i*b_j - delta_ij/3)
```

Here `Delta_p = p_perpendicular - p_parallel`; papers using `Pi` in the
momentum flux write `Pi = -tau`. The implementation uses centered normal and
transverse velocity derivatives without a slope limiter, following the
Cartesian centered approach described by
[Kempf, Rincon & Clerc, Appendix B](https://arxiv.org/html/2306.12460v2#A2).
At each face, density and velocity are arithmetic averages, normal B is the
current CT face field, and transverse B is averaged from adjacent cell-centered
fields. The momentum flux receives `-tau`; ideal-gas total energy receives
the matching conservative work flux `-v_face dot tau`. Explicit and RKL2 use the same
operator with current-stage fields. Braginskii damping, heating and RKL2
benchmarks are also described by
[Berlok, Pakmor & Pfrommer (2020)](https://arxiv.org/abs/1911.05694);
their moving-mesh reconstruction and energy time integration are not copied here.

The continuum heating rate is `3*rho*nu_aniso*S^2`, but the centered discrete
operator does not guarantee local thermal positivity for arbitrary states.
At exactly zero reconstructed face B, the anisotropic stress is skipped;
this is not a physical transition to unmagnetized viscosity.

`Viscosity::NewTimeStep` supplies `1/(rate_iso + rate_aniso)`, minimized over
blocks, where the sums and dimension count include only active directions:

```text
rate_iso   = 2*nu_iso*(sum_d dx_d^-2 + max_d(dx_d^-2)/3)
rate_aniso = (8/3)*ndim*nu_aniso*max_d(dx_d^-2)
```

The mesh applies the CFL factor. Adding rates accounts for simultaneous
isotropic and parallel viscosity. This geometric bound is not a
variable-density positivity guarantee.

The driver places an RKL2 half-sweep on either side of the fluid update.
Hydro/MHD STS tasks reuse the conservative flux and boundary exchange paths.
RKL methods and their application to viscosity are described by
[Meyer, Balsara & Aslam (2014)](https://www.sciencedirect.com/science/article/pii/S0021999113005597);
see also [Vaidya et al. (2017)](https://arxiv.org/abs/1702.05487)
for anisotropic-diffusion STS.

STS adds safeguards, not a new stress discretization:

- `Viscosity::InspectRawState` checks finite density, momentum and, where
  applicable, positive internal energy and the configured EOS floors before
  EOS conversion. MHD checks use current CT faces.
- `Driver::ExecuteViscositySTSSweep` restores rejected trials and halves their
  intervals collectively across ranks. Boundary exchanges finish before retry.
  Invalid starting states or exhausted retry limits terminate with an error.
- `Driver::ExecutePreSTSSweep` refreshes fluid, source, user and explicit-
  diffusion bounds after viscous heating. If needed, it restores the cycle
  start and repeats the first half-sweep at a smaller cycle length.
  Source application and work callbacks are not executed inside these retries.

These checks do not estimate time-integration error or guarantee dissipation
for arbitrary density contrasts.

## Refinement and callbacks

The refinement support below is for isotropic viscosity only. Parallel
Braginskii viscosity currently rejects both static and adaptive refinement.

For Newtonian ideal Hydro with viscosity STS and conserved prolongation,
`HydroProlongTheta` contracts inadmissible reconstructed siblings toward their
parent using a common factor. Both ghost interpolation and active AMR refinement
use it. Sibling conserved means are preserved; already admissible reconstructions
are unchanged. Invalid parent data are rejected rather than repaired with floors.
This safeguard is not an MHD/CT prolongation repair.

User boundaries and user timestep setters require explicit code enrollment via
`user_bcs_retry_safe` and `user_dt_retry_safe` in `ProblemGenerator`.
A boundary callback may only deterministically update snapshotted ghost fields;
a timestep callback may only deterministically recompute `dtnew`.
Neither may have persistent side effects. These are contracts, not input switches,
and do not certify boundary time accuracy.

Viscosity STS rejects primitive prolongation, cyclic zoom, simultaneous Hydro/MHD
owners, and co-selected STS resistivity. General STS restrictions also exclude
relativity, radiation, ion-neutral physics, shearing boxes and orbital advection.
Nonlinear conduction must be the only STS process, so cannot share an STS sweep
with viscosity. See [thermal_conduction.md](thermal_conduction.md) for conduction's
own restrictions.
