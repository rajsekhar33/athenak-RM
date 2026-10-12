# Isotropic viscosity

Newtonian Hydro and MHD support constant isotropic kinematic shear viscosity.
Set `nu_iso` in the fluid block, in code units of length squared per time.
For a dimensional coefficient, use
`nu_code = nu_cgs * time_cgs / length_cgs^2`; the input is not converted
automatically. Dynamic viscosity is `rho * nu_iso`.

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

Use `<mhd>` instead for MHD. STS requires positive `nu_iso`.
Malformed, negative or nonfinite coefficients are rejected. Nonzero
`nu_aniso` is rejected: Braginskii viscosity is not implemented.
There is no hyperviscosity or temperature-dependent viscosity closure.

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

`Viscosity::NewTimeStep` supplies the geometric bound
`1 / [2*nu_iso*(sum_d dx_d^-2 + max_d(dx_d^-2)/3)]`, minimized over blocks.
The mesh applies the CFL factor. This bound is not a variable-density
positivity guarantee.

The driver places an RKL2 half-sweep on either side of the fluid update.
Hydro/MHD STS tasks reuse the conservative flux and boundary exchange paths.
RKL methods and their application to viscosity are described by
[Meyer, Balsara & Aslam (2014)](https://www.sciencedirect.com/science/article/pii/S0021999113005597);
see also [Vaidya et al. (2017)](https://arxiv.org/abs/1702.05487)
for anisotropic-diffusion STS.

The implementation adds safeguards, not a new stress discretization:

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
