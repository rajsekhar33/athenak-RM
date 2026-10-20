# Viscosity

Newtonian Hydro and MHD support constant isotropic kinematic shear viscosity.
Uniform-grid Newtonian MHD also supports leading parallel Braginskii viscosity
with ideal or isothermal equations of state. Set `nu_iso` and/or `nu_aniso`
in the fluid block, in code units of length squared per time.
For a dimensional coefficient, use
`nu_code = nu_cgs * time_cgs / length_cgs^2`; the input is not converted
automatically. The corresponding dynamic coefficients are `rho * nu_iso`
and `rho * nu_aniso`.

Ideal-gas Hydro/MHD additionally support constant **dynamic** fourth-order
hyperviscosity, `mu_hyper4`, on uniform periodic meshes. Its code units are
`density * length^4 / time`; convert a dimensional coefficient with
`mu4_code = mu4_cgs * time_cgs / (density_cgs * length_cgs^4)`.
The local kinematic coefficient is `mu_hyper4 / rho`, not a constant input.

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
RKL2. All viscosity coefficients share `viscosity_integrator`;
STS requires at least one positive coefficient. Malformed, negative or
nonfinite coefficients are rejected.
Nonzero `nu_aniso` requires MHD without SMR/AMR, in either integration mode.
There is no temperature-dependent viscosity, gyroviscosity or mirror/firehose
pressure-anisotropy clipping.

For fourth-order hyperviscosity, add the following to the fluid block:

```ini
<hydro>
eos = ideal
nu_iso = 0
mu_hyper4 = 0.0001
viscosity_integrator = explicit
```

Use `<mhd>` for MHD; `viscosity_integrator = sts` uses the RKL2 settings above.
`mu_hyper4` defaults to zero and may be combined with the supported ordinary
viscosities. Nonzero hyperviscosity requires periodic boundaries in every
active direction, no SMR/AMR, at least two ghost cells and `fofc = false`.
It is not available with an isothermal EOS or relativistic physics.

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

### Fourth-order stress and ideal-gas energy

`Viscosity::AddViscousFluxHyper4` uses centered face strains and their discrete
Laplacian. In continuum notation, with repeated indices summed:

```text
S_ij = (d_j v_i + d_i v_j)/2 - delta_ij div(v)/3
tau4_ij = -2 mu4 Laplacian(S_ij)
H_ijk = 2 mu4 d_k S_ij
J_j = H_abj S_ab
F_momentum(i),j = -tau4_ij
F_energy,j = -v_i tau4_ij - J_j
```

The general-order constant-dynamic hyperstress is described by
[Haugen & Brandenburg (2004), equations 2--5](https://arxiv.org/html/astro-ph/0402301v2#S2).
Their simulations use sixth-order hyperviscosity and an isothermal EOS;
the fourth-order specialization here is not a reproduction of those runs.
Higher-stress mechanical work in an ideal-gas total-energy balance has a
continuum precedent in
[Necas, Novotny & Silhavy (1989), equation 2.12 and Example 2.21](https://dml.cz/bitstream/handle/10338.dmlcz/106776/CommentatMathUnivCarol_030-1989-3_14.pdf).
These are continuum precedents, not a published implementation of this
finite-volume stencil or a proof of its numerical properties.

For constant `mu4`, the extra mechanical-work term is
`J = mu4 grad(S:S)`. Including it yields the continuum heat production
`Q4 = 2 mu4 sum_ijk (d_k S_ij)^2`; using only `-v dot tau4` would give a
different local thermal-energy equation. This is not thermal conduction.
The code evaluates the momentum and complete energy contributions as shared
face fluxes using current-stage primitive velocities, without a new evolved
field. The stencil uses two primitive ghost cells in each active direction.
Viscosity itself does not update the CT magnetic faces.

In the continuum at uniform density, transverse Fourier velocity modes decay at
`(mu4/rho)*|k|^4`; the longitudinal viscous rate is `4/3` times larger.
Sound and magnetosonic waves require the coupled fluid dispersion relation,
not this isolated velocity-decay law. The finite-volume stencil has its own
modified-wavenumber response. Periodic global kinetic dissipation is a spatial
operator property, not a finite-step RKL2 guarantee; positive continuum heating
does not establish multidimensional cellwise thermal positivity for the
discrete update.

### Timestep selection and RKL2

`Viscosity::NewTimeStep` supplies `1/(rate_iso + rate_aniso + rate_hyper4)`,
minimized over blocks, where the sums and dimension count include only active
directions:

```text
rate_iso   = 2*nu_iso*(sum_d dx_d^-2 + max_d(dx_d^-2)/3)
rate_aniso = (8/3)*ndim*nu_aniso*max_d(dx_d^-2)
rate_hyper4 = (32/3)*(mu_hyper4/rho_min)*(sum_d dx_d^-2)^2
```

The mesh applies the CFL factor. Adding rates accounts for simultaneous
isotropic, parallel and hyperviscosity. For hyperviscosity, `rho_min` is the
current positive active-cell minimum on the local pack; the mesh minimizes
the timestep across ranks. Density and the bound are refreshed after fluid
evolution. This bound is not a variable-density positivity guarantee.
The zero-hyperviscosity path retains the ordinary-viscosity timestep arithmetic.

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
for arbitrary density contrasts. In particular, a stable large RKL2 timestep
can poorly resolve stiff fourth-order modes. Check timestep convergence of
both velocity and thermal energy; passing raw-state checks or agreeing across
CPU/GPU backends is not an accuracy criterion.

## Refinement and callbacks

The refinement support below is for isotropic viscosity only. Parallel
Braginskii viscosity and fourth-order hyperviscosity currently reject both
static and adaptive refinement. Hyperviscosity also rejects nonperiodic
boundaries; the callback contracts below do not override that restriction.

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

## Focused hyperviscosity regression

The test-only `tests/hyperviscosity_probe` problem generator and
`tst/functional/hyperviscosity.py` provide a small single-rank regression.
Configure a double-precision build with
`-DPROBLEM=tests/hyperviscosity_probe`, then run:

```sh
python tst/functional/hyperviscosity.py --selfcheck
python tst/functional/hyperviscosity.py --binary /path/to/athena --output /path/to/new/results
```

NumPy is required. The tests check the one-dimensional tensor symbol and
complete energy-work flux, coefficient-off behavior, circular-shear decay
and generated heat with explicit/RKL2 integration, and selected input guards.
The evolution oracle uses the discrete spatial operator and time-integrator
polynomial; it is not a general nonlinear or continuum-convergence test.
Use the same script with a GPU executable and `--gpu` for that backend.
This custom problem build is not suitable for the standard regression suite,
which requires `-DPROBLEM=built_in_pgens`.
