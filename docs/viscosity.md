# Viscosity

Newtonian Hydro/MHD support constant isotropic kinematic shear viscosity, `nu_iso`.
Uniform-grid MHD also supports parallel Braginskii viscosity, `nu_aniso`, with
ideal or isothermal equations of state. Both have code units `length^2/time`
and dynamic coefficients `rho*nu`. Convert dimensional inputs manually using
`nu_code = nu_cgs*time_cgs/length_cgs^2`.

Ideal-gas Hydro/MHD support constant **dynamic** fourth-order hyperviscosity,
`mu_hyper4`, on uniform meshes with periodic or outflow boundaries, in code
units `density*length^4/time`:
`mu4_code = mu4_cgs*time_cgs/(density_cgs*length_cgs^4)`.
The local kinematic coefficient is `mu_hyper4/rho`.

## Configuration

Explicit integration is the default. For example:

```ini
<mhd>
eos = ideal
nu_iso = 0.01
nu_aniso = 0
mu_hyper4 = 0
viscosity_integrator = explicit
```

Use `<hydro>` for Hydro. For RKL2 super-time-stepping (STS), change the fluid
selector and supply:

```ini
<mhd>
viscosity_integrator = sts
<time>
integrator = rk2
sts_integrator = rkl2
sts_max_dt_ratio = 1
```

All coefficients default to zero and share one `viscosity_integrator`.
Malformed, negative or nonfinite values are rejected; STS requires a positive
coefficient. Nonzero `nu_aniso` requires MHD without SMR/AMR in either mode.
Nonzero `mu_hyper4` requires ideal gas, periodic or outflow boundaries in every
active direction, no SMR/AMR and at least two ghost cells. FOFC can be enabled
with `viscosity_integrator = sts`; explicit hyperviscosity requires `fofc = false`.
At physical outflow faces, the hyperviscous momentum and energy fluxes are zero.
Adjacent interior faces use the usual extrapolated fluid ghosts. Advective,
conductive and ordinary-viscous fluxes retain their existing boundary treatment.
There is no temperature-dependent viscosity, gyroviscosity or mirror/firehose
pressure-anisotropy clipping.

The positive `sts_max_dt_ratio` caps the full cycle relative to the combined
CFL-scaled STS diffusion bound; `-1` removes that cap. Start conservatively and
check timestep convergence before increasing it, including refinement timing.

## Spatial operators

The `Viscosity::AddViscousFlux*` methods in `src/diffusion/viscosity.cpp` share the
same conservative face fluxes between explicit and STS modes, without directly
updating CT magnetic faces.

`AddViscousFluxIso` uses centered gradients and arithmetic face density:

```text
tau_ij = rho*nu_iso*(d_j v_i + d_i v_j - 2*delta_ij div(v)/3)
F_momentum(i),j = -tau_ij
F_energy,j = -v_face,i tau_ij
```

Face velocity is arithmetic; isothermal fluids have no evolved energy equation.
`AddViscousFluxAniso` evaluates the parallel Braginskii stress:

```text
b = B/|B|,  S = b_i b_j d_j v_i - div(v)/3
Delta_p = 3*rho*nu_aniso*S
tau_ij = Delta_p*(b_i b_j - delta_ij/3)
```

Here `Delta_p = p_perpendicular - p_parallel`; papers writing `Pi` in the
momentum flux have `Pi = -tau`. Gradients are centered without slope limiting, following
[Kempf, Rincon & Clerc, Appendix B](https://arxiv.org/html/2306.12460v2#A2).
Density and velocity are arithmetic face averages. Normal B is the current CT
face field; transverse B is averaged from neighboring cell-centered fields.
Fluxes have the same forms as above, using current-stage fields. At zero face B,
the anisotropic stress is zero; unmagnetized viscosity requires an isotropic
coefficient. The continuum heating rate is `3*rho*nu_aniso*S^2`; discrete local
thermal positivity remains dependent on the state and discretization. See
[Berlok, Pakmor & Pfrommer (2020)](https://arxiv.org/abs/1911.05694) for Braginskii
damping, heating and RKL2 benchmarks on a moving mesh.

`AddViscousFluxHyper4` uses centered face strains and their discrete Laplacian.
In continuum notation, summing repeated indices:

```text
S_ij = (d_j v_i + d_i v_j)/2 - delta_ij div(v)/3
tau4_ij = -2 mu4 Laplacian(S_ij)
J_j = 2 mu4 S_ab d_j S_ab
F_momentum(i),j = -tau4_ij
F_energy,j = -v_i tau4_ij - J_j
```

For constant `mu4`, `J = mu4 grad(S:S)` yields continuum heat production
`Q4 = 2 mu4 sum_ijk (d_k S_ij)^2`; omitting J changes the local thermal equation.
The stencil uses two primitive ghost cells per active direction and the existing
fluid variables. The continuum formulation follows
[Haugen & Brandenburg (2004), equations 2–5](https://arxiv.org/html/astro-ph/0402301v2#S2)
and [Necas, Novotny & Silhavy (1989), equation 2.12 and Example 2.21](https://dml.cz/bitstream/handle/10338.dmlcz/106776/CommentatMathUnivCarol_030-1989-3_14.pdf).
Haugen & Brandenburg use sixth-order isothermal hyperviscosity; AthenaK uses
the fourth-order specialization with ideal-gas energy evolution.

Uniform-density transverse velocity modes decay at `(mu4/rho)*|k|^4` in the
continuum; longitudinal rates are `4/3` times transverse rates. Coupled waves
and discrete stencils have their own dispersion relations. Cellwise thermal
positivity and finite-step RKL2 dissipation require separate numerical checks.

## Timestep and STS safeguards

`Viscosity::NewTimeStep` returns the reciprocal summed rate, minimized over
blocks; sums and `ndim` include active directions only:

```text
rate_iso = 2*nu_iso*(sum_d dx_d^-2 + max_d(dx_d^-2)/3)
rate_aniso = (8/3)*ndim*nu_aniso*max_d(dx_d^-2)
rate_hyper4 = (32/3)*(mu_hyper4/rho_min)*(sum_d dx_d^-2)^2
```

The mesh applies CFL and minimizes across ranks. Hyperviscosity uses the current
positive active-cell minimum density on the local pack; density and bounds are
refreshed after fluid evolution. The rate bound controls diffusion stability;
variable-density thermal positivity is checked separately.
Zero hyperviscosity preserves the ordinary-viscosity timestep arithmetic.

The driver places an RKL2 half-sweep before and after the fluid update, reusing
conservative fluxes and boundary exchanges. References are
[Meyer, Balsara & Aslam (2014)](https://www.sciencedirect.com/science/article/pii/S0021999113005597)
and [Vaidya et al. (2017)](https://arxiv.org/abs/1702.05487).
`InspectRawState` checks finite density/momentum and applicable positive internal
energy/EOS floors before EOS conversion, using current CT faces in MHD.
`Driver::ExecuteViscositySTSSweep` restores rejected trials and halves intervals
collectively after boundary exchanges finish. Invalid starting states and
exhausted retries are fatal. `ExecutePreSTSSweep` refreshes fluid, source, user
and explicit-diffusion bounds after heating and repeats from the cycle start
if needed; source application and work callbacks are not executed within retries.
These state checks do not estimate truncation error or guarantee dissipation.
Stiff modes require timestep-convergence checks of velocity and thermal energy.

## Refinement and callback restrictions

Refinement support is isotropic only. For ideal Hydro viscosity STS with conserved
prolongation, `HydroProlongTheta` contracts inadmissible siblings toward their
parent by one factor, preserving their conserved mean in ghost interpolation and
active AMR. Admissible reconstructions remain unchanged; invalid parents
terminate the run. MHD/CT prolongation is unchanged.

User callbacks require code enrollment through `ProblemGenerator` flags
`user_bcs_retry_safe` and `user_dt_retry_safe`. Boundaries may deterministically
update only snapshotted ghost fields; timestep callbacks may deterministically
update only `dtnew`. Persistent side effects are forbidden. Enrollment is set
in the problem generator code; time accuracy depends on the boundary implementation.

Viscosity STS rejects primitive prolongation, cyclic zoom, simultaneous Hydro/MHD
owners and co-selected STS resistivity. General STS restrictions exclude
relativity, radiation, ion-neutral physics, shearing boxes and orbital advection.
Nonlinear conduction cannot share an STS sweep with viscosity; see
[thermal_conduction.md](thermal_conduction.md). Hyperviscosity supports periodic
and outflow boundaries only; callback enrollment does not extend that set.

See the [feature validation report](https://github.com/rajsekhar33/athenak-RM/blob/feature/viscosity/docs/feature-validation/viscosity/README.md) for results, figures and reproduction instructions.
