# Ohmic resistivity

AthenaK evolves Ohmic magnetic diffusion with constrained transport and a
conservative energy flux. Constant magnetic diffusivity and a scalar Spitzer
law support explicit integration and RKL2 super-time-stepping (STS).

## Equations and spatial operator

With magnetic energy density $B^2/2$ and $\boldsymbol J=\nabla\times\boldsymbol B$,

$$
\boldsymbol E_\eta=\eta_m\boldsymbol J,\qquad
\partial_t\boldsymbol B=\nabla\times(\boldsymbol v\times\boldsymbol B)
-\nabla\times(\eta_m\boldsymbol J),\qquad
\boldsymbol F_{E,\eta}=\eta_m\boldsymbol J\times\boldsymbol B.
$$

Here $\eta_m$ is magnetic diffusivity, with dimensions length squared per time.
The conservative flux converts magnetic energy into gas heat while preserving
total energy in a periodic domain. This already accounts for Joule heating.

The CT electric field uses discrete currents at edges. Temperature-dependent
cell diffusivities are averaged arithmetically to those edges, following
[Athena++'s Ohmic implementation](https://github.com/PrincetonUniversity/athena/blob/main/src/field/field_diffusion/diffusivity.cpp).
The Poynting flux uses the same edge coefficients. This retains spatial
coefficient gradients in the full curl operator.

## Inputs and units

For constant diffusion, set `mhd/eta_ohm` in code units. For temperature-dependent
diffusion, use the code-temperature variable $T=p/\rho$ and
$\eta_m=\eta_0T^{-3/2}$:

```text
<mhd>
eos = ideal
gamma = 1.6666666666666667
eta_ohm = 0
eta_ad = 0
eta_spitzer = true
spitzer_eta0_code = 0.01
resistivity_integrator = explicit
```

Select exactly one of `spitzer_eta0_code` and `spitzer_eta0_cgs`. The CGS
normalization defines $\eta_m[\mathrm{cm^2\,s^{-1}}]
=\eta_{0,\mathrm{cgs}}T[\mathrm K]^{-3/2}$ and includes the chosen composition
and Coulomb logarithm. Supply an explicit `<units>` block for CGS input.
The Units module supplies length $L_0$, time $t_0$ and temperature $T_0$, giving

$$
\eta_{0,\mathrm{code}}=
\eta_{0,\mathrm{cgs}}\frac{t_0}{L_0^2T_0^{3/2}}.
$$

The Spitzer option requires finite positive temperature and normalization,
Newtonian ideal-gas MHD, and zero constant Ohmic/ambipolar coefficients.
The temperature law follows the collisional transport treatment of
[Spitzer & Härm (1953)](https://doi.org/10.1103/PhysRev.89.977).

## Time integration

Explicit Spitzer integration supports SSP `rk1`, `rk2` and `rk3`, with
`0 < cfl_number <= 1`. The Ohmic forward-Euler bound is the minimum over active
cells, the coefficient halo and active coordinate directions of
$\Delta x_d^2/(2N_d\eta_m)$, where $N_d$ is the number of active dimensions.
The mesh applies the CFL factor and reduces the timestep across MPI ranks.

For RKL2, change `resistivity_integrator` to `sts` and set:

```text
<time>
integrator = rk2
sts_integrator = rkl2
sts_max_dt_ratio = 16
```

RKL2 diffusion half-sweeps surround the fluid update. All selected STS processes
contribute to the summed inverse diffusion bound. Spitzer coefficients are
reevaluated from each current stage. The full-state guard checks raw internal
energy before EOS floors and verifies that the current coefficients fit the
selected stage count. A failed trial completes communication, restores fluid
U/W, cell-centered fields and fine/coarse CT fields, then halves its interval.
The recurrence and split integration follow
[Meyer, Balsara & Aslam (2014)](https://doi.org/10.1016/j.jcp.2013.08.021).

The same guard supports joint viscosity and nonlinear conduction. Saturated
conduction retains its unaccelerated donor bound; see
[thermal_conduction.md](thermal_conduction.md) and [viscosity.md](viscosity.md).
STS accuracy should be checked by timestep refinement.

## Refinement and boundaries

Ohmic diffusion uses existing CT EMF/flux corrections and conserved-variable
mesh transfer. Full-state STS requires conserved prolongation, one fluid owner
and Newtonian supported physics. It rejects cyclic zoom. User boundary and
timestep callbacks require `user_bcs_retry_safe` and `user_dt_retry_safe`
enrollment in the problem generator. Boundary callbacks may deterministically
update snapshotted ghost fields; timestep callbacks may update only `dtnew`.
Ambipolar diffusion retains explicit integration; Hall diffusion is unimplemented.

## Implementation

| Source | Responsibilities |
| --- | --- |
| [resistivity.cpp](../src/diffusion/resistivity.cpp), [resistivity.hpp](../src/diffusion/resistivity.hpp) | Input/CGS conversion; cell and edge diffusivities; `AddEMFOhmic`, `AddFluxOhmic`, `NewTimeStep` and raw-state/stage-budget inspection. |
| [mhd.cpp](../src/mhd/mhd.cpp) | Register the Ohmic process and its explicit/STS timestep contribution. |
| [mhd_sts.cpp](../src/mhd/mhd_sts.cpp), [mhd_tasks.cpp](../src/mhd/mhd_tasks.cpp) | Execute conservative fluid/CT updates, synchronize boundaries and inspect stages before EOS conversion. |
| [driver.cpp](../src/driver/driver.cpp), [driver.hpp](../src/driver/driver.hpp) | Validate selections, combine current-stage bounds, snapshot shared fields and retry rejected sweeps. |

Physical tests, analytic expectations, convergence measurements and figures are
in the [feature validation report](https://github.com/rajsekhar33/athenak-RM/blob/feature/resistivity-sts/docs/feature-validation/resistivity/README.md).
