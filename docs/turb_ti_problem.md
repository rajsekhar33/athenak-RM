# `turb_ti`: Turbulent Thermal Instability Problem Generator

`src/pgen/turb_ti.cpp` sets up a homogeneous, isotropically-driven
turbulence box with realistic (`ISMCoolFn`) radiative cooling in the
thermally-unstable regime.

## Setup

The box is initialized at uniform density (`n0`, cm⁻³) and temperature
(`temp`, K), optionally perturbed by a cell-by-cell log-normal random kick
(`sigma_perturb`) or, as in the shipped example, a power-law log-density
fluctuation spectrum set up via a `<dens_init>` block. Turbulence is driven
via the generic `<turb_driving>` mechanism. Cooling uses the tabulated ISM cooling function `ISMCoolFn`
(metallicity-dependent), switched off above a density ceiling
(`n0_ceiling_cool`) or outside configurable temperature bounds. Because the
cooling curve is thermally unstable across most of this range, gas
segregates into cold/hot phases (`temp_cold`/`temp_hot` are reference
temperatures for diagnostics, not hard boundaries).

Radiative losses can be offset by equilibrium heating
(`use_equ_heating = true`): every `equ_heat_cycle` cycles, a heating rate is
recomputed to balance a `heat_fraction` of net cooling, letting the box
reach a statistically steady multiphase state instead of cooling away.

`inputs/problem/turb_ti/turb_ti.athinput` is taken from a validated production run
(128³, `n0 = 3e-3 cm⁻³`, `T = 10⁶ K`).

## Reference

Mohapatra, Dutta & Sharma, "Multiphase gas in Circumgalactic cloud
complexes: Insights from kiloparsec-scale Magnetohydrodynamic Turbulence
Simulations" ([arXiv:2511.00229](https://arxiv.org/abs/2511.00229))
