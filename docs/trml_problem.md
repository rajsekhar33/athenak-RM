# `TRML`: Turbulent Radiative Mixing Layer Problem Generator

`src/pgen/TRML.cpp` is an AthenaK port of Drummond Fielding's turbulent
radiative mixing layer setup.

## Setup

Hot gas (`T_hot = pgas_0/rho_0`) and cold gas (`T_cold = T_hot/contrast`)
are initialized on either side of an interface at `x3 = z_interface` (the
mixing-layer-normal direction is `x3`), smoothed over `smoothing_thickness`,
and sheared past each other at `±velocity/2`. `x1`/`x2` are periodic; `x3`
uses outflow boundaries. `contrast` is a temperature contrast — density
contrast is derived by balancing total (thermal + magnetic) pressure across
the interface.

Cooling is a piecewise power-law function of temperature normalized to peak
at `T_peak = T_cold * T_peak_over_T_cold`, with the normalization set by
`xi = t_shear / t_cool_0`, where `t_shear = 1/velocity` and `t_cool_0` is
the cooling time at `T_peak` (set either directly, or via `xi`, which
derives `t_cool_0 = t_shear/xi`). `beta_lo`/`beta_hi` (magnetic-to-thermal
pressure ratio bounds at `T_cold`/`T_hot`) set the cooling curve's slopes
between the two phases.

Optional turbulence (`<turb_driving>`) can stir the layer. Optional frame
tracking (`use_frame_tracking = true`) periodically shifts the domain in
`x3` to keep gas near `T_peak` centered.

`inputs/problem/TRML/TRML.athinput` is a validated example (`contrast = 100`,
`t_cool_0 = 0.01`).
