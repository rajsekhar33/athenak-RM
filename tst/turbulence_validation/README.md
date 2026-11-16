# Turbulence forcing checks

## Forcing scheme

The acceleration is applied inside every RK stage with weight
\(\beta_s\Delta t\). In power normalization the scale \(s\) of the O-U pattern
\(\mathbf g\) solves

\[
s^2\,\frac{\Delta t}{2}\int\rho|\mathbf g|^2dV + s\int\rho\mathbf v\cdot\mathbf g\,dV
= \mathrm{dedt},
\]

the energy gain of a constant acceleration over one step. The root of
smallest magnitude is used. When the flow does negative work against the pattern
this root is negative, flipping the pattern's sign for that step only; the stored
O-U field is unchanged.

Each stage adds \(\rho\mathbf v\cdot\mathbf a\,\beta_s\Delta t\) to the total
energy, evaluated with that stage's velocity. Integrators with two or more stages
then integrate the work of a constant acceleration exactly, including its
\(\tfrac12\rho|\mathbf a|^2\Delta t^2\) part. RK1 adds that part explicitly.

The timestep is capped so that a step ends on the next multiple of
`dt_turb_update`. The cap is a scheduling limit and is not multiplied by the CFL
number. A time within `dt_turb_thresh` below a boundary counts as having reached it.

## Unit checks

`per_stage.py` compares default power forcing with a reference executable, field by field and
bit for bit. It also checks the volume RMS normalization for RK1–RK4, nonunit
volume, nonuniform density, zero amplitude and invalid inputs. Each RMS field is
reconstructed independently from the stored raw O-U field.

```bash
python3 tst/turbulence_validation/per_stage.py \
  --binary /path/to/athena --source-manifest /path/to/source.json \
  --baseline-binary /path/to/reference/athena \
  --baseline-source-manifest /path/to/reference/source.json \
  --out /path/to/new-test-output
```

## Timestep and energy checks

`forcing_checks.py` requires every step length to equal the smallest of twice the
previous step, the CFL-limited hydro step recomputed from the saved state, and the
turbulence cap. The O-U update count must follow the same boundary rule, and a
restart one rounding error below a boundary must take the same step and update as a
restart on it. From rest, the first driven step must inject
\(\mathrm{dedt}\,\Delta t\) to roundoff with RK1 and RK2. RK3 and RK4 differ by the
work done on the flow's own evolution within the step, about \(3\times10^{-4}\) to
\(7\times10^{-4}\) of the injection here; the check accepts \(2\times10^{-3}\).
From a stationary shear running with or against the pattern, the applied
acceleration must equal the smallest-magnitude root, solved independently from the
saved state; for negative work that root is negative. Invalid power inputs and a
zero forcing pattern with positive power must abort.

```bash
python3 tst/turbulence_validation/forcing_checks.py \
  --binary /path/to/athena --out /path/to/new-test-output
```

## AMR checks

The AMR checks exercise the per-stage turbulence source through refinement,
coarsening, block redistribution and restart. They measure active conserved
fields, cell volumes, forcing fields and random-generator state from native
restart outputs. Hydro and MHD runs cover both stored and regenerated forcing.

```bash
python3 tst/turbulence_validation/amr.py \
  --binary /path/to/athena --out /path/to/new-test-output \
  --ranks 2 --launcher 'mpirun -np {ranks}'
```

For `normalization = rms_acceleration`, the prescribed quantity is

\[
a_{\rm rms}=\left(\frac{\int |\mathbf a|^2\,dV}{\int dV}\right)^{1/2}.
\]

The acceleration has its density-weighted mean removed before normalization.
The relevant checks are the volume-weighted RMS and
\(\int\rho\mathbf a\,dV=0\), with mixed cell volumes and nonuniform density.
This mode specifies an acceleration amplitude; the energy input depends on
the velocity field. `amr.py --normalization rms_acceleration` selects it.

`edge_cases.py duration` ends driving (`turb_flag = 1`) between mesh changes and
checks that energy and momentum stop changing at the end time, through later
refinement and restarts. `edge_cases.py replay` measures the cost of rebuilding the
forcing after a mesh change. The rebuild replays every O-U update since \(t=0\),
so its cost grows with the number of updates already made.

## Driven-flow comparison

`realistic.py` runs a 128³ periodic ideal-gas flow and compares runs, for example
two executables. The input has
\(\rho_0=1\), \(p_0=1/\gamma\), \(\gamma=5/3\), solenoidal driving at
\(1\leq k\leq3\), correlation time 0.5 and forcing updates every 0.005.
RK3 with CFL 0.3 evolves the unit box to time 2.

For power normalization, `dedt = 0.1` is the total energy injection rate of
the box. The expected integrated gain is

\[
E(t)-E(0)=\mathrm{dedt}\sum_{n\in\text{driven steps}}\Delta t_n.
\]

The initial undriven cycle is excluded. Per-cycle histories retain sufficient
precision to measure this budget; sparse raw checkpoints independently check
the energy integrals. The report gives the cumulative budget and the separate
time-1-to-2 budget, each with a 1% acceptance threshold. Mass conservation,
net momentum, pressure bounds, repair counters, Mach number and kinetic and
internal energy are also recorded. Velocity spectra use
\(E_v(k)=\tfrac12\sum_{\mathbf k\in k}|\widetilde{\mathbf v}|^2\), with a
Fourier normalization satisfying
\(\sum_k E_v(k)=\tfrac12\langle|\mathbf v|^2\rangle_V\).

```bash
python3 tst/turbulence_validation/realistic.py run \
  --binary /path/to/athena --manifest /path/to/build-manifest.json \
  --out /path/to/new-run --label per-stage

python3 tst/turbulence_validation/realistic.py compare \
  /path/to/first-run /path/to/second-run \
  --out /path/to/new-comparison
```

The build manifest must contain the executable SHA-256 and its build-time
source identity. Each run retains the effective input, command, native log,
histories and checkpoints. `--pilot-cycles 32` limits a timing pilot to 32
cycles. Such a pilot samples startup and cannot establish a developed-flow
result. Spectra and flow statistics support the physical comparison;
pointwise equality is unsuitable once the turbulent trajectories diverge.
