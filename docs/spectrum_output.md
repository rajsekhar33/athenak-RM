# Power-spectrum output

`file_type=spec` computes shell-binned power spectra at the output cadence
using a distributed Kokkos/MPI FFT. It supports uniform grids on CPU and GPU
without an external FFT library or extra build flags.

## Physical quantities

For a field `q` on `N` transformed cells, the unnormalized forward transform
is `F(k)`, with power `P(k)=abs(F(k))**2/N`. Wave numbers are in cycles
per unit length: `k_axis=fftfreq(n)*n/L`, with `L=n*dx`. Even-length
Nyquist frequencies have negative labels. The transform origin is the first
selected cell.

| Mode | Quantity |
|---|---|
| `scalar`, `weights=ones` | Shell sum of `P_q(k)` for each variable. |
| `scalar`, field weight `w` | Shell sum of `P_q(k)*P_w(k)`. This spectral product differs from the spectrum of `w*q`. |
| `helmholtz` | Three vector components in x1/x2/x3 order, split into `F_comp=(k dot F)*k/abs(k)**2` and `F_sol=F-F_comp`. The zero mode belongs to the solenoidal component. |

The internal real half-spectrum includes conjugate partners in shell sums
and mode counts. Helmholtz projections use each partner's wavevector,
including on Nyquist planes.

For `strat=true`, `skip` removes a distance from each x3 end, with indices
obtained by truncation. A Hamming window along retained x3 cells, normalized
to unit mean square, multiplies each variable. Weights remain unwindowed.

Shell edges cover every nonzero Fourier mode:

- Linear spacing uses `linspace(0,kmax,nbin)`: the input specifies the number
  of edges, giving `nbin-1` bins including the zero mode.
- Log spacing uses `nbin` bins starting at `0.25/L_max` over active axes,
  excluding zero. The final edge includes the corner wave number.

`kmax` is the larger of the corner wave-number magnitude and the smallest
`n/L` over active axes. Header `nbin` is the actual number of bins.
`k` contains arithmetic midpoints for linear bins and geometric midpoints
for log bins.

`E` is a shell sum; `nk` counts Fourier modes. For an unweighted scalar
field, Parseval gives `sum(E)=sumsq` for linear bins or `sumsq-P_DC` for
log bins, to roundoff. `sumsq=sum(q**2)` includes the window when enabled,
and `P_DC` is zero-mode power. The same identity applies to
`E_comp+E_sol`, with `sumsq` summed over components. Divide by `N` for
mean-square rather than spatially summed power.

## Configuration

```ini
<output6>
file_type = spec
id        = velocity
mode      = helmholtz
variables = hydro_w_vx, hydro_w_vy, hydro_w_vz
nbin      = 200
logscale  = false
strat     = false
skip      = 0.0
transport = device
persistent_buffers = false
dt        = 0.1
```

The default mode is `scalar`; it accepts comma-separated variables and
`weights` (default `ones`), with at most eight weight/variable combinations.
Default `nbin` is 256 for scalar and 200 for Helmholtz. Helmholtz requires
exactly three variables and the single weight `ones`. Each name must select
one ordinary output component; derived fields and computed
`pressure/temperature` are unavailable for this output.

Each spectrum block needs a distinct `id`, defaulting to its block name.
The full mesh is transformed, with `strat/skip` as the supported crop.
SMR/AMR, shearing boundaries, `slice_x1/2/3`, `gid`, `ghost_zones`, and
spatial bounds differing from mesh bounds are rejected.

`transport=device` (default) sends device buffers through MPI and requires
GPU-aware MPI for GPU builds. `transport=host` stages exchanges through host
memory. CPU builds use host-resident arrays for both choices.

By default, `persistent_buffers=false` releases the FFT plan and large
buffers after each output. Set true to reuse them and avoid setup cost;
each spectrum block retains its own buffers. Small bin, window, and result
arrays remain allocated in both modes.

## Files and reader

Files are `spec_<id>/<basename>.NNNNN.spec`. An ASCII `key=value` header
ends with `header offset=N`, the binary payload start. Metadata records
time/cycle, variables, transformed shape and lengths, index bounds, FFT
convention, pencil decomposition, and transport/buffer settings.

Timings are maximums over ranks for setup, transforms, binning, and stages
of the last transform. Device/host memory entries count this output's arrays
and host staging at peak allocation on the largest rank. Other simulation
and runtime allocations need separate memory budgeting.

C-ordered float64 payload arrays use the header's byte order:

```text
scalar:
  kbins[nbin+1] k[nbin] nk[nbin]
  E[nweight,nvar,nbin] P_DC[nweight,nvar] sumsq[nvar]
helmholtz:
  kbins[nbin+1] k[nbin] nk[nbin]
  E_comp[nbin] E_sol[nbin] P_DC[1] sumsq[1]
```

Use [sf_spectra_read.py](../vis/python/sf_spectra_read.py):

```python
import sys
import numpy as np

sys.path.insert(0, "/path/to/athenak/vis/python")
from sf_spectra_read import read_spec

meta, a = read_spec("spec_velocity/run.00010.spec")
power = a["E_comp"] + a["E_sol"]
N = np.prod([int(n) for n in meta["shape"].split(",")])
mean_square_per_shell = power / N
mean_power_per_mode = np.divide(
    power, a["nk"], out=np.full_like(power, np.nan), where=a["nk"] > 0)
```

For scalar mode, select `power=a["E"][weight_index,variable_index]`.
`E/nk` is a per-mode average; `4*pi*k**2*E/nk` is a further radial
normalization that should match the intended comparison.

Full transform, Parseval, and reference-spectrum comparisons are retained
with the [feature-branch tests](https://github.com/rajsekhar33/athenak-RM/tree/feature/onthefly-sf-spectra/tests).
