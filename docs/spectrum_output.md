# Power spectra

`file_type = spec` computes Fourier power spectra during a simulation. It supports
uniform Cartesian meshes, scalar fields and Helmholtz decomposition of a vector
field. Kokkos executes the local transforms; MPI distributes the transform across
ranks. The feature uses the normal AthenaK build options.

## Definition and normalization

For a real field sampled at cell indices **j** on a grid with `N = nx*ny*nz` cells,
the forward transform and power are

$$
F(\mathbf m)=\sum_{\mathbf j}q_{\mathbf j}
 \exp\!\left[-2\pi i\sum_d m_d j_d/n_d\right],
\qquad P(\mathbf m)=\frac{|F(\mathbf m)|^2}{N}.
$$

The wavevector components are `k_d = m_d/L_d`, in cycles per unit length, where
`L_d = n_d*dx_d` is the transformed length. Fourier indices above the positive
half of the axis represent negative frequencies; the even-length Nyquist mode
has a negative index. The coordinate origin for the transform is the first
selected cell.

Each shell contains modes with the same range of `|k|`. The output records their
summed power `E`, count `nk`, representative wavenumber `k` and edges `kbins`.
For occupied shells, `E/nk` is the mean power per mode. A shell-integrated power
per unit wavenumber is `E/diff(kbins)`. Dividing either quantity by `N` converts
the squared-field sum normalization to a mean-square normalization.

With unit weights, Parseval's relation gives

$$
\sum_b E_b=\sum_{\mathbf j}q_{\mathbf j}^2.
$$

Linear bins include the zero-frequency mode. Logarithmic bins exclude it, so
`sum(E) = sumsq - P_DC`; `P_DC` stores the zero-mode power separately.
Floating-point accumulation introduces roundoff in these identities.

### Modes and weights

- `mode = scalar` produces a spectrum for every variable and weight combination.
  `weights = ones` gives the field power defined above. A field-valued weight
  produces the per-mode product `P_q(k)*P_w(k)` before shell summation.
- `mode = helmholtz` takes three vector components and `weights = ones`.
  For nonzero **k**, the compressive component is
  `F_comp = k*(k·F)/|k|²`; the solenoidal component is `F - F_comp`.
  The zero mode is assigned to the solenoidal spectrum.

Real-field conjugate partners contribute their full multiplicities to `nk` and
power. Helmholtz projection evaluates each partner with its own wavevector,
including on Nyquist planes.

### Bins and stratified output

For `logscale = false`, input `nbin` specifies the number of linearly spaced
edges, giving `nbin - 1` histogram bins. For `logscale = true`, it specifies the
number of logarithmic bins. The lower logarithmic edge is `0.25/L_max`, using
the longest transformed active axis. The upper edge includes the largest
represented Fourier magnitude. The output header always reports the actual
number of histogram bins.

`strat = true` crops a physical distance `skip` from each end of x3, converts the
crop limits to grid indices by truncation, and applies a Hamming window along the
remaining x3 interval. The window is scaled to unit mean square. It multiplies
each variable before transformation; field-valued weights retain their original
values. `sumsq` then refers to the windowed field.

For nonperiodic simulation boundaries, the transform describes the periodic
extension of the sampled box. Cropping and windowing can reduce the effect of
an end-to-end discontinuity in a stratified domain.

## Configuration

```ini
<output6>
file_type = spec
id = velocity
mode = scalar
variables = hydro_w_vx, hydro_w_vy
weights = ones
nbin = 256
logscale = false
strat = false
skip = 0.0
transport = device
persistent_buffers = false
dt = 0.1
```

`variables` and any field-valued `weights` must each name individual supported
field components. MHD examples include `mhd_w_vx` and `mhd_w_d`. A Helmholtz block
uses the three velocity components in x1, x2, x3 order. Defaults for `nbin` are
256 in scalar mode and 200 in Helmholtz mode.

The transform covers the complete uniform mesh, with the optional x3 crop above.
Refinement, derived variables, slices, individual MeshBlock selection, ghost-zone
output, restricted spatial bounds and shearing-periodic boundaries are unsupported.
Real-valued input parameters must be finite numbers.

`transport = device` requires GPU-aware MPI for device-buffer communication.
`transport = host` stages the communication through host memory. Both compute
the same transform.

## Files and memory

Files are written as `spec_<id>/<basename>.NNNNN.spec`. Each has an ASCII
`key=value` header ending in `header offset=N`, followed by float64 arrays in
the order specified by `payload`:

```text
scalar:
  kbins[nbin+1], k[nbin], nk[nbin], E[nweight,nvar,nbin],
  P_DC[nweight,nvar], sumsq[nvar]
helmholtz:
  kbins[nbin+1], k[nbin], nk[nbin], E_comp[nbin], E_sol[nbin],
  P_DC[1], sumsq[1]
```

The header records the variables, transformed grid, lengths, MPI pencil grid,
and setup, transform and binning timings. Timings are maxima over ranks.
`memory_device_bytes_max_rank` and `memory_host_bytes_max_rank` count the
spectrum's allocated buffers on the largest rank; total simulation memory also
includes the fluid state and other modules.

By default, large transform buffers and plans are released after each output.
`persistent_buffers = true` retains them between outputs, reducing repeated
setup at the cost of persistent memory use. Each spectrum output block owns its
buffers.

Use `read_spec()` in [sf_spectra_read.py](../vis/python/sf_spectra_read.py) to read
the header and named arrays.

## Implementation and verification

The [pencil transform](../src/fft/pencil_fft.hpp) redistributes MeshBlocks into
x-pencils, transforms x, exchanges into y-pencils, and transforms y. Three-dimensional
meshes add a second exchange and the z transform. Active ranks form an
automatically selected two-dimensional pencil grid. Local transforms use
Stockham stages with radices 2–8 and Bluestein transforms for other factors.

The [validation tests](https://github.com/rajsekhar33/athenak-RM/tree/feature/onthefly-sf-spectra/tests) cover Fourier coefficients, analytic
single modes, scalar power, Helmholtz projection, Parseval normalization, odd and
even dimensions, Nyquist modes, final-bin inclusion, MPI redistribution and restart
output. These tests establish both the physical normalization and the complete
Fourier-mode inventory.
