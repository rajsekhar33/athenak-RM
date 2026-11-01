# On-the-fly Power Spectrum Output (`file_type = spec`)

Computes shell-binned power spectra **during the run** with an in-house distributed FFT
(`src/fft/`, plain Kokkos and MPI, no external library and no build flags). The
conventions follow `athena_research/operations/spectra.py` exactly. The distributed
FFT is described under "Distributed FFT" below.

Status: first release. Uniform grids only, no derived variables.

## What is computed

For each variable `q` (and weight `w`), with `N` the number of transformed cells:

- `F(k)` is numpy's unnormalised forward FFT of `q` over the selected cells, with index
  0 at the first selected cell.
- `P(k) = |F(k)|^2 / N`.
- `k = fftfreq(n) * n / L` per axis, in cycles per unit length. The even-length Nyquist
  is labelled negative.
- `L = n * dx` is the length actually transformed.
- Shell histograms in `|k|` with Python's `_spectral_grid` edges:
  - **linear:** `linspace(0, kmax, nbin)`, so `nbin - 1` bins that include `k = 0`;
  - **log:** `nbin` bins from `0.25 / L_max`, which exclude `k = 0`.

**Modes** (`mode`):

- `scalar`: one spectrum per (weight, variable).
  - `weights = ones` gives the plain spectrum.
  - A weight field gives the legacy per-mode product `P_q(k) * P_w(k)`, exactly as
    `get_spectrum_mb(weight_name=...)`. The weight is not windowed.
- `helmholtz`: three velocity-like components. The compressive part is
  `(k.F) k / |k|^2`; the solenoidal part is the remainder. The `k = 0` mode is
  solenoidal.

**`strat = true`:**
- crops `skip` from both z ends with Python's `get_refined_data` index rule
  (`int()` truncation);
- applies a z Hamming window scaled to unit mean square, to `q` only.

**Parseval:** `sumsq` is the sum of the (windowed) squared field. For unweighted spectra,
`sum(E) = sumsq` with linear bins and `sumsq - P_DC` with log bins, exactly: every
non-DC mode is binned.

**One deliberate difference from spectra.py:** its last log edge `10**log10(kmax)` can
round 1 ulp below the corner `|k|`, which then drops out of the histogram (up to an
order-unity fraction of the power for a field concentrated there). Here the final edge
is `max(10**log10(kmax), kmax)`.
- Every other edge, and all linear bins, are identical to spectra.py.
- Changing `kbins[-1]` to that value in `_spectral_grid` makes the two codes agree
  exactly.

Only `kx >= 0` is stored internally (real input). Every other mode also stands for its
conjugate partner:
- counts and powers are weighted accordingly;
- in `helmholtz` mode the partner is evaluated with its own wavevector, which differs
  from `-k` on Nyquist planes.

The totals therefore equal the full complex transform's.

## Input block

One key per line. The whole mesh is transformed; the only crop is `strat`/`skip` in x3.
`x1_min` ... `x3_max` (other than the mesh bounds), `slice_x1/2/3`, `gid`,
`ghost_zones` and shearing-periodic boundaries are fatal errors. Every variable or
weight name must give exactly one field component in the run. Real-valued keys must
be single finite numbers.

```ini
<output6>
file_type  = spec
id         = vel                    # directory spec_<id>/ (default: block name)
mode       = scalar                 # scalar | helmholtz
variables  = hydro_w_vx, hydro_w_vy
weights    = ones, hydro_w_d        # scalar only; "ones" = unweighted
nbin       = 256                    # default 256 (scalar), 200 (helmholtz), as spectra.py
logscale   = false
strat      = false                  # z crop by 'skip' plus Hamming window
skip       = 0.0
transport  = device                 # device (CUDA/HIP-aware MPI) | host (staged)
persistent_buffers = false          # true: keep the FFT plan and buffers between outputs
dt         = 0.1
```

## Output files

`spec_<id>/<basename>.NNNNN.spec`: an ASCII `key=value` header ending in
`header offset=N`. The header records:
- the selection (`index_lo`, `shape`, `length`);
- the FFT convention;
- the pencil grid;
- timings, all maximum over ranks:
  - `time_setup` (plan construction, when it happened in this output);
  - `time_fft` (every transform, weights included);
  - `time_bin`;
  - per-stage times of the last transform;
- `memory_device_bytes_max_rank` and `memory_host_bytes_max_rank`: the bytes held in
  the buffers listed below at the peak of the output, on the rank holding the most.
  They count these buffers only; they are not a measurement of total device memory use.
  - Device memory covers the transform input, the pencils, the exchange buffers, the
    FFT plans with their work buffers, the coefficient copies and the weight powers.
  - Host memory covers the staging buffers.

**Memory lifetime:** by default (`persistent_buffers = false`) the FFT plan and the
large transform buffers are built at the start of each output and freed at its end.
Small arrays stay allocated (bin edges, the z window, host result arrays). With `true`
the transform buffers persist, which saves the setup time at the cost of holding the
memory. Several spec blocks each own their
buffers.

Then float64 arrays in the order of the header's `payload=` key:

```text
scalar:    kbins[nbin+1] k[nbin] nk[nbin] E[nweight,nvar,nbin] P_DC[nweight,nvar] sumsq[nvar]
helmholtz: kbins[nbin+1] k[nbin] nk[nbin] E_comp[nbin] E_sol[nbin] P_DC[1] sumsq[1]
```

- `nbin` in the header is the number of histogram bins.
- Python's `E/nk` and `4*pi*k^2*E/nk` follow directly.
- Use `read_spec()` from [sf_spectra_read.py](../vis/python/sf_spectra_read.py).

## Distributed FFT

`fft::PencilFFT` (`src/fft/pencil_fft.hpp`) arranges `Py x Pz` active ranks (chosen
automatically) and runs:

1. **R0, blocks to x-pencils:** point-to-point; every rank sends.
2. **x transform** (half spectrum kept).
3. **R1** within rows.
4. **y transform.**
5. **R2** within columns (3D only).
6. **z transform.**

The local transforms use `fft::LocalFFT` (Stockham radices 2-8, Bluestein for other
primes).

## Validation

The [test guide](../tests/README.md) lists the self-contained checks and the
optional comparisons that require a separate `athena_research` checkout.

- `tests/fft/run_pencil_suite.sh`: every coefficient vs `numpy.fft.rfftn`. Covers:
  - 1-8 ranks, crops, host and device transport;
  - non-cubic, 2D and 1D meshes;
  - more ranks than pencils.
- `tests/spec/spec_reference.py`: every array vs spectra.py's own helpers and formulas
  (full complex FFTs), plus Parseval. Input: `inputs/tests/spec_unit_test.athinput`
  (anisotropic cells, odd crop lengths, linear/log bins, weighted, Helmholtz).

Both pass on CPU (1 and 3 ranks) and on GPU (2 ranks, CUDA-aware MPI), to about 1e-14
relative.
