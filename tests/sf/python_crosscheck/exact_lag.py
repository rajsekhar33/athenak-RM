#!/usr/bin/env python3
"""Independent exact all-pairs sums on a periodic uniform grid by lag enumeration (CuPy).

Every ordered pair (x, x + l) of distinct cells is visited once, with l over the
nx1*nx2*nx3 - 1 nonzero lags l_i in (-n_i/2, n_i/2]; unordered sums are half the
ordered sums. The displacement of a pair is taken from its two wrapped endpoints with the
production minimum-image rule (wrap only when |d| > L/2, so a half-box component keeps
the sign of x2 - x1), as AthenaK's SFMinImage and Python's kernels do; it is not the
enumeration lag. The longitudinal projection is then invariant under swapping the
endpoints. Each sum is split by population: E1 = both cells in the same MeshBlock
(physical block, also when a block spans a periodic axis), E2 = different MeshBlocks.

Fields are rebuilt in double precision with tests/sf/sf_reference.build_fields (the
values AthenaK's SF uses), or rounded to float32 with --f32 (the values in the .bin
snapshot Python reads).

Output .npz: rbins, and for pop in (E1, E2):
  scalar: T_scalar[pop, w(ones,dens), q(vx,vy,vz), p, b], W_scalar[pop, w, b]
  longtrans: T_lt[pop, q(L,T), p, b]
  C[pop, b]   (unordered pair counts)
Usage: python3 exact_lag.py SF_FILE OUT.npz [--f32]
"""
import argparse
import os
import sys

import cupy as cp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sf_reference import build_fields, read_sf  # noqa: E402

NV = 35  # per-lag sums: 24 scalar T, 2 scalar W, 8 long_trans T, 1 count

KERNEL = r'''
extern "C" __global__
void lag_sums(const double* d, const double* vx, const double* vy, const double* vz,
              const int* lags, const int nx, const int ny, const int nz,
              const int bx, const int by, const int bz,
              const double hx, const double hy, const double hz,
              double* out) {
  // one CUDA block per lag; out[lag][pop][NV]
  const int lag = blockIdx.x;
  const int lx = lags[3*lag], ly = lags[3*lag+1], lz = lags[3*lag+2];
  const double rx = lx*hx, ry = ly*hy, rz = lz*hz;
  const double r = sqrt(rx*rx + ry*ry + rz*rz);
  double acc[2][35];
  for (int p = 0; p < 2; ++p) for (int v = 0; v < 35; ++v) acc[p][v] = 0.0;
  const int n = nx*ny*nz;
  for (int c = threadIdx.x; c < n; c += blockDim.x) {
    const int i = c % nx, j = (c / nx) % ny, k = c / (nx*ny);
    const int ui = i + lx, uj = j + ly, uk = k + lz;           // unwrapped partner
    const int i2 = (ui % nx + nx) % nx, j2 = (uj % ny + ny) % ny,
              k2 = (uk % nz + nz) % nz;
    const int c2 = i2 + nx*(j2 + ny*k2);
    // same physical MeshBlock (wrapped partner), including blocks spanning an axis
    const bool same = (i/bx == i2/bx) && (j/by == j2/by) && (k/bz == k2/bz);
    // displacement x2 - x1 of the wrapped endpoints, strict minimum image
    double dx = (i2 - i)*hx, dy = (j2 - j)*hy, dz = (k2 - k)*hz;
    const double Lx = nx*hx, Ly = ny*hy, Lz = nz*hz;
    if (fabs(dx) > 0.5*Lx) dx -= (dx > 0.0) ? Lx : -Lx;
    if (fabs(dy) > 0.5*Ly) dy -= (dy > 0.0) ? Ly : -Ly;
    if (fabs(dz) > 0.5*Lz) dz -= (dz > 0.0) ? Lz : -Lz;
    double* a = acc[same ? 0 : 1];
    const double q1[3] = {vx[c], vy[c], vz[c]}, q2[3] = {vx[c2], vy[c2], vz[c2]};
    const double w1[2] = {1.0, d[c]}, w2[2] = {1.0, d[c2]};
    for (int w = 0; w < 2; ++w) {
      for (int q = 0; q < 3; ++q) {
        const double df = fabs(w2[w]*q2[q] - w1[w]*q1[q]);
        double pw = df;
        for (int p = 0; p < 4; ++p) { a[(w*3 + q)*4 + p] += pw; pw *= df; }
      }
      a[24 + w] += w1[w] + w2[w];
    }
    const double dux = q2[0]-q1[0], duy = q2[1]-q1[1], duz = q2[2]-q1[2];
    const double du2 = dux*dux + duy*duy + duz*duz;
    const double dl = (dux*dx + duy*dy + duz*dz)/r;
    const double dt = sqrt(fmax(0.0, du2 - dl*dl));
    const double inc[2] = {fabs(dl), dt};
    for (int q = 0; q < 2; ++q) {
      double pw = inc[q];
      for (int p = 0; p < 4; ++p) { a[26 + q*4 + p] += pw; pw *= inc[q]; }
    }
    a[34] += 1.0;
  }
  // block reduction through shared memory, one value at a time
  __shared__ double s[256];
  for (int p = 0; p < 2; ++p) {
    for (int v = 0; v < 35; ++v) {
      s[threadIdx.x] = acc[p][v];
      __syncthreads();
      for (int h = blockDim.x/2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s[threadIdx.x] += s[threadIdx.x + h];
        __syncthreads();
      }
      if (threadIdx.x == 0) out[(lag*2 + p)*35 + v] = s[0];
      __syncthreads();
    }
  }
}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sf_file', help='any .sf file of the run (for mesh, blocks, rbins)')
    ap.add_argument('out')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--noise', type=float, default=1.0)
    a = ap.parse_args()
    meta, raw = read_sf(a.sf_file)
    pos, block, f = build_fields(meta, a.noise)
    nx = [int(v) for v in meta['mesh_nx'].split(',')]
    nb = [int(v) for v in meta['meshblock_nx'].split(',')]
    L = [float(v) for v in meta['domain_length'].split(',')]
    h = [L[i] / nx[i] for i in range(3)]
    flds = [f['hydro_w_d'], f['hydro_w_vx'], f['hydro_w_vy'], f['hydro_w_vz']]
    if a.f32:
        flds = [x.astype(np.float32).astype(np.float64) for x in flds]
    dev = [cp.asarray(np.ascontiguousarray(x)) for x in flds]
    rng = [np.arange(-(n // 2) + 1, n // 2 + 1) for n in nx]   # (-n/2, n/2]
    lz, ly, lx = np.meshgrid(rng[2], rng[1], rng[0], indexing='ij')
    lags = np.stack([lx.ravel(), ly.ravel(), lz.ravel()], 1)
    lags = lags[np.any(lags != 0, axis=1)].astype(np.int32)
    rbins = raw['rbins']
    r = np.sqrt(((lags * np.array(h)) ** 2).sum(1))
    b = np.searchsorted(rbins, r, side='right') - 1
    kern = cp.RawKernel(KERNEL, 'lag_sums')
    out = cp.zeros((len(lags), 2, NV))
    dl = cp.asarray(lags.ravel())
    chunk = 32768
    for s in range(0, len(lags), chunk):
        n = min(chunk, len(lags) - s)
        kern((n,), (256,), (dev[0], dev[1], dev[2], dev[3], dl[3 * s:],
                            np.int32(nx[0]), np.int32(nx[1]), np.int32(nx[2]),
                            np.int32(nb[0]), np.int32(nb[1]), np.int32(nb[2]),
                            np.float64(h[0]), np.float64(h[1]), np.float64(h[2]),
                            out[s:]))
    out = out.get()
    nbin = len(rbins) - 1
    ok = (b >= 0) & (b < nbin)
    acc = np.zeros((2, nbin, NV))
    for pop in range(2):
        for v in range(NV):
            acc[pop, :, v] = 0.5 * np.bincount(b[ok], weights=out[ok, pop, v],
                                               minlength=nbin)
    res = dict(rbins=rbins, f32=a.f32,
               T_scalar=acc[:, :, :24].reshape(2, nbin, 2, 3, 4).transpose(0, 2, 3, 4, 1),
               W_scalar=acc[:, :, 24:26].transpose(0, 2, 1),
               T_lt=acc[:, :, 26:34].reshape(2, nbin, 2, 4).transpose(0, 2, 3, 1),
               C=acc[:, :, 34])
    np.savez(a.out, **res)
    print(f'total pairs {res["C"].sum():.0f}, E1 {res["C"][0].sum():.0f}')


if __name__ == '__main__':
    main()
