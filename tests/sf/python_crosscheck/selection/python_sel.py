#!/usr/bin/env python3
"""Python side of the selection cross-check: exact all-pairs structure functions of an
AthenaK .bin snapshot with a 0/1 density-band mask weight (read-only use of the
athena_research repo).

The band is installed exactly like the production masks (plot_rdshk_mach_transverse_
interface._install_interface_mask, all_radshock_analysis.install_interface_mask):
    ad.data_func['sel_bool'] = lambda ad, mbl, mbh:
        where(lo <= dens <= hi, 1, 0)                 (inclusive, as AthenaK select1_*)
and passed as weights='sel_bool' with sparse_data=True. With npairs = (cells per block x
n_mbs)^2, n_samples1 equals the block size, every block pair takes the exhaustive kernel
(use_exhaustive=True for get_sf_mb; get_sf_helmholtz_mb switches by itself) and the sums
are the exact all-pairs sums over selected cells. This is checked: the raw pair count
must equal Nsel(Nsel-1)/2, else the script exits non-zero.

Writes OUT.npz with, per mode (scalar, longtrans):
  {mode}_sf [1, nq, max_order, nbin], {mode}_W [1, nbin], {mode}_C [1, nbin] (raw count)
and rbins, lo, hi, nsel (mask count), n_on_edge (cells whose float32 density equals an
edge exactly: their AthenaK double value may lie on the other side), time, cycle, and
the exact contribution of the on-edge cells' pairs (edge_C, edge_{mode}_sf), which
compare.py subtracts when AthenaK's ncell_selected shows it excluded them.

Usage: python3 python_sel.py SNAPSHOT.bin --out OUT.npz
       [--lo 0.96875 --hi 0.9990234375 --nbins 24 --max_order 3 --pyrepo DIR]
"""
import argparse
import subprocess
import sys
import time

import numpy as np

PYREPO = '/home/rm3395/analysis_RadShock/athena-research-rm'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('snapshot')
    ap.add_argument('--out', required=True)
    ap.add_argument('--lo', type=float, default=0.96875)
    ap.add_argument('--hi', type=float, default=0.9990234375)
    ap.add_argument('--nbins', type=int, default=24)
    ap.add_argument('--max_order', type=int, default=3)
    ap.add_argument('--pyrepo', default=PYREPO)
    a = ap.parse_args()

    sys.path.insert(0, a.pyrepo)
    from athena_research.core.athena_data import AthenaData
    from athena_research.core.base import xp, asnumpy, cupy_enabled
    import athena_research.operations.structure_functions as S
    if not cupy_enabled:
        print('WARNING: CuPy not enabled; the exhaustive sums run on the CPU (slow)')
    try:
        commit = subprocess.run(['git', '-C', a.pyrepo, 'rev-parse', 'HEAD'],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = 'unknown'

    ad = AthenaData()
    ad.load(a.snapshot)
    lo, hi = a.lo, a.hi

    def sel_bool(self, mbl=None, mbh=None, _lo=lo, _hi=hi):
        dens = self.data('dens', mbl, mbh)
        return xp.where(xp.logical_and(dens >= _lo, dens <= _hi),
                        xp.ones_like(dens), xp.zeros_like(dens))
    ad.data_func['sel_bool'] = sel_bool

    dens = asnumpy(ad.data('dens'))
    nsel = int(np.count_nonzero((dens >= lo) & (dens <= hi)))
    n_on_edge = int(np.count_nonzero((dens == lo) | (dens == hi)))
    ncell = dens.size
    print(f'snapshot {a.snapshot}: time={ad.time} cells={ncell} dtype={dens.dtype} '
          f'selected={nsel} ({nsel / ncell:.4f}) on-edge={n_on_edge}')

    npb = ad.nx1 * ad.nx2 * ad.nx3
    npairs = float(npb * ad.n_mbs) ** 2          # -> n_samples1 = npb (exhaustive)
    expect = nsel * (nsel - 1) // 2
    out = dict(lo=lo, hi=hi, nsel=nsel, n_on_edge=n_on_edge, ncell=ncell,
               time=float(ad.time), cycle=int(getattr(ad, 'cycle', -1)),
               pyrepo_commit=commit, snapshot=a.snapshot)
    status = 0
    for mode in ('scalar', 'longtrans'):
        t0 = time.time()
        if mode == 'scalar':
            rb, sf, ws, nb = S.get_sf_mb(
                ad, ['velx', 'vely', 'velz'], weights=['sel_bool'],
                max_order=a.max_order, npairs=npairs, nbins=a.nbins,
                log_bin_flag=True, sparse_data=True, use_exhaustive=True)
            sf = asnumpy(sf)                                   # [w, q, p, bin]
        else:
            rb, sfc, sfs, ws, nb = S.get_sf_helmholtz_mb(
                ad, 'vel', weights=['sel_bool'], max_order=a.max_order,
                npairs=npairs, nbins=a.nbins, log_bin_flag=True, sparse_data=True)
            sf = np.stack([asnumpy(sfc), asnumpy(sfs)], axis=1)  # [w, (L,T), p, bin]
        if cupy_enabled:
            xp.cuda.Stream.null.synchronize()
        ws, nb = asnumpy(ws).reshape(1, -1), asnumpy(nb).reshape(1, -1)
        sf = sf.reshape((1,) + sf.shape[-3:])
        raw = int(nb.sum())
        ok = raw == expect
        status |= (not ok)
        print(f'{mode}: {time.time() - t0:.1f} s, raw pairs {raw}, '
              f'Nsel(Nsel-1)/2 = {expect}: '
              f'{"exhaustive" if ok else "NOT EXHAUSTIVE (sums are sampled, not exact)"}')
        out[f'{mode}_sf'] = sf
        out[f'{mode}_W'] = ws
        out[f'{mode}_C'] = nb.astype(float)
        out[f'{mode}_seconds'] = time.time() - t0
    # same edges as get_sf_mb / get_sf_helmholtz_mb (rmin = min dx, rmax = half diagonal)
    g = ad.mb_geometry[0, :]
    dx = [(g[1] - g[0]) / ad.nx1, (g[3] - g[2]) / ad.nx2, (g[5] - g[4]) / ad.nx3]
    L = [ad.x1max - ad.x1min, ad.x2max - ad.x2min, ad.x3max - ad.x3min]
    rbins, _ = S.radial_bin_edges(
        np.min(dx), np.sqrt(0.25 * sum(length * length for length in L)), a.nbins, True)
    out['rbins'] = asnumpy(rbins)
    sel = (dens >= lo) & (dens <= hi)
    out.update(edge_contributions(ad, asnumpy, sel, sel & ((dens == lo) | (dens == hi)),
                                  out['rbins'], L, a.max_order))
    np.savez(a.out, **out)
    print(f'wrote {a.out} (athena_research {commit})')
    sys.exit(status)


def edge_contributions(ad, asnumpy, sel, edge, rbins, L, max_order):
    """Exact contribution of the on-edge cells (float32 density == lo or hi) to every
    sum: all pairs with at least one on-edge endpoint, both endpoints selected, with the
    formulas of the Python exhaustive kernels (scalar |dq|^p; long_trans
    |dv.r/|r||^p and |dv - (dv.r)r/|r|^2|^p, minimum-image displacement). compare.py
    subtracts them when AthenaK's ncell_selected shows it excluded these cells.
    sel, edge: boolean masks shaped like ad.data("dens"), edge a subset of sel."""
    nb, P = rbins.size - 1, max_order
    res = dict(edge_scalar_sf=np.zeros((1, 3, P, nb)),
               edge_longtrans_sf=np.zeros((1, 2, P, nb)),
               edge_C=np.zeros((1, nb)))
    if not edge.any():
        return res
    X = [asnumpy(ad.data(c)).ravel() for c in ('x', 'y', 'z')]
    V = np.stack([asnumpy(ad.data(c)).ravel() for c in ('velx', 'vely', 'velz')], axis=1)
    e_idx = np.flatnonzero(edge.ravel())
    other = sel.ravel() & ~edge.ravel()
    done = np.zeros(sel.size, dtype=bool)
    for i in e_idx:
        partners = np.flatnonzero(other | (edge.ravel() & done))   # each pair once
        done[i] = True
        d = np.stack([X[k][partners] - X[k][i] for k in range(3)], axis=1)
        for k in range(3):
            dk = d[:, k]
            wrap = np.abs(dk) > 0.5 * L[k]
            d[wrap, k] -= np.sign(dk[wrap]) * L[k]
        r = np.sqrt((d * d).sum(1))
        b = np.searchsorted(rbins, r, side='right') - 1
        ok = (b >= 0) & (b < nb)
        dv = V[partners] - V[i]
        b, d, r, dv = b[ok], d[ok], r[ok], dv[ok]
        res['edge_C'][0] += np.bincount(b, minlength=nb)
        comp = (d * dv).sum(1) / r
        sol = np.sqrt(np.maximum(0.0, (dv * dv).sum(1) - comp * comp))
        for p in range(P):
            for q in range(3):
                res['edge_scalar_sf'][0, q, p] += np.bincount(
                    b, np.abs(dv[:, q])**(p + 1), nb)
            res['edge_longtrans_sf'][0, 0, p] += np.bincount(b, np.abs(comp)**(p + 1), nb)
            res['edge_longtrans_sf'][0, 1, p] += np.bincount(b, sol**(p + 1), nb)
    print(f'on-edge cells: {e_idx.size}, their pairs: {int(res["edge_C"].sum())}')
    return res


if __name__ == '__main__':
    main()
