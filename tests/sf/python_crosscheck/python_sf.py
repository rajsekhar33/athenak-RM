#!/usr/bin/env python3
"""Run athena_research structure functions on an AthenaK .bin snapshot (read-only use).

Writes an .npz with rbins, sf / sf_comp / sf_sol, weight_sum_bin and num_bin_points.

  --kind exact    every block pair is enumerated exhaustively (exact all-pairs sums):
                  scalar via use_exhaustive=True, long_trans via the automatic exhaustive
                  switch of get_sf_helmholtz_mb (n_samples1 = cells per block).
  --kind sampled  the normal random-pair estimator; --seed != 0 changes the per-block-pair
                  base seed by XOR-ing a constant into structure_functions._pair_seed
                  (monkeypatch at run time, the Python repo is not edited).

  --rst FILE      replace the snapshot's float32 fields by the double-precision state of
                  this restart file (same cycle and MeshBlocks; isothermal hydro: dens =
                  u0, vel = mom * (1/dens) as AthenaK's ConsToPrim); the .bin then only
                  supplies the metadata athena_research reads.

Usage: python3 python_sf.py SNAPSHOT.bin --mode scalar|longtrans --kind exact|sampled
       [--seed S] [--ns1 N] [--vars velx,vely,velz] [--rst FILE] --out FILE.npz
"""
import argparse
import sys
import time

import numpy as np

PYREPO = '/home/rm3395/analysis_RadShock/athena-research-rm'


def replace_from_rst(ad, path):
    """Overwrite dens/velx/vely/velz in ad.data_raw with the restart file's doubles."""
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
    from rst_compare import parse
    meta, arr = parse(path)
    ng, nx1, nx2, nx3 = meta['mb_indcs'][2][:4]
    if 'cycle' in ad.binary and int(ad.binary['cycle']) != meta['ncycle'][2][0]:
        raise ValueError('restart cycle %d != snapshot cycle %s'
                         % (meta['ncycle'][2][0], ad.binary['cycle']))
    n1, n2, n3 = nx1 + 2 * ng, nx2 + 2 * ng, nx3 + 2 * ng
    act = (slice(None), slice(ng, ng + nx3), slice(ng, ng + nx2), slice(ng, ng + nx1))
    u = [arr['hydro_u%d' % v].reshape(-1, n3, n2, n1)[act] for v in range(4)]
    d = u[0]
    di = 1.0 / d
    new = {'dens': d, 'velx': u[1] * di, 'vely': u[2] * di, 'velz': u[3] * di}
    for k, v in new.items():
        if ad.data_raw[k].shape != v.shape:
            raise ValueError('%s: restart shape %s != snapshot %s'
                             % (k, v.shape, ad.data_raw[k].shape))
        # float32 snapshot values must be the rounded restart values (same state)
        if not np.array_equal(ad.data_raw[k].astype(np.float32), v.astype(np.float32)):
            raise ValueError('%s: snapshot is not the rounded restart state' % k)
        ad.data_raw[k] = np.ascontiguousarray(v, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('snapshot')
    ap.add_argument('--mode', choices=['scalar', 'longtrans'], required=True)
    ap.add_argument('--kind', choices=['exact', 'sampled'], required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ns1', type=int, default=1024, help='target n_samples1 per block')
    ap.add_argument('--nbins', type=int, default=24)
    ap.add_argument('--max_order', type=int, default=4)
    ap.add_argument('--vars', default='velx,vely,velz', help='scalar-mode variables')
    ap.add_argument('--rst', default=None)
    ap.add_argument('--pyrepo', default=PYREPO)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.pyrepo)
    from athena_research.core.athena_data import AthenaData
    from athena_research.core.base import xp, asnumpy
    import athena_research.operations.structure_functions as S

    if a.seed != 0:
        orig = S._pair_seed
        salt = (int(S._splitmix64(np.uint64(a.seed))) if hasattr(S, '_splitmix64')
                else a.seed)
        salt &= 0xFFFFFFFFFFFFFFFF
        S._pair_seed = lambda b1, b2: np.uint64(int(orig(b1, b2)) ^ salt)

    ad = AthenaData()
    ad.load(a.snapshot)
    if a.rst:
        replace_from_rst(ad, a.rst)
    npb = ad.nx1 * ad.nx2 * ad.nx3
    ns1 = npb if a.kind == 'exact' else a.ns1
    npairs = float(ns1 * ad.n_mbs) ** 2   # get_sf*_mb: n_samples1 = sqrt(npairs)/n_mbs
    # same rmin/rmax/edges as get_sf_mb / get_sf_helmholtz_mb
    Lx, Ly, Lz = ad.x1max - ad.x1min, ad.x2max - ad.x2min, ad.x3max - ad.x3min
    g = ad.mb_geometry[0, :]
    dx = [(g[1] - g[0]) / ad.nx1, (g[3] - g[2]) / ad.nx2, (g[5] - g[4]) / ad.nx3]
    rmax = np.sqrt(0.25 * (Lx**2 + Ly**2 + Lz**2))
    rbins, _ = S.radial_bin_edges(np.min(dx), rmax, a.nbins, True)

    t0 = time.time()
    out = dict(rbins=asnumpy(rbins), npairs=npairs, seed=a.seed, kind=a.kind, mode=a.mode)
    if a.mode == 'scalar':
        r, sf, ws, nb = S.get_sf_mb(ad, a.vars.split(','), weights=['ones', 'dens'],
                                    max_order=a.max_order, npairs=npairs, nbins=a.nbins,
                                    log_bin_flag=True, use_exhaustive=(a.kind == 'exact'))
        out.update(sf=asnumpy(sf))                      # [w, var, p, bin]
    else:
        r, sfc, sfs, ws, nb = S.get_sf_helmholtz_mb(ad, 'vel', weights=['ones'],
                                                    max_order=a.max_order, npairs=npairs,
                                                    nbins=a.nbins, log_bin_flag=True)
        # [w, (L,T), p, bin]
        out.update(sf=np.stack([asnumpy(sfc), asnumpy(sfs)], axis=1))
    xp.cuda.Stream.null.synchronize()
    out.update(weight_sum_bin=asnumpy(ws), num_bin_points=asnumpy(nb),
               seconds=time.time() - t0)
    np.savez(a.out, **out)
    print(f'{a.mode} {a.kind} seed={a.seed} ns1={ns1}: {out["seconds"]:.1f} s, '
          f'raw pairs {int(out["num_bin_points"].sum())}')


if __name__ == '__main__':
    main()
