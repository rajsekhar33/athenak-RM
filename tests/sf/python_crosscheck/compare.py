#!/usr/bin/env python3
"""Compare AthenaK on-the-fly sf output with athena_research structure functions.

Inputs (defaults under ./run):
  main/sf_{scalar,longtrans}/xchk.00000.sf     primary AthenaK estimate (8 replicates)
  main/... and seeds/s*/...  all *.sf files    ensemble of independent AthenaK estimates
  python/{mode}_exact.npz                      Python exhaustive (exact all-pairs) sums
  python/{mode}_sampled_s*.npz                 Python sampled estimator, several seeds

Quantities (w = weight, q = variable or L/T component, p = order, b = bin):
  S_p  = T/C (AthenaK)  vs  sf/(weight_sum_bin/2) (Python), unit weight only
  T    vs sf, W vs weight_sum_bin, C vs exact pair count (Python exhaustive raw count)
For each comparison z = (X_A - X_ref)/se, se from AthenaK replicates (primary) or from the
spread of independent outputs (ensemble), combined in quadrature with the Python seed
standard error when the reference is the sampled Python mean.

Usage: python3 compare.py [RUN_DIR] [--full]   (--full prints every per-bin row)
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sf_reference import read_sf  # noqa: E402

MODES = {'scalar': dict(qn=['vx', 'vy', 'vz'], wn=['ones', 'dens']),
         'longtrans': dict(qn=['L', 'T'], wn=['ones'])}


def athenak_primary(fname):
    """Replicate-average arrays and their replicate standard errors."""
    meta, a = read_sf(fname)
    R = int(meta['nreplicate'])
    Tt, Wt, Ct = a['T_er'].sum(0), a['W_er'].sum(0), a['C_er'].sum(0)
    assert np.allclose(Tt.mean(0), a['T'], rtol=1e-12, atol=0)
    St = Tt / Ct[:, :, None, None, :]

    def se(x):
        return x.std(0, ddof=1) / np.sqrt(R)

    return meta, a, dict(
        S=a['T'] / a['C'][:, None, None, :], T=a['T'], W=a['W'], C=a['C'],
        se_S=se(St), se_T=se(Tt), se_W=se(Wt), se_C=se(Ct), dof=R - 1)


def athenak_ensemble(files, pop=None):
    """Pool the replicates of independent outputs (each replicate is a complete,
    independent estimator). pop = None: E1 + E2; 0: E1 only; 1: E2 only.
    T, W, C: mean and se = sd/sqrt(n). S = mean(T)/mean(C) with the delta-method se
    sd(T_t - S C_t)/(mean(C) sqrt(n))."""
    T, W, C = [], [], []
    for f in files:
        _, a = read_sf(f)
        sl = slice(None) if pop is None else slice(pop, pop + 1)
        T.append(a['T_er'][sl].sum(0))
        W.append(a['W_er'][sl].sum(0))
        C.append(a['C_er'][sl].sum(0))
    T, W, C = np.concatenate(T), np.concatenate(W), np.concatenate(C)
    n = T.shape[0]
    out = dict(T=T.mean(0), W=W.mean(0), C=C.mean(0), dof=n - 1, n=n)
    for k, v in dict(T=T, W=W, C=C).items():
        out['se_' + k] = v.std(0, ddof=1) / np.sqrt(n)
    Cm = out['C'][:, None, None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        out['S'] = out['T'] / Cm
        out['se_S'] = ((T - out['S'][None] * C[:, :, None, None, :]).std(0, ddof=1)
                       / (Cm * np.sqrt(n)))
    return out


def lag_reference(path, mode, pop=None):
    """Exact sums from exact_lag.py, arranged like the AthenaK arrays."""
    e = np.load(path)
    sl = slice(None) if pop is None else slice(pop, pop + 1)
    C = e['C'][sl].sum(0)
    if mode == 'scalar':
        T, W, C = e['T_scalar'][sl].sum(0), e['W_scalar'][sl].sum(0), np.stack([C, C])
    else:
        T, C = e['T_lt'][sl].sum(0)[None], C[None]
        W = 2.0 * C
    with np.errstate(divide='ignore', invalid='ignore'):
        return dict(T=T, W=W, C=C, S=T / C[:, None, None, :])


def python_sets(pydir, mode):
    ex = np.load(os.path.join(pydir, f'{mode}_exact.npz'))
    exact = dict(T=ex['sf'], W=ex['weight_sum_bin'], C=ex['num_bin_points'].astype(float),
                 rbins=ex['rbins'])
    exact['S'] = ex['sf'] / (0.5 * ex['weight_sum_bin'])[:, None, None, :]
    files = sorted(glob.glob(os.path.join(pydir, f'{mode}_sampled_s*.npz')))
    samp = None
    if len(files) >= 2:
        T = np.array([np.load(f)['sf'] for f in files])
        W = np.array([np.load(f)['weight_sum_bin'] for f in files])
        S = T / (0.5 * W)[:, :, None, None, :]
        n = len(files)
        samp = dict(S=S.mean(0), T=T.mean(0), W=W.mean(0),
                    se_S=S.std(0, ddof=1) / np.sqrt(n),
                    se_T=T.std(0, ddof=1) / np.sqrt(n),
                    se_W=W.std(0, ddof=1) / np.sqrt(n),
                    n=n, rbins=np.load(files[0])['rbins'])
    return exact, samp


def zstats(z, dof):
    z = z.ravel()
    z = z[np.isfinite(z)]
    p_tail = (2 * stats.t.sf(np.abs(z).max(), dof) if dof < 200
              else 2 * stats.norm.sf(np.abs(z).max()))
    return dict(n=z.size, mean=z.mean(), rms=np.sqrt((z**2).mean()),
                maxabs=np.abs(z).max(),
                p_max=min(1.0, p_tail * z.size))   # Bonferroni p-value of the largest |z|


def fmt_row(label, z, rel, dof):
    s = zstats(z, dof)
    return (f'  {label:<34s} n={s["n"]:4d} mean z={s["mean"]:+6.2f} '
            f'rms z={s["rms"]:5.2f} max|z|={s["maxabs"]:5.2f} '
            f'(Bonf. p={s["p_max"]:.2f}) max|rel|={np.nanmax(np.abs(rel)):.2e}')


def compare_block(title, A, ref, keys, wn, qn, dof, ref_se=None, full=False):
    """Compare the AthenaK estimate A with reference ref for each key in keys."""
    print(title)
    rows = {}
    for key in keys:
        se = A['se_' + key]
        if ref_se is not None and ('se_' + key) in ref_se:
            se = np.sqrt(se**2 + ref_se['se_' + key]**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (A[key] - ref[key]) / se
            rel = A[key] / ref[key] - 1.0
        rows[key] = (z, rel)
        if key in ('S', 'T'):
            for iw, w in enumerate(wn):
                if key == 'S' and w != 'ones':
                    continue
                for iq, q in enumerate(qn):
                    for p in range(z.shape[2]):
                        print(fmt_row(f'{key} w={w} q={q} p={p+1}', z[iw, iq, p],
                                      rel[iw, iq, p], dof))
        else:
            for iw, w in enumerate(wn):
                print(fmt_row(f'{key} w={w}', z[iw], rel[iw], dof))
    allz = np.concatenate([rows[k][0][0 if k == 'S' else slice(None)].ravel()
                           for k in keys])
    s = zstats(allz, dof)
    print(f'  ALL: n={s["n"]} rms z={s["rms"]:.2f} max|z|={s["maxabs"]:.2f} '
          f'Bonferroni p(max|z|)={s["p_max"]:.3f}')
    if full:
        print_bins(rows, wn, qn)
    return rows


def print_bins(rows, wn, qn):
    z, rel = rows['S']
    print('  per-bin S_p (unit weight): z [rel]')
    for iq, q in enumerate(qn):
        print(f'   q={q}  ' + ' '.join(
            f'{"p=" + str(p + 1):>20s}' for p in range(z.shape[2])))
        for b in range(z.shape[3]):
            print(f'   b={b:2d}  ' + ' '.join(
                f'{z[0, iq, p, b]:+6.2f} [{rel[0, iq, p, b]:+.1e}]'
                for p in range(z.shape[2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        'run', nargs='?',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run'))
    ap.add_argument('--full', action='store_true')
    a = ap.parse_args()
    pydir = os.path.join(a.run, 'python')
    for mode, info in MODES.items():
        wn, qn = info['wn'], info['qn']
        prim = os.path.join(a.run, 'main', f'sf_{mode}', 'xchk.00000.sf')
        meta, raw, A = athenak_primary(prim)
        ens_files = sorted(
            glob.glob(os.path.join(a.run, 'main', f'sf_{mode}', '*.sf'))
            + glob.glob(os.path.join(a.run, 'seeds', 's*', f'sf_{mode}', '*.sf')))
        exact, samp = python_sets(pydir, mode)
        print(f'================ mode = {mode} ================')
        print(f'AthenaK: {prim}  (R={meta["nreplicate"]}, seed={meta.get("seed")}, '
              f'ndraw_block={meta.get("ndraw_block")}, '
              f'nsample_global={meta.get("nsample_global")}, '
              f'nanchor_global={meta.get("nanchor_global")})')
        d = np.abs(raw['rbins'] - exact['rbins'])
        print(f'rbins: max|AthenaK - Python| = {d.max():.3e}, identical bits: '
              f'{np.array_equal(raw["rbins"], exact["rbins"])}, '
              f'max rel = {(d / exact["rbins"]).max():.2e}')
        # can the ulp-level edge difference (or the different ways the two codes form the
        # separation) move a lattice separation across an edge? distance of every
        # minimum-image lattice separation from the nearest edge:
        nx = [int(v) for v in meta['mesh_nx'].split(',')]
        L = [float(v) for v in meta['domain_length'].split(',')]
        h = [np.arange(n // 2 + 1) * l / n for n, l in zip(nx, L)]
        r = np.unique(np.sqrt(h[0][:, None, None]**2 + h[1][None, :, None]**2 +
                              h[2][None, None, :]**2).ravel())[1:]
        gap = np.min(np.abs(r[:, None] - raw['rbins'][None, :]) / raw['rbins'][None, :])
        print(f'min relative distance of a lattice separation from an edge: {gap:.2e}')
        empty = exact['C'][0] == 0
        empty_ok = bool(np.all(raw["C"][:, empty] == 0)
                        and np.all(raw["W"][:, empty] == 0)
                        and np.all(raw["T"][..., empty] == 0))
        print(f'empty bins (no lattice separation): {np.flatnonzero(empty).tolist()}; '
              f'AthenaK C, W, T there all zero: {empty_ok}')
        tot = exact['C'][0].sum()
        print(f'Python exhaustive pair count = {tot:.0f} '
              f'(N(N-1)/2 = {262144 * 262143 // 2})')
        print(f'AthenaK sum_b C = {raw["C"][0].sum():.6e}  '
              f'(rel to exact {raw["C"][0].sum() / tot - 1:+.2e})')
        keys = ['S', 'T', 'W', 'C']
        compare_block('-- [1] AthenaK primary (8 replicates) vs Python exact '
                      '(z uses AthenaK replicate se, t-dist dof 7)',
                      A, exact, keys, wn, qn, A['dof'], full=a.full)
        lag64 = os.path.join(a.run, 'exact_lag_f64.npz')
        lag32 = os.path.join(a.run, 'exact_lag_f32.npz')
        if os.path.exists(lag32):
            X = lag_reference(lag32, mode)
            half = exact['rbins'][1:] > 0.5 * min(L)   # bins that can hold half-box lags
            msg = []
            for k in ('T', 'W', 'C'):
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = np.abs(exact[k] / X[k] - 1.0)
                msg.append(f'{k}: {np.nanmax(r[..., ~half]):.1e} '
                           f'[half-box bins {np.nanmax(r[..., half]):.1e}]')
            print('-- [0] Python exact vs exact_lag.py (float32 fields), max rel diff: '
                  + ', '.join(msg))
        if os.path.exists(lag64) and os.path.exists(lag32):
            X64 = lag_reference(lag64, mode)
            r = np.nanmax(np.abs(X64['T'] / X['T'] - 1.0))
            print('   float64 vs float32 fields (exact_lag.py), '
                  f'max rel diff of T: {r:.1e}')
        if len(ens_files) > 2:
            E = athenak_ensemble(ens_files)
            compare_block(
                f'-- [2] AthenaK pooled replicates of {len(ens_files)} '
                'independent outputs '
                f'(n={E["n"]}) vs Python exact', E, exact, keys, wn, qn, E['dof'],
                full=a.full)
            if os.path.exists(lag64):
                for pop in (0, 1):
                    Ep = athenak_ensemble(ens_files, pop)
                    compare_block(
                        f'-- [2{"ab"[pop]}] population E{pop + 1} only: pooled AthenaK '
                        'replicates vs exact_lag.py (float64 fields)', Ep,
                        lag_reference(lag64, mode, pop), keys, wn, qn, Ep['dof'],
                        full=a.full)
        if samp is not None:
            P = dict(S=samp['S'], T=samp['T'], W=samp['W'], se_S=samp['se_S'],
                     se_T=samp['se_T'], se_W=samp['se_W'])
            compare_block(f'-- [3] Python sampled ({samp["n"]} seeds) '
                          'vs Python exact (seed se)',
                          P, exact, ['S', 'T', 'W'], wn, qn, samp['n'] - 1)
            compare_block('-- [4] AthenaK primary vs Python sampled mean (combined se)',
                          A, samp, ['S', 'T', 'W'], wn, qn, A['dof'],
                          ref_se=samp, full=a.full)
            # relative precision of each estimator, unit-weight S_p
            print('-- relative standard errors of S_p (unit weight), '
                  'median [max] over q,p,b:')
            for lab, se, val in (('AthenaK primary', A['se_S'], A['S']),
                                 ('Python sampled mean', samp['se_S'], samp['S'])):
                r = (se / val)[0]
                print(f'  {lab:<22s} {np.nanmedian(r):.2e} [{np.nanmax(r):.2e}]')
        print()


if __name__ == '__main__':
    main()
