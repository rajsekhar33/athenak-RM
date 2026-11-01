#!/usr/bin/env python3
"""Acceptance test of the selection cross-check (README.md in this directory).

Inputs: the AthenaK run directory (sf_scalar/, sf_longtrans/ with sel.NNNNN.sf) and the
Python result (python_sel.py --out). The AthenaK file compared is the first sf output
whose header cycle equals --cycle (default 600 = nlim of selection.athinput); its time
must equal the snapshot time recorded by the Python step.

Exact checks (no sampling involved), each must hold:
  X1 select1 in the header is hydro_w_d with the same (lo, hi) the Python mask used
  X2 snapshot and sf output are the same state: header time == Python snapshot time
  X3 selected fraction ncell_selected/ncell_region in [0.10, 0.50]
  X4 selected sets agree: 0 <= nsel_Python - ncell_selected <= n_on_edge. Both band
     edges are float32 numbers and rounding is monotone, so a cell AthenaK selects (on
     its double density) is always in the Python mask (on the float32 density); only a
     cell whose float32 density equals an edge exactly (n_on_edge, counted by the
     Python step) can be in the Python mask but not in AthenaK's selection.
     Resolution: diff 0 -> the sets are identical; diff == n_on_edge -> AthenaK excluded
     every on-edge cell, and the Python reference is corrected exactly by subtracting
     those cells' pairs (edge_* arrays of python_sel.py, validated against the Python
     exhaustive kernels to 5e-14); any other diff (only possible with >= 2 on-edge
     cells) is not resolved and X4b bounds its effect instead.
  X4b (unresolved case only) the unresolved cells change C_b by at most diff x (lattice
     partners of one cell in bin b) pairs, which must be < 0.2 se_C in every bin
  X5 Python sums exhaustive: raw pair count == Nsel(Nsel-1)/2 (each mode)
  X6 bin edges: identical bits, or max relative difference <= 1e-14 and every
     minimum-image lattice separation farther than 1e-12 (relative) from every edge
     (the outer edges are widened by 1e-9 in both codes, so the gap is ~1e-9)
  X7 bins Python finds empty are exactly 0 in AthenaK T, W, C; all AthenaK arrays finite
  X8 unit mask weight: W == 2 C in AthenaK (each replicate, rel 1e-12) and in Python

Statistical test (predefined): for every mode, variable q, order p and bin b populated
in both codes (Python C > 0 and AthenaK C > 0),
  S_p = T/C : z = (S_A - S_P) / se,
    se = sd_r(T_r - S_A C_r) / (C_A sqrt(R))  (delta method)
  T_p       : z = (T_A - T_P) / (sd_r(T_r)/sqrt(R))
and per bin C: z = (C_A - C_P) / (sd_r(C_r)/sqrt(R)), with T_r, C_r the per-replicate
E1 + E2 estimates (R = nreplicate; the stored arrays are their means). With n tests in
total, PASS iff max|z| < t_{R-1}^{-1}(1 - alpha/(2n)), alpha = 1e-3 (two-sided
Bonferroni, family-wise false-alarm rate <= 1e-3 under any correlation between tests).
A test whose se is 0 must agree to 1e-12 relative instead.
Python's S_p is sf/(W/2) = sf/C, since W = 2C for a 0/1 weight on selected pairs.

The float32 snapshot rounds the velocities too (about 3e-8 relative), which moves the
exact sums by ~1e-8 relative, far below the sampling errors (~1e-3).

Usage: python3 compare.py ATHENA_RUN_DIR PYTHON.npz [--cycle 600] [--alpha 1e-3]
Last line: SELECTION CROSSCHECK: PASS | FAIL
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from sf_reference import read_sf  # noqa: E402

MODES = {'scalar': ['vx', 'vy', 'vz'], 'longtrans': ['L', 'T']}


def find_file(run, mode, cycle):
    files = sorted(glob.glob(os.path.join(run, f'sf_{mode}', '*.sf')))
    hits = [f for f in files if read_sf(f)[0].get('cycle') == str(cycle)]
    if not hits:
        sys.exit(f'no sf_{mode} file with cycle={cycle} in {run}')
    return hits


def lattice(meta, rbins):
    """Minimum-image lattice separations of the periodic mesh: their smallest relative
    distance from any edge, and per bin the number of partners one cell has in that bin
    (the most pairs that including or excluding one cell can change there)."""
    nx = [int(v) for v in meta['mesh_nx'].split(',')]
    L = [float(v) for v in meta['domain_length'].split(',')]
    h = [np.minimum(np.arange(n), n - np.arange(n)) * l / n for n, l in zip(nx, L)]
    r = np.sqrt(h[0][:, None, None]**2 + h[1][None, :, None]**2 +
                h[2][None, None, :]**2).ravel()[1:]
    ru = np.unique(r)
    gap = np.min(np.abs(ru[:, None] - rbins[None, :]) / rbins[None, :])
    shell = np.histogram(r, bins=rbins)[0].astype(float)
    return gap, shell


def ztests(a, TP, CP):
    """z arrays (S, T over [q, p, b]; C over [b]) against the Python reference sums
    TP [q, p, b], CP [b], and the expected mask. The tests are fixed by the exact
    reference (CP > 0), not by what AthenaK populated: a bin AthenaK left empty gives a
    nonfinite z there, which the inventory check counts as missing."""
    R = int(a['T_er'].shape[1])
    Tr = a['T_er'].sum(0)[:, 0]            # [R, q, p, b]
    Cr = a['C_er'].sum(0)[:, 0]            # [R, b]
    TA, CA = a['T'][0], a['C'][0]
    pop = CP > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        SA = TA / CA[None, None, :]
        SP = TP / CP[None, None, :]
        seS = (Tr - SA[None] * Cr[:, None, None, :]).std(0, ddof=1) / (CA * np.sqrt(R))
        seT = Tr.std(0, ddof=1) / np.sqrt(R)
        seC = Cr.std(0, ddof=1) / np.sqrt(R)
    out = {}
    for k, A, Pv, se in (('S', SA, SP, seS), ('T', TA, TP, seT), ('C', CA, CP, seC)):
        m = np.broadcast_to(pop, A.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (A - Pv) / se
            rel = A / Pv - 1.0
        out[k] = dict(z=np.where(m & (se > 0), z, np.nan), rel=np.where(m, rel, np.nan),
                      zero_se=m & (se == 0))
    return out, R, pop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('python_npz')
    ap.add_argument('--cycle', type=int, default=600)
    ap.add_argument('--alpha', type=float, default=1e-3)
    ap.add_argument('--full', action='store_true', help='print every per-bin z')
    a = ap.parse_args()
    P = np.load(a.python_npz)
    checks = []

    def check(name, ok, msg):
        checks.append((name, bool(ok)))
        print(f'  {name} {"ok  " if ok else "FAIL"} {msg}')

    print(f'Python: {a.python_npz} (athena_research {P["pyrepo_commit"]}), snapshot '
          f'{P["snapshot"]}, time={float(P["time"])!r}')
    print(f'  band [{float(P["lo"])!r}, {float(P["hi"])!r}] inclusive, mask count '
          f'{int(P["nsel"])} of {int(P["ncell"])}, on-edge cells {int(P["n_on_edge"])}')
    allz, dofs, rows, ninv = [], set(), [], 0
    for mode, qn in MODES.items():
        files = find_file(a.run, mode, a.cycle)
        meta, arr = read_sf(files[0])
        print(f'== {mode}: {files[0]} (R={meta["nreplicate"]}, '
              f'ranks={meta.get("nranks")}, '
              f'sparse={meta.get("sparse")}, time_e1={meta.get("time_e1")}, '
              f'time_e2={meta.get("time_e2")})')
        sel = meta.get('select1', '').split(',')
        check('X1', meta.get('nselect') == '1' and len(sel) == 3 and sel[0] == 'hydro_w_d'
              and float(sel[1]) == float(P['lo']) and float(sel[2]) == float(P['hi']),
              f'select1={meta.get("select1")}')
        tA, tP = float(meta['time']), float(P['time'])
        check('X2', abs(tA - tP) <= 1e-12 * max(1.0, abs(tP)),
              f'sf time={tA!r} cycle={meta["cycle"]}, snapshot time={tP!r}')
        nselA, nreg = int(meta['ncell_selected']), int(meta['ncell_region'])
        frac = nselA / nreg
        check('X3', 0.10 <= frac <= 0.50,
              f'selected fraction {nselA}/{nreg} = {frac:.4f}')
        d, ne = int(P['nsel']) - nselA, int(P['n_on_edge'])
        TP, CP = P[f'{mode}_sf'][0].copy(), P[f'{mode}_C'][0].copy()
        if d == 0:
            how = 'identical sets'
        elif d == ne:
            TP -= P[f'edge_{mode}_sf'][0]
            CP -= P['edge_C'][0]
            how = (f'AthenaK excluded all {ne} on-edge cell(s): their '
                   f'{int(P["edge_C"].sum())} pairs subtracted from the Python sums')
        else:
            how = 'partial, not resolved (bounded by X4b)'
        check('X4', 0 <= d <= ne, f'ncell_selected={nselA}, Python mask count='
              f'{int(P["nsel"])}, diff={d}, on-edge={ne}: {how}')
        n = int(P['nsel'])
        raw = int(P[f'{mode}_C'].sum())
        check('X5', raw == n * (n - 1) // 2,
              f'Python raw pairs {raw} vs Nsel(Nsel-1)/2 = '
              f'{n * (n - 1) // 2}')
        rbA, rbP = arr['rbins'], P['rbins']
        same = np.array_equal(rbA, rbP)
        mrel = float(np.max(np.abs(rbA - rbP) / rbP))
        gap, shell = lattice(meta, rbA)
        check('X6', same or (mrel <= 1e-14 and gap > 1e-12),
              f'rbins identical bits={same}, max rel diff={mrel:.1e}, '
              f'lattice gap={gap:.1e}')
        empty = CP == 0
        finite = all(np.all(np.isfinite(arr[k]))
                     for k in ('T', 'W', 'C', 'T_er', 'W_er', 'C_er'))
        zero_ok = (np.all(arr['C'][:, empty] == 0) and np.all(arr['W'][:, empty] == 0)
                   and np.all(arr['T'][..., empty] == 0))
        check('X7', finite and zero_ok,
              f'empty bins {np.flatnonzero(empty).tolist()} zero in '
              f'AthenaK: {zero_ok}; all finite: {finite}')
        WrA, CrA = arr['W_er'], arr['C_er']
        okA = np.allclose(WrA, 2 * CrA, rtol=1e-12, atol=0)
        okP = np.allclose(P[f'{mode}_W'], 2 * P[f'{mode}_C'], rtol=1e-14, atol=0)
        check('X8', okA and okP, f'W == 2C: AthenaK {okA}, Python {okP}')

        Z, R, pop = ztests(arr, TP, CP)
        dofs.add(R - 1)
        if d not in (0, ne):
            seC = arr['C_er'].sum(0)[:, 0].std(0, ddof=1) / np.sqrt(R)
            with np.errstate(divide='ignore', invalid='ignore'):
                eff = np.where(pop & (seC > 0), d * shell / seC, 0.0)
            check('X4b', np.all(eff < 0.2),
                  f'unresolved selection difference moves C by at '
                  f'most {eff.max():.3f} se')
        print(f'  populated bins: {int(pop.sum())} of {pop.size}')
        for k in ('S', 'T'):
            for iq, q in enumerate(qn):
                for p in range(Z[k]['z'].shape[1]):
                    z = Z[k]['z'][iq, p]
                    rel = Z[k]['rel'][iq, p]
                    print(f'  {k}_{p + 1} {q:<2s} n={np.isfinite(z).sum():3d} '
                          f'rms z={np.sqrt(np.nanmean(z**2)):5.2f} '
                          f'max|z|={np.nanmax(np.abs(z)):5.2f} '
                          f'max|rel|={np.nanmax(np.abs(rel)):.2e} '
                          f'median|rel|={np.nanmedian(np.abs(rel)):.2e}')
                    if a.full:
                        print('     z: ' + ' '.join(f'{v:+.2f}' for v in z))
        zc = Z['C']['z']
        print(f'  C      n={np.isfinite(zc).sum():3d} '
              f'rms z={np.sqrt(np.nanmean(zc**2)):5.2f} '
              f'max|z|={np.nanmax(np.abs(zc)):5.2f} '
              f'max|rel|={np.nanmax(np.abs(Z["C"]["rel"])):.2e}')
        for k in ('S', 'T', 'C'):
            zs = Z[k]['zero_se']
            if np.any(zs):
                rel = np.abs(Z[k]['rel'][zs])
                check('X9', np.all(rel <= 1e-12),
                      f'{k}: {int(zs.sum())} tests with zero replicate spread, '
                      f'max|rel| {rel.max():.1e}')
            z = Z[k]['z']
            idx = np.argwhere(np.isfinite(z))
            for ix in idx:
                rows.append((mode, k, tuple(int(i) for i in ix), float(z[tuple(ix)])))
        for k in ('S', 'T', 'C'):
            allz.append(Z[k]['z'][np.isfinite(Z[k]['z'])].ravel())
        # coverage and the frozen inventory: every bin with reference pairs must hold
        # AthenaK pairs, and every expected statistic must be a finite z or a zero-spread
        # equality (X9); none may drop out of the test count
        miss = np.flatnonzero(pop & (arr['C'][0] <= 0)).tolist()
        check('X10', not miss, f'bins with reference pairs and no AthenaK pairs: {miss}')
        nexp = nhave = 0
        for k in ('S', 'T', 'C'):
            m = np.broadcast_to(pop, Z[k]['z'].shape)
            nexp += int(m.sum())
            nhave += int(np.isfinite(Z[k]['z']).sum() + Z[k]['zero_se'].sum())
        ninv += nexp
        check('X11', nhave == nexp, f'statistics: {nhave} present of {nexp} expected')
        for extra in files[1:]:   # duplicate output of the same state (informational)
            m2, a2 = read_sf(extra)
            Z2, _, _ = ztests(a2, TP, CP)
            z2 = np.concatenate([Z2[k]['z'][np.isfinite(Z2[k]['z'])]
                                 for k in ('S', 'T', 'C')])
            print(f'  (info) independent duplicate output {os.path.basename(extra)}: '
                  f'n={z2.size} '
                  f'rms z={np.sqrt(np.mean(z2**2)):.2f} max|z|={np.abs(z2).max():.2f}')
    allz = np.concatenate(allz)
    ntest = ninv          # the frozen inventory, not the statistics that survived
    dof = min(dofs)
    thr = stats.t.isf(a.alpha / (2 * ntest), dof)
    zmax = float(np.abs(allz).max())
    worst = max(rows, key=lambda r: abs(r[3]))
    pmax = min(1.0, ntest * 2 * stats.t.sf(zmax, dof))
    print('== statistical test')
    print(f'  tests n={ntest} (finite z {allz.size}), dof={dof}, '
          f'rms z={np.sqrt(np.mean(allz**2)):.3f}, '
          f'mean z={allz.mean():+.3f}, max|z|={zmax:.3f} '
          f'at {worst[0]} {worst[1]} idx={worst[2]}')
    print(f'  Bonferroni threshold (alpha={a.alpha:g}): |z| < {thr:.3f}; '
          f'Bonferroni p of max|z| = {pmax:.3g}')
    check('Z', zmax < thr, f'max|z| {zmax:.3f} < {thr:.3f}')
    ok = all(c[1] for c in checks)
    print(f'SELECTION CROSSCHECK: {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
