#!/usr/bin/env python3
"""Fault injection for compare.py: each damaged copy of a passing run must FAIL.

usage: selftest.py OUTDIR NEWDIR
  OUTDIR  a run that passed (OUTDIR/athena/sf_*, OUTDIR/<analysis>/python_exact.npz);
  NEWDIR  must not exist; damaged copies and their compare outputs go there.
Cases: the unmodified copy must PASS; one reference-populated bin emptied in AthenaK
(sums, counts and replicates, both modes); one NaN in a replicate array; one
populated bin's counts doubled (a real disagreement). Last line:
SELECTION SELFTEST: PASS or FAIL.
"""
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'frontier'))
import collect  # noqa: E402


def edit(path, fn):
    """Apply fn(arrays) to one .sf file in place (arrays writable, layout unchanged)."""
    h = collect.header(path)
    off = int(h['header offset'])
    raw = bytearray(open(path, 'rb').read())
    pos, arrs = off, {}
    for name, shape, dt in collect.layout(h, '.sf'):
        n = int(np.prod(shape))
        arrs[name] = np.frombuffer(raw, dtype=dt, count=n, offset=pos).reshape(shape)
        pos += 8 * n
    fn(arrs, h)
    open(path, 'wb').write(bytes(raw))


def main():
    src, new = sys.argv[1], sys.argv[2]
    if os.path.exists(new):
        print('SELECTION SELFTEST: FAIL (%s exists)' % new)
        return 1
    npz = sorted(glob.glob(os.path.join(src, '*', 'python_exact.npz')) +
                 glob.glob(os.path.join(src, 'python_exact.npz')))[0]
    P = np.load(npz)
    b = int(np.flatnonzero(P['scalar_C'][0] > 0)[len(P['scalar_C'][0]) // 2 - 1])

    def empty_bin(a, h):
        for k in ('T', 'W', 'C', 'T_er', 'W_er', 'C_er', 'n_acc_er'):
            a[k][..., b] = 0

    def nan_rep(a, h):
        a['T_er'][1, 0, 0, 0, 0, b] = np.nan

    def double_c(a, h):
        a['C'][..., b] *= 2

    cases = [('unmodified', None, 0), ('empty_bin', empty_bin, 1),
             ('nan_replicate', nan_rep, 1),
             ('double_count', double_c, 1)]
    nfail = 0
    for name, fn, want in cases:
        d = os.path.join(new, name)
        shutil.copytree(os.path.join(src, 'athena'), os.path.join(d, 'athena'),
                        ignore=shutil.ignore_patterns('bin', '*.bin'))
        if fn:
            for f in glob.glob(os.path.join(d, 'athena', 'sf_*', '*.00001.sf')):
                edit(f, fn)
        with open(os.path.join(d, 'compare.txt'), 'w') as out:
            rc = subprocess.call([sys.executable, '-I', os.path.join(HERE, 'compare.py'),
                                  os.path.join(d, 'athena'), npz], stdout=out,
                                 stderr=subprocess.STDOUT)
        last = open(os.path.join(d, 'compare.txt')).read().strip().splitlines()[-1]
        fails = [ln.split()[0] for ln in open(os.path.join(d, 'compare.txt'))
                 if ln.strip() and 'FAIL' in ln.split()[1:2]]
        ok = (rc == 0) == (want == 0)
        nfail += not ok
        print('%-14s bin %d rc=%d %-28s failed checks %s %s' % (
            name, b, rc, last, sorted(set(fails)), 'ok' if ok else 'WRONG'))
    print('SELECTION SELFTEST: %s'
          % ('PASS' if nfail == 0 else 'FAIL (%d cases)' % nfail))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
