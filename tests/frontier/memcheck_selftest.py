#!/usr/bin/env python3
"""Fault injection for memcheck_analyze.py on a valid memcheck.sh run directory.

usage: memcheck_selftest.py RUNDIR NEWDIR NRANK NLIM
Copies RUNDIR (keeping the output files' times) once per case, writes synthetic rocm-smi
samples for card0..card7 over the run, and checks the verdict:
  all cards flat -> PASS; one card never sampled -> INCONCLUSIVE; a card gone before
  the late window -> INCONCLUSIVE; invalid readings inside the windows -> INCONCLUSIVE;
  growth on one card only -> GROWTH; a nonzero Athena exit -> FAIL.
Last line: MEMCHECK SELFTEST: PASS or FAIL.
"""
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def write_samples(d, case):
    ts = [os.path.getmtime(f) for f in glob.glob(os.path.join(d, 'sf_*', '*.sf'))]
    t0, t1 = min(ts) - 2, max(ts) + 2
    dt = (t1 - t0) / 2000
    with open(os.path.join(d, 'mem_samples.txt'), 'w') as f:
        for n in range(2001):
            t = t0 + n * dt
            frac = n / 2000
            f.write('## %.6f\ndevice,VRAM Total Memory (B),'
                    'VRAM Total Used Memory (B)\n' % t)
            for c in range(8):
                if case == 'missing' and c == 5:
                    continue
                if case == 'vanish' and c == 3 and frac > 0.6:
                    continue
                v = 1e10 * (1 + 0.3 * frac if case == 'growth' and c == 6 else 1)
                val = 'nan' if case == 'invalid' and c == 2 and n % 10 == 5 else '%d' % v
                f.write('card%d,68702699520,%s\n' % (c, val))


def main():
    src, new, nrank, nlim = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    if os.path.exists(new):
        print('MEMCHECK SELFTEST: FAIL (%s exists)' % new)
        return 1
    cases = [('flat', 0, 'PASS'), ('missing', 0, 'INCONCLUSIVE'),
             ('vanish', 0, 'INCONCLUSIVE'), ('invalid', 0, 'INCONCLUSIVE'),
             ('growth', 0, 'GROWTH'), ('athena_failed', 3, 'FAIL')]
    env = dict(os.environ, MEMMODE='device', MEMEXPECT='', MEMFILTER='')
    nfail = 0
    for name, rc, want in cases:
        d = os.path.join(new, name)
        shutil.copytree(src, d, ignore=shutil.ignore_patterns('mem_samples.txt'))
        write_samples(d, 'flat' if name == 'athena_failed' else name)
        p = subprocess.run([sys.executable, '-I',
                            os.path.join(HERE, 'memcheck_analyze.py'),
                            d, str(rc), nrank, nlim, 'both', 'hydro'], env=env,
                           capture_output=True, text=True)
        open(os.path.join(d, 'analysis.txt'), 'w').write(p.stdout + p.stderr)
        last = (p.stdout.strip().splitlines() or ['(no output)'])[-1]
        ok = last.startswith('MEMCHECK: ' + want)
        nfail += not ok
        print('%-14s %-40s %s'
              % (name, last[:40], 'ok' if ok else 'WRONG (want %s)' % want))
    print('MEMCHECK SELFTEST: %s' % ('PASS' if nfail == 0 else 'FAIL (%d cases)' % nfail))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
