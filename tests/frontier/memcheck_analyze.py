#!/usr/bin/env python3
"""Judge one memcheck.sh run directory: the run's validity, then memory growth.

usage: memcheck_analyze.py RUNDIR RC NRANK NLIM VARIANT PHYS
env:   MEMMODE   device (physical devices, e.g. rocm-smi cards; the Frontier gate) or
                 process (per-process rows, e.g. nvidia-smi pids on a shared node);
       MEMEXPECT device mode: the device identities that must be sampled, comma
                 separated (default card0..card7); process mode: the minimum number of
                 processes judged (default NRANK);
       MEMFILTER regex; only data lines matching it count.

Reads RUNDIR/log, RUNDIR/mem_samples.txt and the output blocks. Last line:
  MEMCHECK: PASS | GROWTH | FAIL | INCONCLUSIVE  (exit 0, 1, 1, 2)

- FAIL: the run is not valid (exit, rank banner, NLIM, every output with its layout's
  size and finite values), whatever the memory does.
- Windows by completed output events: early = from event max(2, NLIM/10) over the next
  NLIM/5 events; late = the last NLIM/5 cadence events.
- Device mode: every expected device needs at least 3 finite readings in each window,
  else INCONCLUSIVE; a device that is not expected is reported and not judged.
- Process mode: a process present in only one window is reported and not judged
  (another job on a shared node); at least MEMEXPECT processes must be judged.
- GROWTH: a judged device or process grew more than 5% between the window medians.
"""
import collections
import glob
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402


def main():
    d, rc, nrank, nlim, variant, phys = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
        int(sys.argv[4]), sys.argv[5], sys.argv[6]
    os.chdir(d)
    mode = os.environ.get('MEMMODE', 'device')
    if mode not in ('device', 'process'):
        print('MEMCHECK: FAIL (unknown MEMMODE=%s)' % mode)
        return 1
    if mode == 'device':
        expect = os.environ.get('MEMEXPECT') or ','.join('card%d' % i for i in range(8))
        expect = [x.strip() for x in expect.split(',') if x.strip()]
    else:
        expect = int(os.environ.get('MEMEXPECT') or nrank)

    # ---- the run itself ----
    problems = []
    if rc != 0:
        problems.append('athena exit %d' % rc)
    collect.check_log(open('log', errors='replace').read(), 'run', nlim, nrank, problems)
    blocks = collect.expected_blocks(variant, phys)
    want = list(range(0, nlim + 1)) + [nlim]
    collect.check_outputs('.', blocks, want, problems, 'run')

    # ---- output events: completion time of cadence cycle k ----
    done = {}
    for blk in blocks:
        ext = '.sf' if blk.startswith('sf_') else '.spec'
        per = collections.defaultdict(list)
        for f in glob.glob(os.path.join(blk, '*' + ext)):
            per[int(collect.header(f).get('cycle', -1))].append(os.path.getmtime(f))
        for c, ts in per.items():   # at NLIM the earlier file is the cadence write
            done[c] = max(done.get(c, 0.0), min(ts))
    k0, n = max(2, nlim // 10), max(nlim // 5, 1)
    try:
        early = (done[k0], done[k0 + n])
        late = (done[nlim - n], done[nlim])
    except KeyError as e:
        early = late = None
        problems.append('no completion time for output event %s' % e)

    # ---- samples: {time, {key: value}}; a non-numeric or nonfinite reading is nan ----
    flt = os.environ.get('MEMFILTER') or None
    samples, cur, col = [], None, None
    for line in open('mem_samples.txt', errors='replace'):
        line = line.strip()
        if line.startswith('## '):
            cur = (float(line[3:]), {})
            samples.append(cur)
            continue
        parts = [p.strip() for p in line.split(',')]
        if any('used' in p.lower() for p in parts):          # a CSV header
            col = [i for i, p in enumerate(parts) if 'used' in p.lower()][0]
            continue
        if cur is None or col is None or (flt and not re.search(flt, line)):
            continue
        m = (re.match(r'([-+]?[0-9.]+(?:[eE][-+]?\d+)?)', parts[col])
             if col < len(parts) else None)
        v = float(m.group(1)) if m else float('nan')
        cur[1][parts[0]] = cur[1].get(parts[0], 0.0) + v
    keys = sorted({k for _, v in samples for k in v})
    print('mode %s; sampled %s; samples %d' % (mode, keys or 'none', len(samples)))

    verdict, judged, grow = None, 0, False
    if early is None:
        verdict = 'INCONCLUSIVE (no output-event windows)'
    else:
        def window(w, k):
            return [v[k] for t, v in samples if w[0] <= t <= w[1] and k in v]
        if mode == 'device':
            missing = [k for k in expect if k not in keys]
            if missing:
                verdict = 'INCONCLUSIVE (expected devices never sampled: %s)' % missing
            for k in keys:
                if k not in expect:
                    print('  %-10s not an expected device: not judged' % k)
        for k in (expect if mode == 'device' else keys):
            if k not in keys:
                continue
            e, lt = window(early, k), window(late, k)
            pk = max((v[k] for _, v in samples if k in v and math.isfinite(v[k])),
                     default=float('nan'))
            bad = [x for x in e + lt if not math.isfinite(x)]
            if mode == 'process' and (not e or not lt):
                print('  %-10s present in only one window (another process?): '
                      'not judged' % k)
                continue
            if bad or len(e) < 3 or len(lt) < 3:
                verdict = verdict or ('INCONCLUSIVE (%s: %d early and %d late '
                                      'samples, %d invalid readings)'
                                      % (k, len(e), len(lt), len(bad)))
                print('  %-10s %d early, %d late, %d invalid readings'
                      % (k, len(e), len(lt), len(bad)))
                continue
            me, ml = statistics.median(e), statistics.median(lt)
            g = (ml - me) / me if me > 0 else float('inf')
            judged += 1
            grow = grow or g > 0.05
            print('  %-10s early median %.6g (%d), late median %.6g (%d), '
                  'sampled peak %.6g, growth %+.2f%%'
                  % (k, me, len(e), ml, len(lt), pk, 100 * g))
        print('windows: early events %d..%d (%.0f s), late events %d..%d (%.0f s); '
              'units of MEMCMD'
              % (k0, k0 + n, early[1] - early[0], nlim - n, nlim, late[1] - late[0]))
        if verdict is None and mode == 'process' and judged < expect:
            verdict = 'INCONCLUSIVE (%d processes judged, %d expected)' % (judged, expect)
        if verdict is None:
            verdict = 'GROWTH (grew > 5%: investigate)' if grow else \
                ('PASS (%d %s, none grew more than 5%%)'
                 % (judged, 'devices' if mode == 'device' else 'processes'))
    if problems:
        print('run problems:')
        for p in problems:
            print('  ' + p)
        print('MEMCHECK: FAIL (the run is not valid; '
              'memory result above is not accepted)')
        return 1
    print('MEMCHECK: ' + verdict)
    return 0 if verdict.startswith('PASS') else 2 if verdict.startswith('INCON') else 1


if __name__ == '__main__':
    sys.exit(main())
