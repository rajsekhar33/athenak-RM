#!/usr/bin/env python3
"""Validate and summarise a run_variants.sh directory.

usage: collect.py OUTDIR

Validation (any failure makes the exit status 1 and the last line
'SCALING SUMMARY: INVALID ...'):
- every expected <variant>_r<rep> directory exists, exited 0 and reached the final cycle
  (NLIM), from run_info.txt;
- every expected output block wrote files at cycle 0, every DCYCLE up to NLIM, and
  the final output (AthenaK writes a final output at NLIM even when the cadence just
  wrote one, so NLIM is listed twice);
- every sf/spec file has the header keys its layout needs and a payload of exactly the
  size that layout implies (header-only, truncated and trailing-data files are
  rejected); the float64 arrays are finite (n_acc is uint64 and read as such);
- every run's log reports the expected number of MPI ranks (a serial build launched
  under srun would run NRANK independent copies and report 1);
- spectrum sum rules: unweighted scalar and Helmholtz energies add up to the stored
  sumsq (minus P_DC with log bins), within 1e-10 relative.
- every loop time ('cpu time used') is a finite positive number; if only that fails,
  the last line says 'INVALID (timing unavailable ...; outputs valid)'.

Timings (all per output EVENT: one event writes every configured block at once):
- loop: AthenaK's 'cpu time used', the wall time of the cycle loop. It includes the
  outputs at cycles > 0 and the final output (Driver::Finalize writes before the timer
  is read), not the cycle-0 outputs.
- events: output files per block with cycle > 0 (cadence writes plus the final write).
- event cost: (loop - base loop of the same repeat) / events. With two or more repeats
  it is reported with the base scatter (max - min of the base loop times over repeats);
  a difference not above that scatter, or negative, is 'unresolved'.
- components (max over ranks, per output, from stdout/headers): sf E1 and E2 pair
  kernels (they exclude computed-field preparation, histogram set-up, support and
  selection, the final reduction and the write); spec setup, fft and bin, each the max
  over ranks of its own stage (their sum bounds no critical path exactly).
"""
import glob
import math
import os
import re
import statistics
import sys

import numpy as np


def header(path):
    keys = {}
    with open(path, 'rb') as f:
        for raw in f:
            line = raw.decode('latin-1').rstrip('\n')
            k, _, v = line.partition('=')
            keys[k.strip()] = v.strip()
            if k == 'header offset':
                break
    return keys


def layout(h, ext):
    """[(name, shape, dtype)] of the payload, from the header (docs/*_output.md)."""
    nb = int(h['nbin'])
    if ext == '.spec':
        nv, nw = int(h['nvar']), int(h['nweight'])
        if h['mode'] == 'scalar':
            tail = [('E', (nw, nv, nb)), ('P_DC', (nw, nv)), ('sumsq', (nv,))]
        else:
            tail = [('E_comp', (nb,)), ('E_sol', (nb,)), ('P_DC', (1,)), ('sumsq', (1,))]
        return [(n, s, 'f8') for n, s in
                [('kbins', (nb + 1,)), ('k', (nb,)), ('nk', (nb,))] + tail]
    nw, nq, P, R = (int(h['nweight']), int(h['nq']), int(h['max_order']),
                    int(h['nreplicate']))
    if h['mode'] == 'aniso_mhd':
        rb = [('rbins', (int(h['nbin_l']) + 1,)),
              ('rbins_prll', (int(h['nbin_prll']) + 1,)),
              ('rbins_perp', (int(h['nbin_perp']) + 1,))]
    else:
        rb = [('rbins', (nb + 1,))]
    out = [(n, s, 'f8') for n, s in rb + [
        ('T', (nw, nq, P, nb)), ('W', (nw, nb)), ('C', (nw, nb)),
        ('T_er', (2, R, nw, nq, P, nb)), ('W_er', (2, R, nw, nb)),
        ('C_er', (2, R, nw, nb))]]
    out.append(('n_acc_er', (2, R, nw, nb), 'u8'))
    if h.get('histogram', '0') == '1':
        nc = int(h['hist_ncol'])
        out += [('hist_edges', (nw, nq, nc - 1), 'f8'), ('hist_sigma', (nw, nq), 'f8'),
                ('H', (nw, nq, nb, nc), 'f8'), ('H_er', (2, R, nw, nq, nb, nc), 'f8')]
    return out


def read_checked(path, ext, problems):
    """Header and payload arrays of one output file, or None (with a problem recorded)
    when a needed header key is missing or the payload size differs from the layout."""
    try:
        h = header(path)
        off = int(h['header offset'])
        lay = layout(h, ext)
        int(h['cycle'])
    except (KeyError, ValueError) as e:
        problems.append('%s: header unusable (%s: %s)' % (path, type(e).__name__, e))
        return None
    with open(path, 'rb') as f:
        f.seek(off)
        buf = f.read()
    want = sum(8 * int(np.prod(s)) for _, s, _ in lay)
    if len(buf) != want:
        problems.append('%s: payload %d bytes, layout needs %d' % (path, len(buf), want))
        return None
    a, pos = {}, 0
    for name, shape, dt in lay:
        n = int(np.prod(shape))
        a[name] = np.frombuffer(buf, dtype=dt, count=n, offset=pos).reshape(shape)
        pos += 8 * n
        if dt == 'f8' and not np.all(np.isfinite(a[name])):
            problems.append('%s: nonfinite values in %s' % (path, name))
    return h, a


def spec_sum_rule(h, a):
    """Relative error of the strict sum rule, or None when it does not apply."""
    logb = h.get('logscale') == '1'
    if h['mode'] == 'scalar':
        wts = h['weights'].split(',')
        if 'ones' not in wts:
            return None
        iw, err = wts.index('ones'), 0.0
        for iv in range(a['E'].shape[1]):
            ref = a['sumsq'][iv] - (a['P_DC'][iw, iv] if logb else 0.0)
            err = max(err, abs(a['E'][iw, iv].sum() - ref)
                      / max(abs(a['sumsq'][iv]), 1e-300))
        return err
    ref = a['sumsq'][0] - (a['P_DC'][0] if logb else 0.0)
    return (abs(a['E_comp'].sum() + a['E_sol'].sum() - ref)
            / max(abs(a['sumsq'][0]), 1e-300))


def check_outputs(d, blocks, want, problems, tag=None):
    """Validate every expected block of a run directory: each file readable with the
    right payload size and finite values, spectrum sum rules, and the output cycles equal
    to `want`. Returns {block: [(header, arrays)]}."""
    tag = tag or os.path.basename(d)
    got = {}
    for blk in blocks:
        ext = '.sf' if blk.startswith('sf_') else '.spec'
        cycles, got[blk] = [], []
        for f in sorted(glob.glob(os.path.join(d, blk, '*' + ext))):
            ha = read_checked(f, ext, problems)
            if ha is None:
                continue
            h, a = ha
            cycles.append(int(h['cycle']))
            got[blk].append(ha)
            if ext == '.spec':
                e = spec_sum_rule(h, a)
                if e is not None and e > 1e-10:
                    problems.append('%s: spectrum sum rule error %.1e'
                                    % (os.path.relpath(f, d), e))
        if sorted(cycles) != sorted(want):
            problems.append('%s/%s: valid output cycles %s, expected %s'
                            % (tag, blk, sorted(cycles), want))
    return got


def check_log(log, name, nlim, nrank, problems):
    """Common log checks: rank count, finished loop, final cycle."""
    if nrank is not None:
        ranks = re.findall(r'^Number of parallel ranks = (\d+)\s*$', log, re.M)
        if ranks != [str(nrank)]:
            problems.append('%s: rank banner %s, expected exactly one with %d ranks'
                            % (name, ranks or 'missing', nrank))
    if not re.search(r'cpu time used\s*=', log):
        problems.append('%s: no "cpu time used" line (run did not finish)' % name)
    cyc = [int(x) for x in re.findall(r'^\s*time=\S+ cycle=(\d+)\s*$', log, re.M)]
    final = cyc[-1] if cyc else None
    if final != nlim:
        problems.append('%s: final cycle %s, expected %d' % (name, final, nlim))
    return final


def run_info(out):
    info, text = {}, ''
    p = os.path.join(out, 'run_info.txt')
    if os.path.exists(p):
        text = open(p).read()
        for m in re.finditer(r'(\w+)=([^\s;]+)', text):
            info.setdefault(m.group(1), m.group(2))
        m = re.search(r'variants=(.*)$', text, re.M)
        if m:
            info['variants'] = m.group(1).split()
    return info, text


def expected_blocks(variant, phys):
    sf = ['sf_scalar', 'sf_longtrans'] + (['sf_aniso'] if phys == 'mhd' else [])
    spec = ['spec_scal', 'spec_helm']
    return {'base': [], 'sf': sf, 'spec': spec, 'both': sf + spec}[variant]


def parse_run(d, nlim, dc, phys, nrank, problems):
    name = os.path.basename(d)
    variant = name.rsplit('_r', 1)[0]
    r = {'dir': name, 'variant': variant, 'rep': name.rsplit('_r', 1)[1]}
    logp = os.path.join(d, 'log')
    if not os.path.exists(logp):
        problems.append('%s: no log' % name)
        return r
    log = open(logp, errors='replace').read()
    num = r'([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)'
    # the loop time must be a finite positive number; otherwise the outputs may still be
    # valid, but no timing comparison can be made (reported separately)
    m = re.search(r'cpu time used\s*=\s*(\S+)', log)
    try:
        r['loop'] = float(m.group(1)) if m else None
    except ValueError:
        r['loop'] = None
    if r['loop'] is None or not math.isfinite(r['loop']) or r['loop'] <= 0:
        r['timing_problem'] = '%s: loop time %r is not a finite positive number' % (
            name, m.group(1) if m else None)
        r['loop'] = None
    m = re.search(r'zone-cycles/cpu_second\s*=\s*' + num, log)
    r['zcps'] = float(m.group(1)) if m else None
    m = re.search(r'exit=(\d+) wall_launch=' + num, log)
    r['exit'] = int(m.group(1)) if m else None
    r['launch'] = float(m.group(2)) if m else None
    if r['exit'] != 0:
        problems.append('%s: exit %s' % (name, r['exit']))
    r['final_cycle'] = check_log(log, name, nlim, nrank, problems)
    # sf pair-kernel timings: MPI 'E1 a..b s, E2 c..d s' or serial 'E1 a s, E2 b s'
    sf = {}
    for mm in re.finditer(r'sf\[([^\]]*)\] output (\d+): E1 ' + num + r'(?:\.\.' + num +
                          r')? s, E2 ' + num + r'(?:\.\.' + num + r')? s', log):
        e1 = float(mm.group(4) or mm.group(3))
        e2 = float(mm.group(6) or mm.group(5))
        sf.setdefault(mm.group(1), []).append((e1, e2))
    r['sf'] = sf
    want = [0] + list(range(dc, nlim + 1, dc)) + [nlim]
    r['events'] = len(want) - 1
    r['spec'], r['sel'] = {}, {}
    got = check_outputs(d, expected_blocks(variant, phys), want, problems, name)
    for blk, files in got.items():
        for h, _ in files:
            if blk.startswith('spec_'):
                r['spec'].setdefault(blk[5:], []).append(
                    (float(h.get('time_setup', 0)), float(h.get('time_fft', 0)),
                     float(h.get('time_bin', 0)),
                     int(h.get('memory_device_bytes_max_rank', 0)),
                     h.get('pencils', '?'), h.get('time_last_transform', '?'),
                     h.get('time_last_transform_stages', '?')))
            elif 'ncell_selected' in h:
                r['sel'].setdefault(blk[3:], []).append(int(h['ncell_selected']))
    return r


def mean(x):
    return sum(x) / len(x) if x else float('nan')


def main():
    out = sys.argv[1]
    info, text = run_info(out)
    print(text.rstrip())
    problems = []
    try:
        nlim, dc = int(info['nlim']), int(info['dcycle'])
        repeat = int(info.get('repeat', 1))
        variants = info.get('variants', ['base', 'sf', 'spec', 'both'])
        phys = info.get('phys', 'hydro')
        nrank = int(info['nrank'])
    except KeyError as e:
        print('run_info.txt lacks %s' % e)
        print('\nSCALING SUMMARY: INVALID (no run_info)')
        return 1
    runs = []
    for rep in range(1, repeat + 1):
        for v in variants:
            d = os.path.join(out, '%s_r%d' % (v, rep))
            if not os.path.isdir(d):
                problems.append('%s_r%d: missing' % (v, rep))
                continue
            runs.append(parse_run(d, nlim, dc, phys, nrank, problems))
    base = [r['loop'] for r in runs if r['variant'] == 'base' and r.get('loop')]
    scatter = (max(base) - min(base)) if len(base) > 1 else None
    print('\nbase loop times: %s s; scatter (max-min) %s' % (
        ', '.join('%.4g' % x for x in base),
        '%.3g s' % scatter if scatter is not None else 'n/a (one repeat)'))
    print('\n%-9s %4s %9s %9s %11s %6s %14s' %
          ('run', 'exit', 'loop[s]', 'launch[s]', 'zone-cyc/s', 'events',
           'event cost[s]'))
    bymap = {(r['variant'], r['rep']): r for r in runs}

    def fmt(x):
        return '%.4g' % x if x else '-'

    for r in runs:
        cost = ''
        b = bymap.get(('base', r['rep']))
        if r['variant'] != 'base' and b and b.get('loop') and r.get('loop'):
            diff = r['loop'] - b['loop']
            cost = '%.4g' % (diff / r['events'])
            if diff <= 0 or (scatter is not None and diff <= scatter):
                cost += ' unresolved'
        print('%-9s %4s %9s %9s %11s %6s %14s' % (
            r['dir'], r.get('exit'), fmt(r.get('loop')), fmt(r.get('launch')),
            fmt(r.get('zcps')), r.get('events', '-'), cost))
    costs = {}
    for r in runs:
        b = bymap.get(('base', r['rep']))
        if r['variant'] != 'base' and b and b.get('loop') and r.get('loop'):
            costs.setdefault(r['variant'], []).append(
                (r['loop'] - b['loop']) / r['events'])
    if costs:
        print('\nevent cost over repeats (min / median / max) [s]; a spread '
              'comparable to the median means the cost is not resolved:')
        for v, c in costs.items():
            print('  %-5s %s' % (v, ' / '.join('%.4g' % x for x in
                                               (min(c), statistics.median(c), max(c)))))
    print('\ncomponents per output (max over ranks; first output / mean of later ones):')
    for r in runs:
        for oid, ts in sorted(r.get('sf', {}).items()):
            later = ts[1:] or ts
            ns = r['sel'].get(oid)
            print('  %-9s sf[%s] n=%d: E1 %.3g / %.3g s, E2 %.3g / %.3g s%s' % (
                r['dir'], oid, len(ts), ts[0][0], mean([t[0] for t in later]),
                ts[0][1], mean([t[1] for t in later]),
                ', ncell_selected %d..%d' % (min(ns), max(ns)) if ns else ''))
        for oid, v in sorted(r.get('spec', {}).items()):
            later = v[1:] or v
            print('  %-9s spec[%s] n=%d pencils=%s: setup %.3g / %.3g s, '
                  'fft %.3g / %.3g s, '
                  'bin %.3g / %.3g s; owned device memory max %.3g GB; last transform '
                  'stages %s = %s' % (
                      r['dir'], oid, len(v), v[0][4], v[0][0],
                      mean([x[0] for x in later]),
                      v[0][1], mean([x[1] for x in later]), v[0][2],
                      mean([x[2] for x in later]), max(x[3] for x in v) / 1e9, v[-1][6],
                      v[-1][5]))
    timing = [r['timing_problem'] for r in runs if r.get('timing_problem')]
    if timing:
        print('\ntiming unavailable:')
        for p in timing:
            print('  ' + p)
    if problems:
        print('\nproblems:')
        for p in problems:
            print('  ' + p)
        print('\nSCALING SUMMARY: INVALID (%d problems)' % len(problems))
        return 1
    if timing:
        print('\nSCALING SUMMARY: INVALID (timing unavailable in %d runs; outputs valid)'
              % len(timing))
        return 1
    print('\nSCALING SUMMARY: VALID (%d runs; ranks, cycles, output sizes, '
          'finiteness and sum rules checked)' % len(runs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
