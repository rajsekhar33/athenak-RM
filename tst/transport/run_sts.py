"""Portable bounded anisotropic STS regression; never builds or submits jobs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time

import numpy as np

source = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(source / 'tst/transport'), str(source / 'vis/python')]
from oracles import check_mode, conduction_rate, read_metrics  # noqa: E402
import bin_convert  # noqa: E402
from sts_checks import compare_fields  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--launcher', default='', help='Explicit launch prefix; no shell')
parser.add_argument('--expected-ranks', type=int, default=1)
parser.add_argument('--comparison-launcher', default='',
                    help='Optional second decomposition, e.g. mpirun -np 2')
parser.add_argument('--comparison-ranks', type=int, default=2)
parser.add_argument('--device-args', default='', help='Optional Kokkos arguments')
parser.add_argument('--limiter', choices=('vanleer', 'mc', 'both'), default='both')
parser.add_argument('--n3d', type=int, choices=(24, 32), default=24)
parser.add_argument('--only', choices=('all', 'periodic3d'), default='all')
args = parser.parse_args()
if args.expected_ranks < 1 or (args.expected_ranks > 1 and not args.launcher.strip()):
    parser.error('Multiple ranks require a nonempty launcher and positive rank count')
if args.comparison_launcher and (args.comparison_ranks < 1 or
                                 args.comparison_ranks == args.expected_ranks):
    parser.error('Comparison rank count must be positive and differ from primary count')
binary = args.binary.resolve()
if not binary.is_file():
    parser.error(f'Executable does not exist: {binary}')
run = args.root.resolve()
run.mkdir(parents=True, exist_ok=False)
shutil.copy2(binary, run / 'athena')
shutil.copy2(source / 'tst/inputs/transport_mode.athinput', run / 'input.athinput')
report = dict(status='RUNNING', cases={}, comparisons={},
              launcher=args.launcher, expected_ranks=args.expected_ranks,
              comparison_launcher=args.comparison_launcher,
              comparison_ranks=(args.comparison_ranks
                                if args.comparison_launcher else None),
              selection=args.only, n3d=args.n3d,
              binary_sha256=hashlib.sha256((run / 'athena').read_bytes()).hexdigest(),
              input_sha256=hashlib.sha256(
                  (run / 'input.athinput').read_bytes()).hexdigest(),
              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              checker_sha256=hashlib.sha256(
                  (source / 'tst/transport/sts_checks.py').read_bytes()).hexdigest(),
              scope='Uniform-grid STS without diagnostic observers; periodic and '
                    'zero-normal-heat-flux outflow. Not general BC/AMR acceptance.')


def save():
    (run / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


def execute(name, options, ranks, target, field, mode, restart=None, partial=False):
    folder = run / name
    folder.mkdir()
    launcher = shlex.split(args.launcher if ranks == args.expected_ranks
                           else args.comparison_launcher)
    command = [*launcher,
               str(run / 'athena'), '-r' if restart else '-i',
               str(restart or run / 'input.athinput'), '-t', '00:09:00', *options]
    command.extend(shlex.split(args.device_args))
    entry = dict(command=command, status='ERROR')
    report['cases'][name] = entry
    (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    start = time.monotonic()
    with (folder / 'athena.log').open('w') as log:
        process = subprocess.Popen(command, cwd=folder, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=570)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            entry['timeout'] = True
            code = 124
    entry.update(returncode=code, elapsed=time.monotonic() - start)
    save()
    assert code == 0, (name, code)
    log = (folder / 'athena.log').read_text()
    assert re.search(rf'Number of parallel ranks\s*=\s*{ranks}\b', log)
    entry['verified_ranks'] = ranks
    metrics = read_metrics(folder / 'transport.mode.dat')
    actual_time = metrics['time']
    assert ((0 < actual_time < target) if partial
            else abs(actual_time - target) < 1.e-12)
    rate = conduction_rate(tuple(2 * math.pi * k for k in mode),
                           field, 5 / 3, 0, .1)
    failures = check_mode(metrics, rate, actual_time, initial_energy=1.5 + .5 * sum(
        b * b for b in field))
    entry.update(metrics=metrics, failures=failures,
                 status='FAIL' if failures else 'PASS')
    output = sorted((folder / 'bin').glob('transport.mhd_w.*.bin'))[-1]
    data = bin_convert.read_binary_as_athdf(output, dtype=np.float64)
    assert abs(data['Time'] - actual_time) < 1.e-12
    for key in ('dens', 'velx', 'vely', 'velz', 'eint'):
        assert np.isfinite(data[key]).all()
    entry['cycles'] = int(data['NumCycles'])
    save()
    print(name, entry['status'], failures, flush=True)
    assert not failures, (name, failures)
    return entry, data, folder


def compare(name, left, right, temporal=False, reference=False):
    kind = 'reference' if reference else ('temporal' if temporal else 'restart')
    report['comparisons'][name] = compare_fields(
        left[1], right[1], left[0]['metrics'], right[0]['metrics'], kind)
    save()
    assert report['comparisons'][name]['status'] == 'PASS', report['comparisons'][name]


geometries = [('periodic2d', (24, 24, 1), (1, 1, 0), (1, 1, 0), .02),
              ('periodic3d', (args.n3d,) * 3, (1, 1, 1), (1, 1, 1), .02),
              ('insulated2d', (32, 8, 1), (1, 0, 0), (1, 0, 0), .05)]
if args.only != 'all':
    geometries = [g for g in geometries if g[0] == args.only]
limiters = ('vanleer', 'mc') if args.limiter == 'both' else (args.limiter,)
for limiter in limiters:
    for geometry, grid, field, mode, endpoint in geometries:
        group = f'{geometry}-{limiter}'
        options = [f'mhd/anisotropic_limiter={limiter}', 'mhd/alpha_iso=0',
                   'mhd/alpha_aniso=0.1', f'time/tlim={endpoint}',
                   f'output1/dt={endpoint}', 'output2/dt=100', 'time/nlim=-1']
        for axis, (n, b, k) in enumerate(zip(grid, field, mode), 1):
            options += [f'mesh/nx{axis}={n}', f'meshblock/nx{axis}={max(1, n // 2)}',
                        f'problem/b{"xyz"[axis - 1]}={b}',
                        f'problem/mode_{"xyz"[axis - 1]}={k}']
        if geometry == 'insulated2d':
            options += ['mesh/ix1_bc=outflow', 'mesh/ox1_bc=outflow']
        try:
            fine = execute(group + '-explicit01', options + ['time/cfl_number=.1'],
                           args.expected_ranks, endpoint, field, mode)
            coarse = execute(group + '-explicit02', options + ['time/cfl_number=.2'],
                             args.expected_ranks, endpoint, field, mode)
            compare(group + '-reference', fine, coarse, reference=True)
            sts = options + ['time/cfl_number=.4', 'time/sts_integrator=rkl2',
                             'mhd/conductivity_integrator=sts',
                             'time/sts_max_dt_ratio=16']
            r16 = execute(group + '-sts16', sts + ['time/sts_max_dt_ratio=16'],
                          args.expected_ranks, endpoint, field, mode)
            r64 = execute(group + '-sts64', sts + ['time/sts_max_dt_ratio=64'],
                          args.expected_ranks, endpoint, field, mode)
            compare(group + '-temporal16', fine, r16, temporal=True)
            compare(group + '-temporal64', fine, r64, temporal=True)
            assert r16[0]['cycles'] < fine[0]['cycles'], 'STS did not reduce cycle count'
            ranks = (args.comparison_ranks if args.comparison_launcher
                     else args.expected_ranks)
            distributed = r16
            if args.comparison_launcher:
                distributed = execute(group + f'-mpi{ranks}-sts16', sts,
                                      ranks, endpoint, field, mode)
                compare(group + '-mpi', r16, distributed)
            half = distributed[0]['cycles'] // 2
            assert half >= 1, 'Need at least two cycles for a real restart'
            partial = execute(group + '-partial', sts + [f'time/nlim={half}'],
                              ranks, endpoint, field, mode, partial=True)
            checkpoint = sorted((partial[2] / 'rst').glob('*.rst'))[-1]
            checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            resumed = execute(group + '-resumed',
                              ['time/nlim=-1', f'time/tlim={endpoint}'],
                              ranks, endpoint, field, mode, restart=checkpoint)
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
            compare(group + '-restart', distributed, resumed)
        except (AssertionError, OSError, ValueError, IndexError, KeyError) as error:
            report.setdefault('errors', {})[group] = str(error)
            print(group, 'ERROR/FAIL', str(error), flush=True)
            save()
report['status'] = 'FAIL' if report.get('errors') else 'PASS'
report['expected_cases'] = len(limiters) * len(geometries) * (
    7 if args.comparison_launcher else 6)
report['expected_comparisons'] = len(limiters) * len(geometries) * (
    5 if args.comparison_launcher else 4)
if (len(report['cases']) != report['expected_cases'] or
        len(report['comparisons']) != report['expected_comparisons'] or
        any(c['status'] != 'PASS' for c in report['cases'].values()) or
        any(c['status'] != 'PASS' for c in report['comparisons'].values())):
    report['status'] = 'FAIL'
save()
raise SystemExit(0 if report['status'] == 'PASS' else 1)
