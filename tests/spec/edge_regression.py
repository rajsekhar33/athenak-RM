#!/usr/bin/env python3
"""Check spectral endpoint conservation on small noncubic and degenerate grids.

Requires a PROBLEM=sf_unit_test binary. The existing spec_reference.py validates
all bins and Parseval against independent full-complex FFTs with its default gates.
This runner additionally requires exact Fourier mode counts and output inventory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np

from spec_reference import read_spec


CASES = {
    'noncubic': ((8, 8, 8), (4, 4, 4), False, 0.0, None),
    'cropped_odd': ((8, 8, 8), (4, 4, 4), True, 0.1, None),
    'one_dimensional': ((32, 1, 1), (4, 1, 1), False, 0.0, None),
    'global_fallback': ((8, 8, 8), (4, 4, 4), False, 0.0, 10000),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_input(template, case):
    shape, block_shape, strat, skip, bins = CASES[case]
    sections = re.split(r'(?m)^(<[^>]+>)[ \t]*$', template)
    for i in range(1, len(sections), 2):
        name = sections[i][1:-1]
        edits = {}
        if name == 'mesh':
            edits.update({f'nx{d}': shape[d - 1] for d in (1, 2, 3)})
            edits.update({f'x{d}max': value for d, value in enumerate((0.7, 7, 2), 1)})
        elif name == 'meshblock':
            edits.update({f'nx{d}': block_shape[d - 1] for d in (1, 2, 3)})
        elif name.startswith('output'):
            edits.update(strat=str(strat).lower(), skip=skip)
            if bins is not None:
                edits['nbin'] = bins
        body = sections[i + 1]
        for key, value in edits.items():
            pattern = rf'(?m)^[ \t]*{re.escape(key)}[ \t]*=.*$'
            line = f'{key} = {value}'
            if re.search(pattern, body):
                body = re.sub(pattern, lambda match: line, body)
            else:
                body += '\n' + line + '\n'
        sections[i + 1] = body
    return ''.join(sections)


def invariants(run, case, ranks):
    expected_shape = list(CASES[case][0])
    if case == 'cropped_odd':
        expected_shape[2] = 7
    files = sorted(run.glob('spec_*/*.spec'))
    expected = {f'spec_{name}/spt.{number:05d}.spec'
                for name in ('scal', 'scallog', 'helm', 'helmlog') for number in (0, 1)}
    assert {str(path.relative_to(run)) for path in files} == expected, 'output inventory'
    checks = []
    for path in files:
        meta, arrays = read_spec(path)
        assert int(meta['nranks']) == ranks, (path, 'rank count')
        shape = [int(value) for value in meta['shape'].split(',')]
        assert shape == expected_shape, (path, shape, expected_shape)
        assert all(np.isfinite(value).all() for value in arrays.values()), path
        modes = int(np.prod(shape)) - (meta['logscale'] == '1')
        actual = float(arrays['nk'].sum())
        assert actual == modes, (path, 'mode count', actual, modes)
        if case == 'global_fallback':
            assert meta['bin_global_fallback'] == '1', (path, 'fallback not exercised')
        if meta['mode'] == 'scalar':
            iw = meta['weights'].split(',').index('ones')
            total = arrays['E'][iw].sum(axis=-1)
            reference = arrays['sumsq'].copy()
            if meta['logscale'] == '1':
                reference -= arrays['P_DC'][iw]
        else:
            total = arrays['E_comp'].sum() + arrays['E_sol'].sum()
            reference = arrays['sumsq'][0]
            if meta['logscale'] == '1':
                reference -= arrays['P_DC'][0]
        error = np.abs(total - reference)
        assert np.all(error <= 1e-10 * np.abs(reference)), (path, 'Parseval', error)
        checks.append({'file': str(path.relative_to(run)), 'modes': actual,
                       'expected_modes': modes,
                       'parseval_error': np.asarray(error).tolist()})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--launcher', default='', help='MPI command; -np N is appended')
    parser.add_argument('--np', type=int, default=1)
    parser.add_argument('--map-gpus', action='store_true',
                        help='append Kokkos mpi_rank GPU device mapping')
    parser.add_argument('--case', action='append', choices=CASES)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.out = args.out.resolve()
    if args.np < 1 or (args.np != 1 and not args.launcher):
        parser.error('positive --np and a launcher for multiple ranks are required')
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    template = (source / 'inputs/tests/spec_unit_test.athinput').read_text()
    checker = Path(__file__).with_name('spec_reference.py').resolve()
    report = {'binary': str(args.binary), 'binary_sha256': sha256(args.binary),
              'checker_sha256': sha256(checker), 'runner_sha256': sha256(Path(__file__)),
              'np': args.np, 'map_gpus': args.map_gpus, 'cases': []}
    prefix = shlex.split(args.launcher) + ['-np', str(args.np)] if args.launcher else []
    for case in args.case or CASES:
        run = args.out / case
        run.mkdir()
        (run / 'in.athinput').write_text(make_input(template, case))
        command = prefix + [str(args.binary), '-i', 'in.athinput', '-t', '00:05:00']
        if args.map_gpus:
            command.append('--kokkos-map-device-id-by=mpi_rank')
        result = {'case': case, 'command': command}
        try:
            with (run / 'athena.log').open('w') as log:
                subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=360)
            with (run / 'reference.log').open('w') as log:
                reference = subprocess.run(
                    [sys.executable, '-I', str(checker), str(run)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=360)
            result['reference_exit'] = reference.returncode
            result['invariants'] = invariants(run, case, args.np)
            assert reference.returncode == 0, 'spec_reference.py failed'
            result['status'] = 'PASS'
        except (AssertionError, subprocess.SubprocessError, ValueError, OSError) as error:
            result.update(status='FAIL', error=str(error))
        report['cases'].append(result)
        (args.out / 'record.json').write_text(json.dumps(report, indent=2) + '\n')
        print(case, result['status'], result.get('error', ''), flush=True)
    return int(any(case['status'] != 'PASS' for case in report['cases']))


if __name__ == '__main__':
    sys.exit(main())
