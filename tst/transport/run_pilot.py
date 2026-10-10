"""Small conduction tests with an optional explicit launcher. Never submits jobs.

Every physics failure remains FAIL and makes the process exit nonzero.
Reports distinguish numerical failures from harness/execution errors.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import shlex
import socket
import subprocess

from oracles import check_mode, conduction_rate, read_metrics


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--guards-only', action='store_true',
                        help='configuration guards plus small legacy-sentinel restart')
    parser.add_argument('--only', nargs='+',
                        help='rerun named cases; include restart parents')
    parser.add_argument('--wall-minutes', type=int, choices=(2, 9), default=2,
                        help='per-case wall cap; does not alter simulated time or gates')
    parser.add_argument('--limiter', choices=('vanleer', 'mc'), default='vanleer')
    parser.add_argument('--launcher', default='',
                        help='explicit launcher prefix, e.g. "mpirun -np 2"; no shell')
    parser.add_argument('--expected-ranks', type=int, default=1,
                        help='require Athena to report this number of parallel ranks')
    args = parser.parse_args()
    if args.guards_only and args.only:
        parser.error('--guards-only and --only are separate scopes')
    if args.expected_ranks < 1:
        parser.error('--expected-ranks must be positive')
    if args.guards_only or args.only:
        args.extended = True
    selected = set(args.only) if args.only else None
    if selected:
        selected.update(name.removesuffix('_restart') for name in list(selected)
                        if name.endswith('_restart'))
    wall = f'00:{args.wall_minutes:02d}:00'
    timeout = 60*args.wall_minutes+30
    launcher = shlex.split(args.launcher)
    args.root = args.root.resolve()
    source = Path(__file__).resolve().parents[2]
    args.root.mkdir(parents=True, exist_ok=False)
    binary = args.root.resolve()/'athena'
    shutil.copy2(args.binary, binary)
    shutil.copy2(source/'tst/inputs/transport_mode.athinput', args.root/'input.athinput')
    shutil.copytree(Path(__file__).parent, args.root/'checker',
                    ignore=shutil.ignore_patterns('__pycache__'))
    git_head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=False)
    head = git_head.stdout.strip() if git_head.returncode == 0 else None
    report = dict(status='RUNNING', host=socket.gethostname(), source_head=head,
                  scope=('guards_only' if args.guards_only else
                         'selected' if selected else 'smooth_matrix'),
                  selected=sorted(selected) if selected else None,
                  binary_sha256=digest(binary), limiter=args.limiter, launcher=launcher,
                  expected_ranks=args.expected_ranks,
                  input_sha256=digest(args.root/'input.athinput'),
                  source_sha256={str(p.relative_to(source)): digest(p)
                                 for p in sorted((source/'src').rglob('*'))
                                 if p.is_file()},
                  cases={})

    def spec(name, grid=(64, 8, 1), iso=0.0, aniso=0.1, field=(1, 0, 0),
             mode=(1, 0, 0), sts=False, amplitude=0.01, time=0.1):
        return dict(name=name, grid=grid, iso=iso, aniso=aniso, field=field,
                    mode=mode, sts=sts, amplitude=amplitude, time=time)

    cases = [spec('disabled', aniso=0),
             spec('isotropic_n32', grid=(32, 8, 1), iso=0.1, aniso=0),
             spec('isotropic_n64', iso=0.1, aniso=0),
             spec('isotropic_sts', iso=0.1, aniso=0, sts=True),
             spec('isotropic_no_units', iso=0.1, aniso=0),
             spec('anisotropic_parallel'),
             spec('anisotropic_perpendicular', field=(0, 1, 0)),
             spec('anisotropic_out_of_plane', field=(1, 0, 1))]
    if args.extended:
        cases += [spec('anisotropic_reversed', field=(-1, 0, 0)),
                  spec('anisotropic_weak', field=(1.e-12, 0, 0)),
                  spec('anisotropic_rescaled', field=(2, 0, 0)),
                  spec('anisotropic_zero_field', field=(0, 0, 0)),
                  spec('combined_iso_aniso', iso=0.05, aniso=0.05),
                  spec('anisotropic_sts', sts=True),
                  spec('combined_iso_aniso_sts', iso=0.05, aniso=0.05, sts=True),
                  spec('uniform_temperature', field=(1, 1, 1), amplitude=0),
                  spec('legacy_flux_sentinel', iso=0.1, aniso=0),
                  spec('anisotropic_1d', grid=(64, 1, 1), field=(1, 0, 1)),
                  spec('anisotropic_y', grid=(8, 64, 1), field=(0, 1, 0), mode=(0, 1, 0)),
                  spec('oblique_n32', grid=(32, 32, 1), field=(1, 1, 0), mode=(1, 1, 0)),
                  spec('oblique_n64', grid=(64, 64, 1), field=(1, 1, 0), mode=(1, 1, 0)),
                  spec('oblique_perpendicular', grid=(64, 64, 1), field=(1, -1, 0),
                       mode=(1, 1, 0)),
                  spec('anisotropic_z', grid=(8, 8, 32), field=(0, 0, 1),
                       mode=(0, 0, 1)),
                  spec('oblique_3d', grid=(24, 24, 24), field=(1, 1, 1),
                       mode=(1, 1, 1), time=0.02)]
    if args.guards_only:
        cases = [spec('disabled', grid=(16, 8, 1), aniso=0),
                 spec('legacy_flux_sentinel', grid=(16, 8, 1), iso=0.1, aniso=0)]
    for config in cases:
        name, grid = config['name'], config['grid']
        if selected and name not in selected:
            continue
        iso, aniso, field = config['iso'], config['aniso'], config['field']
        case = args.root/name
        case.mkdir()
        command = launcher + [str(binary), '-i',
                              str((args.root/'input.athinput').resolve()),
                              '-t', wall,
                              f'mhd/alpha_iso={iso}', f'mhd/alpha_aniso={aniso}',
                              f'mhd/anisotropic_limiter={args.limiter}',
                              f'problem/bx={field[0]}', f'problem/by={field[1]}',
                              f'problem/bz={field[2]}', f'time/tlim={config["time"]}',
                              f'problem/amplitude={config["amplitude"]}']
        for axis, (n, wave) in enumerate(zip(grid, config['mode']), 1):
            command += [f'mesh/nx{axis}={n}', f'meshblock/nx{axis}={max(1, n//2)}',
                        f'problem/mode_{"xyz"[axis-1]}={wave}']
        if name == 'isotropic_no_units':
            lines, in_units = [], False
            for line in (args.root/'input.athinput').read_text().splitlines():
                if line.startswith('<'):
                    in_units = line.strip() == '<units>'
                if not in_units:
                    lines.append(line)
            local_input = case/'input-no-units.athinput'
            local_input.write_text('\n'.join(lines)+'\n')
            command[command.index('-i')+1] = str(local_input.resolve())
        if name == 'legacy_flux_sentinel':
            command += ['mhd/q_limit=3.40282e38']
        if config['sts']:
            command += ['time/sts_integrator=rkl2', 'mhd/conductivity_integrator=sts']
        (case/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        result = dict(status='ERROR', command=command, configuration=config)
        try:
            with (case/'athena.log').open('w') as log:
                completed = subprocess.run(command, cwd=case, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=timeout)
            result['returncode'] = completed.returncode
            if completed.returncode:
                raise RuntimeError('Athena exited nonzero; see athena.log')
            marker = 'transport advection flux checks: 18 PASS'
            log_text = (case/'athena.log').read_text()
            ranks = re.search(r'Number of parallel ranks\s*=\s*(\d+)', log_text)
            if ranks is None or int(ranks[1]) != args.expected_ranks:
                raise RuntimeError('Athena did not report the expected MPI rank count')
            result['verified_ranks'] = int(ranks[1])
            if marker not in log_text:
                raise RuntimeError('Missing device advection-flux unit-test PASS marker')
            metrics = read_metrics(case/'transport.mode.dat')
            rate = conduction_rate(tuple(2*math.pi*n for n in config['mode']),
                                   field, 5/3, iso, aniso)
            energy = 1/(5/3-1)+0.5*sum(b*b for b in field)
            failures = check_mode(metrics, rate, config['time'],
                                  amplitude=config['amplitude'], initial_energy=energy)
            expected = config['amplitude']*math.exp(-rate*config['time'])
            result.update(status='FAIL' if failures else 'PASS', failures=failures,
                          metrics=metrics, rate=rate,
                          expected_cosine=expected)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            result['error'] = str(error)
        report['cases'][name] = result
        (args.root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        detail = result.get('failures', result.get('error', ''))
        print(name, result['status'], detail, flush=True)
    if args.extended:
        # Reject unsupported/invalid requests explicitly, rather than silently
        # treating them as zero transport. A crash is never a valid rejection.
        negative = [
            ('spitzer_missing_coefficient', ['mhd/alpha_spitzer=true'],
             'Spitzer requires exactly one'),
            ('spitzer_sts_missing_coefficient',
             ['mhd/alpha_spitzer=true', 'mhd/conductivity_integrator=sts',
              'time/sts_integrator=rkl2'],
             'Spitzer requires exactly one'),
            ('legacy_flux_limit_rejected', ['mhd/q_limit=1'],
             'q_limit must be finite'),
            ('legacy_flux_limit_sts_rejected',
             ['mhd/q_limit=1', 'mhd/conductivity_integrator=sts',
              'time/sts_integrator=rkl2'],
             'q_limit must be finite'),
            ('negative_flux_limit', ['mhd/q_limit=-1'],
             'q_limit must be finite'),
            ('nan_flux_limit', ['mhd/q_limit=nan'],
             'q_limit must be finite'),
            ('infinite_flux_limit', ['mhd/q_limit=inf'],
             'q_limit must be finite'),
            ('nan_alpha', ['mhd/alpha_aniso=nan'],
             'Conduction requires finite nonnegative alpha'),
            ('infinite_alpha', ['mhd/alpha_iso=inf'],
             'Conduction requires finite nonnegative alpha'),
            ('invalid_limiter', ['mhd/anisotropic_limiter=invalid'],
             'anisotropic_limiter must be vanleer or mc'),
            ('zero_conduction_sts_rejected',
             ['mhd/alpha_iso=0', 'mhd/alpha_aniso=0',
              'mhd/conductivity_integrator=sts', 'time/sts_integrator=rkl2'],
             'STS requires positive thermal conduction'),
            ('anisotropic_sr_rejected', ['mhd/alpha_aniso=0.1',
                                         'coord/special_rel=true'],
             'Anisotropic/nonlinear conduction requires Newtonian physics'),
            ('anisotropic_gr_rejected', ['mhd/alpha_aniso=0.1',
                                         'coord/general_rel=true'],
             'Anisotropic/nonlinear conduction requires Newtonian physics'),
            ('negative_alpha', ['mhd/alpha_aniso=-0.1'],
             'Conduction requires finite nonnegative alpha'),
            ('missing_sts_integrator_rejected',
             ['mhd/alpha_iso=0', 'mhd/alpha_aniso=0.1',
              'mhd/conductivity_integrator=sts', 'time/sts_integrator=none'],
             'selects STS, but <time>/sts_integrator = none')]
        for name, overrides, message in negative:
            if selected and name not in selected:
                continue
            case = args.root/name
            case.mkdir()
            command = launcher + [str(binary), '-i',
                                  str((args.root/'input.athinput').resolve()),
                                  '-t', wall]+overrides
            result = dict(status='ERROR', command=command, required_message=message)
            (case/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            try:
                with (case/'athena.log').open('w') as log:
                    completed = subprocess.run(command, cwd=case, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=timeout)
                text = (case/'athena.log').read_text()
                rejected = completed.returncode == 1 and message in text
                result.update(returncode=completed.returncode,
                              status='PASS' if rejected else 'FAIL')
            except (OSError, subprocess.TimeoutExpired) as error:
                result['error'] = str(error)
            report['cases'][name] = result
            (args.root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
            print(name, result['status'], flush=True)

        parents = (('legacy_flux_sentinel',) if args.guards_only else
                   ('isotropic_n64', 'isotropic_sts', 'anisotropic_parallel',
                    'anisotropic_sts', 'combined_iso_aniso_sts'))
        if selected and 'legacy_flux_sentinel_restart' in selected:
            parents += ('legacy_flux_sentinel',)
        for parent in parents:
            name = parent+'_restart'
            if selected and name not in selected:
                continue
            case = args.root/name
            case.mkdir()
            restart = (args.root/parent/'rst/transport.00001.rst').resolve()
            command = launcher + [str(binary), '-r', str(restart), '-t', wall,
                                  'time/tlim=0.1']
            (case/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            result = dict(status='ERROR', command=command)
            try:
                if report['cases'][parent]['status'] != 'PASS':
                    raise RuntimeError('Restart parent failed validation')
                result['restart_sha256'] = digest(restart)
                with (case/'athena.log').open('w') as log:
                    completed = subprocess.run(command, cwd=case, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=timeout)
                result['returncode'] = completed.returncode
                if completed.returncode:
                    raise RuntimeError('Restart exited nonzero')
                metrics = read_metrics(case/'transport.mode.dat')
                reference = report['cases'][parent]
                failures = check_mode(metrics, reference['rate'], 0.1)
                differences = {k: abs(metrics[k]-reference['metrics'][k])
                               for k in metrics}
                if max(differences.values()) > 1.e-10:
                    failures.append('restart equivalence')
                result.update(status='FAIL' if failures else 'PASS', metrics=metrics,
                              differences=differences, failures=failures)
            except (OSError, ValueError, RuntimeError,
                    subprocess.TimeoutExpired) as error:
                result['error'] = str(error)
            report['cases'][name] = result
            (args.root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
            detail = result.get('failures', result.get('error', ''))
            print(name, result['status'], detail, flush=True)
    missing = sorted(selected-set(report['cases'])) if selected else []
    report['missing_selected_cases'] = missing
    all_pass = bool(report['cases']) and not missing and all(
        r['status'] == 'PASS' for r in report['cases'].values())
    report['status'] = 'PASS' if all_pass else 'FAIL'
    (args.root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
