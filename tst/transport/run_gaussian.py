"""Low-resolution 2D/3D explicit Gaussian verification; no scheduler operations.

Accuracy gates are coarse screening, not production accuracy or a claimed order
of convergence. At N16/N32 the initial sigma has 1.6/3.2 cells. Conservation,
stage coverage, no floors and fixture checks are separate, near-roundoff gates.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from gaussian_oracle import metrics, periodic_gaussian, separable_gaussian, tensor
from run_patch import digest, validate_stages


def configurations(smoke=False):
    result = {}
    for limiter in ('vanleer', 'mc'):
        for dim in (2, 3):
            for n in ((16,) if smoke else (16, 32)):
                for name, iso, aniso in ([('parallel', 0., .1)] if smoke else
                                         [('isotropic', .1, 0.), ('parallel', 0., .1),
                                          ('ratio100', .001, .099)]):
                    key = f'{limiter}_{dim}d_{name}_n{n}'
                    result[key] = dict(dimension=dim, n=n, block=16, limiter=limiter,
                                       gamma=2., rho=1., iso=iso, aniso=aniso,
                                       field=[1., 2., 3.], stop=.02)
        if not smoke:
            baseline = result[f'{limiter}_3d_parallel_n16']
            for name, change in [('reversed', dict(field=[-1., -2., -3.])),
                                 ('blocks8', dict(block=8)), ('density4', dict(rho=4.)),
                                 ('gamma53', dict(gamma=5/3)),
                                 ('axis', dict(field=[1., 0., 0.])),
                                 ('disabled', dict(aniso=0.))]:
                result[f'{limiter}_3d_{name}_n16'] = dict(baseline, **change)
    return result


def validate(case, c):
    data = np.loadtxt(case/'gaussian.gaussian.dat', ndmin=2)
    shape = (c['n'],)*c['dimension']
    if (data.shape != (int(np.prod(shape)), 9) or not np.isfinite(data).all()):
        raise ValueError('malformed or nonfinite cell table')
    # Sort into x,y,z array order, independently of meshblock/output order.
    data = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))]
    axes = [(np.arange(n)+.5)/n for n in shape]
    if c['dimension'] == 2:
        axes.append(np.array([.5]))
    coords = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    if not np.allclose(data[:, :3], coords, rtol=0, atol=1.e-14):
        raise ValueError('missing/duplicate/misplaced cells')
    summary = np.loadtxt(case/'gaussian.gaussian.summary', ndmin=2)
    if summary.shape != (1, 2) or not np.isfinite(summary).all():
        raise ValueError('malformed time/cycle summary')
    endpoint, cycles = summary[0]
    failures = []
    if abs(endpoint-c['stop']) > 1.e-13 or cycles < 1 or cycles != int(cycles):
        failures.append('time/cycle coverage')
    stage_metrics, stage_failures = validate_stages(case, 'gaussian', cycles, 2, 1., 1.1)
    failures += stage_failures
    log = case/'gaussian.log'
    events = log.read_text().splitlines() if log.is_file() else ['MISSING']
    if any(line.strip() and not line.startswith('#') for line in events):
        failures.append('EOS floor/event log')
    initial = separable_gaussian(shape)
    diffusion = tensor(c['dimension'], c['gamma'], c['iso'], c['aniso'], c['field'])
    reference = periodic_gaussian(shape, c['stop'], diffusion)
    observed = data[:, 5].reshape(shape)
    result = metrics(observed, reference, initial)
    result.update(stage_metrics, time=float(endpoint), cycles=int(cycles))
    volume = 1/np.prod(shape)
    if np.max(np.abs(data[:, 3]-volume)) > 1.e-15:
        failures.append('cell volume')
    if np.max(np.abs(data[:, 4]-c['rho'])) > 1.e-12:
        failures.append('density drift')
    if np.max(np.abs(data[:, 6].reshape(shape)-initial)) > 5.e-14:
        failures.append('cell-average initial fixture')
    result['max_divb'] = float(np.max(np.abs(data[:, 8])))
    if result['max_divb'] > 1.e-12:
        failures.append('divB')
    energy = float(np.sum(data[:, 3]*data[:, 7]))
    energy0 = c['rho']*float(np.mean(initial))/(c['gamma']-1)+.5*np.dot(c['field'], c['field'])
    result.update(energy=energy, initial_energy=float(energy0),
                  energy_relative_error=abs(energy-energy0)/energy0)
    if result['energy_relative_error'] > 1.e-11:
        failures.append('total energy conservation')
    if result['heat_error_over_initial_excess'] > 1.e-9:
        failures.append('excess heat conservation')
    if observed.min() < 1.-1.e-10 or observed.max() > 1.1+1.e-10:
        failures.append('final temperature bounds')
    if c['iso']+c['aniso'] == 0 and np.max(np.abs(observed-initial)) > 5.e-14:
        failures.append('disabled transport changed field')
    gate = .10 if c['n'] == 16 else .05
    peak_gate = .05 if c['n'] == 16 else .03
    result['screening_gates'] = dict(l1=gate, l2=gate, linf_over_initial_amplitude=peak_gate)
    for key, limit in result['screening_gates'].items():
        if result[key] > limit:
            failures.append('coarse accuracy screening: '+key)
    return result, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    args.root.mkdir(parents=True, exist_ok=False)
    root = args.root.resolve()
    binary = root/'athena'
    shutil.copy2(args.binary, binary)
    shutil.copy2(source/'tst/inputs/transport_gaussian.athinput', root/'input.athinput')
    shutil.copytree(Path(__file__).parent, root/'checker',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('source.sha256', 'binary.sha256'):
        manifest = args.binary.parent.parent/name
        if manifest.is_file():
            shutil.copy2(manifest, root/name)
    cases = configurations(args.smoke)
    plan = dict(cases=cases, accuracy_gates='N16 L1/L2 <= .10, Linf/amp <= .05; '
                'N32 L1/L2 <= .05, Linf/amp <= .03; two-level error ratio <= .8',
                symmetry_gate=1.e-11, limits='constant tensor, kinematic RK2; '
                'no STS, MPI exchange, nonlinear closure or coupled waves')
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    report = dict(status='RUNNING', cases={}, comparisons={}, rejections={},
                  binary_sha256=digest(binary), plan_sha256=digest(root/'plan.json'),
                  source_sha256={str(p.relative_to(source)): digest(p)
                                 for p in sorted((source/'src').rglob('*')) if p.is_file()})

    def save():
        (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')

    def command(c):
        cmd = [str(binary), '-i', str(root/'input.athinput'), '-t', '00:02:00',
               f'mhd/anisotropic_limiter={c["limiter"]}', f'mhd/gamma={c["gamma"]}',
               f'problem/rho0={c["rho"]}', f'mhd/alpha_iso={c["iso"]}',
               f'mhd/alpha_aniso={c["aniso"]}', f'time/tlim={c["stop"]}']
        for a in range(3):
            n = c['n'] if a < c['dimension'] else 1
            block = c['block'] if a < c['dimension'] else 1
            cmd += [f'mesh/nx{a+1}={n}', f'meshblock/nx{a+1}={block}',
                    f'problem/b{"xyz"[a]}={c["field"][a]}']
        return cmd

    save()
    for name, c in cases.items():
        case = root/name
        case.mkdir()
        result = dict(status='RUNNING', configuration=c, command=command(c))
        report['cases'][name] = result
        save()
        started = time.monotonic()
        print('START', name, flush=True)
        try:
            with (case/'athena.log').open('w') as stream:
                proc = subprocess.run(result['command'], cwd=case, stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=150)
            result['returncode'] = proc.returncode
            if proc.returncode != 0:
                raise ValueError('nonzero Athena exit')
            measured, failures = validate(case, c)
            result.update(status='FAIL' if failures else 'PASS', metrics=measured,
                          failures=failures)
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result.update(status='ERROR', error=str(error))
        result['seconds'] = time.monotonic()-started
        save()
        print(name, result['status'], result.get('failures', result.get('error', '')),
              flush=True)
    if not args.smoke:
        for limiter in ('vanleer', 'mc'):
            for dim in (2, 3):
                for mode in ('isotropic', 'parallel', 'ratio100'):
                    key = f'{limiter}_{dim}d_{mode}'
                    pair = [report['cases'][key+f'_n{n}'] for n in (16, 32)]
                    ratio = (pair[1]['metrics']['l1']/pair[0]['metrics']['l1']
                             if all('metrics' in p for p in pair) else None)
                    report['comparisons'][key+'_refinement'] = dict(
                        status='PASS' if ratio is not None and ratio <= .8 else 'FAIL',
                        l1_fine_over_coarse=ratio, note='Two levels, not asymptotic order')
            base = np.loadtxt(root/f'{limiter}_3d_parallel_n16/gaussian.gaussian.dat')
            base = base[np.lexsort((base[:, 2], base[:, 1], base[:, 0]))]
            for name in ('reversed', 'blocks8', 'density4'):
                data = np.loadtxt(root/f'{limiter}_3d_{name}_n16/gaussian.gaussian.dat')
                data = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))]
                delta = float(np.max(np.abs(base[:, 5]-data[:, 5])))
                report['comparisons'][limiter+'_'+name] = dict(
                    status='PASS' if delta <= 1.e-11 else 'FAIL', temperature_max_delta=delta)
            # Deliberate negative energy on the LAST z plane must be caught
            # before floors. A crash, unrelated error or normal exit is NOT PASS.
            name = limiter+'_last_plane_fault'
            case = root/name
            case.mkdir()
            cmd = command(cases[f'{limiter}_3d_parallel_n16'])
            cmd.append('problem/stage_probe_negative=true')
            with (case/'athena.log').open('w') as stream:
                proc = subprocess.run(cmd, cwd=case, stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=150)
            text = (case/'athena.log').read_text()
            report['rejections'][name] = dict(
                status='PASS' if proc.returncode == 1 and
                'transport raw stage FAILED: stage=1 min=-1' in text else 'FAIL',
                returncode=proc.returncode, command=cmd)
            save()
    checks = [*report['cases'].values(), *report['comparisons'].values(),
              *report['rejections'].values()]
    report['status'] = 'PASS' if all(r['status'] == 'PASS' for r in checks) else 'FAIL'
    save()
    print('GAUSSIAN_MATRIX', report['status'], flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
