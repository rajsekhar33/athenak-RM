"""Standalone nonlinear explicit-conduction tests. No scheduler operations.

Predeclared diagnostic gates: energy/thermal integrals 1e-9 relative; no floors;
cycle extrema within 1e-10; divB <1e-12; ring first moment within 5% of its initial
value of the circular-diffusion reference; fine-ring leakage <30%, decreasing
under refinement; radial-equilibrium normalized L1 <1%, decreasing with resolution.
These short ring runs are not a reproduction of the paper's late-time benchmark.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from result_policy import annotate_report


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def detected_floor_failure(returncode, log):
    return returncode == 1 and re.search(
        r'transport patch extrema/floor check FAILED: [^\n]*events=[1-9][0-9]*', log
    ) is not None


def initial(profile, x, y, cold, hot):
    radius = math.hypot(x, y)
    if profile == 'square':
        inside = 0 < x < 0.5 and 0 < y < 0.5
    elif profile == 'ring':
        inside = 0.5 < radius < 0.7 and x < 0 and abs(y) < -x*math.tan(math.pi/12)
    else:
        return cold+(hot-cold)*math.exp(-((radius-0.6)/0.12)**2)
    return hot if inside else cold


def rows(path, columns):
    result = [[float(v) for v in line.split()] for line in path.read_text().splitlines()
              if line.strip() and not line.startswith('#')]
    if not result or any(len(row) != columns or
                         not all(math.isfinite(v) for v in row) for row in result):
        raise ValueError('missing, malformed or nonfinite diagnostics: '+str(path))
    return result


def validate_stages(case, basename, cycles, stages, lower, upper):
    data = rows(case/(basename+'.stages'), 4)
    failures = []
    if len(data) != stages or any(r[0] != s+1 or r[1] != cycles
                                  for s, r in enumerate(data)) or cycles < 1:
        failures.append('raw stage coverage')
    lo, hi = min(r[2] for r in data), max(r[3] for r in data)
    if lo < lower-1.e-10 or hi > upper+1.e-10:
        failures.append('raw stage extrema')
    return dict(raw_stage_minimum=lo, raw_stage_maximum=hi), failures


def magnetic_energy(config, x, y):
    if config['profile'] == 'square':
        by = 0 if 'problem/by=0' in config['extra'] else -1
        return 0.5*(1+by*by)
    half = 1/config['n']

    def potential(a, b):
        return 0.5*max(0, 0.81-a*a-b*b)

    bx = sum((potential(a, y+half)-potential(a, y-half))/(2*half)
             for a in (x-half, x+half))/2
    by = -sum((potential(x+half, b)-potential(x-half, b))/(2*half)
              for b in (y-half, y+half))/2
    return 0.5*(bx*bx+by*by)


def validate(case, config):
    data = rows(case/'patch.patch.0.dat', 8)
    summaries = rows(case/'patch.patch.summary', 4)
    if len(summaries) != 1:
        raise ValueError('expected one summary row')
    time, samples, minimum, maximum = summaries[0]
    failures = []
    cold, hot = config['cold'], 10.0
    nstages = 1 if 'time/integrator=rk1' in config['extra'] else 2
    stage_metrics, stage_failures = validate_stages(
        case, 'patch', samples-1, nstages, cold, hot)
    failures += stage_failures
    if len(data) != config['n']**2 or len({(r[0], r[1]) for r in data}) != len(data):
        failures.append('cell coverage')
    if abs(time-config['time']) > 1.e-12 or samples < 2:
        failures.append('time/cycle coverage')
    if minimum < cold-1.e-10 or maximum > hot+1.e-10:
        failures.append('cycle extrema')
    events = case/'patch.log'
    if not events.is_file() or any(line.strip() and not line.startswith('#')
                                   for line in events.read_text().splitlines()):
        failures.append('EOS floor/event log')
    volume = math.fsum(r[2] for r in data)
    thermal = math.fsum(r[2]*r[4] for r in data)
    thermal0 = math.fsum(r[2]*initial(config['profile'], r[0], r[1], cold, hot)
                         for r in data)
    energy = math.fsum(r[2]*r[6] for r in data)
    energy0 = thermal0+math.fsum(r[2]*magnetic_energy(config, r[0], r[1]) for r in data)
    if abs(energy-energy0) > 1.e-9*max(1, energy0):
        failures.append('total energy conservation')
    if abs(volume-4) > 1.e-12 or abs(thermal-thermal0) > 1.e-9*max(1, thermal0):
        failures.append('volume/thermal conservation')
    if max(abs(r[3]-1) for r in data) > 1.e-12:
        failures.append('density drift')
    if max(abs(r[7]) for r in data) > 1.e-12:
        failures.append('divB')
    if max(abs(r[5]-initial(config['profile'], r[0], r[1], cold, hot))
           for r in data) > 1.e-12:
        failures.append('initial-condition fixture')
    l1 = math.fsum(r[2]*abs(r[4]-r[5]) for r in data)/(volume*(hot-cold))
    variance = math.fsum(r[2]*(r[4]-thermal/volume)**2 for r in data)
    variance0 = math.fsum(r[2]*(r[5]-thermal0/volume)**2 for r in data)
    if variance > variance0*(1+1.e-10):
        failures.append('variance growth')
    if config['alpha'] == 0 and max(abs(r[4]-r[5]) for r in data) > 1.e-10:
        failures.append('disabled transport changed temperature')
    if config['profile'] == 'square' and config['alpha'] > 0:
        if variance >= 0.99*variance0:
            failures.append('hot square did not diffuse')
    metrics = dict(time=time, cycle_samples=samples, minimum=minimum, maximum=maximum,
                   energy=energy, energy_initial=energy0,
                   thermal=thermal, thermal_initial=thermal0, normalized_l1=l1,
                   variance_ratio=variance/variance0,
                   max_divb=max(abs(r[7]) for r in data))
    metrics.update(stage_metrics)
    if config['profile'] == 'ring':
        initial_excess = thermal0-cold*volume
        leakage = math.fsum(r[2]*(r[4]-cold) for r in data
                            if not 0.5 < math.hypot(r[0], r[1]) < 0.7)/initial_excess
        moment = math.fsum(r[2]*(r[4]-cold)*r[0]/math.hypot(r[0], r[1]) for r in data)
        # gamma=2 and rho=1: chi=alpha. At each radius the angular n=1 mode
        # decays as exp(-chi*t/r^2). Integrate the sampled initial hot sector.
        moment0 = math.fsum(r[2]*(r[5]-cold)*r[0]/math.hypot(r[0], r[1]) for r in data)
        exact = math.fsum(r[2]*(r[5]-cold)*r[0]/math.hypot(r[0], r[1]) *
                          math.exp(-config['alpha']*time/(r[0]**2+r[1]**2)) for r in data)
        error = abs(moment-exact)/abs(moment0)
        metrics.update(leakage=leakage, angular_moment=moment,
                       reference_moment=exact, moment_error=error)
        if error > 0.05 or leakage < -1.e-9 or leakage > 0.30:
            failures.append('ring moment/leakage diagnostic gate')
    if config['profile'] == 'radial' and l1 > 0.01:
        failures.append('radial equilibrium drift')
    return metrics, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--limiter', choices=('vanleer', 'mc'), default='vanleer')
    parser.add_argument('--leakage-study', action='store_true',
                        help='run only extra spatial/time-resolution controls')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    args.root.mkdir(parents=True, exist_ok=False)
    binary = args.root.resolve()/'athena'
    shutil.copy2(args.binary, binary)
    shutil.copy2(source/'tst/inputs/transport_patch.athinput', args.root/'input.athinput')
    shutil.copy2(__file__, args.root/'checker.py')
    shutil.copy2(Path(__file__).with_name('result_policy.py'),
                 args.root/'result_policy.py')
    report = dict(status='RUNNING', binary_sha256=digest(binary), limiter=args.limiter,
                  cases={})

    def spec(name, profile, n=32, cold=0.1, alpha=0.1, time=0.2, extra=()):
        return dict(name=name, profile=profile, n=n, cold=cold, alpha=alpha,
                    time=time, extra=extra)

    cases = [spec('square_disabled', 'square', cold=1.e-5, alpha=0),
             spec('square_diagonal', 'square', cold=1.e-5),
             spec('square_diagonal64', 'square', n=64, cold=1.e-5),
             spec('square_axis', 'square', cold=1.e-5, extra=('problem/by=0',)),
             spec('square_rk1', 'square', cold=1.e-5,
                  extra=('time/integrator=rk1',)),
             spec('ring32', 'ring', time=1.0),
             spec('ring64', 'ring', n=64, time=1.0),
             spec('ring_reversed', 'ring', n=64, time=1.0,
                  extra=('problem/field_sign=-1',)),
             spec('ring_oneblock', 'ring', n=64, time=1.0,
                  extra=('meshblock/nx1=64', 'meshblock/nx2=64')),
             spec('radial32', 'radial', time=0.5),
             spec('radial64', 'radial', n=64, time=0.5),
             spec('ring_restart', 'ring', n=64, time=1.0),
             spec('floor_detection', 'square', extra=('mhd/pfloor=0.2',))]
    if args.leakage_study:
        cases = [spec('radial_disabled64', 'radial', n=64, alpha=0, time=0.5),
                 spec('radial64_halfstep', 'radial', n=64, time=0.5,
                      extra=('time/cfl_number=0.2',)),
                 spec('radial128', 'radial', n=128, time=0.5),
                 spec('ring128', 'ring', n=128, time=1.0)]
    for config in cases:
        name, n = config['name'], config['n']
        case = args.root/name
        case.mkdir()
        command = [str(binary), '-i', str((args.root/'input.athinput').resolve()),
                   '-t', '00:04:00', f'mesh/nx1={n}', f'mesh/nx2={n}',
                   f'meshblock/nx1={n//2}', f'meshblock/nx2={n//2}',
                   f'problem/profile={config["profile"]}',
                   f'problem/cold={config["cold"]}', f'mhd/alpha_aniso={config["alpha"]}',
                   f'mhd/anisotropic_limiter={args.limiter}',
                   f'time/tlim={config["time"]}', f'output2/dt={config["time"]/2}']
        command += list(config['extra'])
        if name == 'ring_restart':
            restart = (args.root/'ring64/rst/patch.00001.rst').resolve()
            command = [str(binary), '-r', str(restart), '-t', '00:04:00', 'time/tlim=1']
        (case/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        result = dict(status='ERROR', configuration=config, command=command)
        try:
            with (case/'athena.log').open('w') as log:
                completed = subprocess.run(command, cwd=case, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=270)
            result['returncode'] = completed.returncode
            if name == 'floor_detection':
                log_text = (case/'athena.log').read_text()
                if not detected_floor_failure(completed.returncode, log_text):
                    raise RuntimeError('deliberate floor was not detected')
                result.update(status='PASS', expected_rejection=True)
                report['cases'][name] = result
                (args.root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
                print(name, 'PASS (deliberate floor detected)', flush=True)
                continue
            if completed.returncode:
                raise RuntimeError('Athena exited nonzero')
            metrics, failures = validate(case, config)
            result.update(metrics=metrics, failures=failures,
                          status='FAIL' if failures else 'PASS')
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            result['error'] = str(error)
        report['cases'][name] = result
        (args.root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, result['status'], result.get('failures', result.get('error', '')),
              flush=True)

    comparisons = {}
    for name, key in [('ring', 'leakage'), ('radial', 'normalized_l1')]:
        if args.leakage_study:
            break
        try:
            lo = report['cases'][name+'32']['metrics'][key]
            hi = report['cases'][name+'64']['metrics'][key]
            comparisons[name+'_refinement'] = dict(
                ratio=hi/lo, status='PASS' if hi < 0.9*lo else 'FAIL')
        except (KeyError, ZeroDivisionError):
            comparisons[name+'_refinement'] = dict(status='ERROR')
    for name in ('ring_reversed', 'ring_oneblock', 'ring_restart'):
        if args.leakage_study:
            break
        try:
            ref = sorted(rows(args.root/'ring64/patch.patch.0.dat', 8))
            other = sorted(rows(args.root/name/'patch.patch.0.dat', 8))
            passing = all(report['cases'][k]['status'] == 'PASS'
                          for k in ('ring64', name))
            if len(ref) != len(other) or not passing:
                raise ValueError('comparison requires complete passing runs')
            delta = max(abs(a-b) for r, s in zip(ref, other) for a, b in zip(r, s))
            comparisons[name] = dict(max_difference=delta,
                                     status='PASS' if delta < 1.e-10 else 'FAIL')
        except (OSError, ValueError):
            comparisons[name] = dict(status='ERROR')
    if args.reference:
        reference = json.loads(args.reference.read_text())
        for name, case in report['cases'].items():
            if name == 'floor_detection':
                continue
            try:
                if reference['status'] != 'PASS' or case['status'] != 'PASS':
                    raise ValueError('device comparison requires passing references')
                delta = max(abs(v-reference['cases'][name]['metrics'][k])
                            for k, v in case['metrics'].items() if k != 'cycle_samples')
                comparisons[name+'_device'] = dict(
                    max_difference=delta, status='PASS' if delta < 1.e-9 else 'FAIL')
            except (KeyError, ValueError):
                comparisons[name+'_device'] = dict(status='ERROR')
    report['comparisons'] = comparisons
    all_results = list(report['cases'].values())+list(comparisons.values())
    passed = all(r['status'] == 'PASS' for r in all_results)
    report['status'] = 'PASS' if passed else 'FAIL'
    annotate_report(report)
    (args.root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(comparisons, indent=2), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
