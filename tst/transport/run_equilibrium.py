"""Single-process forced equilibrium and density-interface budget tests.

Predeclared gates: relative integrated budget 1e-8, source quadrature 1e-9,
steady resistance T/flux 1e-9; transient L1 decays; Sovinec residual/source <1e-3,
isotropic semi-discrete oracle 2e-7; nonlinear L1 <5%, inferred leakage/parallel
<5% and refinement ratio <0.7. No anisotropic STS or MPI acceptance here.
"""
import argparse
import json
import math
import re
from pathlib import Path
import shutil
import subprocess

from run_patch import digest, rows, validate_stages
from result_policy import annotate_report


def detected_stage_failure(returncode, log):
    match = re.search(r'transport raw stage FAILED: stage=1 min=([^ ]+)', log)
    try:
        return returncode == 1 and match is not None and float(match[1]) == -1.0
    except ValueError:
        return False


def reference(config, x, y):
    if config['profile'] == 'sovinec':
        return 1+math.cos(math.pi*x)*math.cos(math.pi*y)
    ratio = config['ratio']
    resistance = x+0.5 if x < 0 else 0.5+x/ratio
    return 1+resistance/(0.5+0.5/ratio)


def magnetic_energy(config, x, y):
    if config['profile'] == 'resistance':
        return 0.5
    half = 0.5/config['n']

    def potential(a, b):
        return math.cos(math.pi*a)*math.cos(math.pi*b)/math.pi

    bx = sum((potential(a, y+half)-potential(a, y-half))/(2*half)
             for a in (x-half, x+half))/2
    by = -sum((potential(x+half, b)-potential(x-half, b))/(2*half)
              for b in (y-half, y+half))/2
    return 0.5*(bx*bx+by*by)


def validate(case, config):
    data = rows(case/'equilibrium.eq.0.dat', 12)
    summaries = rows(case/'equilibrium.eq.summary', 12)
    if len(summaries) != 1:
        raise ValueError('expected one summary')
    (time, start, samples, lo, hi, e0, energy, heat, out,
     budget, rate, phase) = summaries[0]
    stage_metrics, failures = validate_stages(case, 'equilibrium', samples, 2, 1, 2.1)
    n, ny = config['n'], config['ny']
    dx, dy = 1/n, 1/ny
    volume = math.fsum(r[2] for r in data)
    if (len(data) != n*ny or len({(r[0], r[1]) for r in data}) != n*ny or
            abs(volume-1) > 1.e-12):
        failures.append('cell/volume coverage')
    if (abs(time-config['time']) > 1.e-12 or samples < 1 or phase != 0 or
            not 0 <= start < time):
        failures.append('time/RK phase coverage')
    if lo < 1-1.e-9 or hi > 2.1:
        failures.append('temperature bounds')
    event_file = case/'equilibrium.log'
    event_lines = (event_file.read_text().splitlines()
                   if event_file.is_file() else ['missing'])
    if any(line.strip() and not line.startswith('#') for line in event_lines):
        failures.append('EOS floor/event log')
    independent_energy = math.fsum(r[2]*r[6] for r in data)
    if abs(independent_energy-energy) > 1.e-10*max(1, energy):
        failures.append('summary energy mismatch')
    if max(abs(energy-e0-heat+out), budget) > 1.e-8*max(1, e0):
        failures.append('integrated source/boundary energy budget')
    if max(abs(r[11]) for r in data) > 1.e-11:
        failures.append('divergence')
    initial_energy = 0.0
    initial_l1 = 0.0
    source_sum = 0.0
    residual = 0.0
    flux_boundary = 0.0
    l1 = 0.0
    for x, y, vol, rho, t, target, _, xl, xr, yl, yr, _ in data:
        expected = reference(config, x, y)
        expected_rho = 1 if config['profile'] == 'sovinec' or x < 0 else config['ratio']
        if abs(target-expected) > 1.e-12 or abs(rho-expected_rho) > 1.e-12:
            failures.append('fixture/density mismatch')
        perturb = config['perturbation']*math.sin(math.pi*(x+0.5))
        initial_l1 += vol*abs(perturb)
        initial_energy += vol*(rho*(expected+perturb)+magnetic_energy(config, x, y))
        source = (2*math.pi**2*math.cos(math.pi*x)*math.cos(math.pi*y)
                  if config['profile'] == 'sovinec' else 0)
        source_sum += vol*source
        residual = max(residual, abs(source-(xr-xl)/dx-(yr-yl)/dy))
        l1 += vol*abs(t-expected)
        if x < -0.5+dx:
            flux_boundary -= xl*dy
        if x > 0.5-dx:
            flux_boundary += xr*dy
        if config['profile'] == 'sovinec':
            if y < -0.5+dy:
                flux_boundary -= yl*dx
            if y > 0.5-dy:
                flux_boundary += yr*dx
    if start == 0 and abs(initial_energy-e0) > 1.e-10*max(1, e0):
        failures.append('independent initial energy')
    if abs(heat-source_sum*(time-start)) > 1.e-9*max(1, abs(heat)):
        failures.append('RK source integral')
    metrics = dict(time=time, restart_time=start, minimum=lo, maximum=hi, energy=energy,
                   initial_energy=e0, source_heat=heat, outward_heat=out,
                   max_budget_error=budget, l1=l1, final_residual=residual,
                   last_cycle_change_rate=rate, final_source=source_sum,
                   final_outward_flux=flux_boundary)
    metrics.update(stage_metrics)
    if config['profile'] == 'resistance':
        exact_flux = -0.1/(0.5+0.5/config['ratio'])
        flux_error = max(max(abs(r[7]-exact_flux), abs(r[8]-exact_flux),
                             abs(r[9]), abs(r[10])) for r in data)
        metrics.update(steady_flux_error=flux_error, initial_l1=initial_l1)
        if config['perturbation'] == 0:
            if l1 > 1.e-9 or flux_error > 1.e-9:
                failures.append('steady interface solution/flux')
        elif l1 >= 0.999*initial_l1 or out <= 0:
            failures.append('transient heat did not leave the slab')
    else:
        if residual > 1.e-3*(2*math.pi**2) or l1 > 0.05:
            failures.append('Sovinec stationarity/accuracy')
        central = [r[4]-1 for r in data if abs(r[0]) < dx and abs(r[1]) < dy]
        if len(central) != 4:
            failures.append('central sample coverage')
        else:
            # Even grids have no cell at origin. Correct the four central samples
            # by the known cosine factor; matched isotropic controls use the same proxy.
            metrics['central_amplitude'] = math.fsum(central)/4/math.cos(math.pi*dx/2)**2
        if config['aniso'] == 0:
            eigenvalue = 8*math.sin(math.pi*dx/2)**2/dx**2
            steady = 2*math.pi**2/eigenvalue
            amplitude = steady+(1-steady)*math.exp(-eigenvalue*time)
            error = max(abs(r[4]-1-amplitude*math.cos(math.pi*r[0]) *
                            math.cos(math.pi*r[1])) for r in data)
            metrics['isotropic_discrete_error'] = error
            if error > 2.e-7:
                failures.append('isotropic semi-discrete oracle')
    return metrics, sorted(set(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--sovinec-only', action='store_true')
    parser.add_argument('--restart-study', action='store_true')
    parser.add_argument('--strong', action='store_true',
                        help='ratio-10 Sovinec pilot at 8/16, both limiters and restart')
    parser.add_argument('--stage-probe', action='store_true')
    parser.add_argument('--boundary-only', action='store_true',
                        help='bounded density-interface/heat-budget matrix, both limiters')
    args = parser.parse_args()
    if args.boundary_only and any((args.extended, args.sovinec_only,
                                  args.restart_study, args.strong, args.stage_probe)):
        parser.error('--boundary-only is a separate matrix')
    if (args.strong or args.stage_probe) and (
            args.extended or args.sovinec_only or args.restart_study or
            (args.strong and args.stage_probe)):
        parser.error('strong/stage-probe are separate matrices')
    args.root = args.root.resolve()
    source = Path(__file__).resolve().parents[2]
    args.root.mkdir(parents=True, exist_ok=False)
    binary = args.root.resolve()/'athena'
    shutil.copy2(args.binary, binary)
    shutil.copy2(source/'tst/inputs/transport_equilibrium.athinput',
                 args.root/'input.athinput')
    shutil.copytree(Path(__file__).parent, args.root/'checker',
                    ignore=shutil.ignore_patterns('__pycache__'))
    report = dict(status='RUNNING', binary_sha256=digest(binary), cases={})

    def spec(name, profile='resistance', n=32, ratio=10, perturbation=0, aniso=0.1,
             limiter='mc', blocks=1):
        return dict(name=name, profile=profile, n=n, ny=n if profile == 'sovinec' else 8,
                    ratio=ratio, perturbation=perturbation, aniso=aniso, limiter=limiter,
                    time=0.5 if profile == 'sovinec' else 0.05, blocks=blocks)

    cases = [spec('resistance_uniform', ratio=1), spec('resistance_jump10'),
             spec('resistance_jump100', ratio=100), spec('resistance_reverse', ratio=0.1),
             spec('resistance_transient', perturbation=0.02),
             spec('sovinec_iso16', profile='sovinec', n=16, aniso=0),
             spec('sovinec_mc16', profile='sovinec', n=16, aniso=1)]
    if args.extended:
        cases += [spec('resistance_blocks', blocks=2), spec('resistance_restart'),
                  spec('sovinec_iso32', profile='sovinec', n=32, aniso=0),
                  spec('sovinec_mc32', profile='sovinec', n=32, aniso=1),
                  spec('sovinec_vl16', profile='sovinec', n=16,
                       aniso=1, limiter='vanleer'),
                  spec('sovinec_vl32', profile='sovinec', n=32,
                       aniso=1, limiter='vanleer')]
    if args.sovinec_only:
        cases = [c for c in cases if c['profile'] == 'sovinec']
    restart_pairs = {}
    if args.restart_study:
        if args.extended or args.sovinec_only:
            parser.error('--restart-study is a separate matrix')
        cases = [spec('resistance_transient', perturbation=0.02),
                 spec('sovinec_iso16', profile='sovinec', n=16, aniso=0)]
        for base in cases[:]:
            restarted = dict(base, name=base['name']+'_restart')
            restart_pairs[restarted['name']] = base['name']
            cases.append(restarted)
    elif args.extended and not args.sovinec_only:
        restart_pairs['resistance_restart'] = 'resistance_jump10'
    if args.strong:
        cases = []
        for n in (8, 16):
            cases += [spec(f'sovinec_iso{n}', profile='sovinec', n=n, aniso=0),
                      spec(f'sovinec_mc{n}', profile='sovinec', n=n, aniso=9),
                      spec(f'sovinec_vl{n}', profile='sovinec', n=n, aniso=9,
                           limiter='vanleer')]
        cases.append(spec('sovinec_mc16_restart', profile='sovinec', n=16, aniso=9))
        restart_pairs['sovinec_mc16_restart'] = 'sovinec_mc16'
    if args.stage_probe:
        cases = [spec('stage_control', ratio=1), spec('stage_negative', ratio=1)]
    if args.boundary_only:
        cases = []
        for limiter in ('vanleer', 'mc'):
            for suffix, changes in (
                    ('uniform', dict(ratio=1)), ('jump10', {}),
                    ('jump100', dict(ratio=100)), ('reverse', dict(ratio=.1)),
                    ('blocks', dict(blocks=2)), ('transient', dict(perturbation=.02))):
                cases.append(spec(f'resistance_{limiter}_{suffix}',
                                  limiter=limiter, **changes))
            parent = cases[-1]
            resumed = dict(parent, name=parent['name']+'_restart')
            restart_pairs[resumed['name']] = parent['name']
            cases.append(resumed)
    for config in cases:
        name = config['name']
        case = args.root/name
        case.mkdir()
        sovinec = config['profile'] == 'sovinec'
        command = [str(binary), '-i', str((args.root/'input.athinput').resolve()),
                   '-t', '00:09:00', f'mesh/nx1={config["n"]}',
                   f'mesh/nx2={config["ny"]}',
                   f'meshblock/nx1={config["n"]//config["blocks"]}',
                   f'meshblock/nx2={config["ny"]}',
                   f'mesh/ix2_bc={"user" if sovinec else "periodic"}',
                   f'mesh/ox2_bc={"user" if sovinec else "periodic"}',
                   f'mhd/alpha_iso={1 if sovinec else 0}',
                   f'mhd/alpha_aniso={config["aniso"]}',
                   f'mhd/anisotropic_limiter={config["limiter"]}',
                   f'problem/profile={config["profile"]}',
                   f'problem/density_ratio={config["ratio"]}',
                   f'problem/perturbation={config["perturbation"]}',
                   f'time/tlim={config["time"]}', f'output2/dt={config["time"]/2}']
        if name == 'stage_negative':
            command.append('problem/stage_probe_negative=true')
        if name in restart_pairs:
            restart = args.root/restart_pairs[name]/'rst/equilibrium.00001.rst'
            restart = restart.resolve()
            command = [str(binary), '-r', str(restart), '-t', '00:09:00',
                       f'time/tlim={config["time"]}']
        (case/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        result = dict(status='ERROR', configuration=config, command=command)
        try:
            with (case/'athena.log').open('w') as log:
                completed = subprocess.run(command, cwd=case, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=570)
            result['returncode'] = completed.returncode
            if name == 'stage_negative':
                if not detected_stage_failure(completed.returncode,
                                              (case/'athena.log').read_text()):
                    raise RuntimeError('deliberate negative raw stage not detected')
                if any(line.strip() and not line.startswith('#')
                       for line in (case/'equilibrium.log').read_text().splitlines()):
                    raise RuntimeError('floor/event occurred before raw-stage rejection')
                result.update(status='PASS', expected_rejection=True)
                report['cases'][name] = result
                (args.root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
                print(name, 'PASS (negative raw stage detected)', flush=True)
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
    if args.restart_study or args.strong or args.boundary_only:
        for name, baseline in restart_pairs.items():
            try:
                a = sorted(rows(args.root/baseline/'equilibrium.eq.0.dat', 12))
                b = sorted(rows(args.root/name/'equilibrium.eq.0.dat', 12))
                if len(a) != len(b):
                    raise ValueError('cell counts differ')
                delta = max(abs(x-y) for r, s in zip(a, b) for x, y in zip(r, s))
                resumed = report['cases'][name]['metrics']['restart_time'] > 0
                complete = all(report['cases'][k]['status'] == 'PASS'
                               for k in (name, baseline))
                comparisons[name] = dict(
                    max_difference=delta, resumed=resumed, complete=complete,
                    status='PASS' if delta < 1.e-9 and resumed and complete else 'FAIL')
            except (OSError, ValueError, KeyError):
                comparisons[name] = dict(status='ERROR')
    if args.extended or args.strong:
        for limiter in ('mc', 'vl'):
            try:
                errors = []
                for n in ((8, 16) if args.strong else (16, 32)):
                    iso_metrics = report['cases'][f'sovinec_iso{n}']['metrics']
                    iso = iso_metrics['central_amplitude']
                    case = report['cases'][f'sovinec_{limiter}{n}']
                    a = case['metrics']['central_amplitude']
                    leak = abs(1/a-1/iso)
                    case['metrics']['inferred_perpendicular_leakage'] = leak
                    errors.append(leak)
                ratio = errors[1]/errors[0]
                comparisons[limiter+'_leakage'] = dict(
                    coarse=errors[0], fine=errors[1], ratio=ratio,
                    parallel=10 if args.strong else 2,
                    status='PASS' if ratio < 0.7 and
                    errors[1]/(10 if args.strong else 2) < 0.05 else 'FAIL')
            except (KeyError, ZeroDivisionError):
                comparisons[limiter+'_leakage'] = dict(status='ERROR')
        if not args.sovinec_only and not args.strong:
            for name in ('resistance_blocks', 'resistance_restart'):
                try:
                    ref_file = args.root/'resistance_jump10/equilibrium.eq.0.dat'
                    a = sorted(rows(ref_file, 12))
                    b = sorted(rows(args.root/name/'equilibrium.eq.0.dat', 12))
                    if len(a) != len(b):
                        raise ValueError('cell counts differ')
                    delta = max(abs(x-y) for r, s in zip(a, b) for x, y in zip(r, s))
                    comparisons[name] = dict(max_difference=delta,
                                             status='PASS' if delta < 1.e-9 else 'FAIL')
                except (OSError, ValueError):
                    comparisons[name] = dict(status='ERROR')
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
