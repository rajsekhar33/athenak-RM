#!/usr/bin/env python3
"""Small ideal-gas hyper4 regression for PROBLEM=tests/hyperviscosity_probe.

Requires NumPy and a double-precision single-rank CPU or GPU executable.
Optional --launcher 'mpirun -np 1' is split without invoking a shell. --gpu
records the requested backend; it does not select a device or certify MPI.
--selfcheck performs only algebra/parser negative controls, never native runs.
The 1D operator check includes nonzero higher-stress work J. Circular shear
uses exact FV data and RK2/Legendre polynomials at checked timesteps. This is
not a multidimensional/AMR oracle; no builds or scheduler actions are included.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
import traceback

import numpy as np

N, MU, AMP, STOP, CFL, RATIO = 32, 1e-4, .8, .1, .3, 8.
GATES = dict(initial=2e-13, flux_absolute=5e-12, flux_relative=5e-10,
             polynomial_relative=5e-10, invariant=5e-11, budget=1e-11,
             ct=5e-12, time=2e-13, timestep_relative=2e-12,
             printed_relative=5.1e-7)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def close(a, b, relative=2e-12, absolute=2e-15):
    return np.isfinite([a, b]).all() and abs(a-b) <= absolute+relative*max(abs(a), abs(b))


def input_text(owner, mode='explicit', evolve=False, mu=MU):
    groups = dict(job=dict(basename='hyper4'), mesh=dict(nghost=2),
                  meshblock={}, time=dict(evolution='dynamic', integrator='rk2',
                  sts_integrator='rkl2' if mode == 'sts' else 'none',
                  sts_max_dt_ratio=RATIO, cfl_number=CFL, tlim=STOP,
                  nlim=-1 if evolve else 0, ndiag=1))
    for d, size in enumerate((N, 1, 1), 1):
        groups['mesh'].update({f'nx{d}': size, f'x{d}min': 0, f'x{d}max': 1,
                               f'ix{d}_bc': 'periodic', f'ox{d}_bc': 'periodic'})
        groups['meshblock'][f'nx{d}'] = size
    groups[owner] = dict(eos='ideal', gamma=5/3, iso_sound_speed=1,
                         reconstruct='plm', rsolver='hlld' if owner == 'mhd' else 'hllc',
                         fofc='false', viscosity_integrator=mode,
                         nu_iso=0, nu_aniso=0, mu_hyper4=mu)
    groups['problem'] = dict(circular_shear=str(evolve).lower(), variable_rho='false',
                             amplitude=AMP, circular_mode1=2, circular_mode2=0,
                             circular_mode3=0, state_output='state',
                             flux_output='flux.txt')
    return ''.join(f'<{group}>\n'+''.join(f'{k}={v}\n' for k, v in values.items())
                   for group, values in groups.items())


def metadata(text):
    def match(pattern):
        found = re.findall(pattern, text, re.M)
        require(len(found) == 1, 'missing/duplicate snapshot metadata')
        return found[0]
    clock = match(r'^# cycle (\d+) dt (\S+)$')
    result = dict(time=float(match(r'^# time (\S+)$')),
                  cycle=int(clock[0]), dt=float(clock[1]))
    require(np.isfinite(list(result.values())).all() and result['dt'] > 0, 'bad clock')
    return result


def arrange(rows, count=N, offset=0):
    require(rows.shape == (count, 12+offset) and np.isfinite(rows).all(), 'bad state')
    expected = np.column_stack((np.zeros(count), np.zeros(count), np.arange(count)+2))
    require(np.array_equal(rows[:, offset:offset+3], expected), 'bad cell indices')
    return rows[:, offset+3:].T


def state(path):
    text = path.read_text()
    meta = metadata(text)
    events = re.findall(r'^# events (.+)$', text, re.M)
    require(len(events) == 1 and [int(v) for v in events[0].split()] == [0]*5,
            'missing event counters or nonzero EOS/floor/FOFC events')
    values = arrange(np.loadtxt(path, ndmin=2))
    require(np.min(values[0]) > 0 and np.min(internal(values)) > 0, 'bad raw state')
    return values, meta


def internal(values):
    kinetic = .5*np.sum(values[1:4]**2, axis=0)/values[0]
    return values[4]-kinetic-.5*np.sum(values[5:8]**2, axis=0)


def ct_faces(path, meta, values):
    require(metadata(path.read_text()) == meta, 'CT snapshot clock differs')
    rows = np.loadtxt(path, ndmin=2)
    keys = [(0, 0, 0, i) for i in range(2, N+3)]
    keys += [(1, 0, j, i) for j in range(2) for i in range(2, N+2)]
    keys += [(2, k, 0, i) for k in range(2) for i in range(2, N+2)]
    require(rows.shape == (len(keys), 5) and np.isfinite(rows).all()
            and np.array_equal(rows[:, :4], keys), 'missing/duplicate/truncated CT faces')
    wanted = np.array([.3 if key[0] == 1 else 0. for key in keys])
    divb = N*np.diff(rows[:N+1, 4])
    require(max(np.max(abs(rows[:, 4]-wanted)), np.max(abs(divb)),
                np.max(abs(divb-values[8]))) <= GATES['ct'], 'CT mismatch')


def initial(owner, circular):
    phase = 2*np.pi*(np.arange(N)+.5)/N
    values = np.zeros((9, N))
    values[0] = 1
    if circular:
        values[2] = AMP*np.sinc(2/N)*np.cos(2*phase)
        values[3] = AMP*np.sinc(2/N)*np.sin(2*phase)
    else:
        for mode, amplitude in [(1, [.2, -.3, .4]), (2, [-.1, .2, .15])]:
            values[1:4] += np.array(amplitude)[:, None]*np.sinc(mode/N)*np.cos(mode*phase)
    values[6] = .3 if owner == 'mhd' else 0
    kinetic = AMP**2 if circular else np.sum(values[1:4]**2, axis=0)
    values[4] = 1.5+.5*kinetic+.5*values[6]**2
    return values


def operator_flux(velocity, mu):
    # Fluxes on the RIGHT face of each cell; native dump starts at the LEFT.
    right = np.roll(velocity, -1, axis=1)
    derivative = N*(right-velocity)

    def lap(a):
        return N*N*(np.roll(a, -1, axis=-1)-2*a+np.roll(a, 1, axis=-1))
    weights = np.array([4/3, 1., 1.])[:, None]
    traction = -mu*weights*lap(derivative)
    centered = .5*N*(right-np.roll(velocity, 1, axis=1))
    extra = mu*N*np.sum(weights*derivative*(np.roll(centered, -1, axis=1)-centered), 0)
    conventional = np.sum(.5*(right+velocity)*traction, axis=0)
    result = np.vstack((np.zeros(N), -traction, -conventional-extra))
    result = np.roll(result, 1, axis=1)
    return np.column_stack((result, result[:, 0])), extra


def kernel(directory, owner, mu):
    values, meta = state(directory/'state.initial')
    require(meta['cycle'] == 0 and meta['time'] == 0 and
            np.max(abs(values-initial(owner, False))) <= GATES['initial'], 'wrong IC')
    rows = np.loadtxt(directory/'flux.txt', ndmin=2)
    keys = np.column_stack((np.zeros((N+1, 3)), np.arange(N+1)+2))
    require(rows.shape == (N+1, 9) and np.isfinite(rows).all() and
            np.array_equal(rows[:, :4], keys), 'invalid/missing operator faces')
    expected, extra = operator_flux(values[1:4], mu)
    error, scale = np.max(abs(rows[:, 4:].T-expected)), np.max(abs(expected))
    require(error <= GATES['flux_absolute']+GATES['flux_relative']*scale, 'wrong flux')
    matches = re.findall(r'^# dtnew (\S+)$', (directory/'flux.txt').read_text(), re.M)
    require(len(matches) == 1, 'missing/duplicate operator timestep')
    dt = float(matches[0])
    wanted = 1/((32/3)*mu*N**4) if mu else float(np.finfo(np.float32).max)
    require(close(dt, wanted, GATES['timestep_relative'], 0.), 'wrong timestep bound')
    if owner == 'mhd':
        ct_faces(directory/'state.initial.faces', meta, values)
    require(mu != 0 or np.all(rows[:, 4:] == 0), 'disabled hyper4 has nonzero flux')
    return dict(passed=True, face_max_error=float(error), dtnew=dt,
                nonzero_J_max=float(np.max(abs(extra))))


def stages(ratio):
    result = max(3, math.ceil((math.sqrt(9+16*ratio)-1)/2))
    return result+int(result % 2 == 0)


def polynomial(z, count):
    den = count*count+count-2
    b = den/(2*count*(count+1))
    coefficients = np.zeros(count+1)
    coefficients[count] = 1
    return float(1-b+b*np.polynomial.legendre.legval(1+4*z/den, coefficients))


def evolution(directory, owner, mode):
    first, m0 = state(directory/'state.initial')
    last, mf = state(directory/'state.final')
    require(m0['time'] == m0['cycle'] == 0 and
            abs(mf['time']-STOP) <= GATES['time'], 'incomplete evolution')
    require(np.max(abs(first-initial(owner, True))) <= GATES['initial'], 'wrong FV IC')
    bound, elapsed, steps = CFL/((32/3)*MU*N**4), 0., []
    while elapsed < STOP:
        dt = min(bound*(RATIO if mode == 'sts' else 1), STOP-elapsed)
        steps.append(dt)
        elapsed += dt
    require(mf['cycle'] == len(steps), 'unexpected cycle count')
    text = (directory/'run.log').read_text()
    records = re.findall(r'\bcycle=(\d+) time=(\S+) dt=(\S+)', text)
    require([int(r[0]) for r in records] == list(range(len(steps)+1)), 'missing cycles')
    elapsed = 0.
    for row, dt in zip(records, steps):
        require(close(float(row[1]), elapsed, GATES['printed_relative'], 1e-14)
                and close(float(row[2]), dt, GATES['printed_relative'], 0.), 'wrong dt')
        elapsed += dt
    lam = MU*(2*N*np.sin(2*np.pi/N))**4
    factor = 1.
    telemetry = [line for line in text.splitlines() if line.startswith('viscosity_sts')]
    require(len(telemetry) == (3*len(steps) if mode == 'sts' else 0), 'missing STS')
    for cycle, dt in enumerate(steps):
        if mode == 'explicit':
            z = -lam*dt
            factor *= 1+z+.5*z*z
            continue
        lines = telemetry[3*cycle:3*cycle+3]
        pre, cfl, post = [dict(re.findall(r'(\w+)=(\S+)', line)) for line in lines]
        for row, name in ((pre, 'pre'), (post, 'post')):
            count = stages(.5*dt/bound)
            require(row == dict(cycle=str(cycle), sweep=name,
                    interval=row.get('interval'),
                    stages=str(count), checked_stages=str(count), status='accepted',
                    diffusion_bound=row.get('diffusion_bound'), reason='none'), 'bad STS')
            require(close(float(row['interval']), dt/2) and
                    close(float(row['diffusion_bound']), bound), 'wrong STS bound')
        require(set(cfl) == {'cycle', 'attempt', 'dt', 'bound', 'status', 'next_dt'} and
                cfl['cycle'] == str(cycle) and cfl['attempt'] == '0' and
                cfl['status'] == 'accepted' and
                close(float(cfl['dt']), dt) and close(float(cfl['next_dt']), dt) and
                np.isfinite(float(cfl['bound'])) and
                float(cfl['bound']) >= dt*(1-2e-12), 'invalid cycle-CFL acceptance')
        factor *= polynomial(-lam*dt/2, count)**2
    expected = first.copy()
    expected[1:4] *= factor
    heat = float(np.mean(internal(expected)-internal(first)))
    momentum = float(np.linalg.norm(last[1:4]-expected[1:4])/np.linalg.norm(first[1:4]))
    thermal = float(np.sqrt(np.mean((internal(last)-internal(expected))**2))/heat)
    budget = float(np.max(abs(np.mean(last[:5]-first[:5], axis=1))))
    require(0 < factor < 1 and heat > 0 and
            max(momentum, thermal) <= GATES['polynomial_relative'], 'failed oracle')
    invariant = [0, 1, 4, 5, 6, 7, 8]
    require(np.max(abs(last[invariant]-first[invariant])) <= GATES['invariant']
            and budget <= GATES['budget'], 'circular invariants/conserved budget differ')
    if owner == 'mhd':
        for suffix, meta, values in [('initial', m0, first), ('final', mf, last)]:
            ct_faces(directory/f'state.{suffix}.faces', meta, values)
    return dict(passed=True, momentum_relative=momentum, generated_heat_relative=thermal,
                expected_heat=heat, polynomial_factor=factor,
                budget_absolute=budget, cycles=len(steps))


def selfcheck():
    velocity = initial('hydro', False)[1:4]
    flux, extra = operator_flux(velocity, MU)
    force = -N*np.diff(flux[1:4], axis=1)
    heat = -N*np.diff(flux[4])-np.sum(velocity*force, axis=0)
    lap = N*N*(np.roll(velocity, -1, axis=1)-2*velocity+np.roll(velocity, 1, axis=1))
    wanted_heat = MU*np.sum(np.array([4/3, 1., 1.])[:, None]*lap**2, axis=0)
    require(np.max(abs(heat-wanted_heat)) < 2e-12, 'heat identity failed')
    gate = GATES['flux_absolute']+GATES['flux_relative']*np.max(abs(flux))
    require(np.max(abs(extra)) > 100*gate, 'missing-J control is not discriminating')
    require(np.max(abs(flux)) > 100*gate, 'missing-hyper control is not discriminating')
    for mode in (1, 2):
        v = np.array([.2, -.3, .4])[:, None]*np.cos(2*np.pi*mode*(np.arange(N)+.5)/N)
        expected = -MU*(2*N*np.sin(np.pi*mode/N))**4*np.array([4/3, 1., 1.])[:, None]*v
        actual = -N*np.diff(operator_flux(v, MU)[0][1:4], axis=1)
        require(np.max(abs(actual-expected)) < 1e-11, 'tensor symbol differs')
    x = np.polynomial.Polynomial([1., 1/7])
    p5 = 8/15+(7/15)*(63*x**5-70*x**3+15*x)/8
    error = max(abs(polynomial(z, 5)-p5(z)) for z in np.linspace(-7, 0, 17))
    require(error < 1e-13, 'RKL5 polynomial differs')
    rows = np.column_stack((np.zeros((N, 2)), np.arange(N)+2, initial('hydro', True).T))
    require(np.array_equal(arrange(rows), initial('hydro', True)), 'valid parser failed')
    require(np.max(abs(arrange(rows)-initial('mhd', True))) > GATES['initial'],
            'wrong-owner control is not discriminating')
    faults = [rows[:-1], np.vstack((rows, rows[-1])), rows[::-1]]
    faults += [rows.copy(), rows.copy()]
    faults[-2][0, 2] = 2.5
    faults[-1][0, 3] = np.nan
    for bad in faults:
        try:
            arrange(bad)
        except ValueError:
            continue
        raise ValueError('malformed state escaped detection')
    return dict(passed=True, native_executed=False, rejected_parser_faults=len(faults),
                nonzero_J_discriminating=True, tensor_symbol_and_heat_identity=True)


def launch(binary, launcher, directory, text, reject=None):
    directory.mkdir()
    path = directory/'input.athinput'
    path.write_text(text)
    before = sha(path)
    command = launcher+[str(binary), '-i', path.name, '-t', '00:04:30']
    start = time.monotonic()
    with (directory/'run.log').open('x') as log:
        child = subprocess.Popen(command, cwd=directory, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=300)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
            raise
    log = (directory/'run.log').read_text()
    require(before == sha(path), 'input changed during execution')
    good = (code > 0 and reject is not None and reject in log
            and not (directory/'state.initial').exists())
    require(good if reject else code == 0, 'wrong exit/guard or reached initializer')
    return dict(command=command, returncode=code, seconds=time.monotonic()-start,
                input_sha256=before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--launcher', default='')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    report = dict(passed=False, native_executed=False, gates=GATES, cases={},
                  requested_backend='gpu' if args.gpu else 'cpu',
                  script_sha256=sha(__file__))
    binary_hash = None
    if args.output:
        args.output = args.output.resolve()
        require(not args.output.exists(), 'output must be a new directory')
        args.output.mkdir(parents=True)
    try:
        report['selfcheck'] = selfcheck()
        if not args.selfcheck:
            require(args.output is not None and args.binary is not None,
                    '--binary and --output are required')
            args.binary = args.binary.resolve(strict=True)
            binary_hash = sha(args.binary)
            report.update(binary=str(args.binary), binary_sha256_before=binary_hash)
            cases = [(f'kernel-{owner}-{mu}', owner, 'explicit', False, mu, None, None)
                     for owner, mu in [('hydro', MU), ('hydro', 0), ('mhd', MU)]]
            cases += [('shear-'+owner+'-'+mode, owner, mode, True, MU, None, None)
                      for owner in ('hydro', 'mhd') for mode in ('explicit', 'sts')]
            for name, replacement, message in [
                    ('negative', ('mu_hyper4=0.0001', 'mu_hyper4=-1'),
                     'Viscosity coefficients must be finite and nonnegative'),
                    ('nonfinite', ('mu_hyper4=0.0001', 'mu_hyper4=nan'),
                     'Viscosity coefficients must be finite and nonnegative'),
                    ('isothermal', ('eos=ideal', 'eos=isothermal'),
                     'mu_hyper4 requires ideal-gas'),
                    ('nonperiodic', ('ix1_bc=periodic\nox1_bc=periodic',
                                     'ix1_bc=outflow\nox1_bc=outflow'),
                     'mu_hyper4 requires ideal-gas')]:
                cases.append((f'reject-{name}', 'hydro', 'explicit', False,
                              MU, replacement, message))
            for name, owner, mode, evolve, mu, replacement, rejection in cases:
                text = input_text(owner, mode, evolve, mu)
                if replacement:
                    require(text.count(replacement[0]) == 1, 'guard template differs')
                    text = text.replace(*replacement)
                directory = args.output/name
                report['cases'][name] = dict(passed=False, status='RUNNING')
                report['native_executed'] = True
                save(args.output/'report.json', report)
                result = launch(args.binary, shlex.split(args.launcher),
                                directory, text, rejection)
                result.update(dict(passed=True, expected_rejection=rejection)
                              if rejection else evolution(directory, owner, mode)
                              if evolve else kernel(directory, owner, mu))
                result['files'] = {p.name: sha(p) for p in directory.iterdir()
                                   if p.is_file()}
                report['cases'][name] = result
                save(args.output/'report.json', report)
            require(len(report['cases']) == 11 and
                    all(c['passed'] for c in report['cases'].values()), 'partial matrix')
        report['passed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
    finally:
        try:
            if binary_hash:
                report['binary_sha256_after'] = sha(args.binary)
                report['passed'] &= binary_hash == report['binary_sha256_after']
            report['passed'] &= report['script_sha256'] == sha(__file__)
            for name, row in report['cases'].items():
                for filename, digest in row.get('files', {}).items():
                    require(sha(args.output/name/filename) == digest, 'artifact changed')
        except Exception:
            report.update(passed=False, provenance_error=traceback.format_exc())
        if args.output:
            save(args.output/'report.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
