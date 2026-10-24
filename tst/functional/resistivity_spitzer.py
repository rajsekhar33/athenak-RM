#!/usr/bin/env python3
"""Explicit Spitzer induction/heating against an independent nonlinear ODE."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.integrate import solve_ivp

import resistivity as h
AMP, PRES, GAMMA, END = 1., .1, 5./3., .1
ETA0 = .25*PRES**1.5


def reference(k2):
    def rhs(t, amplitude):
        temp = PRES+(GAMMA-1)*.5*(AMP**2-amplitude[0]**2)
        return [-k2*ETA0*temp**(-1.5)*amplitude[0]]
    sol = solve_ivp(rhs, (0., END), [AMP], method='DOP853', rtol=2e-13, atol=2e-14,
                    dense_output=True)
    h.require(sol.success, 'Independent ODE reference failed')
    return sol.sol


def launch(
        args,
        name,
        n=64,
        cfl=.4,
        axis=1,
        units_scale=None,
        overrides=(),
        fail_text=None,
        restart=None):
    run = args.root/name
    run.mkdir()
    length = 1. if units_scale is None else 1./units_scale
    text = h.input_text(n, 'explicit', axis, 16, cfl)
    text = text.replace('amp = 0.5', f'amp = {AMP}').replace(
        'pres = 1.0', f'pres = {PRES}')
    text = text.replace('eta_ohm = 0.25', 'eta_ohm = 0.0\neta_spitzer = true')
    if units_scale is None:
        text += f'\n<mhd>\nspitzer_eta0_code = {ETA0:.17g}\n'
    else:
        # The same physical box and time are represented in two unit systems.
        # L0 and t0 scale together, preserving T0 and velocity/pressure units.
        L0, t0, mu = 2.5e11, 2.5e3, .6
        T0 = (L0/t0)**2*mu*1.660538921e-24/1.3806488e-16
        eta0_cgs = ETA0*L0**2/t0*T0**1.5
        text += (f'\n<mhd>\nspitzer_eta0_cgs = {eta0_cgs:.17g}\n'
                 f'<units>\nlength_cgs = {L0*units_scale:.17g}\n'
                 f'time_cgs = {t0*units_scale:.17g}\n'
                 f'mass_cgs = {units_scale**3:.17g}\nmu = {mu}\n')
        for a in (1, 2, 3):
            text = text.replace(f'x{a}max = 1\n', f'x{a}max = {length:.17g}\n')
        text = text.replace('tlim = 0.1', f'tlim = {END/units_scale:.17g}')
        text = text.replace('dt = 0.025', f'dt = {END/4/units_scale:.17g}')
    for override in overrides:
        block_key, value = override.split('=', 1)
        block, key = block_key.split('/', 1)
        text += f'\n<{block}>\n{key} = {value}\n'
    (run/'input.athinput').write_text(text)
    command = ['mpirun', '-np', str(args.ranks), str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
               '-i', 'input.athinput', *overrides]
    if restart is not None:
        command += ['-r', os.path.relpath(restart, run)]
    (run/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (run/'athena.log').open('x') as log:
        result = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT,
                                timeout=270)
    if fail_text is not None:
        h.require(result.returncode != 0 and fail_text in (run/'athena.log').read_text(),
                  'Missing expected rejection: '+name)
        return dict(status='PASS', name=name, expected_rejection=fail_text), None
    h.require(result.returncode == 0, 'Athena failed: '+str(run))
    states = [h.checkpoint(p) for p in sorted((run/'rst').glob('*.rst'))]
    scale = 1. if units_scale is None else units_scale
    h.require(states and abs(states[-1]['time']*scale-END) < 1e-13, 'Incomplete run')
    continuum, discrete = reference((2*np.pi)**2), reference(4*n**2*np.sin(np.pi/n)**2)
    rows = []
    x = (np.arange(n)+.5)/n
    sh = [1, 1, 1]
    sh[3-axis] = n
    phase = 2*np.pi*x.reshape(sh)
    E0 = PRES/(GAMMA-1)+.5*AMP**2
    for state in states:
        t = state['time']*scale
        a = float(continuum(t)[0])
        b, c = state['bcc'][axis % 3], state['bcc'][(axis+1) % 3]
        measured = float(np.mean(b*np.cos(phase)+c*np.sin(phase)))
        error = float(np.sqrt(np.mean((b-a*np.cos(phase))**2+(c-a*np.sin(phase))**2)))/AMP
        heat = .5*(AMP**2-a**2)
        row = dict(time=t,
                   amplitude=measured,
                   expected_amplitude=a,
                   relative_field_l2=error,
                   temporal_amplitude_error=abs(measured-float(discrete(t)[0])),
                   thermal_gain=state['thermal']-PRES/(GAMMA-1),
                   expected_thermal_gain=heat,
                   total_relative_error=abs(state['total']-E0)/E0,
                   divb=state['divb'],
                   face_seam=state['seam'],
                   periodic_face=state['periodic'],
                   cycle=state['cycle'])
        h.require(error < .5*(2*np.pi/n)**2, 'Spitzer field error')
        h.require(row['total_relative_error'] < 512*np.finfo(float).eps *
                  (state['cycle']+1), 'Spitzer total energy conservation')
        h.require(abs(row['thermal_gain']-heat)/(.5*AMP**2) < (2*np.pi/n)**2,
                  'Spitzer Joule heating error')
        h.require(max(state['seam'], state['periodic']) < 128 *
                  np.finfo(float).eps*(state['cycle']+1), 'Spitzer CT face mismatch')
        h.require(state['divb'] < 4096*np.finfo(float).eps*n*(state['cycle']+1),
                  'Spitzer CT divergence')
        rows.append(row)
    record = dict(name=name, status='PASS', n=n, cfl=cfl, axis=axis,
                  units_scale=units_scale, samples=rows)
    (run/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(name, rows[-1], flush=True)
    return record, states[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--ranks', type=int, default=1)
    p.add_argument('--extended', action='store_true')
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--binary-manifest', type=Path)
    args = p.parse_args()
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    for manifest in (args.source_manifest, args.binary_manifest):
        if manifest is not None:
            subprocess.run(['sha256sum', '--quiet', '-c', str(manifest)], check=True)
    report = dict(
        status='RUNNING', phase='explicit Spitzer closure', cases=[], binary_sha256=h.sha(
            args.binary), source_manifest_sha256=h.sha(
            args.source_manifest) if args.source_manifest else None)
    (
        args.root /
        'gates.json').write_text(
        json.dumps(
            dict(
                field='0.5*(2*pi/N)^2 relative to initial amplitude',
                spatial_ratio=.35,
                temporal_ratio=.35,
                energy='512*eps*(cycles+1)',
                units_maxabs=1e-12),
            indent=2) +
        '\n')
    try:
        states, errors = {}, []
        for n in (32, 64, 128):
            result, state = launch(args, f'n{n}', n=n)
            report['cases'].append(result)
            states[n] = state
            errors.append(result['samples'][-1]['relative_field_l2'])
        ratios = [b/a for a, b in zip(errors, errors[1:])]
        h.require(max(ratios) < .35, 'Spitzer spatial convergence '+str(ratios))
        report['spatial_ratios'] = ratios
        if args.extended:
            temporal = [report['cases'][1]['samples'][-1]['temporal_amplitude_error']]
            for cfl in (.2, .1):
                result, _ = launch(args, f'time-cfl{cfl}', cfl=cfl)
                report['cases'].append(result)
                temporal.append(result['samples'][-1]['temporal_amplitude_error'])
            ratios = [b/a for a, b in zip(temporal, temporal[1:])]
            report.update(temporal_errors=temporal, temporal_ratios=ratios)
            h.require(max(ratios) < .35, 'Spitzer temporal convergence '+str(ratios))
            for scale in (1., 2.):
                result, state = launch(args, f'cgs-scale{scale}', units_scale=scale)
                report['cases'].append(result)
                difference = max(
                    float(np.max(abs(state['u']-states[64]['u']))),
                    *(float(np.max(abs(a-b)))
                      for a, b in zip(state['faces'], states[64]['faces'])))
                report.setdefault('unit_rescaling_maxabs', []).append(difference)
                h.require(difference < 1e-12, 'CGS/code unit-rescaling discrepancy')
            mid = sorted((args.root/'n64/rst').glob('*.rst'))[2]
            result, restarted = launch(args, 'restart', restart=mid)
            report['cases'].append(result)
            difference = max(
                float(np.max(abs(restarted['u']-states[64]['u']))),
                *(float(np.max(abs(a-b)))
                  for a, b in zip(restarted['faces'], states[64]['faces'])))
            h.require(difference < 1e-12, 'Spitzer restart changed final state')
            report['restart_maxabs'] = difference
            for axis in (2, 3):
                result, _ = launch(args, f'axis{axis}', n=32, axis=axis)
                report['cases'].append(result)
        if args.extended:
            invalid = [
                ('negative', ['mhd/spitzer_eta0_code=-1'],
                 'coefficient must be finite and positive'),
                ('zero', ['mhd/spitzer_eta0_code=0'],
                 'coefficient must be finite and positive'),
                ('mixed', ['mhd/eta_ohm=0.1'], 'zero eta_ohm/eta_ad'),
                ('ambipolar', ['mhd/eta_ad=0.1'], 'zero eta_ohm/eta_ad'),
                ('orphan', ['mhd/eta_spitzer=false'], 'require eta_spitzer=true'),
                ('both-norms', ['mhd/spitzer_eta0_cgs=1'], 'exactly one'),
                ('sts-disabled-controller', ['mhd/resistivity_integrator=sts'],
                 'selects STS, but <time>/sts_integrator = none')]
            for label, overrides, text in invalid:
                result, _ = launch(args, 'invalid-'+label,
                                   overrides=overrides, fail_text=text)
                report['cases'].append(result)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
