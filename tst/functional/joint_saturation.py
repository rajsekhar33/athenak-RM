#!/usr/bin/env python3
"""Saturated heat transport in pressure-balanced, density-modulated MHD.

The initial conductive flux and donor-pressure Jacobian have independent 1D
expressions. Compare the coupled RKL2 solution with fine explicit references,
and check that its outer timestep retains the unaccelerated donor bound.
References: Cowie & McKee (1977), doi:10.1086/154911; Mignone et al. (2012),
doi:10.1088/0067-0049/198/1/7; Meyer et al. (2014), doi:10.1016/j.jcp.2013.08.021.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

import viscosity_spitzer as v
from resistivity_evolution import difference

END = .005


def donor_oracle(state, alpha, phi):
    """Frozen-classical-flux derivative dF/dP_donor, gamma=5/3."""
    rho = state['u'][0, 0, 0]
    pressure = (2./3.)*state['internal'][0, 0]
    temp = pressure/rho
    dx = 1./len(rho)
    rho_face = .5*(rho+np.roll(rho, 1))
    classical = -alpha*rho_face*(temp-np.roll(temp, 1))/dx
    donor = np.where(classical > 0., np.roll(pressure, 1), pressure)
    q = 5*phi*donor**1.5/np.sqrt(rho_face)
    ratio = abs(classical)/(abs(classical)+q)
    speed = 1.5*(2./3.)*q/donor*ratio**2
    return 1./float(np.max((speed+np.roll(speed, -1))/dx))


def run(args, method, cfl, restart=None, ratio=-1.):
    name = f'{method}-cfl{cfl}-ratio{ratio}'+('-restart' if restart else '')
    root = args.root/name
    root.mkdir()
    sections = v.input_text(args.n, 'mhd', method, cfl, 1, 1.)
    sections['time'].update(tlim=END, sts_max_dt_ratio=ratio)
    sections['problem'].update(helix_bfield=0., helix_magnetic_amp=.5,
                               helix_density_amp=args.contrast, amp=args.amplitude)
    sections['mhd'].update(mu_spitzer='false', nu_iso=args.viscosity, eta_ohm=1e-4,
                           resistivity_integrator=method, conductivity_integrator=method,
                           alpha_iso=args.alpha, conduction_saturation='cowie_mckee',
                           saturation_phi=args.phi)
    sections['output1']['dt'] = END/4
    text = '\n'.join(f'<{block}>\n'+'\n'.join(f'{k} = {value}' for k, value
                     in values.items())+'\n' for block, values in sections.items())
    (root/'input.athinput').write_text(text)
    command = ['mpirun', '-np', str(args.ranks), str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank', '-i', 'input.athinput', '-t', '00:10:00']
    if restart:
        command += ['-r', os.path.relpath(restart, root)]
    (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'athena.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=630)
    paths = sorted((root/'rst').glob('*.rst'))
    states = [v.h.checkpoint(p) for p in paths]
    v.h.require(abs(states[-1]['time']-END) < 1e-13, 'Incomplete saturated evolution')
    for state in states:
        gate = 512*np.finfo(float).eps*(state['cycle']+1)
        energy = .275+.5*args.amplitude**2
        v.h.require(abs(state['total']/energy-1.) < gate, 'Saturated energy budget')
        v.h.require(abs(float(state['u'][0].mean())-1.) < gate, 'Saturated mass budget')
        v.h.require(state['divb'] < 8*args.n*gate, 'Saturated CT divergence')
    events = [list(map(int, line.split())) for line in (root/'visc_helix.log')
              .read_text().splitlines() if line.strip() and not line.startswith('#')]
    v.h.require(all(not any(row[1:6]) for row in events), 'Saturated EOS floors')
    log = (root/'athena.log').read_text()
    trials = [line for line in log.splitlines() if line.startswith('transport_sts cycle=')]
    row = dict(status='PASS', name=name, method=method, cfl=cfl, ratio=ratio,
               checkpoint=str(paths[-1]), cycle=states[-1]['cycle'],
               rejected_trials=sum('status=retry' in line for line in trials),
               rejection_reasons=[line.split('reason=', 1)[1] for line in trials
                                  if 'status=retry' in line])
    if method == 'sts' and not restart:
        bound = donor_oracle(states[0], args.alpha, args.phi)
        match = re.search(r'cycle=0 time=\S+ dt=(\S+)', log)
        v.h.require(match is not None, 'Initial timestep diagnostic absent')
        initial_dt = float(match[1])
        row.update(initial_dt=initial_dt, initial_donor_bound=bound,
                   initial_cap_ratio=initial_dt/(cfl*bound))
        # Athena prints outer dt with six significant digits.
        v.h.require(initial_dt <= cfl*bound*(1+2e-6), 'Donor bound was STS accelerated')
        if ratio < 0.:
            v.h.require(abs(initial_dt/(cfl*bound)-1.) < 2e-6,
                        'Selected state did not exercise the donor timestep cap')
    (root/'result.json').write_text(json.dumps(row, indent=2)+'\n')
    print(row, flush=True)
    return row, states[-1], paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--phi', type=float, default=1.)
    parser.add_argument('--contrast', type=float, default=.8)
    parser.add_argument('--amplitude', type=float, default=.5)
    parser.add_argument('--viscosity', type=float, default=1e-4)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--recovery', action='store_true')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', cases=[], binary_sha256=v.h.sha(args.binary),
                  harness_sha256=v.h.sha(Path(__file__)),
                  parameters=dict(n=args.n, alpha=args.alpha, phi=args.phi,
                                  contrast=args.contrast, end_time=END,
                                  amplitude=args.amplitude, viscosity=args.viscosity))
    (args.root/'gates.json').write_text(json.dumps(dict(
        cap_relative_tolerance=2e-6, temporal_ratio_max=.4, state_error_max=.005,
        reference_uncertainty_fraction=.15, restart_maxabs=1e-12), indent=2)+'\n')
    try:
        if args.pilot or args.recovery:
            row, uninterrupted, paths = run(args, 'sts', .3)
            report['cases'].append(row)
            if args.recovery:
                v.h.require('current saturated transport cap' in row['rejection_reasons'],
                            'Saturation recovery case did not reject a donor-limited trial')
                row, state, _ = run(args, 'sts', .3, restart=paths[1])
                report['cases'].append(row)
                delta = max(float(np.max(abs(state['u']-uninterrupted['u']))),
                            *(float(np.max(abs(x-y)))
                              for x, y in zip(state['faces'], uninterrupted['faces'])))
                report['restart_maxabs'] = delta
                v.h.require(delta < 1e-12, 'Recovered saturation restart discrepancy')
                report['scope'] = 'Trial rejection, state recovery and restart integrity'
            report['status'] = 'PASS' if args.recovery else 'CAP_PILOT_PASS'
            return
        if args.reference_root:
            previous = json.loads((args.reference_root/'summary.json').read_text())
            v.h.require(previous['status'] == 'PASS', 'Saturation reference did not pass')
            expected = dict(previous['parameters'])
            expected.setdefault('amplitude', .5)
            expected.setdefault('viscosity', 1e-4)
            v.h.require(expected == report['parameters'], 'Saturation reference parameters differ')
            refrow = min((r for r in previous['cases'] if r['method'] == 'explicit'),
                         key=lambda r: r['cfl'])
            ref = v.h.checkpoint(Path(refrow['checkpoint']))
            uncertainty = previous['reference_uncertainty']
            report['reference_summary_sha256'] = v.h.sha(args.reference_root/'summary.json')
        else:
            refs = []
            for cfl in (.00625, .003125):
                row, state, _ = run(args, 'explicit', cfl)
                report['cases'].append(row)
                refs.append(state)
            ref, uncertainty = refs[-1], difference(*refs)
        errors = []
        for cfl in (.3, .15, .075):
            row, state, paths = run(args, 'sts', cfl)
            report['cases'].append(row)
            errors.append(difference(state, ref))
            if args.reference_root:
                matching = next(r for r in previous['cases'] if r['name'] == row['name'])
                row['backend_relative_difference'] = difference(
                    state, v.h.checkpoint(Path(matching['checkpoint'])))
                v.h.require(row['backend_relative_difference'] < 1e-11,
                            'Saturation backend field discrepancy')
            if cfl == .15:
                checkpoint, uninterrupted = paths[2], state
        ratios = [y/x for x, y in zip(errors, errors[1:])]
        report.update(reference_uncertainty=uncertainty, sts_errors=errors, ratios=ratios)
        v.h.require(uncertainty < .15*min(errors), 'Saturation reference unresolved')
        v.h.require(max(ratios) < .4 and errors[-1] < .005, 'Saturation time convergence')
        row, state, _ = run(args, 'sts', .15, restart=checkpoint)
        report['cases'].append(row)
        delta = max(float(np.max(abs(state['u']-uninterrupted['u']))),
                    *(float(np.max(abs(x-y)))
                      for x, y in zip(state['faces'], uninterrupted['faces'])))
        report['restart_maxabs'] = delta
        v.h.require(delta < 1e-12, 'Saturated restart discrepancy')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
