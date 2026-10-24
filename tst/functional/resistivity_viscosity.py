#!/usr/bin/env python3
"""Coupled viscous/resistive helices: two energy reservoirs heat the same gas.

Parallel transverse circular velocity and magnetic fields have zero ideal
induction and Lorentz force. At rho=1 their amplitudes obey V'=-mu(T)k^2 V,
A'=-eta(T)k^2 A, with T fixed by the conserved kinetic+magnetic+thermal energy.
This additional analytic reduction checks the conservative operators described
by Stone et al. (2020), doi:10.3847/1538-4365/ab929b, integrated with the RKL2
method of Meyer et al. (2014), doi:10.1016/j.jcp.2013.08.021.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.integrate import solve_ivp

import viscosity_spitzer as v

END, V0, A0, T0 = .05, .5, .5, .1
NU, ETA = .25, .25


def selection_arguments(parser):
    parser.add_argument('--omit-viscosity', action='store_true')
    parser.add_argument('--omit-resistivity', action='store_true')
    v.conduction_arguments(parser)


def configure_selection(args):
    global NU, ETA
    if args.omit_viscosity:
        NU = 0.
    if args.omit_resistivity:
        ETA = 0.


def reference(times, k2, nonlinear):
    def rhs(time, state):
        vel, mag = state
        temp = T0+(2./3.)*.5*(V0*V0-vel*vel+A0*A0-mag*mag)
        nu = NU*(temp/T0)**2.5 if nonlinear else NU
        eta = ETA*(temp/T0)**(-1.5) if nonlinear else ETA
        return [-k2*nu*vel, -k2*eta*mag]
    result = solve_ivp(rhs, (0., END), [V0, A0], method='DOP853',
                       dense_output=True, rtol=2e-13, atol=2e-14)
    v.h.require(result.success, 'Coupled reference integration')
    return result.sol(times)


def run(args, method, n, cfl, nonlinear):
    law = 'spitzer' if nonlinear else 'constant'
    name = f'{law}-{method}-n{n}-cfl{cfl}'
    root = args.root/name
    root.mkdir()
    sections = v.input_text(n, 'mhd', method, cfl, 1, 1.)
    v.add_conduction(sections, args, 'mhd', method)
    sections['problem'].update(helix_bfield=0., helix_magnetic_amp=A0)
    if nonlinear:
        sections['mhd'].update(spitzer_mu0_code=NU/T0**2.5,
                               eta_spitzer='true', spitzer_eta0_code=ETA*T0**1.5)
    else:
        sections['mhd'].update(mu_spitzer='false', nu_iso=NU, eta_ohm=ETA)
    sections['mhd']['resistivity_integrator'] = method
    if getattr(args, 'omit_viscosity', False):
        sections['mhd'].pop('spitzer_mu0_code', None)
        sections['mhd'].update(mu_spitzer='false', nu_iso=0., viscosity_integrator='explicit')
    if getattr(args, 'omit_resistivity', False):
        sections['mhd'].pop('eta_spitzer', None)
        sections['mhd'].pop('spitzer_eta0_code', None)
        sections['mhd'].update(eta_ohm=0., resistivity_integrator='explicit')
    text = '\n'.join(f'<{block}>\n'+'\n'.join(f'{k} = {value}' for k, value
                     in values.items())+'\n' for block, values in sections.items())
    (root/'input.athinput').write_text(text)
    command = ['mpirun', '-np', str(args.ranks), str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
               '-i', 'input.athinput']
    (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'athena.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=270)
    states = [v.h.checkpoint(p) for p in sorted((root/'rst').glob('*.rst'))]
    times = [s['time'] for s in states]
    v.h.require(abs(times[-1]-END) < 1e-13, 'Incomplete coupled helix')
    exact = reference(times, (2*np.pi)**2, nonlinear)
    discrete = reference(times, 4*n*n*np.sin(np.pi/n)**2, nonlinear)
    phase = 2*np.pi*(np.arange(n)+.5)/n
    basis = np.stack([np.zeros(n), np.cos(phase), np.sin(phase)])[:, None, None, :]
    samples = []
    for i, state in enumerate(states):
        vel = state['u'][1:4]/state['u'][0]
        mag = state['bcc']
        error = float(np.sqrt(np.mean(np.sum((vel-exact[0, i]*basis)**2, axis=0))/V0**2
                              + np.mean(np.sum((mag-exact[1, i]*basis)**2, axis=0))/A0**2))
        vm = float(np.mean(np.sum(vel*basis, axis=0)))
        am = float(np.mean(np.sum(mag*basis, axis=0)))
        heat = .5*(V0**2-exact[0, i]**2+A0**2-exact[1, i]**2)
        thermal_gain = state['thermal']-T0/(2./3.)
        energy_error = abs(state['total']/.4-1.)
        row = dict(time=state['time'], velocity=vm, magnetic=am,
                   expected_velocity=float(exact[0, i]), expected_magnetic=float(exact[1, i]),
                   thermal_gain=thermal_gain, expected_thermal_gain=float(heat),
                   field_error=error, energy_relative_error=energy_error,
                   temporal_error=float(np.linalg.norm(np.array([vm, am])-discrete[:, i])),
                   divb=state['divb'], min_internal=float(state['internal'].min()))
        samples.append(row)
        v.h.require(error < (2*np.pi/n)**2, 'Coupled velocity/magnetic accuracy')
        v.h.require(abs(thermal_gain-heat)/.25 < (2*np.pi/n)**2, 'Coupled heating accuracy')
        v.h.require(energy_error < 512*np.finfo(float).eps*(state['cycle']+1),
                    'Coupled total energy budget')
        v.h.require(state['divb'] == 0., 'Axis-aligned coupled CT divergence')
    events = [list(map(int, line.split())) for line in (root/'visc_helix.log')
              .read_text().splitlines() if line.strip() and not line.startswith('#')]
    v.h.require(all(not any(row[1:6]) for row in events), 'Coupled EOS floors')
    row = dict(status='PASS', name=name, method=method, n=n, cfl=cfl,
               nonlinear=nonlinear, samples=samples)
    (root/'result.json').write_text(json.dumps(row, indent=2)+'\n')
    print(name, samples[-1], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=('explicit', 'sts'),
                        default=['explicit', 'sts'])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    selection_arguments(parser)
    args = parser.parse_args()
    configure_selection(args)
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', cases=[], binary_sha256=v.h.sha(args.binary),
                  source_manifest_sha256=v.h.sha(args.source_manifest)
                  if args.source_manifest else None)
    (args.root/'gates.json').write_text(json.dumps(dict(
        field='(2*pi/N)^2', spatial_ratio_max=.4,
        conservation='512*eps*(cycles+1)', floors=0), indent=2)+'\n')
    try:
        for method in args.methods:
            laws = (True,) if args.omit_viscosity and args.omit_resistivity else (False, True)
            for nonlinear in laws:
                errors = []
                for n in ((32,) if args.pilot else (32, 64, 128)):
                    row = run(args, method, n, .1, nonlinear)
                    report['cases'].append(row)
                    errors.append(row['samples'][-1]['field_error'])
                if not args.pilot:
                    ratios = [y/x for x, y in zip(errors, errors[1:])]
                    report[method+('-spitzer' if nonlinear else '-constant')] = ratios
                    v.h.require(max(ratios) < .4, 'Coupled spatial convergence')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
