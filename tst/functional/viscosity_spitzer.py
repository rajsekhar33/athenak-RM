#!/usr/bin/env python3
"""Spitzer dynamic-viscosity decay and heating in a circular transverse shear.

The conservative Navier-Stokes equations give V'=-k^2*mu(T)*V at rho=1 and
T=T_initial+(gamma-1)*(V_initial^2-V^2)/2. References:
Stone et al. (2020), doi:10.3847/1538-4365/ab929b; Spitzer ion normalization:
https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node147.html
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import resistivity as h

AMPLITUDE, PRESSURE, END = .5, .1, .05
MU0 = .25/PRESSURE**2.5


def conduction_arguments(parser):
    parser.add_argument('--conduction', choices=('off', 'constant', 'spitzer', 'saturated'),
                        default='off')
    parser.add_argument('--conduction-geometry', choices=('auto', 'isotropic', 'field_aligned'),
                        default='auto')
    parser.add_argument('--conduction-limiter', choices=('vanleer', 'mc'), default='vanleer')
    parser.add_argument('--conduction-coefficient', type=float, default=.025)
    parser.add_argument('--saturation-phi', type=float, default=.001)
    parser.add_argument('--sts-ratio', type=float, default=16.)


def add_conduction(sections, args, fluid, method, scale=1., temperature=PRESSURE):
    if hasattr(args, 'sts_ratio'):
        sections['time']['sts_max_dt_ratio'] = args.sts_ratio
    kind = getattr(args, 'conduction', 'off')
    if kind == 'off':
        return
    geometry = args.conduction_geometry
    if geometry == 'auto':
        geometry = 'field_aligned' if fluid == 'mhd' else 'isotropic'
    block = sections[fluid]
    block.update(conductivity_integrator=method, anisotropic_limiter=args.conduction_limiter)
    if kind == 'constant':
        block['alpha_iso' if geometry == 'isotropic' else 'alpha_aniso'] = (
            args.conduction_coefficient/scale)
    else:
        block.update(alpha_spitzer='true', spitzer_geometry=geometry,
                     spitzer_kappa0_code=args.conduction_coefficient/temperature**2.5/scale)
    if kind == 'saturated':
        block.update(conduction_saturation='cowie_mckee', saturation_phi=args.saturation_phi)


def reference(times, k2):
    rate = k2*MU0

    def elapsed(q):
        def integrand(z):
            temp = PRESSURE+(2./3.)*.5*AMPLITUDE**2*(2*z-z*z)
            return 1./(rate*temp**2.5*(1-z))
        return quad(integrand, 0., q, epsabs=1e-15, epsrel=5e-13)[0]

    values = []
    for time in times:
        upper = .5
        while elapsed(upper) < time:
            upper = .5*(1+upper)
            h.require(upper < 1., 'Reference velocity underflows')
        q = brentq(
            lambda z: elapsed(z)-time, 0., upper, xtol=1e-14, rtol=1e-14
        ) if time else 0.
        values.append(AMPLITUDE*(1-q))
    return np.array(values)


def input_text(n, fluid, method, cfl, axis, scale):
    length = 1./scale
    dims = [n, 1, 1] if axis == 1 else ([16, n, 1] if axis == 2 else [16, 16, n])
    sections = dict(
        job=dict(basename='visc_helix'), mesh=dict(nghost=3), meshblock={},
        time=dict(evolution='dynamic', integrator='rk2', cfl_number=cfl,
                  tlim=END/scale, nlim=-1, ndiag=1,
                  sts_integrator='rkl2' if method == 'sts' else 'none',
                  sts_max_dt_ratio=16),
        problem=dict(pgen_name='diffusion', viscosity_profile='helix',
                     amp=AMPLITUDE, rho=1, pres=PRESSURE, helix_axis=axis,
                     helix_bfield=1 if fluid == 'mhd' else 0),
        output1=dict(file_type='rst', dt=END/(4*scale)),
        output2=dict(file_type='log', dcycle=1))
    sections[fluid] = dict(eos='ideal', gamma=5./3., reconstruct='wenoz',
                           rsolver='hlld' if fluid == 'mhd' else 'hllc',
                           mu_spitzer='true', viscosity_integrator=method)
    for a in (1, 2, 3):
        sections['mesh'].update({f'nx{a}': dims[a-1], f'x{a}min': 0,
                                 f'x{a}max': length,
                                 f'ix{a}_bc': 'periodic', f'ox{a}_bc': 'periodic'})
        sections['meshblock'][f'nx{a}'] = min(dims[a-1], 16)
    return sections


def launch(args, fluid, method, name, n=64, cfl=.4, axis=1,
           units_scale=None, restart=None, overrides=(), fail_text=None):
    root = args.root/name
    root.mkdir()
    scale = units_scale if units_scale is not None else 1.
    sections = input_text(n, fluid, method, cfl, axis, scale)
    add_conduction(sections, args, fluid, method, scale=scale)
    if units_scale is None:
        sections[fluid]['spitzer_mu0_code'] = MU0
    else:
        L0, t0, rho0, molecular_weight = 2.5e11, 2.5e3, 1e-27, .6
        T0 = (L0/t0)**2*molecular_weight*1.660538921e-24/1.3806488e-16
        sections[fluid]['spitzer_mu0_cgs'] = MU0*rho0*L0**2/t0/T0**2.5
        sections['units'] = dict(length_cgs=L0*scale, time_cgs=t0*scale,
                                 mass_cgs=rho0*(L0*scale)**3, mu=molecular_weight)
    if restart:
        # The new helix reads physical tlim for fresh starts and restarts.
        sections['time']['tlim'] = END/scale
    for override in overrides:
        key, value = override.split('=', 1)
        block, parameter = key.split('/', 1)
        sections.setdefault(block, {})[parameter] = value
    text = '\n'.join(f'<{block}>\n'+'\n'.join(f'{k} = {v}' for k, v in values.items())
                     + '\n' for block, values in sections.items())
    (root/'input.athinput').write_text(text)
    command = ['mpirun', '-np', str(args.ranks), str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
               '-i', 'input.athinput']
    if restart:
        command += ['-r', os.path.relpath(restart, root)]
    (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'athena.log').open('x') as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                timeout=270)
    if fail_text is not None:
        h.require(result.returncode != 0 and fail_text in (root/'athena.log').read_text(),
                  'Missing expected rejection '+name)
        return dict(status='PASS', name=name, expected_rejection=fail_text), None, []
    h.require(result.returncode == 0, 'Athena failed: '+str(root))
    paths = sorted((root/'rst').glob('*.rst'))
    states = [h.checkpoint(p, mhd=fluid == 'mhd') for p in paths]
    times = [s['time']*scale for s in states]
    h.require(abs(times[-1]-END) < 1e-13, 'Incomplete viscous helix')
    continuum = reference(times, (2*np.pi)**2)
    discrete = reference(times, 4*n*n*np.sin(np.pi/n)**2)
    phase = 2*np.pi*(np.arange(n)+.5)/n
    shape = [1, 1, 1]
    shape[3-axis] = n
    phase = phase.reshape(shape)
    E0 = PRESSURE/(2./3.)+.5*AMPLITUDE**2+(.5 if fluid == 'mhd' else 0.)
    rows = []
    for state, t, exact, exact_discrete in zip(states, times, continuum, discrete):
        velocity = state['u'][1:4]/state['u'][0]
        first, second = velocity[axis % 3], velocity[(axis+1) % 3]
        amplitude = float(np.mean(first*np.cos(phase)+second*np.sin(phase)))
        error = float(np.sqrt(np.mean((first-exact*np.cos(phase))**2 +
                                      (second-exact*np.sin(phase))**2))/AMPLITUDE)
        heat = .5*(AMPLITUDE**2-exact**2)
        row = dict(time=t, amplitude=amplitude, expected_amplitude=float(exact),
                   field_error=error, temporal_error=float(abs(amplitude-exact_discrete)),
                   thermal_gain=state['thermal']-PRESSURE/(2./3.),
                   expected_thermal_gain=float(heat),
                   energy_relative_error=abs(state['total']-E0)/E0,
                   divb=state['divb'], min_internal=float(state['internal'].min()))
        h.require(error < .5*(2*np.pi/n)**2, 'Viscous velocity error')
        h.require(row['energy_relative_error'] <
                  512*np.finfo(float).eps*(state['cycle']+1), 'Viscous total energy')
        h.require(abs(row['thermal_gain']-heat)/(.5*AMPLITUDE**2) < (2*np.pi/n)**2,
                  'Viscous heating error')
        if fluid == 'mhd':
            h.require(np.max(abs(state['bcc']-states[0]['bcc'])) < 1e-12,
                      'Passive magnetic field changed')
        rows.append(row)
    events = [list(map(int, line.split()))
              for line in (root/'visc_helix.log').read_text().splitlines()
              if line.strip() and not line.startswith('#')]
    h.require(all(not any(row[1:6]) for row in events), 'Viscous EOS floors/failures')
    log = (root/'athena.log').read_text()
    trials = [dict(re.findall(r'(\w+)=(\S+)', line)) for line in log.splitlines()
              if line.startswith(('viscosity_sts cycle=', 'transport_sts cycle='))]
    h.require(method != 'sts' or trials, 'Missing viscosity STS execution evidence')
    result = dict(status='PASS', name=name, fluid=fluid, method=method, n=n, cfl=cfl,
                  axis=axis, units_scale=units_scale, samples=rows,
                  accepted=sum(t['status'] == 'accepted' for t in trials),
                  rejected=sum(t['status'] == 'retry' for t in trials))
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(name, rows[-1], 'rejected', result['rejected'], flush=True)
    return result, states[-1], paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--fluids', nargs='+', choices=('hydro', 'mhd'),
                        default=['hydro', 'mhd'])
    parser.add_argument('--methods', nargs='+', choices=('explicit', 'sts'),
                        default=['explicit', 'sts'])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    conduction_arguments(parser)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    for manifest in (args.source_manifest, args.binary_manifest):
        if manifest:
            subprocess.run(['sha256sum', '--quiet', '-c', str(manifest)], check=True)
    report = dict(status='RUNNING', cases=[], binary_sha256=h.sha(args.binary),
                  source_manifest_sha256=(h.sha(args.source_manifest)
                                          if args.source_manifest else None),
                  harness_sha256=h.sha(__file__))
    (args.root/'gates.json').write_text(json.dumps(dict(
        field='.5*(2*pi/N)^2', spatial_ratio_max=.4, temporal_ratio_max=.4,
        restart_maxabs=1e-12, units_maxabs=1e-12,
        conservation='512*eps*(cycles+1)'), indent=2)+'\n')
    try:
        for fluid in args.fluids:
            for method in args.methods:
                errors, states, checkpoints = [], {}, {}
                for n in ((32,) if args.pilot else (32, 64, 128)):
                    row, state, paths = launch(args, fluid, method,
                                               f'{fluid}-{method}-n{n}',
                                               n=n)
                    report['cases'].append(row)
                    states[n], checkpoints[n] = state, paths
                    errors.append(row['samples'][-1]['field_error'])
                if args.pilot:
                    continue
                ratios = [b/a for a, b in zip(errors, errors[1:])]
                report[fluid+'-'+method+'_spatial_ratios'] = ratios
                h.require(max(ratios) < .4, 'Viscous spatial convergence '+str(ratios))
                if args.extended:
                    temporal = [next(c for c in report['cases']
                                     if c['name'] == f'{fluid}-{method}-n64')
                                ['samples'][-1]['temporal_error']]
                    for cfl in (.2, .1):
                        row, _, _ = launch(args, fluid, method,
                                           f'{fluid}-{method}-cfl{cfl}', cfl=cfl)
                        report['cases'].append(row)
                        temporal.append(row['samples'][-1]['temporal_error'])
                    ratios = [b/a for a, b in zip(temporal, temporal[1:])]
                    report[fluid+'-'+method+'_temporal_ratios'] = ratios
                    h.require(max(ratios) < .4, 'Viscous time convergence '+str(ratios))
                    for label, kwargs in [('restart', dict(restart=checkpoints[64][2])),
                                          ('cgs', dict(units_scale=2.))]:
                        row, state, _ = launch(args, fluid, method,
                                               f'{fluid}-{method}-{label}', **kwargs)
                        report['cases'].append(row)
                        delta = float(np.max(abs(state['u']-states[64]['u'])))
                        report[f'{fluid}-{method}-{label}_maxabs'] = delta
                        h.require(delta < 1e-12, 'Viscous '+label+' discrepancy')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
