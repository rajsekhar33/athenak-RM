#!/usr/bin/env python3
"""Joint nonlinear RKL2 timestep convergence, CGS scaling and cycle restarts."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import resistivity_viscosity as rv


def repeat(args, name, text, restart=None):
    root = args.root/name
    root.mkdir()
    (root/'input.athinput').write_text(text)
    command = ['mpirun', '-np', str(args.ranks), str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
               '-i', 'input.athinput']
    if restart:
        command += ['-r', os.path.relpath(restart, root)]
    (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'athena.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=270)
    paths = sorted((root/'rst').glob('*.rst'))
    state = rv.v.h.checkpoint(paths[-1])
    rows = [list(map(int, line.split())) for line in (root/'visc_helix.log').read_text()
            .splitlines() if line.strip() and not line.startswith('#')]
    rv.v.h.require(all(not any(row[1:6]) for row in rows), 'Joint control EOS floors')
    return state


def delta(a, b):
    return max(float(np.max(abs(a['u']-b['u']))),
               *(float(np.max(abs(x-y))) for x, y in zip(a['faces'], b['faces'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    rv.selection_arguments(parser)
    args = parser.parse_args()
    rv.configure_selection(args)
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', cases=[], binary_sha256=rv.v.h.sha(args.binary))
    (args.root/'gates.json').write_text(json.dumps(dict(
        temporal_ratio_max=.4, units_maxabs=1e-12, restart_maxabs=1e-12,
        reference='discrete Laplacian coupled amplitude/heating ODE'), indent=2)+'\n')
    try:
        errors = []
        for cfl in (.4, .2, .1):
            row = rv.run(args, 'sts', 32, cfl, True)
            report['cases'].append(row)
            errors.append(row['samples'][-1]['temporal_error'])
        ratios = [y/x for x, y in zip(errors, errors[1:])]
        report.update(temporal_errors=errors, temporal_ratios=ratios)
        rv.v.h.require(max(ratios) < .4, 'Joint helix timestep convergence '+str(ratios))
        original = args.root/'spitzer-sts-n32-cfl0.1'
        paths = sorted((original/'rst').glob('*.rst'))
        expected = rv.v.h.checkpoint(paths[-1])
        text = (original/'input.athinput').read_text()
        state = repeat(args, 'restart', text, restart=paths[2])
        report['restart_maxabs'] = delta(state, expected)
        rv.v.h.require(report['restart_maxabs'] < 1e-12, 'Joint helix restart')
        L0, t0, rho0 = 2.5e11, 2.5e3, 1e-27
        Tunit = (L0/t0)**2*.6*1.660538921e-24/1.3806488e-16
        mu0_cgs = rv.NU/rv.T0**2.5*rho0*L0**2/t0/Tunit**2.5
        eta0_cgs = rv.ETA*rv.T0**1.5*L0**2/t0*Tunit**1.5
        lines = []
        for line in text.splitlines():
            if line.startswith('spitzer_mu0_code ='):
                line = f'spitzer_mu0_cgs = {mu0_cgs:.17g}'
            elif line.startswith('spitzer_eta0_code ='):
                line = f'spitzer_eta0_cgs = {eta0_cgs:.17g}'
            elif line.startswith('spitzer_kappa0_code ='):
                speed = L0/t0
                normalization = float(line.split('=', 1)[1])
                kappa_cgs = normalization*rho0*speed**3*L0/Tunit**3.5
                line = f'spitzer_kappa0_cgs = {kappa_cgs:.17g}'
            elif line.startswith(('alpha_iso =', 'alpha_aniso =')):
                key, value = line.split('=', 1)
                line = f'{key.strip()} = {float(value)/2:.17g}'
            elif line.startswith(('x1max =', 'x2max =', 'x3max =', 'tlim =', 'dt =')):
                key, value = line.split('=', 1)
                line = f'{key.strip()} = {float(value)/2:.17g}'
            lines.append(line)
        scaled = '\n'.join(lines)+(f'\n<units>\nlength_cgs = {2*L0:.17g}\n'
                                   f'time_cgs = {2*t0:.17g}\n'
                                   f'mass_cgs = {rho0*(2*L0)**3:.17g}\nmu = .6\n')
        state = repeat(args, 'cgs-rescaled', scaled)
        rv.v.h.require(abs(state['time']*2-rv.END) < 1e-13, 'Joint unit-scaled time')
        report['units_maxabs'] = delta(state, expected)
        rv.v.h.require(report['units_maxabs'] < 1e-12, 'Joint CGS coefficient scaling')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
