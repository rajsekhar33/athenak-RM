#!/usr/bin/env python3
"""Independent initial Faraday-law residual for spatially varying Spitzer eta.

Pressure is uniform, density varies periodically, and the force-free helix has
zero initial velocity. Two tiny kinematic steps isolate the induction residual.
Richardson extrapolation removes the leading finite-step difference. Compare
both the discrete CT curl and its continuum limit, including grad(eta) terms.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

import resistivity as h
AMP, PRES, GAMMA, RHOAMP = 1., .1, 5./3., .2
ETA0 = .25*PRES**1.5


def unique_faces(state):
    out = []
    for a, face in enumerate(state['faces']):
        sl = [slice(None)]*3
        sl[2-a] = slice(0, -1)
        out.append(face[tuple(sl)])
    return np.stack(out)


def discrete_rhs(state):
    b = unique_faces(state)
    dims = np.array(b.shape[1:][::-1])
    eta = ETA0*((GAMMA-1)*state['internal']/state['u'][0])**(-1.5)

    def backward(f, axis):
        return (f-np.roll(f, 1, axis=2-axis)) * \
            dims[axis] if dims[axis] > 1 else np.zeros_like(f)
    current = np.stack([backward(b[2], 1)-backward(b[1], 2),
                        backward(b[0], 2)-backward(b[2], 0),
                        backward(b[1], 0)-backward(b[0], 1)])
    electric = []
    for axis in range(3):
        coeff = eta.copy()
        for normal in range(3):
            if normal != axis and dims[normal] > 1:
                coeff = .5*(coeff+np.roll(coeff, 1, axis=2-normal))
        electric.append(coeff*current[axis])

    def forward(f, axis):
        return (np.roll(f, -1, axis=2-axis)-f) * \
            dims[axis] if dims[axis] > 1 else np.zeros_like(f)
    return np.stack([-forward(electric[2], 1)+forward(electric[1], 2),
                     forward(electric[2], 0)-forward(electric[0], 2),
                    -forward(electric[1], 0)+forward(electric[0], 1)])


def continuum_rhs(dims):
    wave = 2*np.pi
    out = []
    for component in range(3):
        coords = []
        for a, n in enumerate(dims):
            q = (np.arange(n)+(0. if a == component else .5))/n
            if n == 1:
                q = np.zeros(1)
            shape = [1]*3
            shape[2-a] = n
            coords.append(q.reshape(shape))
        cosines = [np.cos(wave*q) for q in coords]
        rho = 1+RHOAMP*cosines[0]*cosines[1]*cosines[2]
        eta = ETA0*(PRES/rho)**(-1.5)
        gradient = []
        for a, n in enumerate(dims):
            drho = -wave*RHOAMP*np.sin(wave*coords[a])
            for other in range(3):
                if other != a:
                    drho = drho*cosines[other]
            gradient.append(1.5*eta/rho*drho if n > 1 else np.zeros_like(rho))
        by, bz = AMP*np.cos(wave*coords[0]), AMP*np.sin(wave*coords[0])
        value = (wave*(bz*gradient[1]-by*gradient[2]) if component == 0 else
                 (-wave*bz*gradient[0]-wave**2*eta*by) if component == 1 else
                 (wave*by*gradient[0]-wave**2*eta*bz))
        out.append(np.broadcast_to(value, tuple(dims[::-1])))
    return np.stack(out)


def run(args, n, ndim, dt):
    root = args.root/f'dim{ndim}-n{n}-dt{dt}'
    root.mkdir()
    text = h.input_text(n, 'explicit', 1, 16, .4)
    text = text.replace('amp = 0.5', 'amp = 1.0').replace('pres = 1.0', 'pres = 0.1')
    text = text.replace('evolution = dynamic', 'evolution = kinematic')
    text = text.replace('rsolver = hlld', 'rsolver = advect')
    text = text.replace(
        'eta_ohm = 0.25',
        'eta_ohm = 0\neta_spitzer = true\n'
        f'spitzer_eta0_code = {ETA0:.17g}')
    text = text.replace('tlim = 0.1', f'tlim = {dt:.17g}')
    text = text.replace('dt = 0.025', f'dt = {dt:.17g}')
    text += f'\n<problem>\nhelix_density_amp = {RHOAMP}\n'
    for a in range(2, ndim+1):
        text = text.replace(f'nx{a} = 1\n', f'nx{a} = {n}\n', 1)
        text = text.replace(f'nx{a} = 1\n', f'nx{a} = 16\n', 1)
    (root/'input.athinput').write_text(text)
    command = ['mpirun',
               '-np',
               str(args.ranks),
               str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank',
               '-t',
               '00:04:00',
               '-i',
               'input.athinput']
    (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'athena.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=270)
    paths = sorted((root/'rst').glob('*.rst'))
    initial, final = h.checkpoint(paths[0]), h.checkpoint(paths[-1])
    h.require(final['cycle'] == 1 and abs(final['time']-dt) < 1e-15, 'Expected one step')
    h.require(final['divb'] < 4096*np.finfo(float).eps*n, 'CT divergence')
    return initial, final, (unique_faces(final)-unique_faces(initial))/dt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--ranks', type=int, default=1)
    args = p.parse_args()
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', cases=[], binary_sha256=h.sha(args.binary))
    gates = dict(discrete_rhs_relative=2e-7, continuum_refinement_ratio=.35,
                 multidim_bx_change_min=1e-9)
    (args.root/'gates.json').write_text(json.dumps(gates, indent=2)+'\n')
    try:
        for ndim in (1, 2, 3):
            errors = []
            for n in (16, 32):
                initial, final, d1 = run(args, n, ndim, 1e-6)
                initial2, final2, d2 = run(args, n, ndim, 5e-7)
                np.testing.assert_array_equal(initial['u'], initial2['u'])
                derivative = 2*d2-d1
                expected = discrete_rhs(initial)
                normalization = np.sqrt(np.mean(expected**2))
                discrete_error = float(
                    np.sqrt(np.mean((derivative-expected)**2))/normalization)
                dims = [n if a < ndim else 1 for a in range(3)]
                continuum = continuum_rhs(dims)
                continuum_error = float(np.sqrt(np.mean((derivative-continuum)**2)) /
                                        np.sqrt(np.mean(continuum**2)))
                h.require(discrete_error < 2e-7,
                          'Variable-eta discrete induction residual')
                bx_change = float(np.max(abs(final2['faces'][0]-initial['faces'][0])))
                if ndim > 1:
                    h.require(bx_change > 1e-9,
                              'Did not exercise transverse eta gradient')
                row = dict(ndim=ndim, n=n, discrete_rhs_error=discrete_error,
                           continuum_rhs_error=continuum_error, max_divb=final2['divb'],
                           max_bx_change=bx_change, status='PASS')
                report['cases'].append(row)
                errors.append(continuum_error)
                print(row, flush=True)
            ratio = errors[1]/errors[0]
            h.require(ratio < .35, 'Variable-eta continuum convergence')
            report[f'dim{ndim}_ratio'] = ratio
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
