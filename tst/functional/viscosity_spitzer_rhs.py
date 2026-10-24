#!/usr/bin/env python3
"""Variable-temperature viscosity: momentum and energy-work residuals in 2D/3D.

Subtract an inviscid run of the identical initial state to isolate diffusion.
Two short timesteps and Richardson extrapolation measure the initial derivative.
Compare the tensor-form face flux and independent continuum derivatives.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

import viscosity_spitzer as v

h = v.h
B = np.array([1., 2., 3.])/np.sqrt(14.)


def discrete(state, geometry):
    velocity = state['u'][1:4]/state['u'][0]
    dims = np.array(state['u'].shape[1:][::-1])
    ndim = int(sum(dims > 1))
    mu = v.MU0*((2./3.)*state['internal']/state['u'][0])**2.5
    momentum, energy = np.zeros_like(velocity), np.zeros_like(mu)
    for normal in range(ndim):
        left = np.roll(velocity, 1, axis=3-normal)
        coeff = .5*(mu+np.roll(mu, 1, axis=2-normal))
        grad = np.zeros((3, 3)+mu.shape)
        for direction in range(ndim):
            if direction == normal:
                grad[:, direction] = (velocity-left)*dims[direction]
            else:
                axis = 3-direction
                grad[:, direction] = .25*dims[direction]*(
                    np.roll(velocity, -1, axis=axis)-np.roll(velocity, 1, axis=axis)
                    + np.roll(left, -1, axis=axis)-np.roll(left, 1, axis=axis))
        trace = np.einsum('ii...->...', grad)
        unit = np.eye(3)[normal]
        if geometry == 'isotropic':
            stress = coeff*(grad[:, normal]+grad[normal, :] -
                            (2./3.)*unit[:, None, None, None]*trace)
        else:
            strain = np.einsum('i,ij...,j->...', B, grad, B)-trace/3.
            stress = coeff*strain*(3*B*B[normal]-unit)[:, None, None, None]
        work = np.sum(.5*(velocity+left)*stress, axis=0)
        momentum += (np.roll(stress, -1, axis=3-normal)-stress)*dims[normal]
        energy += (np.roll(work, -1, axis=2-normal)-work)*dims[normal]
    return momentum, energy


def continuum(n, ndim, geometry):
    shape = tuple(n if a < ndim else 1 for a in (2, 1, 0))
    coords = []
    for direction in range(3):
        size = n if direction < ndim else 1
        sh = [1, 1, 1]
        sh[2-direction] = size
        coords.append(((np.arange(size)+.5)/size).reshape(sh))
    k = 2*np.pi
    rho = np.ones(shape)
    product = np.ones(shape)
    for direction in range(ndim):
        product *= np.cos(k*coords[direction])
    rho += .2*product
    mu = v.MU0*(v.PRESSURE/rho)**2.5
    dmu = []
    for direction in range(ndim):
        derivative = -.2*k*np.sin(k*coords[direction])*np.ones(shape)
        for other in range(ndim):
            if other != direction:
                derivative *= np.cos(k*coords[other])
        dmu.append(-2.5*mu*derivative/rho)
    while len(dmu) < 3:
        dmu.append(np.zeros(shape))
    velocity = np.stack([np.zeros(shape),
                         np.broadcast_to(v.AMPLITUDE*np.cos(k*coords[0]), shape),
                         np.broadcast_to(v.AMPLITUDE*np.sin(k*coords[0]), shape)])
    dxv = np.stack([np.zeros(shape), -k*velocity[2], k*velocity[1]])
    if geometry == 'isotropic':
        momentum = np.stack([dmu[1]*dxv[1]+dmu[2]*dxv[2],
                             dmu[0]*dxv[1]-k*k*mu*velocity[1],
                             dmu[0]*dxv[2]-k*k*mu*velocity[2]])
        return momentum, np.zeros(shape)
    strain = B[0]*np.einsum('i,i...->...', B, dxv)
    dxstrain = -k*k*B[0]*np.einsum('i,i...->...', B, velocity)
    terms = strain*np.stack(dmu)
    terms[0] += mu*dxstrain
    projector = np.outer(B, B)-np.eye(3)/3.
    momentum = 3*np.einsum('ij,j...->i...', projector, terms)
    energy = np.sum(velocity*momentum, axis=0)+3*mu*strain**2
    return momentum, energy


def run(args, n, ndim, geometry, dt, inviscid):
    name = f'{geometry}-dim{ndim}-n{n}-dt{dt}'+('-inviscid' if inviscid else '')
    root = args.root/name
    root.mkdir()
    sections = v.input_text(n, 'mhd', 'explicit', .4, 1, 1.)
    sections['time']['tlim'] = dt
    sections['output1']['dt'] = dt
    sections['problem'].update(helix_density_amp=.2,
                               helix_bx=B[0], helix_by=B[1], helix_bz=B[2])
    sections['mhd']['spitzer_mu0_code'] = v.MU0
    sections['mhd']['spitzer_viscosity_geometry'] = geometry
    if inviscid:
        sections['mhd'].pop('spitzer_mu0_code')
        sections['mhd'].pop('spitzer_viscosity_geometry')
        sections['mhd'].update(mu_spitzer='false', nu_iso=0.)
    for direction in range(ndim):
        a = direction+1
        sections['mesh'][f'nx{a}'] = n
        sections['meshblock'][f'nx{a}'] = n//2
    text = '\n'.join(f'<{block}>\n'+'\n'.join(f'{k} = {value}'
                     for k, value in values.items())+'\n'
                     for block, values in sections.items())
    (root/'input.athinput').write_text(text)
    command = ['mpirun', '-np', str(args.ranks), str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
               '-i', 'input.athinput']
    (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'athena.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=270)
    paths = sorted((root/'rst').glob('*.rst'))
    initial, final = h.checkpoint(paths[0]), h.checkpoint(paths[-1])
    h.require(final['cycle'] == 1 and abs(final['time']-dt) < 1e-15, 'Expected one step')
    h.require(final['divb'] < 4096*np.finfo(float).eps*n, 'Viscous RHS CT divergence')
    events = [list(map(int, line.split()))
              for line in (root/'visc_helix.log').read_text().splitlines()
              if line.strip() and not line.startswith('#')]
    h.require(all(not any(row[1:6]) for row in events), 'Viscous RHS EOS floors')
    return initial, (final['u'][1:5]-initial['u'][1:5])/dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--geometries', nargs='+', choices=('isotropic', 'parallel'),
                        default=['isotropic', 'parallel'])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
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
        discrete_momentum_relative_max=2e-7, discrete_work_scaled_max=2e-7,
        continuum_momentum_refinement_max=.4), indent=2)+'\n')
    try:
        for geometry in args.geometries:
            for ndim in (2, 3):
                errors = []
                for n in (16, 32):
                    derivatives = []
                    for dt in (1e-6, 5e-7):
                        initial, a = run(args, n, ndim, geometry, dt, False)
                        control, b = run(args, n, ndim, geometry, dt, True)
                        np.testing.assert_array_equal(initial['u'], control['u'])
                        derivatives.append(a-b)
                    derivative = 2*derivatives[1]-derivatives[0]
                    expected_m, expected_e = discrete(initial, geometry)
                    scale = float(np.sqrt(np.mean(expected_m**2)))
                    dm = float(np.sqrt(np.mean((derivative[:3]-expected_m)**2))/scale)
                    de = float(np.sqrt(np.mean((derivative[3]-expected_e)**2)) /
                               (scale*v.AMPLITUDE))
                    continuum_m, continuum_e = continuum(n, ndim, geometry)
                    error = float(np.sqrt(np.mean((derivative[:3]-continuum_m)**2)) /
                                  np.sqrt(np.mean(continuum_m**2)))
                    h.require(dm < 2e-7 and de < 2e-7, 'Viscous tensor/work residual')
                    row = dict(status='PASS', geometry=geometry, ndim=ndim, n=n,
                               discrete_momentum_error=dm, discrete_work_error=de,
                               continuum_momentum_error=error,
                               continuum_work_scaled_error=float(np.sqrt(np.mean(
                                   (derivative[3]-continuum_e)**2))/(scale*v.AMPLITUDE)))
                    report['cases'].append(row)
                    errors.append(error)
                    print(row, flush=True)
                ratio = errors[1]/errors[0]
                report[f'{geometry}-dim{ndim}_ratio'] = ratio
                h.require(ratio < .4, 'Viscous continuum refinement '+str(ratio))
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
