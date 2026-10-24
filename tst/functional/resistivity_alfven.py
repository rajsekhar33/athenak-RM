#!/usr/bin/env python3
"""Measure resistive Alfven damping and phase against the complex dispersion.

Zier, Springel & Mayer (2024), section 3.3 and Appendix B3:
https://doi.org/10.1093/mnras/stad3200.
Use small perturbations in ideal-gas MHD. The reference evolves the measured
initial Fourier velocity/field with the exact linear two-variable matrix.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import expm

import resistivity as h

ETA, AMP, PRESSURE, END = .02, 1e-4, .3, .25


def input_text(n, method, length, spitzer):
    sections = dict(
        job=dict(basename='alfven'),
        mesh=dict(nghost=3), meshblock={},
        time=dict(evolution='dynamic', integrator='rk2', cfl_number=.3,
                  tlim=END/length, nlim=-1, ndiag=1,
                  sts_integrator='rkl2' if method == 'sts' else 'none',
                  sts_max_dt_ratio=16),
        mhd=dict(eos='ideal', gamma=5./3., reconstruct='wenoz', rsolver='hlld',
                 eta_ohm=0. if spitzer else ETA, resistivity_integrator=method),
        problem=dict(pgen_name='linear_wave', wave_flag=1, amp=AMP,
                     dens=1., pgas=PRESSURE, bx0=1., by0=0., bz0=0.,
                     along_x1='true'),
        output1=dict(file_type='rst', dt=END/8),
        output2=dict(file_type='log', dcycle=1))
    if spitzer:
        sections['mhd'].update(eta_spitzer='true',
                               spitzer_eta0_code=ETA*PRESSURE**1.5)
    for axis in (1, 2, 3):
        sections['mesh'].update({
            f'nx{axis}': n if axis == 1 else 1, f'x{axis}min': 0,
            f'x{axis}max': length if axis == 1 else 1,
            f'ix{axis}_bc': 'periodic', f'ox{axis}_bc': 'periodic'})
        sections['meshblock'][f'nx{axis}'] = 16 if axis == 1 else 1
    return '\n'.join(f'<{name}>\n'+'\n'.join(f'{k} = {v}' for k, v in values.items())
                     + '\n' for name, values in sections.items())


def run(args, n, method, length=1., spitzer=False):
    name = f'{method}-n{n}-L{length}'+('-spitzer' if spitzer else '')
    root = args.root/name
    root.mkdir()
    (root/'input.athinput').write_text(input_text(n, method, length, spitzer))
    command = ['mpirun', '-np', str(args.ranks), str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
               '-i', 'input.athinput']
    (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'athena.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=270)
    states = [h.checkpoint(path) for path in sorted((root/'rst').glob('*.rst'))]
    h.require(abs(states[-1]['time']-END) < 1e-13, 'Incomplete Alfven evolution')
    initial = states[0]
    component = 1+int(np.argmax(np.mean(initial['bcc'][1:]**2, axis=(1, 2, 3))))
    phase = np.exp(-2j*np.pi*(np.arange(n)+.5)/n)

    def mode(state):
        velocity = state['u'][1+component, 0, 0]/state['u'][0, 0, 0]
        field = state['bcc'][component, 0, 0]
        return 2*np.array([np.mean(velocity*phase), np.mean(field*phase)])

    k = 2*np.pi/length
    matrix = np.array([[0., 1j*k], [1j*k, -ETA*k*k]], dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    y0 = mode(initial)
    h.require(np.linalg.norm(y0) > AMP/2, 'Missing initial Alfven perturbation')
    coefficients0 = np.linalg.solve(eigenvectors, y0)
    index = int(np.argmax(abs(coefficients0)))
    rows, coefficients = [], []
    energy0 = initial['total']
    for state in states:
        measured = mode(state)
        expected = expm(matrix*state['time'])@y0
        error = float(np.linalg.norm(measured-expected)/np.linalg.norm(y0))
        coefficient = np.linalg.solve(eigenvectors, measured)[index]/coefficients0[index]
        coefficients.append(coefficient)
        energy_error = abs(state['total']-energy0)/energy0
        h.require(energy_error < 512*np.finfo(float).eps*(state['cycle']+1),
                  'Alfven total energy conservation')
        h.require(state['divb'] < 4096*np.finfo(float).eps*n*(state['cycle']+1),
                  'Alfven CT divergence')
        rows.append(dict(time=state['time'], normalized_mode_error=error,
                         velocity=[float(measured[0].real), float(measured[0].imag)],
                         magnetic=[float(measured[1].real), float(measured[1].imag)],
                         energy_relative_error=energy_error, divb=state['divb']))
    times = np.array([row['time'] for row in rows])
    damping = -float(np.polyfit(times, np.log(abs(np.array(coefficients))), 1)[0])
    frequency = abs(float(np.polyfit(times, np.unwrap(np.angle(coefficients)), 1)[0]))
    expected_damping = ETA*k*k/2
    expected_frequency = float(np.sqrt(k*k-expected_damping**2))
    log = (root/'alfven.log').read_text()
    events = [list(map(int, line.split())) for line in log.splitlines()
              if line.strip() and not line.startswith('#')]
    h.require(all(not any(row[1:6]) for row in events), 'Alfven EOS floors/failures')
    h.require(rows[-1]['normalized_mode_error'] < .05, 'Alfven damping/phase error')
    result = dict(status='PASS', name=name, n=n, method=method, length=length,
                  spitzer=spitzer, samples=rows, damping=damping,
                  expected_damping=expected_damping, frequency=frequency,
                  expected_frequency=expected_frequency,
                  damping_relative_error=abs(damping/expected_damping-1),
                  frequency_relative_error=abs(frequency/expected_frequency-1))
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(name, 'mode error', rows[-1]['normalized_mode_error'],
          'damping error', result['damping_relative_error'],
          'frequency error', result['frequency_relative_error'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--methods', nargs='+', choices=('explicit', 'sts'),
                        default=['explicit', 'sts'])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    for manifest in (args.source_manifest, args.binary_manifest):
        if manifest is not None:
            subprocess.run(['sha256sum', '--quiet', '-c', str(manifest)], check=True)
    report = dict(status='RUNNING', cases=[], binary_sha256=h.sha(args.binary),
                  source_manifest_sha256=(h.sha(args.source_manifest)
                                          if args.source_manifest else None),
                  harness_sha256=h.sha(__file__))
    (args.root/'gates.json').write_text(json.dumps(dict(
        mode_error_max=.05, spatial_ratio_max=.4,
        finest_damping_frequency_relative_max=.02,
        doubled_k_damping_ratio_relative_max=.02), indent=2)+'\n')
    try:
        for method in args.methods:
            results = [run(args, n, method) for n in (32, 64, 128)]
            report['cases'].extend(results)
            errors = [r['samples'][-1]['normalized_mode_error'] for r in results]
            ratios = [b/a for a, b in zip(errors, errors[1:])]
            report[method+'_spatial_ratios'] = ratios
            h.require(max(ratios) < .4, 'Alfven convergence '+str(ratios))
            fine = results[-1]
            h.require(max(fine['damping_relative_error'],
                          fine['frequency_relative_error']) < .02,
                      'Resolved damping/frequency')
            high_k = run(args, 128, method, length=.5)
            report['cases'].append(high_k)
            h.require(abs(high_k['damping']/fine['damping']/4-1) < .02,
                      'Damping does not scale as k squared')
            nonlinear = run(args, 64, method, spitzer=True)
            report['cases'].append(nonlinear)
            current = nonlinear['samples'][-1]
            previous = results[1]['samples'][-1]
            initial = results[1]['samples'][0]
            delta = np.linalg.norm(np.array(current['velocity']+current['magnetic']) -
                                   np.array(previous['velocity']+previous['magnetic']))
            normalization = np.linalg.norm(initial['velocity']+initial['magnetic'])
            nonlinear['linear_limit_mode_difference'] = float(delta/normalization)
            h.require(delta/normalization < 1e-5,
                      'Spitzer linear-limit dispersion')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
