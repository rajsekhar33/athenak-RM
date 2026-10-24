#!/usr/bin/env python3
"""Parallel Spitzer-viscosity acoustic damping and phase in ideal-gas MHD.

For B along the wavevector, Braginskii stress gives longitudinal viscosity
4*nu/3 and sound-wave eigenvalues -2*nu*k^2/3 +/- i*sqrt(cs^2*k^2-(2*nu*k^2/3)^2).
The matrix exponential also advances the entropy component of the measured
initial state. See Braginskii (1965) and Berlok et al. (2020),
https://doi.org/10.1093/mnras/stz3115.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import expm

import resistivity_alfven as alfven

h = alfven.h
NU, PRESSURE, GAMMA, END = .02, .3, 5./3., .25


def run(args, n, method, constant=False, matched=False):
    name = f'{method}-n{n}'+('-constant' if constant else '')
    name += '-matched' if matched else ''
    root = args.root/name
    root.mkdir()
    sound = np.sqrt(GAMMA*PRESSURE)
    text = alfven.input_text(n, method, 1., False)
    text += ('\n<mhd>\neta_ohm = 0\nresistivity_integrator = explicit\n'
             f'viscosity_integrator = {method}\n'
             f'<time>\ntlim = {END*sound:.17g}\n'
             f'cfl_number = {.1 if constant else .3}\n'
             '<problem>\nwave_flag = 2\n')
    if constant:
        text += f'<mhd>\nnu_aniso = {NU}\n'
    else:
        text += ('<mhd>\nmu_spitzer = true\nspitzer_viscosity_geometry = parallel\n'
                 f'spitzer_mu0_code = {NU/PRESSURE**2.5:.17g}\n')
    if matched:
        # Force the parabolic cap to choose both cycle lengths. With CFL .3
        # and .1 the new/legacy bounds then select matching three-stage trials.
        text += '<time>\nsts_max_dt_ratio = 1\n'
    (root/'input.athinput').write_text(text)
    command = ['mpirun', '-np', str(args.ranks), str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
               '-i', 'input.athinput']
    (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'athena.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=270)
    states = [h.checkpoint(path) for path in sorted((root/'rst').glob('*.rst'))]
    h.require(abs(states[-1]['time']-END) < 1e-13, 'Incomplete parallel-viscosity wave')
    phase = np.exp(-2j*np.pi*(np.arange(n)+.5)/n)

    def mode(state):
        density = state['u'][0, 0, 0]
        velocity = state['u'][1, 0, 0]/density
        pressure = (GAMMA-1)*state['internal'][0, 0]
        return 2*np.array([np.mean((density-1)*phase), np.mean(velocity*phase),
                           np.mean((pressure-PRESSURE)*phase)])

    k = 2*np.pi
    matrix = np.array([[0., -1j*k, 0.], [0., -(4./3.)*NU*k*k, -1j*k],
                       [0., -1j*k*GAMMA*PRESSURE, 0.]], dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    y0 = mode(states[0])
    initial_coefficient = np.linalg.solve(eigenvectors, y0)
    index = int(np.argmax(abs(initial_coefficient)))
    h.require(abs(eigenvalues[index].imag) > 0., 'Selected entropy instead of sound mode')
    rows, coefficients = [], []
    for state in states:
        measured = mode(state)
        expected = expm(matrix*state['time'])@y0
        error = float(np.linalg.norm(measured-expected)/np.linalg.norm(y0))
        coefficients.append(np.linalg.solve(eigenvectors, measured)[index] /
                            initial_coefficient[index])
        h.require(error < .05, 'Parallel viscous wave error')
        h.require(abs(state['total']-states[0]['total'])/states[0]['total'] <
                  512*np.finfo(float).eps*(state['cycle']+1), 'Parallel viscous energy')
        rows.append(dict(time=state['time'], normalized_mode_error=error,
                         mode_real=measured.real.tolist(),
                         mode_imag=measured.imag.tolist()))
    times = np.array([row['time'] for row in rows])
    damping = -float(np.polyfit(times, np.log(abs(np.array(coefficients))), 1)[0])
    frequency = abs(float(np.polyfit(times, np.unwrap(np.angle(coefficients)), 1)[0]))
    gamma = (2./3.)*NU*k*k
    omega = float(np.sqrt(GAMMA*PRESSURE*k*k-gamma*gamma))
    events = [list(map(int, line.split()))
              for line in (root/'alfven.log').read_text().splitlines()
              if line.strip() and not line.startswith('#')]
    h.require(all(not any(row[1:6]) for row in events), 'Parallel viscous EOS floors')
    row = dict(status='PASS', name=name, n=n, method=method, constant=constant,
               samples=rows, damping=damping, frequency=frequency,
               damping_relative_error=abs(damping/gamma-1),
               frequency_relative_error=abs(frequency/omega-1))
    (root/'result.json').write_text(json.dumps(row, indent=2)+'\n')
    print(name, 'mode error', rows[-1]['normalized_mode_error'],
          'damping error', row['damping_relative_error'], flush=True)
    return row


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
        if manifest:
            subprocess.run(['sha256sum', '--quiet', '-c', str(manifest)], check=True)
    report = dict(status='RUNNING', cases=[], binary_sha256=h.sha(args.binary),
                  source_manifest_sha256=(h.sha(args.source_manifest)
                                          if args.source_manifest else None),
                  harness_sha256=h.sha(__file__))
    (args.root/'gates.json').write_text(json.dumps(dict(
        mode_error_max=.05, spatial_ratio_max=.4,
        fine_damping_frequency_relative_max=.02,
        constant_weak_mode_difference_max=1e-6), indent=2)+'\n')
    try:
        for method in args.methods:
            results = []
            for n in (32, 64, 128):
                row = run(args, n, method)
                results.append(row)
                report['cases'].append(row)
            errors = [row['samples'][-1]['normalized_mode_error'] for row in results]
            ratios = [b/a for a, b in zip(errors, errors[1:])]
            report[method+'_spatial_ratios'] = ratios
            h.require(max(ratios) < .4, 'Parallel viscous convergence '+str(ratios))
            h.require(max(results[-1]['damping_relative_error'],
                          results[-1]['frequency_relative_error']) < .02,
                      'Parallel viscous damping/frequency')
            candidate = results[1]
            if method == 'sts':
                candidate = run(args, 64, method, matched=True)
                report['cases'].append(candidate)
            baseline = run(args, 64, method, constant=True, matched=method == 'sts')
            report['cases'].append(baseline)

            def vector(row):
                value = row['samples'][-1]
                return np.array(value['mode_real'])+1j*np.array(value['mode_imag'])
            initial = candidate['samples'][0]
            norm = np.linalg.norm(np.array(initial['mode_real']) +
                                  1j*np.array(initial['mode_imag']))
            delta = float(np.linalg.norm(vector(baseline)-vector(candidate))/norm)
            report[method+'_constant_mode_difference'] = delta
            h.require(delta < 1e-6, 'Parallel Spitzer weak-wave limit')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
