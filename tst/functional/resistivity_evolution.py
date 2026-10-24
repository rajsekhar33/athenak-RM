#!/usr/bin/env python3
"""Evolve spatial Spitzer gradients with ideal MHD and compare RKL2 to RK2.

The periodic helix starts force-free, at uniform pressure and modulated density.
Its temperature-dependent diffusivity develops a longitudinal magnetic field.
The equations and edge-coefficient placement follow the Athena++ diffusion
operator (Stone et al. 2020, doi:10.3847/1538-4365/ab929b). This additional test
measures temporal convergence to the same-grid explicit solution, conservation,
CT consistency and restart recovery; there is no closed-form nonlinear solution.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import resistivity as h
END = .01


def difference(a, b):
    if 'native_fields' in a or 'native_fields' in b:
        h.require(np.array_equal(a['locs'], b['locs']) and
                  np.array_equal(a['weights'], b['weights']),
                  'Composite reference topology mismatch')
        weights = b['weights']
        fields = list(zip(a['native_fields'], b['native_fields']))

        def norm(value):
            return np.sum(np.mean(value**2, axis=tuple(range(1, value.ndim)))*weights)

        numerator = sum(norm(x-y) for x, y in fields)
        return float(np.sqrt(numerator/sum(norm(y) for _, y in fields)))
    fields = [(a['u'], b['u'])]+list(zip(a['faces'], b['faces']))
    return float(np.sqrt(sum(np.mean((x-y)**2) for x, y in fields) /
                         sum(np.mean(y**2) for _, y in fields)))


def run(args, ndim, method, cfl, restart=None):
    n = 32
    name = f'dim{ndim}-{method}-cfl{cfl}'+('-restart' if restart else '')
    root = args.root/name
    root.mkdir()
    text = h.input_text(n, method, 1, 16, cfl)
    text = text.replace('amp = 0.5', 'amp = 1.0').replace('pres = 1.0', 'pres = 0.1')
    text = text.replace('eta_ohm = 0.25',
                        'eta_ohm = 0\neta_spitzer = true\n'
                        f'spitzer_eta0_code = {.25*.1**1.5:.17g}')
    text = text.replace('tlim = 0.1', f'tlim = {END}')
    text = text.replace('dt = 0.025', f'dt = {END/2}')
    text += ('\n<problem>\nhelix_density_amp = 0.2\n'
             '<output3>\nfile_type = log\ndcycle = 1\n')
    for axis in range(2, ndim+1):
        text += f'\n<mesh>\nnx{axis} = {n}\n<meshblock>\nnx{axis} = 16\n'
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
    final = h.checkpoint(paths[-1])
    h.require(abs(final['time']-END) < 1e-13, 'Incomplete evolution')
    lines = (root/'helix.log').read_text().splitlines()
    rows = [list(map(int, line.split())) for line in lines
            if line.strip() and not line.startswith('#')]
    # EventLogOutput writes a header-only file when all event counts are zero.
    h.require(all(not any(row[1:6]) for row in rows), 'EOS floors/failures')
    expected_energy = .1/(2./3.)+.5
    energy_error = abs(final['total']-expected_energy)/expected_energy
    h.require(energy_error < 512*np.finfo(float).eps*(final['cycle']+1),
              'Periodic total energy changed')
    div_gate = 4096*np.finfo(float).eps*n*(final['cycle']+1)
    h.require(final['divb'] < div_gate, 'CT divergence')
    h.require(max(final['seam'], final['periodic']) < div_gate/n, 'CT face mismatch')
    longitudinal = float(np.max(abs(final['faces'][0])))
    h.require(longitudinal > 1e-4, 'Transverse diffusivity gradients not exercised')
    mass_error = abs(float(final['u'][0].mean())-1.)
    h.require(mass_error < 512*np.finfo(float).eps*(final['cycle']+1),
              'Periodic mass changed')
    row = dict(name=name, ndim=ndim, n=n, method=method, cfl=cfl,
               cycles=final['cycle'], energy_relative_error=energy_error,
               mass_absolute_error=mass_error, divb=final['divb'],
               longitudinal_field_max=longitudinal,
               internal_energy_min=float(final['internal'].min()),
               checkpoint=str(paths[-1]), status='PASS')
    (root/'result.json').write_text(json.dumps(row, indent=2)+'\n')
    print(row, flush=True)
    return row, final, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--ndim', type=int, choices=(2, 3), default=2)
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
        refinement_ratio_max=.4, reference_uncertainty_fraction=.15,
        finest_relative_state_error_max=.005, restart_maxabs=1e-12,
        conservation='512*eps*(cycles+1)',
        ct_divergence='4096*eps*N*(cycles+1)'), indent=2)+'\n')
    try:
        explicit = []
        for cfl in (.1, .05):
            row, state, _ = run(args, args.ndim, 'explicit', cfl)
            report['cases'].append(row)
            explicit.append(state)
        uncertainty = difference(*explicit)
        errors = []
        restart_state = restart_path = None
        for cfl in (.4, .2, .1):
            row, state, paths = run(args, args.ndim, 'sts', cfl)
            error = difference(state, explicit[-1])
            row['explicit_reference_error'] = error
            report['cases'].append(row)
            errors.append(error)
            if cfl == .2:
                restart_state, restart_path = state, paths[1]
        ratios = [b/a for a, b in zip(errors, errors[1:])]
        report.update(reference_uncertainty=uncertainty, sts_errors=errors,
                      sts_error_ratios=ratios)
        h.require(uncertainty < .15*min(errors), 'Explicit reference is unresolved')
        h.require(max(ratios) < .4, 'RKL2 timestep convergence '+str(ratios))
        h.require(errors[-1] < .005, 'RKL2/explicit field discrepancy')
        row, state, _ = run(args, args.ndim, 'sts', .2, restart=restart_path)
        report['cases'].append(row)
        restart_error = max(float(np.max(abs(state['u']-restart_state['u']))),
                            *(float(np.max(abs(a-b)))
                              for a, b in zip(state['faces'], restart_state['faces'])))
        report['restart_maxabs'] = restart_error
        h.require(restart_error < 1e-12, 'Restart field discrepancy')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
