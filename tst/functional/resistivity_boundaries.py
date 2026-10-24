#!/usr/bin/env python3
"""Nonlinear Spitzer fast-wave evolution at physical domain boundaries.

Start a finite-amplitude fast magnetosonic perturbation with tangential B.
Compare RKL2 against two fine same-grid RK2 references for outflow, diode and
reflecting boundaries. Reflecting walls additionally close mass/energy budgets.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import resistivity_alfven as alfven
from resistivity_evolution import difference

h = alfven.h
alfven.AMP, alfven.END = .1, .1


def run(args, boundary, method, cfl, restart=None):
    name = f'{boundary}-{method}-cfl{cfl}'+('-restart' if restart else '')
    root = args.root/name
    root.mkdir()
    # B normal to the reflecting wall must vanish. Use a fast mode with
    # B=(0,1,0); the pgen interprets its time limit as wave periods.
    speed = np.sqrt(1.+(5./3.)*alfven.PRESSURE)
    # LinearWave rescales fresh-start periods, but reads physical restart time.
    input_tlim = alfven.END if restart else alfven.END*speed
    text = alfven.input_text(64, method, 1., True)
    text += (f'\n<mesh>\nix1_bc = {boundary}\nox1_bc = {boundary}\n'
             f'<time>\ncfl_number = {cfl}\ntlim = {input_tlim:.17g}\n'
             '<problem>\nwave_flag = 0\nbx0 = 0\nby0 = 1\nbz0 = 0\n')
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
    states = [h.checkpoint(p) for p in paths]
    initial, final = states[0], states[-1]
    h.require(abs(final['time']-alfven.END) < 1e-13, 'Incomplete boundary run')
    for state in states:
        gate = 512*np.finfo(float).eps*(state['cycle']+1)
        h.require(state['seam'] < gate, 'Block-interface CT mismatch')
        h.require(state['divb'] < gate*64*8, 'Physical-boundary CT divergence')
        if boundary == 'reflect':
            h.require(abs(state['total']-initial['total'])/initial['total'] < gate,
                      'Closed reflecting-wall energy budget')
            h.require(abs(float(state['u'][0].mean()) -
                          float(initial['u'][0].mean())) < gate,
                      'Closed reflecting-wall mass budget')
    events = [list(map(int, line.split()))
              for line in (root/'alfven.log').read_text().splitlines()
              if line.strip() and not line.startswith('#')]
    h.require(all(not any(row[1:6]) for row in events), 'Boundary EOS floors/failures')
    result = dict(status='PASS', name=name, boundary=boundary, method=method,
                  cfl=cfl, cycles=final['cycle'], total_energy=final['total'],
                  mean_density=float(final['u'][0].mean()),
                  internal_energy_min=float(final['internal'].min()),
                  divb=final['divb'])
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result, flush=True)
    return result, final, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--boundaries', nargs='+',
                        choices=('outflow', 'diode', 'reflect'),
                        default=['outflow', 'diode', 'reflect'])
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
        temporal_ratio_max=.4, reference_uncertainty_fraction=.15,
        finest_state_error_max=.005, restart_maxabs=1e-12,
        reflecting_budgets='512*eps*(cycles+1)'), indent=2)+'\n')
    try:
        for boundary in args.boundaries:
            explicit, errors = [], []
            for cfl in (.05, .025):
                row, state, _ = run(args, boundary, 'explicit', cfl)
                report['cases'].append(row)
                explicit.append(state)
            reference_error = difference(*explicit)
            for cfl in (.3, .15, .075):
                row, state, paths = run(args, boundary, 'sts', cfl)
                report['cases'].append(row)
                errors.append(difference(state, explicit[-1]))
                if cfl == .15:
                    checkpoint, uninterrupted = paths[4], state
            ratios = [b/a for a, b in zip(errors, errors[1:])]
            report[boundary] = dict(explicit_uncertainty=reference_error,
                                    sts_errors=errors, sts_error_ratios=ratios)
            h.require(reference_error < .15*min(errors), 'Boundary reference unresolved')
            h.require(max(ratios) < .4, 'Boundary time convergence '+str(ratios))
            h.require(errors[-1] < .005, 'Boundary RKL2/explicit discrepancy')
            row, state, _ = run(args, boundary, 'sts', .15, restart=checkpoint)
            report['cases'].append(row)
            delta = max(float(np.max(abs(state['u']-uninterrupted['u']))),
                        *(float(np.max(abs(a-b)))
                          for a, b in zip(state['faces'], uninterrupted['faces'])))
            report[boundary]['restart_maxabs'] = delta
            h.require(delta < 1e-12, 'Boundary restart discrepancy')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
