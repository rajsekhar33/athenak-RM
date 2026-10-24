#!/usr/bin/env python3
"""Finite-amplitude magnetosonic waves with temperature-dependent viscosity.

Compare RKL2 to two fine, same-grid RK2 references at periodic, open and
reflecting boundaries. A tangential magnetic field is compatible with the
reflecting CT parity; compression exercises the parallel Braginskii stress.
The wave starts with amplitude 0.1, so the coefficient evolves nonlinearly.
References: Berlok et al. (2020), doi:10.1093/mnras/stz3115; Stone et al.
(2020), doi:10.3847/1538-4365/ab929b. Physical-boundary variants are additional
checks of the same transport equations, with numerical temporal references.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import resistivity_alfven as a
from resistivity_evolution import difference
import viscosity_spitzer as v

h = a.h
END = .025


def run(args, geometry, boundary, method, cfl, restart=None):
    name = f'{geometry}-{boundary}-{method}-{cfl}'+('-restart' if restart else '')
    root = args.root/name
    root.mkdir()
    text = a.input_text(32, 'explicit', 1., False)
    bx = .5 if args.oblique_field else 0.
    sound2 = (5./3.)*a.PRESSURE
    total = 1.+bx*bx+sound2
    speed = np.sqrt(.5*(total+np.sqrt(total*total-4*sound2*bx*bx)))
    # LinearWave's fresh-start time is a number of periods, restart time is physical.
    tlim = END if restart else END*speed
    text += (f'\n<mesh>\nix1_bc = {boundary}\nox1_bc = {boundary}\n'
             f'<time>\ncfl_number = {cfl}\ntlim = {tlim:.17g}\n'
             f'sts_integrator = {"rkl2" if method == "sts" else "none"}\n'
             '<mhd>\neta_ohm = 0\nmu_spitzer = true\n'
             f'spitzer_mu0_code = {.02/a.PRESSURE**2.5:.17g}\n'
             f'spitzer_viscosity_geometry = {geometry}\nviscosity_integrator = {method}\n'
             f'<problem>\nwave_flag = 0\nbx0 = {bx}\nby0 = 1\nbz0 = 0\namp = .1\n'
             f'<output1>\ndt = {END/4}\n')
    ohmic = getattr(args, 'ohmic', 'off')
    if ohmic != 'off':
        text += f'<mhd>\nresistivity_integrator = {method}\n'
        if ohmic == 'constant':
            text += 'eta_ohm = .02\n'
        else:
            text += f'eta_spitzer = true\nspitzer_eta0_code = {.02*a.PRESSURE**1.5:.17g}\n'
    conduction = dict(time={}, mhd={})
    v.add_conduction(conduction, args, 'mhd', method, temperature=a.PRESSURE)
    text += '\n'.join(f'<{block}>\n'+'\n'.join(f'{k} = {value}' for k, value
                      in values.items())+'\n' for block, values in conduction.items())
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
    h.require(abs(final['time']-END) < 1e-13, 'Incomplete viscous wave')
    for state in states:
        gate = 512*np.finfo(float).eps*(state['cycle']+1)
        h.require(state['seam'] < gate, 'Viscous wave CT interface mismatch')
        h.require(state['divb'] < gate*32*8, 'Viscous wave CT divergence')
        if boundary in ('periodic', 'reflect'):
            h.require(abs(state['total']-initial['total'])/initial['total'] < gate,
                      'Closed viscous wave energy budget')
            h.require(abs(float(state['u'][0].mean()-initial['u'][0].mean())) < gate,
                      'Closed viscous wave mass budget')
    events = [list(map(int, line.split())) for line in (root/'alfven.log').read_text()
              .splitlines() if line.strip() and not line.startswith('#')]
    h.require(all(not any(row[1:6]) for row in events), 'Viscous wave EOS floors')
    temp = (2./3.)*final['internal']/final['u'][0]
    row = dict(status='PASS', name=name, geometry=geometry, boundary=boundary,
               method=method, cfl=cfl, cycle=final['cycle'], divb=final['divb'],
               min_internal=float(final['internal'].min()),
               viscosity_contrast=float((temp.max()/temp.min())**2.5),
               checkpoint=str(paths[-1]))
    (root/'result.json').write_text(json.dumps(row, indent=2)+'\n')
    print(row, flush=True)
    return row, final, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--ohmic', choices=('off', 'constant', 'spitzer'), default='off')
    parser.add_argument('--oblique-field', action='store_true')
    v.conduction_arguments(parser)
    parser.add_argument('--geometries', nargs='+', choices=('isotropic', 'parallel'),
                        default=['isotropic', 'parallel'])
    parser.add_argument('--boundaries', nargs='+',
                        choices=('periodic', 'outflow', 'diode', 'reflect'),
                        default=['periodic', 'outflow', 'diode', 'reflect'])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    args = parser.parse_args()
    h.require(not args.oblique_field or 'reflect' not in args.boundaries,
              'Oblique background field is incompatible with reflecting CT parity')
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', cases=[], binary_sha256=h.sha(args.binary),
                  conduction=args.conduction, conduction_geometry=args.conduction_geometry,
                  conduction_limiter=args.conduction_limiter, oblique_field=args.oblique_field)
    (args.root/'gates.json').write_text(json.dumps(dict(
        temporal_ratio_max=.4, reference_uncertainty_fraction=.15,
        finest_state_error_max=.005, restart_maxabs=1e-12), indent=2)+'\n')
    try:
        for geometry in args.geometries:
            for boundary in args.boundaries:
                references, errors = [], []
                for cfl in (.0125, .00625):
                    row, state, _ = run(args, geometry, boundary, 'explicit', cfl)
                    report['cases'].append(row)
                    references.append(state)
                uncertainty = difference(*references)
                for cfl in (.3, .15, .075):
                    row, state, paths = run(args, geometry, boundary, 'sts', cfl)
                    report['cases'].append(row)
                    errors.append(difference(state, references[-1]))
                    if cfl == .15:
                        checkpoint, uninterrupted = paths[2], state
                ratios = [b/a for a, b in zip(errors, errors[1:])]
                report[geometry+'-'+boundary] = dict(
                    reference_uncertainty=uncertainty, sts_errors=errors, ratios=ratios)
                h.require(uncertainty < .15*min(errors), 'Viscous wave reference unresolved')
                h.require(max(ratios) < .4, 'Viscous wave time convergence '+str(ratios))
                h.require(errors[-1] < .005, 'Viscous wave RKL2/explicit discrepancy')
                row, state, _ = run(args, geometry, boundary, 'sts', .15,
                                    restart=checkpoint)
                report['cases'].append(row)
                delta = max(float(np.max(abs(state['u']-uninterrupted['u']))),
                            *(float(np.max(abs(x-y)))
                              for x, y in zip(state['faces'], uninterrupted['faces'])))
                report[geometry+'-'+boundary]['restart_maxabs'] = delta
                h.require(delta < 1e-12, 'Viscous wave restart discrepancy')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
