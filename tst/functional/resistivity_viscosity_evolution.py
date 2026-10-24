#!/usr/bin/env python3
"""Joint nonlinear viscosity/resistivity in multidimensional dynamical MHD.

Modulated density gives spatial temperature/coefficient gradients. A circular
velocity/B helix then develops nonzero longitudinal fields and compressive
motions. Compare RKL2 with two fine same-grid RK2 references and exact restarts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from resistivity_evolution import difference
import viscosity_spitzer as v
import resistivity_refinement as refined

END = .02


def run(args, method, cfl, restart=None):
    name = f'dim{args.ndim}-{method}-cfl{cfl}'+('-restart' if restart else '')
    root = args.root/name
    sections = v.input_text(args.n, 'mhd', method, cfl, 1, 1.)
    v.add_conduction(sections, args, 'mhd', method)
    sections['time']['tlim'] = END
    sections['output1']['dt'] = END/4
    sections['problem'].update(helix_bfield=0., helix_magnetic_amp=.5,
                               helix_density_amp=.2)
    for a, value in enumerate(args.bulk, 1):
        sections['problem'][f'helix_bulk_v{a}'] = value
    sections['mhd'].update(spitzer_mu0_code=.025/.1**2.5, eta_spitzer='true',
                           spitzer_eta0_code=.025*.1**1.5, resistivity_integrator=method,
                           rsolver=args.rsolver, spitzer_viscosity_geometry=args.geometry)
    for axis in range(1, args.ndim+1):
        sections['mesh'][f'nx{axis}'] = args.n
        sections['meshblock'][f'nx{axis}'] = min(args.n//2, 16)
    if args.static_refinement:
        sections['mesh']['nghost'] = 4
        sections['meshblock'].update(nx1=args.n//4, nx2=args.n//4)
        sections['mesh_refinement'] = dict(refinement='static', prolong_primitives='false')
        sections['refined_region1'] = dict(
            level=1, x1min=.25, x1max=.75, x2min=0., x2max=1.)
    text = '\n'.join(f'<{block}>\n'+'\n'.join(f'{k} = {value}' for k, value
                     in values.items())+'\n' for block, values in sections.items())
    resumed_from = None
    reused = False
    if args.resume_root:
        previous = args.resume_root/name
        if previous.is_dir():
            v.h.require((previous/'input.athinput').read_text() == text,
                        'Resume input differs from requested evolution')
            saved = sorted((previous/'rst').glob('*.rst'))
            if saved:
                last = read_state(args, saved[-1])
                if abs(last['time']-END) < 1e-13:
                    root, reused = previous, True
                else:
                    v.h.require(last['time'] < END, 'Resume checkpoint exceeds target')
                    resumed_from = saved[-1]
                    restart = resumed_from
    if not reused:
        root.mkdir()
        (root/'input.athinput').write_text(text)
    command = ['mpirun', '-np', str(args.ranks), str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank', '-t',
               f'00:{args.walltime_seconds//60:02d}:{args.walltime_seconds % 60:02d}',
               '-i', 'input.athinput']
    if restart:
        command += ['-r', os.path.relpath(restart, root)]
    if not reused:
        (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (root/'athena.log').open('x') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.walltime_seconds+30)
    paths = sorted((root/'rst').glob('*.rst'))
    states = [read_state(args, p) for p in paths]
    v.h.require(abs(states[-1]['time']-END) < 1e-13, 'Incomplete joint MHD evolution')
    # Composite midpoint quadrature of the bulk/shear cross term differs from
    # its continuum mean. Conservation compares the actual initial cell sum.
    expected_energy = (states[0]['total'] if args.static_refinement else
                       .4+.5*sum(value*value for value in args.bulk))
    for state in states:
        gate = 512*np.finfo(float).eps*(state['cycle']+1)
        v.h.require(abs(state['total']/expected_energy-1.) < gate, 'Joint MHD energy budget')
        mass = state.get('mass', float(state['u'][0].mean()))
        v.h.require(abs(mass-1.) < gate, 'Joint MHD mass budget')
        v.h.require(state['divb'] < gate*args.n*8, 'Joint MHD CT divergence')
    final = states[-1]
    longitudinal = float(np.max(abs(final['bcc'][0])))
    v.h.require(longitudinal > 1e-7, 'Joint transverse coefficient gradients absent')
    logs = [root/'visc_helix.log']
    if resumed_from:
        logs.append(args.resume_root/name/'visc_helix.log')
    events = [list(map(int, line.split())) for log in logs
              for line in log.read_text().splitlines()
              if line.strip() and not line.startswith('#')]
    v.h.require(all(not any(row[1:6]) for row in events), 'Joint MHD EOS floors')
    row = dict(status='PASS', name=name, method=method, cfl=cfl, ndim=args.ndim, n=args.n,
               cycle=final['cycle'], longitudinal_field_max=longitudinal,
               min_internal=float(final['internal'].min()), divb=final['divb'],
               checkpoint=str(paths[-1]))
    if resumed_from:
        row['resumed_from'] = str(resumed_from)
        row['resumed_checkpoint_sha256'] = v.h.sha(resumed_from)
    if reused:
        row['reused_run'] = str(root)
    else:
        (root/'result.json').write_text(json.dumps(row, indent=2)+'\n')
    print(row, flush=True)
    return row, final, paths


def read_state(args, path):
    return refined.checkpoint(path) if args.static_refinement else v.h.checkpoint(path)


def main():
    global END
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--ndim', type=int, choices=(2, 3), default=2)
    parser.add_argument('--n', type=int, choices=(16, 32, 64), default=32)
    parser.add_argument('--geometry', choices=('isotropic', 'parallel'), default='isotropic')
    parser.add_argument('--rsolver', choices=('hlld', 'hlle', 'llf'), default='hlld')
    parser.add_argument('--bulk', nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--static-refinement', action='store_true')
    parser.add_argument('--end-time', type=float, default=.02)
    parser.add_argument('--resume-root', type=Path)
    parser.add_argument('--walltime-seconds', type=int, default=240)
    parser.add_argument('--reference-cfl', nargs=2, type=float,
                        default=[.003125, .0015625])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    v.conduction_arguments(parser)
    args = parser.parse_args()
    v.h.require(np.isfinite(args.end_time) and args.end_time > 0., 'Invalid end time')
    v.h.require(not args.static_refinement or
                (args.ndim == 2 and args.geometry == 'isotropic'),
                'Composite test supports 2D isotropic viscosity')
    END = args.end_time
    args.binary = args.binary.resolve()
    if args.resume_root:
        args.resume_root = args.resume_root.resolve()
    v.h.require(0 < args.walltime_seconds < 3600, 'Invalid local execution limit')
    if args.resume_root:
        old = json.loads((args.resume_root/'summary.json').read_text())
        v.h.require(old['binary_sha256'] == v.h.sha(args.binary),
                    'Resume executable differs from original campaign')
    args.root.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', cases=[], binary_sha256=v.h.sha(args.binary),
                  harness_sha256=v.h.sha(Path(__file__)),
                  parameters=dict(n=args.n, ndim=args.ndim, bulk=args.bulk,
                                  static_refinement=args.static_refinement, end_time=END,
                                  geometry=args.geometry, rsolver=args.rsolver,
                                  conduction=args.conduction,
                                  conduction_geometry=args.conduction_geometry,
                                  conduction_limiter=args.conduction_limiter,
                                  conduction_coefficient=args.conduction_coefficient,
                                  saturation_phi=args.saturation_phi))
    (args.root/'gates.json').write_text(json.dumps(dict(
        temporal_ratio_max=.4, reference_uncertainty_fraction=.15,
        state_error_max=.005, restart_maxabs=1e-12, backend_difference_max=1e-11),
        indent=2)+'\n')
    try:
        if args.reference_root:
            reference = json.loads((args.reference_root/'summary.json').read_text())
            v.h.require(reference['status'] == 'PASS', 'Joint reference campaign failed')
            defaults = dict(static_refinement=False, end_time=.02)
            for key, value in report['parameters'].items():
                v.h.require(reference['parameters'].get(key, defaults.get(key)) == value,
                            'Joint reference parameter mismatch: '+key)
            refrow = min((c for c in reference['cases'] if c['method'] == 'explicit'),
                         key=lambda c: c['cfl'])
            v.h.require(reference['parameters'].get('static_refinement', False) ==
                        args.static_refinement, 'Reference refinement mismatch')
            ref = read_state(args, Path(refrow['checkpoint']))
            uncertainty = reference['reference_uncertainty']
            report['reference_summary_sha256'] = v.h.sha(args.reference_root/'summary.json')
        else:
            refs = []
            for cfl in args.reference_cfl:
                row, state, _ = run(args, 'explicit', cfl)
                report['cases'].append(row)
                refs.append(state)
            ref, uncertainty = refs[-1], difference(*refs)
        errors = []
        for cfl in (.3, .15, .075):
            row, state, paths = run(args, 'sts', cfl)
            report['cases'].append(row)
            errors.append(difference(state, ref))
            if args.reference_root:
                refrow = next(c for c in reference['cases'] if c['name'] == row['name'])
                delta = difference(state, read_state(args, Path(refrow['checkpoint'])))
                row['backend_relative_difference'] = delta
                v.h.require(delta < 1e-11, 'Joint MHD backend discrepancy')
            if cfl == .15:
                checkpoint, uninterrupted = paths[2], state
        ratios = [y/x for x, y in zip(errors, errors[1:])]
        report.update(reference_uncertainty=uncertainty, sts_errors=errors, ratios=ratios)
        v.h.require(uncertainty < .15*min(errors), 'Joint MHD reference unresolved')
        v.h.require(max(ratios) < .4 and errors[-1] < .005, 'Joint MHD time convergence')
        row, state, _ = run(args, 'sts', .15, restart=checkpoint)
        report['cases'].append(row)
        delta = max(float(np.max(abs(state['u']-uninterrupted['u']))),
                    *(float(np.max(abs(x-y)))
                      for x, y in zip(state['faces'], uninterrupted['faces'])))
        report['restart_maxabs'] = delta
        v.h.require(delta < 1e-12, 'Joint MHD restart discrepancy')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
