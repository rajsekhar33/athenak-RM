#!/usr/bin/env python3
"""Repeat nonlinear boundary STS cases against saved fine explicit references.

The reference campaign must have passed temporal convergence, conservation,
positivity and restart checks. This runner checks the same physical solution
and RKL2 timestep convergence on a second backend, including exact restarts.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import viscosity_spitzer_boundaries as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--ohmic', choices=('off', 'constant', 'spitzer'), default='off')
    parser.add_argument('--oblique-field', action='store_true')
    b.v.conduction_arguments(parser)
    parser.add_argument('--boundaries', nargs='+',
                        choices=('periodic', 'outflow', 'diode', 'reflect'),
                        default=['periodic', 'outflow', 'diode', 'reflect'])
    parser.add_argument('--geometries', nargs='+', choices=('isotropic', 'parallel'),
                        default=['isotropic', 'parallel'])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    args = parser.parse_args()
    b.h.require(not args.oblique_field or 'reflect' not in args.boundaries,
                'Oblique background field is incompatible with reflecting CT parity')
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    reference = json.loads((args.reference_root/'summary.json').read_text())
    b.h.require(reference['status'] == 'PASS', 'Reference campaign must pass')
    for key in ('conduction', 'conduction_geometry', 'conduction_limiter', 'oblique_field'):
        b.h.require(reference.get(key, getattr(args, key)) == getattr(args, key),
                    'Boundary reference option mismatch: '+key)
    report = dict(status='RUNNING', cases=[], binary_sha256=b.h.sha(args.binary),
                  reference_summary_sha256=b.h.sha(args.reference_root/'summary.json'),
                  reference_binary_sha256=reference['binary_sha256'])
    try:
        for geometry in args.geometries:
            for boundary in args.boundaries:
                label = geometry+'-'+boundary
                refrow = next(c for c in reference['cases'] if
                              c['name'] == label+'-explicit-0.00625')
                state_ref = b.h.checkpoint(Path(refrow['checkpoint']))
                errors, backend = [], []
                for cfl in (.3, .15, .075):
                    row, state, paths = b.run(args, geometry, boundary, 'sts', cfl)
                    report['cases'].append(row)
                    errors.append(b.difference(state, state_ref))
                    matching = next(c for c in reference['cases'] if c['name'] == row['name'])
                    backend.append(b.difference(state, b.h.checkpoint(
                        Path(matching['checkpoint']))))
                    if cfl == .15:
                        checkpoint, uninterrupted = paths[2], state
                ratios = [y/x for x, y in zip(errors, errors[1:])]
                result = dict(sts_errors=errors, ratios=ratios,
                              backend_relative_errors=backend,
                              reference_uncertainty=reference[label]['reference_uncertainty'])
                report[label] = result
                b.h.require(max(ratios) < .4, 'Backend wave temporal convergence')
                b.h.require(result['reference_uncertainty'] < .15*min(errors),
                            'Backend wave reference uncertainty')
                b.h.require(errors[-1] < .005, 'Backend wave accuracy')
                b.h.require(max(backend) < 1e-11, 'Backend wave field discrepancy')
                row, state, _ = b.run(args, geometry, boundary, 'sts', .15,
                                      restart=checkpoint)
                report['cases'].append(row)
                delta = max(float(np.max(abs(state['u']-uninterrupted['u']))),
                            *(float(np.max(abs(x-y)))
                              for x, y in zip(state['faces'], uninterrupted['faces'])))
                result['restart_maxabs'] = delta
                b.h.require(delta < 1e-12, 'Backend wave restart discrepancy')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
