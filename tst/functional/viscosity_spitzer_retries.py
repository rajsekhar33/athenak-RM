#!/usr/bin/env python3
"""Rapid viscous heating: nonlinear stage-budget recovery and analytic heating."""
import argparse
import json
from pathlib import Path
import subprocess

import viscosity_spitzer as v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--cfl', type=float, default=.4)
    parser.add_argument('--n', type=int, choices=(32, 64, 128), default=64)
    parser.add_argument('--fluid', choices=('hydro', 'mhd'), default='mhd')
    v.conduction_arguments(parser)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    for manifest in (args.source_manifest, args.binary_manifest):
        if manifest:
            subprocess.run(['sha256sum', '--quiet', '-c', str(manifest)], check=True)
    v.AMPLITUDE, v.END = 1.5, .005
    report = dict(status='RUNNING', cases=[], binary_sha256=v.h.sha(args.binary),
                  harness_sha256=v.h.sha(__file__),
                  source_manifest_sha256=(v.h.sha(args.source_manifest)
                                          if args.source_manifest else None))
    (args.root/'gates.json').write_text(json.dumps(dict(
        heat_gain_relative_max=.01, field='.5*(2*pi/N)^2',
        raw_state='positive thermal energy, no EOS floors',
        retry_count='reported from completed trial log'), indent=2)+'\n')
    try:
        row, _, _ = v.launch(args, args.fluid, 'sts', 'strong-heating',
                             n=args.n, cfl=args.cfl)
        report['cases'].append(row)
        final = row['samples'][-1]
        error = abs(final['thermal_gain']/final['expected_thermal_gain']-1.)
        report['heat_gain_relative_error'] = error
        v.h.require(error < .01, 'Strong heating error exceeds one percent')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
