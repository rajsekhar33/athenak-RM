#!/usr/bin/env python3
"""Rotate the nonlinear viscous-heating helix into the y and z directions."""
import argparse
import json
from pathlib import Path

import viscosity_spitzer as v


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
    report = dict(status='RUNNING', cases=[], binary_sha256=v.h.sha(args.binary))
    try:
        for axis in (2, 3):
            for method in args.methods:
                row, _, _ = v.launch(args, 'mhd', method,
                                     f'axis{axis}-{method}', n=32, axis=axis)
                report['cases'].append(row)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
