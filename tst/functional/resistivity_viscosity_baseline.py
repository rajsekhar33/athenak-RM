#!/usr/bin/env python3
"""Compare explicit joint transport with the preserved pre-coupling executable.

Both binaries restart the identical cycle-zero state, so the new coupled helix
initializer does not affect the comparison. This isolates driver coupling
changes from inherited explicit-MHD behavior.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import resistivity as h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline-binary', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', cases=[], binary_sha256=h.sha(args.binary),
                  baseline_binary_sha256=h.sha(args.baseline_binary))
    try:
        for cfl in (.0125, .00625):
            reference = args.reference_root/f'dim2-explicit-cfl{cfl}'
            initial = sorted((reference/'rst').glob('*.rst'))[0].resolve()
            states = []
            for label, binary in [('current', args.binary), ('baseline', args.baseline_binary)]:
                root = args.root/f'{label}-cfl{cfl}'
                root.mkdir()
                (root/'input.athinput').write_text((reference/'input.athinput').read_text())
                command = ['mpirun', '-np', str(args.ranks), str(binary.resolve()),
                           '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
                           '-i', 'input.athinput', '-r', os.path.relpath(initial, root)]
                (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
                with (root/'athena.log').open('x') as log:
                    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=270)
                path = sorted((root/'rst').glob('*.rst'))[-1]
                state = h.checkpoint(path)
                h.require(abs(state['time']-.02) < 1e-13, 'Incomplete baseline comparison')
                states.append(state)
            a, b = states
            delta = max(float(np.max(abs(a['u']-b['u']))),
                        *(float(np.max(abs(x-y))) for x, y in zip(a['faces'], b['faces'])))
            row = dict(cfl=cfl, maxabs=delta, status='PASS',
                       initial_checkpoint_sha256=h.sha(initial))
            report['cases'].append(row)
            h.require(delta < 1e-12, 'Explicit transport changed with joint STS implementation')
            print(row, flush=True)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
