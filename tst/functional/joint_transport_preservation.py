#!/usr/bin/env python3
"""Compare established conduction paths with the preserved pair executable."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

import resistivity as h


def run(args, binary, label, case, text):
    root = args.root/(case+'-'+label)
    root.mkdir()
    (root/'input.athinput').write_text(text)
    command = ['mpirun', '-np', str(args.ranks), str(binary.resolve()),
               '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
               '-i', 'input.athinput']
    (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'athena.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=270)
    state = h.checkpoint(sorted((root/'rst').glob('*.rst'))[-1])
    h.require(abs(state['time']-.02) < 1e-13, 'Incomplete preservation run')
    gate = 512*np.finfo(float).eps*(state['cycle']+1)
    h.require(abs(state['total']/.99375-1.) < gate, 'Preservation energy budget')
    h.require(state['divb'] < gate*32*8, 'Preservation CT divergence')
    events = [list(map(int, line.split())) for line in (root/'visc_helix.log').read_text()
              .splitlines() if line.strip() and not line.startswith('#')]
    h.require(all(not any(row[1:6]) for row in events), 'Preservation EOS floors')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline-binary', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--cases', nargs='+',
                        choices=('pure-spitzer', 'pure-saturated', 'mixed', 'joint'),
                        default=['pure-spitzer', 'pure-saturated', 'mixed'])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    original = args.input.read_text()
    report = dict(status='RUNNING', cases=[], binary_sha256=h.sha(args.binary),
                  baseline_binary_sha256=h.sha(args.baseline_binary),
                  input_sha256=h.sha(args.input), harness_sha256=h.sha(Path(__file__)))
    try:
        for case in args.cases:
            if case in ('mixed', 'joint'):
                text = original
                if case == 'mixed':
                    text += '\n<mhd>\nconductivity_integrator = explicit\n'
            else:
                text = '\n'.join(line for line in original.splitlines()
                                 if not line.startswith(('spitzer_mu0_', 'spitzer_eta0_',
                                                         'spitzer_viscosity_geometry')))
                text += ('\n<mhd>\nmu_spitzer = false\nnu_iso = 0\n'
                         'viscosity_integrator = explicit\neta_spitzer = false\n'
                         'eta_ohm = 0\nresistivity_integrator = explicit\n')
                if case == 'pure-saturated':
                    text += 'conduction_saturation = cowie_mckee\nsaturation_phi = .001\n'
            a = run(args, args.binary, 'current', case, text)
            b = run(args, args.baseline_binary, 'baseline', case, text)
            delta = max(float(np.max(abs(a['u']-b['u']))),
                        *(float(np.max(abs(x-y))) for x, y in zip(a['faces'], b['faces'])))
            row = dict(name=case, status='PASS', maxabs=delta)
            report['cases'].append(row)
            h.require(delta < 1e-12, 'Established conduction path changed: '+case)
            print(row, flush=True)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
