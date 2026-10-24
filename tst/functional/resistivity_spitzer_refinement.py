#!/usr/bin/env python3
"""Spitzer RK2/RKL2 helix across static coarse/fine interfaces."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

import resistivity_refinement as refinement
import resistivity_sts as sts
h = refinement.h
h.AMP, h.PRESSURE, h.END = 1., .1, .02
sts.s.END, sts.s.PRES, sts.s.ETA0 = .02, .1, .25*.1**1.5
reference = sts.reference((2*np.pi)**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--methods', nargs='+', choices=('explicit', 'sts'),
                        default=['explicit', 'sts'])
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
        spatial_ratio_max=.4, positivity='raw density and internal energy',
        energy_relative='512*eps*(cycles+1)',
        divb='4096*eps*A0*Nfine*(cycles+1)'), indent=2)+'\n')
    try:
        for method in args.methods:
            errors = []
            for n in (32, 64):
                root = args.root/f'{method}-n{n}'
                root.mkdir()
                text = h.input_text(n, method, 2, 8 if n == 32 else 16, .2)
                text = text.replace('nghost = 3', 'nghost = 4')
                text = text.replace('helix_axis = 2', 'helix_axis = 1')
                text = text.replace('nx1 = 16\n', f'nx1 = {n}\n', 1)
                text = text.replace('eta_ohm = 0.25',
                                    'eta_ohm = 0\neta_spitzer = true\n'
                                    f'spitzer_eta0_code = {sts.s.ETA0:.17g}')
                text += ('\n<mesh_refinement>\nrefinement = static\n'
                         'prolong_primitives = false\n<refined_region1>\nlevel = 1\n'
                         'x1min = 0.25\nx1max = 0.75\nx2min = 0\nx2max = 1\n'
                         '<output3>\nfile_type = log\ndcycle = 1\n')
                (root/'input.athinput').write_text(text)
                command = ['mpirun', '-np', str(args.ranks), str(args.binary),
                           '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
                           '-i', 'input.athinput']
                (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
                with (root/'athena.log').open('x') as log:
                    subprocess.run(command, cwd=root, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=270)
                rows = [refinement.read(path, lambda t: float(reference(t)[0]))
                        for path in sorted((root/'rst').glob('*.rst'))]
                h.require(abs(rows[-1]['time']-h.END) < 1e-13, 'Incomplete run')
                for row in rows:
                    h.require(row['levels'] == [0, 1], 'Missing refinement interface')
                    h.require(abs(row['total']-.65)/.65 <
                              512*np.finfo(float).eps*(row['cycle']+1),
                              'Total energy conservation')
                    h.require(row['divb'] <
                              4096*np.finfo(float).eps*(2*n)*(row['cycle']+1),
                              'Refinement-interface CT divergence')
                log = (root/'helix.log').read_text()
                events = [list(map(int, line.split())) for line in log.splitlines()
                          if line.strip() and not line.startswith('#')]
                h.require(all(not any(row[1:6]) for row in events), 'EOS floors/failures')
                result = dict(method=method, n=n, samples=rows, status='PASS')
                report['cases'].append(result)
                errors.append(rows[-1]['normalized_field_l2'])
                print(method, n, rows[-1], flush=True)
            ratio = errors[1]/errors[0]
            report[method+'_spatial_ratio'] = ratio
            h.require(ratio < .4, 'Spitzer coarse/fine convergence '+str(ratio))
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
