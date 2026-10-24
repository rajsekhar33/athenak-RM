#!/usr/bin/env python3
"""Spitzer RKL2 across runtime refinement and coarsening, with AMR restarts.

The nonlinear helical amplitude follows the independent energy/induction ODE.
Topology changes occur during evolution using AthenaK's existing location and
density criteria. Native checkpoints retain block levels and refinement ages.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import resistivity_spitzer_refinement as sr

h = sr.h


def run(args, n, scenario, restart=None):
    name = f'{scenario}-n{n}'+('-restart' if restart else '')
    root = args.root/name
    root.mkdir()
    text = h.input_text(n, 'sts', 2, 8 if n == 32 else 16, .2)
    text = text.replace('nghost = 3', 'nghost = 4')
    text = text.replace('helix_axis = 2', 'helix_axis = 1')
    text = text.replace('nx1 = 16\n', f'nx1 = {n}\n', 1)
    text = text.replace('eta_ohm = 0.25',
                        'eta_ohm = 0\neta_spitzer = true\n'
                        f'spitzer_eta0_code = {sr.sts.s.ETA0:.17g}')
    text += ('\n<mesh_refinement>\nrefinement = adaptive\nnum_levels = 2\n'
             'prolong_primitives = false\nmax_nmb_per_rank = 64\n'
             'refinement_interval = 3\nncycle_check = 1\n'
             '<output3>\nfile_type = log\ndcycle = 1\n')
    if n == 32:
        text += '<output1>\ndcycle = 2\n'
    if scenario == 'refine':
        text += ('<amr_criterion1>\nmethod = location\nlocation_x1 = 0.5\n'
                 'location_x2 = 0.5\nlocation_rad = 0.2\n')
    else:
        text += ('<refined_region1>\nlevel = 1\nx1min = 0.25\nx1max = 0.75\n'
                 'x2min = 0\nx2max = 1\n'
                 '<amr_criterion1>\nmethod = min_max\nvariable = mhd_u_d\n'
                 'value_max = 2.0\n')
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
    rows = [sr.refinement.read(p, lambda t: float(sr.reference(t)[0])) for p in paths]
    h.require(abs(rows[-1]['time']-.02) < 1e-13, 'Incomplete AMR run')
    h.require(rows[-1]['topology'] > 0, 'No runtime topology change')
    if restart is None:
        if scenario == 'refine':
            h.require(rows[0]['levels'] == [0] and rows[-1]['levels'] == [0, 1],
                      'Did not refine during evolution')
        else:
            h.require(rows[0]['levels'] == [0, 1] and rows[-1]['levels'] == [0],
                      'Did not coarsen during evolution')
    for row in rows:
        h.require(abs(row['total']-.65)/.65 <
                  512*np.finfo(float).eps*(row['cycle']+1), 'AMR total energy')
        h.require(row['divb'] <
                  4096*np.finfo(float).eps*(2*n)*(row['cycle']+1), 'AMR CT divergence')
    events = [list(map(int, line.split()))
              for line in (root/'helix.log').read_text().splitlines()
              if line.strip() and not line.startswith('#')]
    h.require(all(not any(row[1:6]) for row in events), 'AMR EOS floors/failures')
    result = dict(status='PASS', name=name, n=n, scenario=scenario, samples=rows)
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(name, rows[-1], flush=True)
    return result, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
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
        spatial_ratio_max=.4, restart='identical active U and CT faces',
        topology='at least one runtime change',
        conservation='512*eps*(cycles+1)'), indent=2)+'\n')
    try:
        for scenario in ('refine', 'coarsen'):
            errors = []
            for n in (32, 64):
                result, paths = run(args, n, scenario)
                report['cases'].append(result)
                errors.append(result['samples'][-1]['normalized_field_l2'])
                if n == 32:
                    before = next(path for path, row in zip(paths, result['samples'])
                                  if row['cycle'] > 0 and row['topology'] == 0)
                    repeated, _ = run(args, n, scenario, restart=before)
                    report['cases'].append(repeated)
                    h.require(result['samples'][-1]['state_sha256'] ==
                              repeated['samples'][-1]['state_sha256'],
                              'AMR restart changed active fields')
                    h.require(result['samples'][-1]['topology'] ==
                              repeated['samples'][-1]['topology'],
                              'AMR restart changed topology history')
            report[scenario+'_spatial_ratio'] = errors[1]/errors[0]
            h.require(errors[1]/errors[0] < .4, 'AMR spatial convergence')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
