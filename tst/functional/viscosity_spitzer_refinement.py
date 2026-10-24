#!/usr/bin/env python3
"""Nonlinear viscous heating across static interfaces and AMR topology changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import resistivity_refinement as r
import viscosity_spitzer as v
import resistivity_viscosity as rv

v.END = .02
COUPLED = False


def measure(path):
    if COUPLED:
        row = r.read(path, lambda t: float(rv.reference([t], (2*np.pi)**2, True)[1, 0]))
        row['magnetic_field_error'] = row.pop('normalized_field_l2')
    else:
        row = r.read(path)
        row.pop('normalized_field_l2')
    data = path.read_bytes()
    off = data.index(b'<par_end>\n')+len(b'<par_end>\n')
    head = v.h.HEADER.unpack_from(data, off)
    off += v.h.HEADER.size
    nb, root = head[:2]
    dims, block, ng = np.array(head[12:15]), np.array(head[31:34]), head[11]
    locs = np.frombuffer(data, '<i4', count=4*nb, offset=off).reshape(nb, 4)
    off += 20*nb
    lifecycle = v.h.LIFECYCLE.unpack_from(data, off)
    off += v.h.LIFECYCLE.size+4*lifecycle[-1]
    stride = int(np.frombuffer(data, '<u8', count=1, offset=off)[0])
    off += 8
    shape = tuple(b+2*ng if a == 0 or b > 1 else 1
                  for a, b in enumerate(block))[::-1]
    active = tuple(slice(ng, ng+b) if a == 0 or b > 1 else slice(0, 1)
                   for a, b in enumerate(block))[::-1]
    if COUPLED:
        amplitude = rv.reference([row['time']], (2*np.pi)**2, True)[0, 0]
    else:
        amplitude = v.reference([row['time']], (2*np.pi)**2)[0]
    err, mass = 0., 0.
    for m, loc in enumerate(locs):
        dx = np.where(dims > 1, 1./dims/2**(loc[3]-root), 1.)
        u = np.frombuffer(data, '<f8', count=5*int(np.prod(shape)),
                          offset=off+m*stride).reshape((5,)+shape)
        u = u[(slice(None),)+active]
        x = (loc[0]*block[0]+np.arange(block[0])+.5)*dx[0]
        exact = np.stack([np.zeros_like(x), amplitude*np.cos(2*np.pi*x),
                          amplitude*np.sin(2*np.pi*x)])
        err += float(np.sum((u[1:4]/u[0]-exact[:, None, None, :])**2))*np.prod(dx)
        mass += float(np.sum(u[0]))*np.prod(dx)
    row.update(velocity_field_error=float(np.sqrt(err)/v.AMPLITUDE), mass=mass)
    row['combined_field_error'] = float(np.hypot(
        row['velocity_field_error'], row.get('magnetic_field_error', 0.)))
    return row


def run(args, n, scenario, method, restart=None):
    name = f'{scenario}-{method}-n{n}'+('-restart' if restart else '')
    root = args.root/name
    root.mkdir()
    sections = v.input_text(n, 'mhd', method, .15, 1, 1.)
    v.add_conduction(sections, args, 'mhd', method)
    sections['mhd']['spitzer_mu0_code'] = v.MU0
    if COUPLED:
        sections['mhd'].update(eta_spitzer='true',
                               spitzer_eta0_code=args.initial_nu*.1**1.5,
                               resistivity_integrator=method)
        sections['problem'].update(helix_bfield=0., helix_magnetic_amp=.5)
    sections['mesh'].update(nx2=n, nghost=4)
    sections['meshblock'].update(nx1=n//4, nx2=n//4)
    sections['mesh_refinement'] = dict(
        refinement='static' if scenario == 'static' else 'adaptive',
        num_levels=2, prolong_primitives='false', max_nmb_per_rank=128,
        refinement_interval=3, ncycle_check=1)
    if scenario != 'refine':
        sections['refined_region1'] = dict(
            level=1, x1min=.25, x1max=.75, x2min=0, x2max=1)
    if scenario == 'refine':
        sections['amr_criterion1'] = dict(
            method='location', location_x1=.5, location_x2=.5, location_rad=.2)
    if scenario == 'coarsen':
        sections['amr_criterion1'] = dict(
            method='min_max', variable='mhd_u_d', value_max=2.)
    if n == 32 and scenario != 'static':
        sections['output1']['dcycle'] = 2
    text = '\n'.join(f'<{block}>\n'+'\n'.join(f'{k} = {value}' for k, value
                     in values.items())+'\n' for block, values in sections.items())
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
    rows = [measure(path) for path in paths]
    v.h.require(abs(rows[-1]['time']-v.END) < 1e-13, 'Incomplete refined viscous run')
    energy = v.PRESSURE/(2./3.)+.5*v.AMPLITUDE**2+(.125 if COUPLED else .5)
    for row in rows:
        gate = 512*np.finfo(float).eps*(row['cycle']+1)
        v.h.require(abs(row['total']/energy-1.) < gate, 'Refined viscous energy budget')
        v.h.require(abs(row['mass']-1.) < gate, 'Refined viscous mass budget')
        v.h.require(row['divb'] < gate*2*n*8, 'Refined viscous CT divergence')
        if scenario == 'static':
            v.h.require(row['levels'] == [0, 1], 'Static interface absent')
    if scenario != 'static':
        v.h.require(rows[-1]['topology'] > 0, 'No runtime viscous topology change')
        if not restart:
            before, after = ([0], [0, 1]) if scenario == 'refine' else ([0, 1], [0])
            v.h.require(rows[0]['levels'] == before and rows[-1]['levels'] == after,
                        'Wrong viscous AMR transition')
    events = [list(map(int, line.split())) for line in (root/'visc_helix.log')
              .read_text().splitlines() if line.strip() and not line.startswith('#')]
    v.h.require(all(not any(row[1:6]) for row in events), 'Refined viscous EOS floors')
    row = dict(status='PASS', name=name, n=n, method=method, scenario=scenario, samples=rows)
    (root/'result.json').write_text(json.dumps(row, indent=2)+'\n')
    print(name, rows[-1], flush=True)
    return row, paths


def main():
    global COUPLED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--initial-nu', type=float, default=.025)
    parser.add_argument('--coupled', action='store_true')
    v.conduction_arguments(parser)
    parser.add_argument('--scenarios', nargs='+', choices=('static', 'refine', 'coarsen'),
                        default=['static', 'refine', 'coarsen'])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    args = parser.parse_args()
    v.h.require(np.isfinite(args.initial_nu) and args.initial_nu > 0.,
                'Initial kinematic viscosity must be positive')
    v.MU0 = args.initial_nu/v.PRESSURE**2.5
    COUPLED = args.coupled
    rv.NU, rv.ETA, rv.END = args.initial_nu, args.initial_nu, v.END
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', cases=[], binary_sha256=v.h.sha(args.binary),
                  initial_kinematic_viscosity=args.initial_nu, coupled=COUPLED,
                  conduction=args.conduction, conduction_geometry=args.conduction_geometry)
    (args.root/'gates.json').write_text(json.dumps(dict(
        spatial_ratio_max=.4, energy_mass='512*eps*(cycles+1)',
        restart='identical active U/CT and topology'), indent=2)+'\n')
    try:
        for scenario in args.scenarios:
            for method in (('explicit', 'sts') if scenario == 'static' else ('sts',)):
                errors = []
                for n in (32, 64):
                    row, paths = run(args, n, scenario, method)
                    report['cases'].append(row)
                    errors.append(row['samples'][-1]['combined_field_error'])
                    if n == 32 and scenario != 'static':
                        checkpoint = next(p for p, sample in zip(paths, row['samples'])
                                          if sample['cycle'] > 0 and sample['topology'] == 0)
                        repeated, _ = run(args, n, scenario, method, restart=checkpoint)
                        report['cases'].append(repeated)
                        a, b = row['samples'][-1], repeated['samples'][-1]
                        v.h.require(a['state_sha256'] == b['state_sha256'] and
                                    a['topology'] == b['topology'], 'Viscous AMR restart')
                ratio = errors[1]/errors[0]
                report[scenario+'-'+method+'_ratio'] = ratio
                v.h.require(ratio < .4, 'Refined viscous spatial convergence '+str(ratio))
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
