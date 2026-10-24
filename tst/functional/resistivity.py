#!/usr/bin/env python3
"""Constant-Ohmic dynamical helix: decay, Joule heating and CT conservation.

Use native double-precision checkpoints, including CT face fields. The analytic
solution follows the conservative resistive-MHD equations used in Stone et al.
(2020), https://doi.org/10.3847/1538-4365/ab929b. This is an additional analytic
check; published Gaussian and Alfven-wave benchmarks remain separate tests.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

HEADER = struct.Struct('<ii9d19i19iddi')
LIFECYCLE = struct.Struct('<QQddii')
ETA, AMP, PRESSURE, GAMMA, END = 0.25, 0.5, 1., 5./3., 0.1


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(path, mhd=True):
    """Uniform ideal Hydro/MHD lifecycle-v1 restart, without scalars."""
    data = path.read_bytes()
    marker = b'<par_end>\n'
    begin = data.find(marker)
    require(sys.byteorder == 'little' and 0 < begin < 131072, 'Unsupported checkpoint')
    text = data[:begin].decode('ascii')
    require('checkpoint_lifecycle_version' in text and '<turb_driving>' not in text,
            'Unexpected checkpoint physics/lifecycle')
    begin += len(marker)
    h = HEADER.unpack_from(data, begin)
    blocks = h[0]
    dims, block, ng = tuple(h[12:15]), tuple(h[31:34]), h[11]
    require(ng == h[30] and blocks == math.prod(d//b for d, b in zip(dims, block)),
            'Unexpected mesh header')
    t, next_dt, cycle = h[49:52]
    offset = begin+HEADER.size
    locs = np.frombuffer(data, '<i4', count=4*blocks, offset=offset).reshape(blocks, 4)
    require(len(set(locs[:, 3])) == 1, 'Refinement requires a different reader')
    expected = {(i, j, k) for i in range(dims[0]//block[0])
                for j in range(dims[1]//block[1]) for k in range(dims[2]//block[2])}
    require(set(map(tuple, locs[:, :3])) == expected, 'Invalid block coverage')
    offset += 20*blocks
    magic, version, dtold, last_dt, topology, ages = LIFECYCLE.unpack_from(data, offset)
    require(magic == 0x414b43504c494645 and version == 1 and ages == 0,
            'Invalid lifecycle extension')
    require(np.isfinite([t, next_dt, dtold, last_dt]).all(), 'Nonfinite time state')
    offset += LIFECYCLE.size
    stride = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    shape = tuple(b+2*ng if b > 1 or a == 0 else 1
                  for a, b in enumerate(block))[::-1]
    fs = [tuple(n+(a == 2-d) for d, n in enumerate(shape))
          for a in range(3)] if mhd else []
    count_u = 5*math.prod(shape)
    require(stride == 8*(count_u+sum(math.prod(s) for s in fs)) and
            len(data) == offset+blocks*stride, 'Unexpected checkpoint stride/length')
    u = np.full((5,)+dims[::-1], np.nan)
    faces = [np.full(tuple(n+(a == 2-d) for d, n in enumerate(dims[::-1])), np.nan)
             for a in range(3 if mhd else 0)]
    hits = [np.zeros(f.shape, dtype=np.int8) for f in faces]
    active = tuple(slice(ng, ng+b) if b > 1 or a == 0 else slice(0, 1)
                   for a, b in enumerate(block))[::-1]
    seam = 0.
    for m, loc in enumerate(locs):
        starts = tuple(int(loc[a])*block[a] for a in range(3))[::-1]
        region = tuple(slice(s, s+b) for s, b in zip(starts, block[::-1]))
        pos = offset+m*stride
        raw = np.frombuffer(data, '<f8', count=count_u, offset=pos).reshape((5,)+shape)
        require(np.isfinite(raw).all(), 'Nonfinite conserved state')
        u[(slice(None),)+region] = raw[(slice(None),)+active]
        pos += 8*count_u
        for a in range(len(fs)):
            count = math.prod(fs[a])
            raw = np.frombuffer(data, '<f8', count=count, offset=pos).reshape(fs[a])
            pos += 8*count
            require(np.isfinite(raw).all(), 'Nonfinite CT field')
            src = tuple(slice(s.start, s.stop+(a == 2-d)) for d, s in enumerate(active))
            dst = tuple(slice(s, s+b+(a == 2-d))
                        for d, (s, b) in enumerate(zip(starts, block[::-1])))
            val, old = raw[src], hits[a][dst] > 0
            target = faces[a][dst]
            if old.any():
                seam = max(seam, float(np.max(abs(target[old]-val[old]))))
            target[~old] = val[~old]
            hits[a][dst] += 1
    require(np.isfinite(u).all() and all(np.all(h > 0)
            for h in hits), 'Missing cells/faces')
    bcc = np.stack([.5*(faces[0][..., :-1]+faces[0][..., 1:]),
                    .5*(faces[1][:, :-1, :]+faces[1][:, 1:, :]),
                    .5*(faces[2][:-1]+faces[2][1:])]) if mhd else np.zeros(
                        (3,)+dims[::-1])
    ke = .5*np.sum(u[1:4]**2, axis=0)/u[0]
    me = .5*np.sum(bcc**2, axis=0)
    internal = u[4]-ke-me
    require(u[0].min() > 0 and internal.min() > 0, 'Nonpositive raw state')
    dx = [1./n for n in dims]
    divb = sum(np.diff(faces[a], axis=2-a)/dx[a] for a in range(len(faces)))
    periodic = max((float(np.max(abs(np.take(faces[a], 0, axis=2-a) -
                                     np.take(faces[a], -1, axis=2-a))))
                    for a in range(len(faces))), default=0.)
    return dict(u=u, bcc=bcc, faces=faces, internal=internal,
                time=float(t), cycle=int(cycle), seam=seam, periodic=periodic,
                divb=float(np.max(abs(divb))), magnetic=float(np.mean(me)),
                thermal=float(np.mean(internal)), kinetic=float(np.mean(ke)),
                total=float(np.mean(u[4])), path=str(path))


def input_text(n, method, axis, block, cfl):
    dims = [n, 1, 1] if axis == 1 else ([16, n, 1] if axis == 2 else [16, 16, n])
    blocks = [min(d, block) for d in dims]
    sections = dict(
        job=dict(
            basename='helix'),
        mesh=dict(
            nghost=3),
        meshblock={},
        time=dict(
            evolution='dynamic',
            integrator='rk2',
            cfl_number=cfl,
            tlim=END,
            nlim=-1,
            ndiag=1,
            sts_integrator='rkl2' if method == 'sts' else 'none',
            sts_max_dt_ratio=16),
        mhd=dict(
            eos='ideal',
            gamma=GAMMA,
            reconstruct='wenoz',
            rsolver='hlld',
            eta_ohm=ETA,
            resistivity_integrator=method),
        problem=dict(
            pgen_name='diffusion',
            resistivity_profile='helix',
            helix_axis=axis,
            amp=AMP,
            rho=1,
            pres=PRESSURE),
        output1=dict(
            file_type='rst',
            dt=END/4),
        output2=dict(
            file_type='hst',
            dt=END/4,
            data_format='%24.16e'))
    for a in (1, 2, 3):
        sections['mesh'].update({f'nx{a}': dims[a-1], f'x{a}min': 0, f'x{a}max': 1,
                                 f'ix{a}_bc': 'periodic', f'ox{a}_bc': 'periodic'})
        sections['meshblock'][f'nx{a}'] = blocks[a-1]
    return '\n'.join(f'<{s}>\n'+'\n'.join(f'{k} = {v}' for k, v in v.items())+'\n'
                     for s, v in sections.items())


def measure(state, n, axis):
    x = (np.arange(n)+.5)/n
    sh = [1, 1, 1]
    sh[3-axis] = n
    phase = 2*np.pi*x.reshape(sh)
    bc, bs = state['bcc'][axis % 3], state['bcc'][(axis+1) % 3]
    amp = float(np.mean(bc*np.cos(phase)+bs*np.sin(phase)))
    quadrature = float(np.mean(bs*np.cos(phase)-bc*np.sin(phase)))
    exact_amp = AMP*np.exp(-ETA*(2*np.pi)**2*state['time'])
    exact_heat = .5*AMP**2*(1-np.exp(-2*ETA*(2*np.pi)**2*state['time']))
    magnetic_error = float(np.sqrt(np.mean((bc-exact_amp*np.cos(phase))**2 +
                                           (bs-exact_amp*np.sin(phase))**2))/AMP)
    heat = state['thermal']-PRESSURE/(GAMMA-1)
    return dict(time=state['time'],
                cycle=state['cycle'],
                amplitude=amp,
                exact_amplitude=float(exact_amp),
                phase_quadrature=quadrature,
                relative_field_l2=magnetic_error,
                thermal_gain=heat,
                exact_thermal_gain=float(exact_heat),
                normalized_heating_error=abs(heat-exact_heat)/(.5*AMP**2),
                total_energy=state['total'],
                kinetic_energy=state['kinetic'],
                divb=state['divb'],
                face_seam=state['seam'],
                periodic_face=state['periodic'])


def run(args, root, n, method, axis=1, block=16, cfl=.4, restart=None, name=None):
    label = name or f'{method}-axis{axis}-n{n}-b{block}-cfl{cfl}'
    directory = root/label
    directory.mkdir()
    (directory/'input.athinput').write_text(input_text(n, method, axis, block, cfl))
    command = ['mpirun',
               '-np',
               str(args.ranks),
               str(args.binary),
               '--kokkos-map-device-id-by=mpi_rank',
               '-t',
               '00:04:00',
               '-i',
               'input.athinput']
    if restart is not None:
        command += ['-r', os.path.relpath(restart, directory)]
    (directory/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    start = time.monotonic()
    with (directory/'athena.log').open('x') as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=270)
    states = [checkpoint(p) for p in sorted((directory/'rst').glob('*.rst'))]
    require(states and abs(states[-1]['time']-END) < 1e-13, 'Incomplete integration')
    rows = [measure(s, n, axis) for s in states]
    eps = np.finfo(float).eps
    etot = PRESSURE/(GAMMA-1)+.5*AMP**2
    for row in rows:
        count = row['cycle']+1
        require(abs(row['total_energy']-etot)/etot <= 512*eps*count,
                'Periodic total-energy conservation failed')
        require(max(row['face_seam'], row['periodic_face']) <= 128*eps*AMP*count,
                'CT shared/periodic face mismatch')
        require(row['divb'] <= 4096*eps*AMP*n*count,
                'CT divergence exceeds roundoff bound')
        require(row['relative_field_l2'] <= .5*(2*np.pi/n)**2,
                'Helical field exceeds second-order spatial error envelope')
        require(row['normalized_heating_error'] <= (2*np.pi/n)**2,
                'Joule heating exceeds second-order spatial error envelope')
    result = dict(status='PASS', case=label, n=n, axis=axis, method=method,
                  seconds=time.monotonic()-start, samples=rows)
    (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result, states[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--rotations', action='store_true')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--binary-manifest', type=Path)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    for manifest in (args.source_manifest, args.binary_manifest):
        if manifest is not None:
            subprocess.run(['sha256sum', '--quiet', '-c', str(manifest)], check=True)
    report = dict(
        status='RUNNING', binary=str(
            args.binary), binary_sha256=sha(
            args.binary), source_manifest_sha256=sha(
            args.source_manifest) if args.source_manifest else None, cases=[])
    (
        args.root /
        'gates.json').write_text(
        json.dumps(
            dict(
                field_l2_envelope='0.5*(2*pi/N)^2 relative to initial amplitude',
                heating_envelope='(2*pi/N)^2 relative to initial magnetic energy',
                convergence_ratio_max=.35,
                energy='512*eps*(cycles+1) relative',
                temporal_ratio_max=.35,
                decomposition_maxabs=1e-12,
                ct_faces='128*eps*B0*(cycles+1)',
                divb='4096*eps*B0*N*(cycles+1)'),
            indent=2) +
        '\n')
    try:
        final = {}
        for method in ('explicit', 'sts'):
            errors = []
            for n in (32, 64, 128):
                result, state = run(args, args.root, n, method)
                report['cases'].append(result)
                errors.append(result['samples'][-1]['relative_field_l2'])
                final[method, n] = state
                print(method, n, result['samples'][-1], flush=True)
            ratios = [hi/lo for lo, hi in zip(errors, errors[1:])]
            require(all(r < .35 for r in ratios),
                    f'{method} spatial convergence {ratios}')
            report[method+'_convergence_ratios'] = ratios
        # Restart at an actual intermediate completed cycle, preserving its dt state.
        directory = args.root/'sts-axis1-n64-b16-cfl0.4'
        mid = sorted((directory/'rst').glob('*.rst'))[2]
        result, restarted = run(args, args.root, 64, 'sts',
                                restart=mid, name='sts-restart')
        report['cases'].append(result)
        ref = final['sts', 64]
        differences = dict(u=float(np.max(abs(restarted['u']-ref['u']))),
                           faces=max(float(np.max(abs(a-b)))
                                     for a, b in zip(restarted['faces'], ref['faces'])))
        require(max(differences.values()) < 1e-12,
                'Restart differs from uninterrupted run')
        report['restart_maxabs'] = differences
        if args.extended:
            # The finite-difference CT Laplacian has eigenvalue -4*sin(k*dx/2)^2/dx^2.
            # Comparing with this reference removes its leading spatial error.
            discrete_amp = AMP*np.exp(-ETA*4*64**2*np.sin(np.pi/64)**2*END)
            temporal = [dict(cfl=.4, error=abs(
                report['cases'][4]['samples'][-1]['amplitude']-discrete_amp))]
            for cfl in (.2, .1):
                result, _ = run(args, args.root, 64, 'sts', cfl=cfl)
                report['cases'].append(result)
                temporal.append(dict(cfl=cfl, error=abs(
                    result['samples'][-1]['amplitude']-discrete_amp)))
            ratios = [b['error']/a['error'] for a, b in zip(temporal, temporal[1:])]
            report.update(temporal_errors=temporal, temporal_convergence_ratios=ratios)
            require(all(r < .35 for r in ratios),
                    'RKL2 temporal convergence '+str(ratios))
            result, decomposed = run(args, args.root, 64, 'sts', block=32)
            report['cases'].append(result)
            differences = dict(
                u=float(np.max(abs(decomposed['u']-ref['u']))),
                faces=max(float(np.max(abs(a-b)))
                          for a, b in zip(decomposed['faces'], ref['faces'])))
            require(max(differences.values()) < 1e-12,
                    'MeshBlock decomposition changed solution')
            report['decomposition_maxabs'] = differences
        if args.rotations:
            for axis in (2, 3):
                result, _ = run(args, args.root, 32, 'sts', axis=axis)
                report['cases'].append(result)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        report['harness_sha256'] = sha(__file__)
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print('HELIX_PASS', args.root, flush=True)


if __name__ == '__main__':
    main()
