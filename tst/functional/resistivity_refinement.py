#!/usr/bin/env python3
"""Native-field checks for a resistive helix crossing static refinement interfaces."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

import numpy as np

import resistivity as h
h.END = 0.02


def read(path, amplitude_at_time=None, include_fields=False):
    data = path.read_bytes()
    off = data.index(b'<par_end>\n')+len(b'<par_end>\n')
    head = h.HEADER.unpack_from(data, off)
    off += h.HEADER.size
    nb, root = head[:2]
    dims, block, ng = np.array(head[12:15]), np.array(head[31:34]), head[11]
    t, dt, cycle = head[49:52]
    locs = np.frombuffer(data, '<i4', count=4*nb, offset=off).reshape(nb, 4)
    off += 20*nb
    magic, version, dtold, lastdt, topology, ages = h.LIFECYCLE.unpack_from(data, off)
    h.require(magic == 0x414b43504c494645 and version == 1, 'Wrong lifecycle')
    off += h.LIFECYCLE.size+4*ages
    stride = struct.unpack_from('<Q', data, off)[0]
    off += 8
    shape = tuple(b+2*ng if a == 0 or b > 1 else 1 for a, b in enumerate(block))[::-1]
    active = tuple(slice(ng, ng+b) if a == 0 or b > 1 else slice(0, 1)
                   for a, b in enumerate(block))[::-1]
    sizes = [tuple(n+(a == 2-d) for d, n in enumerate(shape)) for a in range(3)]
    cells = math.prod(shape)
    h.require(stride == 8*(5*cells+sum(math.prod(s) for s in sizes)) and
              len(data) == off+nb*stride, 'Wrong native payload size')
    records, total, mag, kin, thermal, volume, err2, div = [], 0., 0., 0., 0., 0., 0., 0.
    amin, rmin = math.inf, math.inf
    state_hash = hashlib.sha256()
    native_fields, native_weights = [], []
    mass = 0.
    for m, loc in enumerate(locs):
        level = int(loc[3])-root
        dx = np.where(dims > 1, 1./dims/2**level, 1.)
        origin = loc[:3]*block*dx
        pos = off+m*stride
        raw = np.frombuffer(data, '<f8', count=5*cells, offset=pos).reshape((5,)+shape)
        u = raw[(slice(None),)+active]
        pos += 8*5*cells
        faces = []
        for a, s in enumerate(sizes):
            raw = np.frombuffer(data, '<f8', count=math.prod(s), offset=pos).reshape(s)
            pos += 8*math.prod(s)
            sl = tuple(slice(v.start, v.stop+(a == 2-d)) for d, v in enumerate(active))
            faces.append(raw[sl])
        h.require(np.isfinite(u).all() and all(np.isfinite(f).all() for f in faces),
                  'Nonfinite native active state')
        state_hash.update(loc.tobytes())
        for field in [u]+faces:
            state_hash.update(np.ascontiguousarray(field).tobytes())
        bc = np.stack([.5*(faces[0][..., :-1]+faces[0][..., 1:]),
                       .5*(faces[1][:, :-1]+faces[1][:, 1:]),
                       .5*(faces[2][:-1]+faces[2][1:])])
        ke = .5*np.sum(u[1:4]**2, axis=0)/u[0]
        me = .5*np.sum(bc**2, axis=0)
        ei = u[4]-ke-me
        amin, rmin = min(amin, float(ei.min())), min(rmin, float(u[0].min()))
        dv = math.prod(dx)
        mass += float(np.sum(u[0]))*dv
        if include_fields:
            native_fields.append([u]+faces)
            native_weights.append(math.prod(block)*dv)
        total += float(np.sum(u[4]))*dv
        mag += float(np.sum(me))*dv
        kin += float(np.sum(ke))*dv
        thermal += float(np.sum(ei))*dv
        volume += math.prod(block)*dv
        local_div = sum(np.diff(faces[a], axis=2-a)/dx[a] for a in range(3))
        div = max(div, float(np.max(abs(local_div))))
        x = origin[0]+(np.arange(block[0])+.5)*dx[0]
        amp = (h.AMP*np.exp(-h.ETA*(2*np.pi)**2*t) if amplitude_at_time is None
               else amplitude_at_time(t))
        exact = np.stack([np.zeros_like(x), amp*np.cos(2*np.pi*x), amp*np.sin(2*np.pi*x)])
        err2 += float(np.sum((bc-exact[:, None, None, :])**2))*dv
        records.append(dict(loc=loc.tolist(), dx=dx.tolist(), origin=origin.tolist()))
    h.require(abs(volume-1.) < 1e-13 and amin >
              0 and rmin > 0, 'Coverage/positivity failed')
    result = dict(time=t,
                  cycle=cycle,
                  blocks=nb,
                  levels=sorted(map(int,
                                    set(locs[:,
                                             3]-root))),
                  total=total,
                  magnetic=mag,
                  kinetic=kin,
                  thermal=thermal,
                  normalized_field_l2=np.sqrt(err2)/h.AMP,
                  divb=div,
                  min_internal=amin,
                  min_density=rmin,
                  topology=topology,
                  state_sha256=state_hash.hexdigest())
    if include_fields:
        order = sorted(range(nb), key=lambda m: tuple(locs[m]))
        result.update(mass=mass, locs=locs[order], weights=np.array(native_weights)[order],
                      native_fields=[np.stack([native_fields[m][a] for m in order])
                                     for a in range(4)])
    return result


def checkpoint(path):
    """Composite-grid active U/CT arrays and volume-weighted diagnostics."""
    result = read(path, include_fields=True)
    fields = result['native_fields']
    u = np.moveaxis(fields[0], 0, 1)
    faces = fields[1:]
    bcc = np.stack([.5*(faces[0][..., :-1]+faces[0][..., 1:]),
                    .5*(faces[1][..., :-1, :]+faces[1][..., 1:, :]),
                    .5*(faces[2][:, :-1]+faces[2][:, 1:])])
    internal = u[4]-.5*np.sum(u[1:4]**2, axis=0)/u[0]-.5*np.sum(bcc**2, axis=0)
    result.update(u=u, faces=faces, bcc=bcc, internal=internal)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--ranks', type=int, default=2)
    p.add_argument('--analyze-only', action='store_true')
    args = p.parse_args()
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=args.analyze_only)
    report = dict(status='RUNNING', binary=str(args.binary),
                  binary_sha256=h.sha(args.binary), cases=[])
    gates = dict(energy_relative='512*eps*(cycle+1)', divb='4096*eps*A0*Nfine*(cycle+1)',
                 spatial_ratio=.4, explicit_sts_error_difference=0.01)
    (args.root/'gates.json').write_text(json.dumps(gates, indent=2)+'\n')
    try:
        for method in ('explicit', 'sts'):
            errors = []
            for n in (32, 64):
                run = args.root/f'{method}-n{n}'
                run.mkdir(exist_ok=args.analyze_only)
                text = h.input_text(n, method, 2, 8 if n == 32 else 16, .3)
                text = text.replace('nghost = 3', 'nghost = 4')
                # Keep the helix along x and refine a slab normal to x; diffusion
                # must cross both coarse/fine interfaces. Periodic y stays resolved.
                text = text.replace('helix_axis = 2', 'helix_axis = 1')
                text += ('\n<mesh_refinement>\nrefinement = static\n'
                         'prolong_primitives = false\n<refined_region1>\nlevel = 1\n'
                         'x1min = 0.25\nx1max = 0.75\nx2min = 0\nx2max = 1\n')
                # input_text(axis=2) uses Nx=16, Ny=N. Set Nx=N, Ny=N as well.
                text = text.replace('nx1 = 16\n', f'nx1 = {n}\n', 1)
                if not args.analyze_only:
                    (run/'input.athinput').write_text(text)
                else:
                    h.require((run/'input.athinput').read_text() == text, 'Input changed')
                command = ['mpirun', '-np', str(args.ranks), str(args.binary),
                           '--kokkos-map-device-id-by=mpi_rank', '-t', '00:04:00',
                           '-i', 'input.athinput']
                if not args.analyze_only:
                    (run/'command.json').write_text(json.dumps(command, indent=2)+'\n')
                    with (run/'athena.log').open('x') as log:
                        subprocess.run(
                            command,
                            cwd=run,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=270)
                rows = [read(f) for f in sorted((run/'rst').glob('*.rst'))]
                h.require(abs(rows[-1]['time']-h.END) < 1e-13, 'Incomplete integration')
                for row in rows:
                    h.require(row['levels'] == [0, 1], 'Did not exercise refinement')
                    h.require(
                        abs(row['total']-rows[0]['total'])/rows[0]['total'] <
                        512*np.finfo(float).eps*(row['cycle']+1), 'Energy conservation')
                    h.require(row['divb'] < 4096*np.finfo(float).eps *
                              h.AMP*(2*n)*(row['cycle']+1), 'CT divergence')
                result = dict(method=method, n=n, samples=rows, status='PASS')
                report['cases'].append(result)
                errors.append(rows[-1]['normalized_field_l2'])
                print(method, n, rows[-1], flush=True)
            ratio = errors[1]/errors[0]
            report[method+'_spatial_ratio'] = ratio
            h.require(ratio < gates['spatial_ratio'],
                      'Refined-grid convergence '+str(ratio))
        for n in (32, 64):
            errors = [c['samples'][-1]['normalized_field_l2']
                      for c in report['cases'] if c['n'] == n]
            h.require(abs(errors[0]-errors[1]) < gates['explicit_sts_error_difference'],
                      'Explicit/STS refined-grid errors disagree')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        name = 'summary-reanalysis.json' if args.analyze_only else 'summary.json'
        (args.root/name).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
