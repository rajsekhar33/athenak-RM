#!/usr/bin/env python3
"""Portable, bounded harmonic-Spitzer/Cowie-McKee regression.

Runs an existing Athena executable only: no build, host restriction, scheduler,
or site-specific helper imports. The small numerical matrix is a same-grid
comparison, not a continuum/order or all-stage/CT/retry-injection acceptance test.
Face and raw/configuration guards are single-rank; numerical controls support MPI.
All tolerances precede execution and retain the earlier targeted-test conventions.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time

import numpy as np


SOURCE = Path(__file__).resolve().parents[2]
GAMMA, AMPLITUDE, END = 5.0/3.0, .2, .04
FIELD = np.array([1., .5, .25])
ENERGY = 1./(GAMMA-1)+.5*float(FIELD@FIELD)
SUMMARY = ('time', 'mean_temp', 'cosine', 'sine', 'min_temp', 'max_temp',
           'total_energy')
GATES = {'time': 5e-14, 'energy_relative': 1e-12,
         'saved_field_scaled': 1e-6, 'double_extrema_slack': 1e-10,
         'explicit_halfstep_L1_over_amplitude': .001,
         'sts_L1_over_amplitude': .005, 'sts_Linf_over_amplitude': .01,
         'sts_double_over_amplitude': .005, 'operator_scaled': 2e-11,
         'restart_double_absolute': max(1e-12, 64*256*np.finfo(float).eps)}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def source_manifest():
    paths = sorted(p for p in (SOURCE/'src').rglob('*')
                   if p.is_file() and p.suffix in ('.cpp', '.hpp', '.h', '.cu'))
    paths += [SOURCE/'CMakeLists.txt', SOURCE/'src/CMakeLists.txt',
              SOURCE/'vis/python/bin_convert.py', Path(__file__).resolve()]
    return {str(p.relative_to(SOURCE)): digest(p) for p in paths}


def input_text(blocks):
    def value(item):
        return str(item).lower() if isinstance(item, bool) else str(item)
    return '\n'.join(f'<{name}>\n'+'\n'.join(
        f'{key} = {value(item)}' for key, item in block.items())
        for name, block in blocks.items())+'\n'


def parameters(lines):
    block, result = '', {}
    for line in lines:
        line = line.split('#', 1)[0].strip()
        if line.startswith('<'):
            block = line[1:-1]
        elif '=' in line:
            key, value = map(str.strip, line.split('=', 1))
            result[f'{block}/{key}'] = value
    return result


def blocks(name='pulse', cfl=.05, sts=False):
    """The previously checked N16 oblique MC finite-amplitude pulse."""
    data = {'job': {'basename': name}, 'mesh': {'nghost': 2}, 'meshblock': {},
            'time': {'evolution': 'kinematic', 'integrator': 'rk2',
                     'cfl_number': cfl, 'tlim': END, 'nlim': -1, 'ndiag': 1,
                     'sts_integrator': 'rkl2' if sts else 'none',
                     'sts_max_dt_ratio': 16},
            'mhd': {'eos': 'ideal', 'gamma': GAMMA, 'reconstruct': 'plm',
                    'rsolver': 'advect', 'alpha_iso': 0., 'alpha_aniso': 0.,
                    'alpha_spitzer': True, 'spitzer_kappa0_code': .1,
                    'spitzer_geometry': 'field_aligned',
                    'conduction_saturation': 'cowie_mckee', 'saturation_phi': .03,
                    'conductivity_integrator': 'sts' if sts else 'explicit',
                    'anisotropic_limiter': 'mc', 'q_limit': '3.40282e38'},
            'problem': {'pgen_name': 'transport_mode', 'profile': 'fourier',
                        'rho0': 1., 'temp0': 1., 'amplitude': AMPLITUDE,
                        'mode_x': 1, 'mode_y': 1, 'mode_z': 0,
                        'bx': 1., 'by': .5, 'bz': .25,
                        'check_conduction_operator': False,
                        'check_advection_flux': False,
                        'check_saturated_operator': False},
            'output1': {'file_type': 'bin', 'variable': 'mhd_u_bcc', 'id': 'cons',
                        'dt': END/4, 'single_file_per_rank': False},
            'output2': {'file_type': 'rst', 'dt': END/2,
                        'single_file_per_rank': False},
            'output3': {'file_type': 'log', 'dcycle': 1}}
    for d, n in enumerate((16, 16, 1), 1):
        data['mesh'].update({f'nx{d}': n, f'x{d}min': 0., f'x{d}max': 1.,
                             f'ix{d}_bc': 'periodic', f'ox{d}_bc': 'periodic'})
        data['meshblock'][f'nx{d}'] = 8 if n > 1 else 1
    return data


def operator_cases():
    result = []
    for name, saturated, null, limiter in [
            ('face_iso3d', False, False, 'vanleer'),
            ('face_combined3d', True, False, 'vanleer'),
            *[(f'null_{s}_{lim}', s, True, lim)
              for s in (False, True) for lim in ('vanleer', 'mc')]]:
        data = blocks(name)
        fluid, problem = data['mhd'], data['problem']
        fluid.update(anisotropic_limiter=limiter,
                     conduction_saturation='cowie_mckee' if saturated else 'none')
        if not saturated:
            fluid.pop('saturation_phi')
        problem.update(check_saturated_operator=True,
                       check_saturated_perpendicular_null=null)
        data['time'].update(nlim=0, tlim=.001)
        data.pop('output1')
        data.pop('output2')
        data.pop('output3')
        for d, n in enumerate((4, 4, 1 if null else 4), 1):
            data['mesh'][f'nx{d}'] = data['meshblock'][f'nx{d}'] = n
        if null:
            fluid['gamma'] = 2.
            problem.update(bx=1., by=1., bz=0.)
        else:
            problem.update(bx=1., by=2., bz=3.)
            if saturated:
                fluid.update(spitzer_kappa0_code=1e8, alpha_iso=3e7,
                             alpha_aniso=2e7, saturation_phi=.3)
            else:
                fluid['spitzer_geometry'] = 'isotropic'
        result.append((name, data, null))
    return result


def guard_cases():
    """Exact current-API diagnostics; no obsolete nonlinear-STS rejection."""
    result = []

    def add(name, edits, diagnostic, raw=False):
        data = blocks(name)
        data.pop('output1')
        data.pop('output2')
        data.pop('output3')
        for section, key, value in edits:
            if value is None:
                data[section].pop(key, None)
            else:
                data[section][key] = value
        result.append((name, data, diagnostic, raw))

    add('bad_model', [('mhd', 'conduction_saturation', 'invalid')],
        'conduction_saturation must be none or cowie_mckee')
    for label, phi in [('negative', -1), ('nan', 'nan')]:
        add(f'bad_phi_{label}', [('mhd', 'saturation_phi', phi)],
            'saturation_phi must be finite and nonnegative')
    add('phi_without_model', [('mhd', 'conduction_saturation', 'none')],
        'saturation_phi requires conduction_saturation=cowie_mckee')
    add('missing_k0', [('mhd', 'spitzer_kappa0_code', None)],
        'Spitzer requires exactly one')
    add('ambiguous_k0', [('mhd', 'spitzer_kappa0_cgs', .1)],
        'Spitzer requires exactly one')
    add('bad_geometry', [('mhd', 'spitzer_geometry', 'invalid')],
        'spitzer_geometry must be isotropic or field_aligned')
    add('non_ssp_rk', [('time', 'integrator', 'rk4')],
        'Spitzer/saturation requires SSP rk1, rk2 or rk3')
    add('missing_sts', [('mhd', 'conductivity_integrator', 'sts')],
        'selects STS, but <time>/sts_integrator = none')
    add('zero_sts', [('mhd', 'spitzer_kappa0_code', 0.),
                     ('mhd', 'conductivity_integrator', 'sts'),
                     ('time', 'sts_integrator', 'rkl2')],
        'STS requires positive thermal conduction')
    add('simultaneous_sts_viscosity', [
        ('mhd', 'conductivity_integrator', 'sts'),
        ('time', 'sts_integrator', 'rkl2'), ('mhd', 'nu_iso', .1),
        ('mhd', 'viscosity_integrator', 'sts')],
        'nonlinear conduction must be the only STS process and fluid')
    add('phi_zero_raw_floor', [
        ('mhd', 'alpha_spitzer', False), ('mhd', 'spitzer_kappa0_code', None),
        ('mhd', 'spitzer_geometry', None), ('mhd', 'alpha_iso', .1),
        ('mhd', 'saturation_phi', 0.), ('mhd', 'pfloor', 1.3)],
        'raw conserved state is invalid or would require an EOS floor', raw=True)
    return result+range_guard_cases()


def range_guard_cases():
    """Fail closed on active rate terms lost before heat-capacity scaling."""
    base = {
        'job': {'basename': 'range'},
        'mesh': {'nghost': 2}, 'meshblock': {},
        'time': {'evolution': 'kinematic', 'integrator': 'rk2',
                 'cfl_number': .4, 'tlim': 1e-4, 'nlim': 0, 'ndiag': 1},
        'hydro': {'eos': 'ideal', 'gamma': 2., 'rsolver': 'advect',
                  'reconstruct': 'plm', 'alpha_iso': 0., 'alpha_aniso': 0.,
                  'alpha_spitzer': True, 'spitzer_kappa0_code': 1e-200,
                  'spitzer_geometry': 'isotropic',
                  'conductivity_integrator': 'explicit',
                  'dfloor': 1e-310, 'pfloor': 1e-310,
                  'tfloor': 1e-310, 'sfloor': 1e-310},
        'problem': {'pgen_name': 'transport_mode', 'profile': 'fourier',
                    'rho0': 1e-300, 'temp0': 1., 'amplitude': 0.,
                    'mode_x': 1, 'mode_y': 0, 'mode_z': 0}}
    for d, n in enumerate((8, 1, 1), 1):
        base['mesh'].update({f'nx{d}': n, f'x{d}min': 0.,
                             f'x{d}max': 8e64 if d == 1 else 1.,
                             f'ix{d}_bc': 'periodic', f'ox{d}_bc': 'periodic'})
        base['meshblock'][f'nx{d}'] = n
    changes = {
        'spitzer_zero': {},
        'spitzer_zero_masked_by_alpha': {
            'problem/rho0': 1e-100, 'hydro/alpha_iso': 1e100},
        'alpha_zero_masked_by_spitzer': {
            'problem/rho0': 1e-100, 'hydro/alpha_iso': 1e-100,
            'hydro/spitzer_kappa0_code': 1e-170},
        'density_row_zero_masked_by_spitzer': {
            'hydro/alpha_iso': 1e200, 'hydro/spitzer_kappa0_code': 1e-172},
        'spitzer_subnormal': {
            'mesh/x1max': 8e10, 'hydro/spitzer_kappa0_code': 1e-300},
        'density_row_subnormal': {
            'mesh/x1max': 8e10, 'hydro/alpha_iso': 1e100,
            'hydro/spitzer_kappa0_code': 1e-172}}
    result = []
    for label, edits in changes.items():
        data = {section: dict(values) for section, values in base.items()}
        name = 'range_'+label
        data['job']['basename'] = name
        for key, value in edits.items():
            section, item = key.split('/')
            require(item in data[section], 'unknown range override: '+key)
            data[section][item] = value
        require(data['problem']['rho0'] > data['hydro']['dfloor'] > 0. and
                data['problem']['rho0']*data['problem']['temp0'] >
                data['hydro']['pfloor'] > 0. and
                data['problem']['temp0'] > data['hydro']['tfloor'] > 0.,
                'range fixture would require an EOS floor')
        result.append((name, data, 'invalid explicit conduction timestep', False))
    return result


def inspect_operator(directory, data, null):
    path = directory/(data['job']['basename']+'.saturated_operator.dat')
    rows = np.loadtxt(path, ndmin=2)
    ndim = 2 if null else 3
    require(rows.shape == (7*ndim*5*4**(ndim-1), 20) and
            np.isfinite(rows).all(), 'operator table shape/finite data')
    meta = path.read_text().splitlines()[1]
    require('stencil=harmonic_full_conductivity_limited_temperature' in meta and
            'states=7' in meta, 'incorrect operator stencil version')
    require(len(np.unique(rows[:, :5], axis=0)) == len(rows), 'duplicate face')
    for state in range(7):
        for direction in range(ndim):
            require(np.count_nonzero((rows[:, 0] == state) &
                    (rows[:, 1] == direction)) == 5*4**(ndim-1),
                    'missing operator state/direction')
    require(np.all(rows[:, 5:11] > 0), 'invalid oracle thermal state')
    require(rows[:, 19].max() <= GATES['operator_scaled'], 'native oracle gap')
    classical = rows[:, 12:15]
    magnitude = np.linalg.norm(classical, axis=1)
    normal = classical[np.arange(len(rows)), rows[:, 1].astype(int)]
    factor = np.ones(len(rows))
    saturated = data['mhd']['conduction_saturation'] == 'cowie_mckee'
    if saturated:
        active = magnitude > 0
        factor[active] = rows[active, 11]/(rows[active, 11]+magnitude[active])
    expected = normal*factor
    gap = float(np.max(np.abs(rows[:, 16]-expected)/np.maximum(1, abs(expected))))
    require(gap <= GATES['operator_scaled'], 'independent whole-vector cap gap')
    require(np.array_equal(rows[:, 18]-rows[:, 17], rows[:, 16]),
            'preexisting energy flux was not preserved')
    if null:
        selected = rows[:, 0] == 6
        require('exact_null_check=1' in meta and
                np.all(classical[selected] == 0) and
                np.all(rows[selected, 15:19] == 0), 'affine null is not exact')
        require(np.all(abs(rows[selected, 8]-rows[selected, 7]) == .0625),
                'null fixture lacks a temperature gradient')
    elif saturated:
        affine = rows[:, 0] == 0
        require(np.min(magnitude[affine]/rows[affine, 11]) >= 1e6,
                'combined face fixture is not strongly saturated')
    require('saturated conduction operator: 7 states' in
            (directory/'stdout.log').read_text(), 'operator did not finish')
    return {'faces': len(rows), 'scaled_error': gap, 'exact_null_required': null}


def read_state(reader, path):
    data = reader.read_binary(str(path))
    names = ['dens', 'mom1', 'mom2', 'mom3', 'ener', 'bcc1', 'bcc2', 'bcc3']
    require(set(data['var_names']) == set(names), 'wrong conserved field inventory')
    coords, fields, volumes = [], [], []
    for block, geo in enumerate(data['mb_geometry']):
        shape = np.asarray(data['mb_data']['dens'][block]).shape
        widths = [(geo[2*d+1]-geo[2*d])/shape[2-d] for d in range(3)]
        axes = [geo[2*d]+(np.arange(shape[2-d])+.5)*widths[d] for d in range(3)]
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
        coords.append(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
        fields.append(np.column_stack([np.asarray(data['mb_data'][v][block],
                      dtype=float).ravel() for v in names]))
        volumes.extend([math.prod(widths)]*x.size)
    xyz = np.concatenate(coords)
    values, volume = np.concatenate(fields), np.array(volumes)
    order = np.lexsort((xyz[:, 0], xyz[:, 1], xyz[:, 2]))
    xyz, values, volume = xyz[order], values[order], volume[order]
    require(xyz.shape == (256, 3) and len(np.unique(xyz, axis=0)) == 256,
            'missing or duplicate pulse cells')
    require(np.isfinite(values).all() and np.all(values[:, 0] > 0),
            'invalid raw density/fields')
    thermal = values[:, 4]-.5*np.sum(values[:, 1:4]**2, axis=1)/values[:, 0]
    thermal -= .5*np.sum(values[:, 5:8]**2, axis=1)
    temperature = (GAMMA-1)*thermal/values[:, 0]
    require(np.isfinite(temperature).all() and np.all(thermal > 0),
            'nonpositive raw thermal state')
    require(abs(volume.sum()-1) < 1e-12, 'incorrect pulse volume')
    expected = np.array([(x, y, .5) for y in (np.arange(16)+.5)/16
                         for x in (np.arange(16)+.5)/16])
    require(np.array_equal(xyz, expected), 'incorrect actual cell geometry')
    return {'time': float(data['time']), 'cycle': int(data['cycle']),
            'xyz': xyz, 'fields': values, 'temperature': temperature,
            'volume': volume, 'effective': parameters(data['header']),
            'path': str(path), 'sha256': digest(path)}


def sweep_trace(text, stamps):
    attempts = []
    for line in text.splitlines():
        if 'nonlinear_sts cycle=' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line.split('nonlinear_sts ', 1)[1]))
        row = {k: int(fields[k]) for k in ('cycle', 'stages', 'checked_stages')}
        row.update(interval=float(fields['interval']),
                   **{k: fields[k] for k in ('sweep', 'status', 'reason')})
        require(row['sweep'] in ('pre', 'post') and
                row['status'] in ('accepted', 'retry'), 'invalid STS attempt')
        require(math.isfinite(row['interval']) and row['interval'] > 0 and
                row['stages'] >= 3 and row['stages'] % 2 == 1 and
                0 <= row['checked_stages'] <= row['stages'], 'invalid STS stages')
        if row['status'] == 'accepted':
            require(row['checked_stages'] == row['stages'], 'unchecked accepted stage')
        attempts.append(row)
    expected = [(int(stamp[0]), sweep) for stamp in stamps[:-1]
                for sweep in ('pre', 'post')]
    actual = list(dict.fromkeys((r['cycle'], r['sweep']) for r in attempts))
    require(actual == expected, 'missing/reordered nonlinear sweeps')
    for stamp in stamps[:-1]:
        for sweep in ('pre', 'post'):
            rows = [r for r in attempts if r['cycle'] == int(stamp[0]) and
                    r['sweep'] == sweep]
            accepted = [r for r in rows if r['status'] == 'accepted']
            dt = float(stamp[2])
            require(accepted and rows[-1]['status'] == 'accepted', 'unfinished sweep')
            require(abs(sum(r['interval'] for r in accepted)-.5*dt) <= 2e-6*dt,
                    'accepted intervals do not cover the half-cycle')
    return attempts


def inspect_pulse(reader, directory, data, ranks, resumed=False):
    text = (directory/'stdout.log').read_text()
    require('Terminating on time limit' in text, 'not a normal endpoint')
    require(re.search(rf'Number of parallel ranks\s*=\s*{ranks}\b', text),
            'actual MPI rank count differs')
    stamps = re.findall(r'cycle=(\d+)\s+time=([+\-\d.eE]+)\s+dt=([+\-\d.eE]+)', text)
    require(len(stamps) >= 2, 'missing evolved cycle trace')
    cycles = [int(s[0]) for s in stamps]
    require(cycles == list(range(cycles[0], cycles[-1]+1)), 'cycle trace incomplete')
    require(all(math.isfinite(float(s[2])) and float(s[2]) > 0 for s in stamps[:-1]),
            'invalid evolved timestep')
    require(all(float(b[1]) > float(a[1]) for a, b in zip(stamps, stamps[1:])),
            'nonmonotone evolved time')
    require((cycles[0] > 0 and 0 < float(stamps[0][1]) < END) if resumed else
            cycles[0] == 0 and float(stamps[0][1]) == 0, 'incorrect restart/start')
    name = data['job']['basename']
    table = np.loadtxt(directory/(name+'.mode.dat'), ndmin=2)
    require(table.shape == (1, 7) and np.isfinite(table).all(), 'invalid mode summary')
    summary = dict(zip(SUMMARY, map(float, table[0])))
    states = [read_state(reader, p) for p in sorted((directory/'bin').glob('*.bin'))]
    require(states, 'missing saved conserved fields')
    states.sort(key=lambda s: (s['time'], s['cycle']))
    final = states[-1]
    require(abs(final['time']-END) <= GATES['time'] and
            abs(summary['time']-END) <= GATES['time'] and
            final['cycle'] == cycles[-1], 'unmatched actual final time/cycle')
    if not resumed:
        require(states[0]['time'] == states[0]['cycle'] == 0, 'missing initial field')
    require(abs(summary['total_energy']/ENERGY-1) <= GATES['energy_relative'] and
            abs(summary['mean_temp']-1) <= GATES['energy_relative'], 'double budget')
    require(1-AMPLITUDE-GATES['double_extrema_slack'] <= summary['min_temp'] and
            summary['max_temp'] <= 1+AMPLITUDE+GATES['double_extrema_slack'],
            'double final thermal envelope')
    require(0 < summary['cosine'] < AMPLITUDE*(1-1e-7), 'no resolved pulse decay')
    event_lines = (directory/(name+'.log')).read_text().splitlines()
    labels = ('cycle', 'eos_dfloor', 'eos_efloor', 'eos_tfloor', 'eos_vceil',
              'eos_fail', 'c2p_it', 'fofc')
    # Athena writes the header once and omits zero-counter rows. Header-only
    # output is the expected zero-event result, not missing per-cycle sampling.
    require(len(event_lines) >= 2 and event_lines[0] == '# Athena event counter data' and
            tuple(event_lines[1].lstrip('# ').split()) == labels,
            'missing native EOS event counters')
    events = [list(map(int, line.split())) for line in event_lines[2:]
              if line.strip() and not line.startswith('#')]
    require(all(len(r) == 8 and min(r) >= 0 for r in events),
            'invalid native EOS event row')
    require(all(r[j] == 0 for r in events for j in (1, 2, 3, 4, 5, 7)),
            'native EOS repair/failure or FOFC event')
    for state in states:
        field, temp = state['fields'], state['temperature']
        require(np.max(abs(field[:, 0]-1)) <= 1e-6 and
                np.max(abs(field[:, 1:4])) <= 1e-6 and
                np.max(abs(field[:, 5:8]-FIELD)) <= 1e-6, 'unchanged fields moved')
        require(temp.min() >= 1-AMPLITUDE-1e-6 and
                temp.max() <= 1+AMPLITUDE+1e-6, 'saved thermal envelope')
        require(abs(float(field[:, 4]@state['volume'])/ENERGY-1) <= 1e-6,
                'saved energy budget')
        if state['time'] == 0:
            exact = 1+AMPLITUDE*np.cos(2*np.pi*(state['xyz'][:, 0]+state['xyz'][:, 1]))
            require(np.max(abs(temp-exact)) <= 1e-6, 'incorrect initial pulse')
    expected = parameters(input_text(data).splitlines())
    for key in ('mhd/alpha_spitzer', 'mhd/spitzer_geometry',
                'mhd/conduction_saturation', 'mhd/conductivity_integrator',
                'mhd/anisotropic_limiter', 'time/sts_integrator', 'time/integrator'):
        require(final['effective'].get(key) == expected[key], 'effective key: '+key)
    for key in ('mhd/spitzer_kappa0_code', 'mhd/saturation_phi',
                'time/cfl_number', 'time/sts_max_dt_ratio', 'output3/dcycle'):
        require(float(final['effective'][key]) == float(expected[key]),
                'effective value: '+key)
    require(final['effective'].get('output3/file_type') == 'log',
            'native EOS event output not enabled')
    sts = data['time']['sts_integrator'] == 'rkl2'
    attempts = sweep_trace(text, stamps) if sts else []
    require(sts or 'nonlinear_sts cycle=' not in text, 'explicit control used STS')
    metrics = {'summary': summary, 'cycle_trace': [list(s) for s in stamps],
               'attempts': attempts, 'samples': [
                   {k: s[k] for k in ('path', 'sha256', 'cycle', 'time')}
                   for s in states],
               'actual_ranks': ranks, 'effective': final['effective'],
               'native_eos_events': events}
    return metrics, final


def thermal_gap(left, right):
    ma, a = left
    mb, b = right
    require(np.array_equal(a['xyz'], b['xyz']) and
            abs(a['time']-b['time']) <= GATES['time'], 'unmatched comparison grid/time')
    delta = abs(a['temperature']-b['temperature'])
    double = max(abs(ma['summary'][k]-mb['summary'][k])
                 for k in ('cosine', 'min_temp', 'max_temp'))
    return {'L1_over_amplitude': float(np.mean(delta))/AMPLITUDE,
            'Linf_over_amplitude': float(np.max(delta))/AMPLITUDE,
            'double_over_amplitude': double/AMPLITUDE}


def gate_thermal(gap, reference=False):
    require(gap['L1_over_amplitude'] <= GATES[
        'explicit_halfstep_L1_over_amplitude' if reference else 'sts_L1_over_amplitude'],
        'same-grid L1 gap')
    require(gap['Linf_over_amplitude'] <= GATES['sts_Linf_over_amplitude'] and
            gap['double_over_amplitude'] <= GATES['sts_double_over_amplitude'],
            'same-grid maximum/double gap')


def restart_gap(left, right):
    gap = thermal_gap(left, right)
    ma, a = left
    mb, b = right
    scale = np.maximum(1, np.maximum(abs(a['fields']), abs(b['fields'])))
    delta = abs(a['fields']-b['fields'])/scale
    require(a['cycle'] == b['cycle'] and delta.max() <= GATES['saved_field_scaled'],
            'restart conserved fields/cycles differ')
    require(np.max(abs(a['temperature']-b['temperature'])) <= 1e-6,
            'restart temperature differs')
    for key in SUMMARY[1:-1]:
        require(abs(ma['summary'][key]-mb['summary'][key]) <=
                GATES['restart_double_absolute'], 'restart double scalar: '+key)
    require(abs(ma['summary']['total_energy']/mb['summary']['total_energy']-1) <=
            GATES['energy_relative'], 'restart double energy')
    first = int(mb['cycle_trace'][0][0])
    require(mb['cycle_trace'] == [r for r in ma['cycle_trace'] if int(r[0]) >= first],
            'restart continued time/dt trace differs')
    require(mb['attempts'] == [r for r in ma['attempts'] if r['cycle'] >= first],
            'restart continued STS attempt trace differs')
    return dict(gap, saved_field_scaled=float(delta.max()),
                exact_parsed_fields=bool(np.array_equal(a['fields'], b['fields'])))


def self_test():
    tests = []

    def rejects(label, call):
        try:
            call()
        except (AssertionError, ValueError, KeyError):
            tests.append(label)
        else:
            raise AssertionError('checker accepted '+label)

    good = {'L1_over_amplitude': 0., 'Linf_over_amplitude': 0.,
            'double_over_amplitude': 0.}
    gate_thermal(good)
    for key in good:
        rejects('thermal_'+key, lambda k=key: gate_thermal(dict(good, **{k: 1.})))
    rejects('explicit_halfstep', lambda: gate_thermal(
        dict(good, L1_over_amplitude=.002), reference=True))
    stamp = [('0', '0', '.04'), ('1', '.04', '0')]
    line = ('nonlinear_sts cycle=0 sweep={} interval=.02 stages=3 '
            'checked_stages=3 status=accepted reason=none\n')
    text = line.format('pre')+line.format('post')
    require(len(sweep_trace(text, stamp)) == 2, 'positive trace control')
    corruptions = [('missing_sweep', line.format('pre')),
                   ('unchecked_stage', text.replace('checked_stages=3',
                                                    'checked_stages=2')),
                   ('wrong_interval', text.replace('interval=.02', 'interval=.03')),
                   ('nonfinite_interval', text.replace('interval=.02', 'interval=nan'))]
    for label, corrupted in corruptions:
        rejects(label, lambda t=corrupted: sweep_trace(t, stamp))
    ranges = range_guard_cases()
    require(len(ranges) == 6 and len({row[0] for row in ranges}) == 6,
            'range guard inventory')
    for name, data, diagnostic, raw in ranges:
        parsed = parameters(input_text(data).splitlines())
        require(parsed['time/nlim'] == '0' and 'mhd' not in data and
                parsed['hydro/gamma'] == '2.0' and
                parsed['problem/amplitude'] == '0.0' and
                diagnostic == 'invalid explicit conduction timestep' and not raw,
                'range guard semantics')
    return {'status': 'PASS', 'controls': ['valid_gap', 'valid_attempts'],
            'rejected_mutations': tests, 'prepared_range_inputs': len(ranges)}


def plan(group):
    groups = ('operators', 'numerical', 'guards') if group == 'all' else (group,)
    counts = {'operators': 6, 'numerical': 4, 'guards': 18}
    return {'groups': groups, 'native_launches': sum(counts[g] for g in groups),
            'gates': GATES, 'numerical_fixture': blocks(),
            'limitations': [
                'No continuum or measured time-order claim.',
                'Saved binary fields are float32 U/cell-centered B, not full CT faces.',
                'Stage counts/intervals checked; no independently dumped raw stages.',
                'No deliberate internal-stage fault/retry injection or AMR coverage.',
                'Guard failures are startup rejection, not retry recovery.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--launcher', default='')
    parser.add_argument('--expected-ranks', type=int, default=1)
    parser.add_argument('--group', choices=('all', 'operators', 'numerical', 'guards'),
                        default='all')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test or args.plan:
        print(json.dumps(self_test() if args.self_test else plan(args.group), indent=2))
        return 0
    require(args.binary and args.root, '--binary and --root required')
    require(args.expected_ranks >= 1 and
            (args.expected_ranks == 1 or args.launcher.strip()),
            'MPI requires an explicit launcher/rank count')
    require(args.expected_ranks == 1 or args.group == 'numerical',
            'operator/configuration fixtures require one rank')
    binary, root = args.binary.resolve(strict=True), args.root.resolve()
    require(os.access(binary, os.X_OK), 'binary is not executable')
    require(not root.exists() or not any(root.iterdir()),
            'new empty result root required')
    root.mkdir(parents=True, exist_ok=True)
    pinned = root/'athena.pinned'
    shutil.copy2(binary, pinned)
    binary_sha, sources = digest(binary), source_manifest()
    report = {'status': 'RUNNING', 'plan': plan(args.group), 'binary': str(binary),
              'binary_sha256': binary_sha, 'source_manifest': sources,
              'source_build_identity': ('Caller supplies matching executable; '
                                        'source hashes describe checkout.'),
              'expected_ranks': args.expected_ranks, 'launcher': args.launcher,
              'cases': {}, 'checks': {'checker_controls': self_test()}}
    cache = {}

    def save():
        text = json.dumps(report, indent=2, allow_nan=False)+'\n'
        (root/'report.json').write_text(text)

    def launch(name, data, inspect, checkpoint=None):
        folder = root/name
        folder.mkdir()
        inp = folder/'input.athinput'
        inp.write_text(input_text(data))
        command = shlex.split(args.launcher)+[str(pinned)]
        command += (['-r', str(checkpoint), 'job/basename='+name] if checkpoint else
                    ['-i', str(inp)])
        row = {'status': 'RUNNING', 'command': command, 'input_sha256': digest(inp)}
        report['cases'][name] = row
        save()
        start = time.monotonic()
        checkpoint_sha = digest(checkpoint) if checkpoint else None
        try:
            with (folder/'stdout.log').open('w') as log:
                process = subprocess.Popen(command, cwd=folder, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    row['returncode'] = process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise AssertionError('native timeout; owned process group terminated')
            result = inspect(folder, row['returncode'])
            if checkpoint:
                require(digest(checkpoint) == checkpoint_sha, 'checkpoint changed')
                row.update(restart=str(checkpoint), restart_sha256=checkpoint_sha)
            metrics = result[0] if isinstance(result, tuple) else result
            row.update(status='PASS', metrics=metrics)
            cache[name] = result
        except Exception as error:
            row.update(status='FAIL', error=str(error))
        row['elapsed_seconds'] = time.monotonic()-start
        row['artifacts'] = {str(p.relative_to(folder)): digest(p)
                            for p in folder.rglob('*') if p.is_file()}
        save()
        print(name, row['status'], flush=True)

    def check(name, dependencies, callback):
        row = {'status': 'BLOCKED', 'dependencies': dependencies}
        report['checks'][name] = row
        if all(k in cache for k in dependencies):
            try:
                row.update(status='PASS', metrics=callback())
            except Exception as error:
                row.update(status='FAIL', error=str(error))
        save()

    if args.group in ('all', 'operators'):
        for name, data, null in operator_cases():
            def inspect(folder, code, d=data, n=null):
                require(code == 0, 'operator process failed')
                return inspect_operator(folder, d, n)
            launch(name, data, inspect)
    if args.group in ('all', 'numerical'):
        spec = importlib.util.spec_from_file_location(
            'portable_nonlinear_reader', SOURCE/'vis/python/bin_convert.py')
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        for name, cfl, sts in [('explicit05', .05, False), ('explicit025', .025, False),
                               ('rkl16', .4, True)]:
            data = blocks(name, cfl, sts)

            def inspect(folder, code, d=data):
                require(code == 0, 'pulse process failed')
                return inspect_pulse(reader, folder, d, args.expected_ranks)
            launch(name, data, inspect)
        checkpoint = root/'rkl16/rst/rkl16.00001.rst'
        if 'rkl16' in cache and checkpoint.is_file():
            data = blocks('rkl16_restart', .4, True)

            def inspect(folder, code):
                require(code == 0, 'restart process failed')
                result = inspect_pulse(reader, folder, data, args.expected_ranks, True)
                middle = next(s for s in cache['rkl16'][0]['samples']
                              if s['time'] >= END/2)
                stamp = result[0]['cycle_trace'][0]
                require(int(stamp[0]) == middle['cycle'] and
                        abs(float(stamp[1])-middle['time']) <= 5e-7*middle['time'],
                        'checkpoint did not resume the actual middle cycle/time')
                return result
            launch('rkl16_restart', data, inspect, checkpoint)
        else:
            report['cases']['rkl16_restart'] = {
                'status': 'BLOCKED', 'error': 'RKL checkpoint unavailable'}

        def compare(a, b, reference=False):
            gap = thermal_gap(cache[a], cache[b])
            gate_thermal(gap, reference)
            return gap
        check('explicit_halfstep', ['explicit05', 'explicit025'],
              lambda: compare('explicit05', 'explicit025', True))
        for name in ('explicit05', 'explicit025'):
            check('rkl_vs_'+name, ['rkl16', name], lambda n=name: compare('rkl16', n))
        check('actual_restart', ['rkl16', 'rkl16_restart'],
              lambda: restart_gap(cache['rkl16'], cache['rkl16_restart']))
    if args.group in ('all', 'guards'):
        for name, data, diagnostic, raw in guard_cases():
            def inspect(folder, code, message=diagnostic, is_raw=raw):
                text = (folder/'stdout.log').read_text()
                require(code > 0 and message in text,
                        'intended rejection diagnostic absent')
                crash = r'Segmentation fault|Floating point exception|SIGSEGV|SIGFPE'
                require(not re.search(crash, text),
                        'unintended native crash')
                if message == 'invalid explicit conduction timestep':
                    require('raw conserved state is invalid' not in text and
                            'Setup complete, executing' not in text,
                            'range rejection masked by raw guard or reached evolution')
                require(not re.search(r'cycle=\s*[1-9]', text) and
                        not list(folder.glob('*.mode.dat')),
                        'guard reached evolution/finalization')
                return {'diagnostic': message, 'startup_raw_guard': is_raw,
                        'initial_thermal_states_positive': True}
            launch(name, data, inspect)
    try:
        require(len(report['cases']) == report['plan']['native_launches'],
                'case inventory differs')
        require(digest(binary) == digest(pinned) == binary_sha, 'binary changed')
        require(source_manifest() == sources, 'source/helper changed during test')
        report['checks']['immutable_provenance'] = {'status': 'PASS'}
    except Exception as error:
        report['checks']['immutable_provenance'] = {'status': 'FAIL', 'error': str(error)}
    report['status'] = (
        'PASS' if all(row['status'] == 'PASS'
                      for kind in ('cases', 'checks')
                      for row in report[kind].values()) else 'FAIL')
    save()
    print('PORTABLE_NONLINEAR', report['status'], root/'report.json', flush=True)
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
