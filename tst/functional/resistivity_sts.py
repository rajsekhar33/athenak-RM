#!/usr/bin/env python3
"""Spitzer RKL2 convergence, raw-state retries, CT recovery and restart tests."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import resistivity_spitzer as s
h = s.h


def reference(k2):
    # Integrate dt/dtheta with theta=1-A/A0. This stays within the physical
    # amplitude interval even for a very small starting thermal energy.
    def solution(t):
        times = np.asarray(t)
        result = np.empty(times.shape)

        def elapsed(theta):
            def integrand(q):
                temp = s.PRES+(s.GAMMA-1)*.5*s.AMP**2*(2*q-q*q)
                return temp**1.5/(k2*s.ETA0*(1-q))
            return quad(integrand, 0., theta, epsabs=1e-14, epsrel=3e-13)[0]
        for index in np.ndindex(times.shape):
            value = float(times[index])
            upper = .5
            while elapsed(upper) < value:
                upper = .5*(1+upper)
                h.require(upper < 1., 'Reference amplitude underflows')
            theta = 0. if value == 0. else brentq(
                lambda q: elapsed(q)-value, 0., upper, xtol=1e-14, rtol=1e-14)
            result[index] = s.AMP*(1-theta)
        return result.reshape((1,)+times.shape)
    return solution


original_reference = s.reference
np.testing.assert_allclose(reference((2*np.pi)**2)(np.linspace(0, .1, 5)),
                           original_reference((2*np.pi)**2)(np.linspace(0, .1, 5)),
                           rtol=2e-12, atol=2e-13)
s.reference = reference


def run(args, name, n=64, cfl=.4, ratio=16, pressure=.1, end=.1,
        units_scale=None, restart=None, axis=1):
    s.PRES, s.END, s.h.END = pressure, end, end
    s.ETA0 = .25*pressure**1.5
    overrides = ['mhd/resistivity_integrator=sts', 'time/sts_integrator=rkl2',
                 f'time/sts_max_dt_ratio={ratio}', 'output3/file_type=log',
                 'output3/dcycle=1']
    result, state = s.launch(args, name, n=n, cfl=cfl, axis=axis, units_scale=units_scale,
                             overrides=overrides, restart=restart)
    log = (args.root/name/'athena.log').read_text()
    trials = [dict(re.findall(r'(\w+)=(\S+)', line)) for line in log.splitlines()
              if line.startswith('resistivity_sts cycle=')]
    h.require(trials and any(t['sweep'] == 'pre' for t in trials) and
              any(t['sweep'] == 'post' for t in trials), 'Missing Spitzer STS evidence')
    events = args.root/name/'helix.log'
    h.require(events.is_file(), 'Missing EOS event log')
    rows = [list(map(int, line.split())) for line in events.read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    h.require(all(not any(row[1:6]) for row in rows), 'EOS floors/failures occurred')
    result['trial_statistics'] = dict(
        accepted=sum(t['status'] == 'accepted' for t in trials),
        rejected=sum(t['status'] == 'retry' for t in trials),
        stages=sorted(set(int(t['stages']) for t in trials)))
    expected = result['samples'][-1]['expected_thermal_gain']
    result['relative_heating_error'] = abs(
        result['samples'][-1]['thermal_gain']-expected)/expected
    (args.root/name/'result-sts.json').write_text(json.dumps(result, indent=2)+'\n')
    print(name, result['trial_statistics'], 'relative heat error',
          result['relative_heating_error'], flush=True)
    return result, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--ranks', type=int, default=1)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--extended', action='store_true')
    p.add_argument('--cold-scan', action='store_true')
    p.add_argument('--units-axes', action='store_true')
    p.add_argument('--guards', action='store_true')
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--binary-manifest', type=Path)
    args = p.parse_args()
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
        spatial_ratio_max=.35, temporal_ratio_max=.35, restart_maxabs=1e-12,
        raw_state='positive native internal energy, zero EOS floors/failures',
        cold_heating_relative_max=.01), indent=2)+'\n')
    try:
        if args.guards:
            base = ['mhd/resistivity_integrator=sts', 'time/sts_integrator=rkl2']
            guards = [
                ('viscosity-sts', ['mhd/nu_iso=0.1', 'mhd/viscosity_integrator=sts'],
                 'retryable viscosity with STS resistivity is not yet supported'),
                ('conduction-sts', ['mhd/alpha_iso=0.1',
                                    'mhd/conductivity_integrator=sts'],
                 'Spitzer resistivity must be the only STS process'),
                ('non-ssp-rk', ['time/integrator=rk4'], 'SSP rk1, rk2 or rk3')]
            for name, overrides, message in guards:
                result, _ = s.launch(args, name, n=32, overrides=base+overrides,
                                     fail_text=message)
                report['cases'].append(result)
        elif args.cold_scan:
            temporal = []
            for cfl in (.01, .005, .0025):
                result, _ = run(args, f'cold-cfl{cfl}', pressure=1e-4, end=.01, cfl=cfl)
                report['cases'].append(result)
                h.require(result['relative_heating_error'] < .01,
                          'Resolved cold heating differs by more than one percent')
                temporal.append(result['samples'][-1]['temporal_amplitude_error'])
            ratios = [b/a for a, b in zip(temporal, temporal[1:])]
            report.update(cold_temporal_errors=temporal, cold_temporal_ratios=ratios)
            h.require(max(ratios) < .35, 'Cold timestep convergence '+str(ratios))
        elif args.units_axes:
            result, reference_state = run(args, 'code-units')
            report['cases'].append(result)
            for scale in (1., 2.):
                result, state = run(args, f'cgs-scale{scale}', units_scale=scale)
                report['cases'].append(result)
                difference = max(float(np.max(abs(state['u']-reference_state['u']))),
                                 *(float(np.max(abs(a-b)))
                                   for a, b in zip(state['faces'],
                                                   reference_state['faces'])))
                report.setdefault('unit_rescaling_maxabs', []).append(difference)
                h.require(difference < 1e-12, 'CGS/code transformation')
            for axis in (2, 3):
                result, _ = run(args, f'axis{axis}', n=32, axis=axis)
                report['cases'].append(result)
        elif args.pilot:
            for label, pres, end in [('warm', .1, .1), ('cold', 1e-4, .01)]:
                result, _ = run(args, label, pressure=pres, end=end)
                report['cases'].append(result)
                if label == 'cold':
                    h.require(result['relative_heating_error'] < .01,
                              'Cold heat-gain error exceeds one percent')
        else:
            errors, states = [], {}
            for n in (32, 64, 128):
                result, state = run(args, f'n{n}', n=n)
                report['cases'].append(result)
                states[n] = state
                errors.append(result['samples'][-1]['relative_field_l2'])
            ratios = [b/a for a, b in zip(errors, errors[1:])]
            report['spatial_ratios'] = ratios
            h.require(max(ratios) < .35, 'Spitzer STS spatial convergence '+str(ratios))
            temporal = [report['cases'][1]['samples'][-1]['temporal_amplitude_error']]
            for cfl in (.2, .1):
                result, _ = run(args, f'time-cfl{cfl}', cfl=cfl)
                report['cases'].append(result)
                temporal.append(result['samples'][-1]['temporal_amplitude_error'])
            ratios = [b/a for a, b in zip(temporal, temporal[1:])]
            report.update(temporal_errors=temporal, temporal_ratios=ratios)
            h.require(max(ratios) < .35, 'Spitzer STS temporal convergence '+str(ratios))
            mid = sorted((args.root/'n64/rst').glob('*.rst'))[2]
            result, state = run(args, 'restart', restart=mid)
            report['cases'].append(result)
            difference = max(float(np.max(abs(state['u']-states[64]['u']))),
                             *(float(np.max(abs(a-b)))
                               for a, b in zip(state['faces'], states[64]['faces'])))
            report['restart_maxabs'] = difference
            h.require(difference < 1e-12, 'Spitzer STS restart discrepancy')
            if args.extended:
                for scale in (1., 2.):
                    result, state = run(args, f'cgs-scale{scale}', units_scale=scale)
                    report['cases'].append(result)
                    difference = max(float(np.max(abs(state['u']-states[64]['u']))),
                                     *(float(np.max(abs(a-b)))
                                       for a, b in zip(state['faces'],
                                                       states[64]['faces'])))
                    h.require(difference < 1e-12, 'Spitzer STS CGS/unit transformation')
                result, _ = run(args, 'cold', pressure=1e-4, end=.01)
                report['cases'].append(result)
                h.require(result['trial_statistics']['rejected'] > 0,
                          'Cold case did not exercise retry')
                h.require(result['relative_heating_error'] < .01,
                          'Cold heat-gain error exceeds one percent')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
