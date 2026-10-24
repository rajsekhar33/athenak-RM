#!/usr/bin/env python3
"""Check the retained scope guards for joint viscosity/Ohmic STS."""
import argparse
import json
from pathlib import Path
import subprocess

import viscosity_spitzer as v


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
    cases = [
        ('isothermal-joint', dict(eos='isothermal', iso_sound_speed=1.),
         'joint viscosity/resistivity STS requires ideal-gas MHD'),
        ('ambipolar', dict(eta_ad=.01), 'Ambipolar diffusion is not supported with STS'),
    ]
    report = dict(status='RUNNING', cases=[], binary_sha256=v.h.sha(args.binary))
    try:
        for name, parameters, message in cases:
            root = args.root/name
            root.mkdir()
            sections = v.input_text(32, 'mhd', 'sts', .3, 1, 1.)
            sections['mhd'].update(mu_spitzer='false', nu_iso=.25, eta_ohm=.25,
                                   resistivity_integrator='sts', **parameters)
            sections['problem'].update(pgen_name='linear_wave', amp=1e-4,
                                       wave_flag=0, dens=1., pgas=.3, bx0=1.,
                                       by0=0., bz0=0., along_x1='true')
            text = '\n'.join(f'<{block}>\n'+'\n'.join(f'{k} = {value}' for k, value
                             in values.items())+'\n' for block, values in sections.items())
            (root/'input.athinput').write_text(text)
            command = ['mpirun', '-np', str(args.ranks), str(args.binary),
                       '-i', 'input.athinput', '-t', '00:01:00']
            (root/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            with (root/'athena.log').open('x') as log:
                result = subprocess.run(command, cwd=root, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=90)
            v.h.require(result.returncode != 0 and message in (root/'athena.log').read_text(),
                        'Missing joint guard '+name)
            report['cases'].append(dict(name=name, status='PASS', expected_rejection=message))
            print(name, 'PASS', flush=True)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
