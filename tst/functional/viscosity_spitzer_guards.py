#!/usr/bin/env python3
"""Reject ambiguous or unsupported temperature-dependent viscosity inputs."""
import argparse
import json
from pathlib import Path

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
        ('disabled-law', 'mhd', ['mhd/mu_spitzer=false'],
         'Spitzer viscosity parameters require mu_spitzer=true'),
        ('two-normalizations', 'mhd', ['mhd/spitzer_mu0_cgs=1'],
         'exactly one spitzer_mu0_code/cgs'),
        ('zero-normalization', 'mhd', ['mhd/spitzer_mu0_code=0'],
         'normalization must be finite and positive'),
        ('negative-normalization', 'mhd', ['mhd/spitzer_mu0_code=-1'],
         'Viscosity coefficients must be finite and nonnegative'),
        ('nan-normalization', 'mhd', ['mhd/spitzer_mu0_code=nan'],
         'Viscosity coefficients must be finite and nonnegative'),
        ('malformed-normalization', 'mhd', ['mhd/spitzer_mu0_code=1bad'],
         'Viscosity coefficients must be finite and nonnegative'),
        ('constant-isotropic', 'mhd', ['mhd/nu_iso=.1'],
         'zero constant nu_iso/nu_aniso'),
        ('constant-parallel', 'mhd', ['mhd/nu_aniso=.1'],
         'zero constant nu_iso/nu_aniso'),
        ('bad-geometry', 'mhd', ['mhd/spitzer_viscosity_geometry=oblique'],
         'spitzer_viscosity_geometry must be isotropic or parallel'),
        ('hydro-parallel', 'hydro', ['hydro/spitzer_viscosity_geometry=parallel'],
         'Braginskii validation currently requires uniform-grid MHD'),
        ('cfl-too-large', 'mhd', ['time/cfl_number=1.1'],
         'Spitzer viscosity requires 0 < CFL <= 1 and SSP rk1/rk2/rk3'),
        ('non-ssp', 'mhd', ['time/integrator=rk4'],
         'Spitzer viscosity requires 0 < CFL <= 1 and SSP rk1/rk2/rk3'),
        ('isothermal', 'mhd', ['mhd/eos=isothermal', 'mhd/iso_sound_speed=1'],
         'Spitzer viscosity requires ideal gas'),
    ]
    report = dict(status='RUNNING', cases=[], binary_sha256=v.h.sha(args.binary))
    try:
        for name, fluid, overrides, message in cases:
            row, _, _ = v.launch(args, fluid, 'explicit', name, n=32,
                                 overrides=overrides, fail_text=message)
            report['cases'].append(row)
            print(name, 'PASS', flush=True)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
