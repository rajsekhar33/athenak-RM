"""Regenerate the report figures from bundled numerical results, without AthenaK.

Usage: python make_figures.py
Dependencies: numpy, matplotlib.  This performs analysis only, not new simulations.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data/validation_results.json').read_text())
PROFILES = json.loads((ROOT / 'data/physical_profiles.json').read_text())
OUT = ROOT / 'figures'
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .23,
                     'savefig.dpi': 170, 'figure.constrained_layout.use': True})


def report(name):
    return DATA['campaigns'][name]['results']


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def dispersion():
    fig, grid = plt.subplots(2, 2, figsize=(12, 8))
    profile, rates = grid[0]
    axes = grid[1]
    wave = PROFILES['hyper_sound']
    profile.plot(wave['x'], wave['continuum_density_over_amplitude'], 'k-', label='Coupled sound eigenmode')
    profile.plot(wave['x'], wave['density_perturbation_over_amplitude'], 'o', ms=3.5,
                 mfc='none', label='AthenaK, N=64')
    profile.set(xlabel=r'$x/L$', ylabel=r'$(\rho-\rho_0)/A$', title='Damped propagating sound wave at t=0.1')
    profile.legend(fontsize=8)
    for kind, color in [('sound', 'C0'), ('alfven', 'C1')]:
        label = {'sound': 'Sound', 'alfven': 'Alfvén'}[kind]
        rows = []
        for key, campaign in DATA['campaigns'].items():
            if not key.startswith('native-waves-') or not key.endswith('cpu-01'):
                continue
            result = campaign['results']
            for case in result.get('cases', []):
                if (case['name'].endswith('-hyper') and case['kind'] == kind
                        and case['n'] == 64):
                    rows.append((case['harmonic'], case))
        rows = sorted(rows, key=lambda r: r[0])
        # Native modal rates come from the measured complex Fourier amplitudes.
        for harmonic, case in rows:
            measured = case['measured_modal_decay_rate']
            expected = -case['lambda_reference'][0]
            rates.plot(harmonic, measured, 'o', color=color, mfc='none')
            rates.plot(harmonic, expected, 'x', color=color)
        if rows:
            harmonics = np.linspace(1, 2, 100)
            first = next(case for harmonic, case in rows if harmonic == 1)
            rates.plot(harmonics, -first['lambda_reference'][0]*harmonics**4,
                       color=color, label=f'{label}: fourth-order prediction')
    rates.set(xlabel=r'Harmonic number $m=kL/(2\pi)$', ylabel=r'$\Gamma$ (inverse code time units)',
              title=r'Shorter waves damp as $k^4$')
    rates.set_xticks([1, 2])
    rates.legend(fontsize=8)
    rates.text(.04, .96, 'Circles: AthenaK; crosses: eigenvalues', transform=rates.transAxes,
               fontsize=8, va='top')
    colors = dict(sound='C0', alfven='C1', fast='C2', slow='C3')
    for backend, marker, style in [('cpu', 'o', '-'), ('gpu', 'x', '--')]:
        for family in ('native-waves', 'native-waves-magnetosonic'):
            d = report(f'{family}-{backend}-01')
            for kind in sorted({p['name'].split('-')[0] for p in d['pairs']}):
                label = {'sound': 'Sound', 'alfven': 'Alfvén', 'fast': 'Fast', 'slow': 'Slow'}[kind]
                rows = [p for p in d['pairs'] if p['name'].startswith(kind+'-')]
                n = [int(re.search(r'-n(\d+)', p['name'])[1]) for p in rows]
                axes[0].plot(n, [p['relative_error'] for p in rows], style+marker,
                             color=colors[kind], label=f'{label}, {backend.upper()}')
                cases = [c for c in d['cases'] if c['name'].startswith(kind+'-')
                         and c['name'].endswith('-hyper')]
                axes[1].plot([c['n'] for c in cases],
                             [c['phase_frequency_relative_error'] for c in cases],
                             style+marker, color=colors[kind])
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xticks([32, 64])
        ax.set_xlabel('Cells per wavelength')
        ax.axhline(.03, color='k', ls=':', label='3% tolerance')
    axes[0].set_ylabel('Relative error in added damping rate')
    axes[0].set_title('Damping relative to the zero-hyperviscosity control')
    axes[1].set_ylabel('Relative phase-frequency error')
    axes[1].set_title('Wave frequency')
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle('Hyperviscosity: wave profiles, damping and dispersion')
    save(fig, 'dispersion.png')


def heating():
    fig, grid = plt.subplots(2, 2, figsize=(12, 8))
    heat, velocity = grid[0]
    axes = grid[1]
    rows = PROFILES['hyper_heating']['profiles']
    fine = rows[-1]
    heat.plot(fine['x'], fine['reference_heat'], 'k-', label='Coupled continuum reference')
    heat.plot(fine['x'], fine['no_j_heat'], 'C3--', label=r'Reference omitting $J$')
    velocity.plot(fine['x'], fine['reference_vy'], 'k-', label='Coupled continuum reference')
    velocity.plot(fine['x'], fine['initial_vy'], ':', color='0.5', label='Initial velocity')
    for row, marker, color in zip(rows, ['o', 'x'], ['C0', 'C1']):
        heat.plot(row['x'], row['heat'], marker, ms=3, mfc='none', color=color,
                  label=f"AthenaK, N={row['n']}")
        velocity.plot(row['x'], row['vy'], marker, ms=3, mfc='none', color=color,
                      label=f"AthenaK, N={row['n']}")
    heat.set(xlabel=r'$x/L$', ylabel=r'Internal-energy density gain $\Delta u$', title='Where dissipated kinetic energy heats the gas')
    velocity.set(xlabel=r'$x/L$', ylabel=r'$v_y$', title='Finite-amplitude two-harmonic shear, t=0.02')
    heat.legend(fontsize=8)
    velocity.legend(fontsize=8)
    for backend, marker, style in [('cpu', 'o', '-'), ('gpu', 'x', '--')]:
        cases = report(f'native-circular-{backend}-01')['cases']
        for owner, color in [('hydro', 'C0'), ('mhd', 'C1')]:
            rows = [c for c in cases if c['owner'] == owner]
            axes[0].plot([c['n'] for c in rows], [c['heating_relative_error'] for c in rows],
                         style+marker, color=color, label=f'{owner}, {backend.upper()}')
        rows = [c for c in report(f'native-nonzero-j-{backend}-01')['cases']
                if c['cfl'] == .3]
        axes[1].plot([c['n'] for c in rows], [c['heat_increment_relative_error'] for c in rows],
                     style+marker, color='C0', label=f'Full energy flux, {backend.upper()}')
        if backend == 'cpu':
            axes[1].plot([c['n'] for c in rows], [c['missing_J_relative_error'] for c in rows],
                         's:', color='C3', label='Reference with J = 0')
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xlabel('Cells per direction')
        ax.set_ylabel('Relative thermal-increment error')
        ax.axhline(.03, color='k', ls=':', label='3% tolerance')
        ax.legend(fontsize=8)
    axes[0].set_xticks([32, 64]); axes[1].set_xticks([64, 128])
    axes[0].set_title('Circular shear: finite-volume analytic reference')
    axes[1].set_title('Nonuniform strain: independent continuum reference')
    fig.suptitle('Hyperviscous heating and mechanical-energy flux J')
    save(fig, 'heating.png')


def sts():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for backend, marker, style in [('cpu', 'o', '-'), ('gpu', 'x', '--')]:
        rows = report(f'nonzero-j-time-diagnostic-{backend}-01')['versus_finest_explicit']
        for ratio, color in [(128., 'C0'), (8., 'C1')]:
            points = sorted([r for r in rows if r['mode'] == 'sts' and r['ratio'] == ratio],
                            key=lambda r: r['cfl'])
            for ax, metric in zip(axes, ('thermal_increment', 'velocity')):
                ax.loglog([r['cfl']*ratio for r in points],
                          [r['gated_metrics'][metric] for r in points], style+marker,
                          color=color, label=f'RKL2 ratio {ratio:g}, {backend.upper()}')
    x = np.array([.6, 4.8])
    axes[0].loglog(x, 6.36819745e-7*(x/.6)**2, 'k:', label='Second-order guide')
    for ax in axes:
        ax.set_xlabel('Nominal dt / unscaled explicit diffusion limit (CFL × ratio)')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Thermal-increment error / reference heat RMS')
    axes[1].set_ylabel('Velocity error / initial velocity RMS')
    axes[0].set_title('Thermal-energy convergence')
    axes[1].set_title('Velocity convergence')
    fig.suptitle('Hyperviscosity: RKL2 timestep convergence')
    save(fig, 'sts_convergence.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Destination for the three PNGs')
    args = parser.parse_args()
    if args.output is not None:
        OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    for plot in (dispersion, heating, sts):
        plot()
    print('Wrote three figures under', OUT)
