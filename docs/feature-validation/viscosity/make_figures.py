"""Regenerate the report figures from bundled numerical results, without AthenaK.

Usage: python make_figures.py
Dependencies: numpy, matplotlib.  This performs analysis only, not new simulations.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data/validation_results.json').read_text())
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .23,
                     'savefig.dpi': 170, 'figure.constrained_layout.use': True})


def report(name):
    return DATA['campaigns'][name]['results']


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def dispersion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = dict(sound='C0', alfven='C1', fast='C2', slow='C3')
    for backend, marker, style in [('cpu', 'o', '-'), ('gpu', 'x', '--')]:
        for family in ('native-waves', 'native-waves-magnetosonic'):
            d = report(f'{family}-{backend}-01')
            for kind in sorted({p['name'].split('-')[0] for p in d['pairs']}):
                rows = [p for p in d['pairs'] if p['name'].startswith(kind+'-')]
                n = [int(re.search(r'-n(\d+)', p['name'])[1]) for p in rows]
                axes[0].plot(n, [p['relative_error'] for p in rows], style+marker,
                             color=colors[kind], label=f'{kind}, {backend.upper()}')
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
    fig.suptitle('Hyperviscous wave dispersion: fundamental mode')
    save(fig, 'dispersion.png')


def heating():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
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
    for plot in (dispersion, heating, sts):
        plot()
    print('Wrote three figures under', OUT)
