"""Turbulence-driver property checks: mode-list completeness and force isotropy.

Deliberately NOT part of the imported_features/FEATURE_CASES suite that
test_public_main_features_{cpu,mpicpu,gpu,mpigpu}.py runs by default -- the
turbulence driver is one small feature among many, and everyone working on any
public-main feature already pays the cost of that shared suite on every run.
These tests live here instead, run manually (`pytest test_suite/turbulence/`)
by anyone actually touching src/srcterms/turb_driver.{cpp,hpp}, not by default.

Checks that BuildModeList() samples the full signed wavevector lattice for
driving_type=0, not just the nonnegative octant, and that the resulting
force field is statistically isotropic (no spurious cross-component
correlation).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from test_suite.imported_features.feature_utils import latest, run_athena
from test_suite.turbulence.turbulence_utils import read_mesh_binary

BACKEND = "cpu"


def test_turbulence_mode_lattice(tmp_path: Path) -> None:
    """BuildModeList should sample the full signed lattice, not just the octant."""
    run_dir = tmp_path / "turbulence_mode_lattice"
    run_athena(run_dir, "public_main_turbulence.athinput", BACKEND, ["time/nlim=0"])
    log_text = (run_dir / "athena.stdout.log").read_text()
    match = re.search(r"mode_count\s*=\s*(\d+)", log_text)
    if match is None:
        raise AssertionError(f"No 'mode_count = ' line in {run_dir / 'athena.stdout.log'}")
    # nlow=1, nhigh=2, driving_type=0: BuildModeList keeps exactly one member of every
    # +-k pair with 1 <= nx^2+ny^2+nz^2 <= 4 inside the [-2,2]^3 lattice -- 16 modes.
    # The pre-fix nonnegative-octant-only sampler gave 10 for this same shell; a
    # regression back to octant-only sampling would silently roughly halve this count.
    mode_count = int(match.group(1))
    assert mode_count == 16, f"unexpected mode_count: {mode_count}"


def test_turbulence_isotropy(tmp_path: Path) -> None:
    """The raw driven-force field should have no spurious cross-component correlation."""
    run_dir = tmp_path / "turbulence_isotropy"
    run_athena(run_dir, "turbulence_isotropy_check.athinput", BACKEND)
    # AddForcing only applies (and NormalizeForcing only populates) the force array once
    # pm->ncycle >= 1 -- force1/2/3 are deliberately all zero on the very first cycle by
    # design (turb_driver.cpp's "set force to zero" branch), so this needs the dump from
    # after the *second* completed cycle, not the initial or first-cycle dump.
    fields = read_mesh_binary(latest(run_dir, "bin/*.turb_force.*.bin"))
    force1, force2, force3 = fields["force1"], fields["force2"], fields["force3"]

    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))

    # Full-lattice sampling cancels cross-component correlation analytically: every
    # +-k pair contributes equal and opposite terms to the projection sum. A single
    # snapshot of 584 random mode amplitudes still carries finite-sample noise of a
    # few percent, so 0.15 sits well above that noise floor while remaining far below
    # the O(0.3-0.5) correlation a nonnegative-octant-only sampler produces for this
    # same projection algebra -- tight enough to catch a regression, loose enough not
    # to be flaky.
    for label, rho in (
        ("1-2", correlation(force1, force2)),
        ("1-3", correlation(force1, force3)),
        ("2-3", correlation(force2, force3)),
    ):
        assert abs(rho) < 0.15, f"force component {label} cross-correlation too large: {rho}"
