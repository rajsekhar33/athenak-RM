"""MPICPU contracts for on-the-fly SF and spectrum outputs."""

import pytest

from test_suite.onthefly_sf_spectra.scenarios import CASES


@pytest.mark.parametrize(
    "scenario", [case for _, case in CASES], ids=[name for name, _ in CASES],
)
def test_onthefly_sf_spectra_mpicpu(tmp_path, scenario):
    scenario(tmp_path, "mpicpu")
