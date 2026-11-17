"""MPI CPU contracts for interface-area diagnostics."""

import pytest

from test_suite.interface_area.scenarios import CASES


@pytest.mark.parametrize("scenario", [case for _, case in CASES],
                         ids=[name for name, _ in CASES])
def test_interface_area_mpicpu(tmp_path, scenario):
    scenario(tmp_path, "mpicpu")
