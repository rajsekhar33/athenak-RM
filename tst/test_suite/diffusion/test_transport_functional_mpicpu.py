"""Distributed CPU conduction regressions; set ATHENAK_TEST_LAUNCHER as needed."""
import pytest
from .transport_functional import run_sts, run_transport


@pytest.mark.parametrize("limiter", ("vanleer", "mc"))
def test_anisotropic_sts_mpicpu(tmp_path, limiter):
    run_sts(tmp_path, limiter, distributed=True)


@pytest.mark.parametrize("limiter", ("vanleer", "mc"))
def test_transport_functional_mpicpu(tmp_path, limiter):
    run_transport(tmp_path, limiter, distributed=True)
