"""Distributed GPU regressions; the launcher must provide suitable GPU binding."""
import pytest
from .transport_functional import run_sts, run_transport


@pytest.mark.parametrize("limiter", ("vanleer", "mc"))
def test_anisotropic_sts_mpigpu(tmp_path, limiter):
    run_sts(tmp_path, limiter, distributed=True)


@pytest.mark.parametrize("limiter", ("vanleer", "mc"))
def test_transport_functional_mpigpu(tmp_path, limiter):
    run_transport(tmp_path, limiter, distributed=True)
