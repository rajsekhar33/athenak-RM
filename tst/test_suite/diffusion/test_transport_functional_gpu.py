"""Single-process GPU conduction functional regressions and actual restarts."""
import pytest
from .transport_functional import run_sts, run_transport


@pytest.mark.parametrize("limiter", ("vanleer", "mc"))
def test_anisotropic_sts_gpu(tmp_path, limiter):
    run_sts(tmp_path, limiter)


@pytest.mark.parametrize("limiter", ("vanleer", "mc"))
def test_transport_functional_gpu(tmp_path, limiter):
    run_transport(tmp_path, limiter)


def test_transport_guards_gpu(tmp_path):
    run_transport(tmp_path, "vanleer", guards=True)
