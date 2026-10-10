"""Bounded CPU localized-pulse and physical thermal-boundary tests."""
import pytest
from .transport_functional import run_auxiliary


@pytest.mark.parametrize("matrix", ("gaussian", "boundary"))
def test_transport_localized_cpu(tmp_path, matrix):
    run_auxiliary(tmp_path, matrix)
