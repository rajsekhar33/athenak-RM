"""Small nonlinear conduction regressions on one CPU rank."""

import pytest

from .transport_nonlinear import run_nonlinear


@pytest.mark.parametrize('group', ('operators', 'numerical', 'guards'))
def test_transport_nonlinear_cpu(tmp_path, group):
    run_nonlinear(tmp_path, group)
