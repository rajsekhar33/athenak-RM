"""Production face kernels and a bounded nonlinear pulse/restart on one GPU."""

import pytest

from .transport_nonlinear import run_nonlinear


@pytest.mark.parametrize('group', ('operators', 'numerical'))
def test_transport_nonlinear_gpu(tmp_path, group):
    run_nonlinear(tmp_path, group)
