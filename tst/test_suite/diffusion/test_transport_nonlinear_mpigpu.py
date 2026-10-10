"""Optional MPI-GPU pulse/restart adapter; launch/binding remains caller-owned."""

from .transport_nonlinear import run_nonlinear


def test_transport_nonlinear_mpigpu(tmp_path):
    run_nonlinear(tmp_path, 'numerical', distributed=True)
