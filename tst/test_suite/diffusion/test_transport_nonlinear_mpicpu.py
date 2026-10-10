"""Matched nonlinear pulse controls and actual restart on caller-selected MPI."""

from .transport_nonlinear import run_nonlinear


def test_transport_nonlinear_mpicpu(tmp_path):
    run_nonlinear(tmp_path, 'numerical', distributed=True)
