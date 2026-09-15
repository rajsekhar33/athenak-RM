"""Single-GPU Bondi user-boundary restart regression."""

from test_suite.gr.bondi_restart import check_bondi_restart


def test_bondi_restart_gpu(tmp_path):
    check_bondi_restart(tmp_path, "gpu")
