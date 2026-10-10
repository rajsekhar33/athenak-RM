"""Discover portable transport reference/checker tests in the normal CPU suite.

These tests exercise analytic references and deliberately corrupted synthetic
data. They do not launch Athena or establish numerical solver acceptance.
Subprocess isolation avoids collisions with other legacy top-level modules.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


TRANSPORT = Path(__file__).resolve().parents[2] / "transport"
MODULES = (
    "test_oracles",
    "test_gaussian_oracle",
    "test_gaussian_checker",
    "test_equilibrium_checker",
    "test_patch_checker",
    "test_result_policy",
    "test_sts_checks",
)


@pytest.mark.parametrize("module", MODULES)
def test_transport_reference_cpu(module, tmp_path):
    """Run each existing unittest module without site paths or host restrictions."""
    environment = os.environ.copy()
    environment["TMPDIR"] = str(tmp_path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", module],
        cwd=TRANSPORT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Ran 0 tests" not in result.stderr, result.stderr
