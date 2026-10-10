"""Discover native constitutive pytest tests in the normal CPU suite.

A C++17 `c++` compiler is required unless callers provide the existing
ATHENAK_CONDUCTION_CLOSURE_EXE, ATHENAK_SPITZER_POTENTIAL_EXE, and
ATHENAK_SATURATED_SPEED_EXE overrides. Only tiny standalone unit-test wrappers
are compiled, into pytest temporary directories; Athena is never rebuilt here.
The potential helper is reference-only; its limiter/temperature/product checks
also exercise utilities used by the production harmonic-temperature operator.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


TRANSPORT = Path(__file__).resolve().parents[2]/'transport'


@pytest.mark.parametrize('module', ('test_conduction_closure_cpu.py',
                                    'test_spitzer_potential_cpu.py',
                                    'test_saturated_speed_cpu.py'))
def test_transport_closure_cpu(module, tmp_path):
    environment = os.environ.copy()
    environment['TMPDIR'] = str(tmp_path)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [sys.executable, '-m', 'pytest', '-q', str(TRANSPORT/module),
               '--basetemp', str(tmp_path/'native-helper')]
    result = subprocess.run(command, env=environment, capture_output=True,
                            text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'passed' in result.stdout, result.stdout


def test_transport_nonlinear_checker_cpu(tmp_path):
    environment = os.environ.copy()
    environment['TMPDIR'] = str(tmp_path)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, str(TRANSPORT/'run_nonlinear.py'),
                             '--self-test'], env=environment, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '"status": "PASS"' in result.stdout, result.stdout
