"""Shared exact-executable entry point for bounded nonlinear conduction tests."""

import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[3]


def run_nonlinear(tmp_path, group, distributed=False):
    """Use an existing caller-selected binary; never build Athena or submit jobs."""
    binary = Path(os.environ.get('ATHENAK_TEST_BINARY',
                                 REPOSITORY/'tst/build/src/athena')).resolve()
    assert binary.is_file() and os.access(binary, os.X_OK), (
        f'Set ATHENAK_TEST_BINARY to an existing executable: {binary}')
    launcher = (os.environ.get('ATHENAK_TEST_LAUNCHER', 'mpirun -np 2')
                if distributed else '')
    ranks = os.environ.get('ATHENAK_TEST_RANKS', '2') if distributed else '1'
    assert not distributed or launcher.strip(), 'MPI needs an explicit launcher'
    assert not distributed or int(ranks) > 1, 'MPI needs multiple ranks'
    root = tmp_path/group
    command = [sys.executable, str(REPOSITORY/'tst/transport/run_nonlinear.py'),
               '--binary', str(binary), '--root', str(root), '--group', group,
               '--launcher', launcher, '--expected-ranks', ranks]
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    # Per-native-process timeout/owned-process-group cleanup is in the runner.
    result = subprocess.run(command, capture_output=True, text=True,
                            env=environment, check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((root/'report.json').read_text())
    assert report['status'] == 'PASS', report
    expected = {'operators': 6, 'numerical': 4, 'guards': 18}[group]
    assert len(report['cases']) == expected, report
    assert all(row['status'] == 'PASS' for kind in ('cases', 'checks')
               for row in report[kind].values()), report
