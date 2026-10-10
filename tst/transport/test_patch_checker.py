"""Synthetic failure-detection checks, not numerical physics validation."""
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_patch
from run_patch import detected_floor_failure, initial, magnetic_energy, rows, validate


class PatchCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name)
        self.config = dict(profile='square', n=4, cold=0.1, time=0.2, alpha=0, extra=())
        self.data = []
        for x in (-0.75, -0.25, 0.25, 0.75):
            for y in (-0.75, -0.25, 0.25, 0.75):
                t = initial('square', x, y, 0.1, 10)
                self.data.append([x, y, 0.25, 1, t, t, t+1, 0])
        (self.case/'patch.log').write_text('# no events\n')
        (self.case/'patch.patch.summary').write_text('0.2 2 0.1 10\n')
        (self.case/'patch.stages').write_text('1 1 0.1 10\n2 1 0.1 10\n')

    def evaluate(self):
        (self.case/'patch.patch.0.dat').write_text(
            '\n'.join(' '.join(str(v) for v in row) for row in self.data)+'\n')
        return validate(self.case, self.config)[1]

    def test_unchanged_fixture_passes(self):
        self.assertEqual(self.evaluate(), [])

    def test_thermal_and_energy_errors_rejected(self):
        self.data[0][4] += 0.01
        self.data[0][6] += 0.01
        failures = self.evaluate()
        self.assertIn('volume/thermal conservation', failures)
        self.assertIn('total energy conservation', failures)
        self.assertIn('disabled transport changed temperature', failures)

    def test_floor_log_rejected(self):
        (self.case/'patch.log').write_text('# floor event\n0 0 1 0 0 0 0 0\n')
        self.assertIn('EOS floor/event log', self.evaluate())

    def test_expected_floor_failure_requires_normal_rejection(self):
        message = 'transport patch extrema/floor check FAILED: 0.2 10 events=2811\n'
        self.assertTrue(detected_floor_failure(1, message))
        self.assertFalse(detected_floor_failure(-11, message))
        self.assertFalse(detected_floor_failure(0, message))
        self.assertFalse(detected_floor_failure(1, message.replace('2811', '0')))
        self.assertFalse(detected_floor_failure(1, 'unrelated failure'))

    def test_cycle_extremum_rejected(self):
        (self.case/'patch.patch.summary').write_text('0.2 2 -0.1 10\n')
        self.assertIn('cycle extrema', self.evaluate())

    def test_incomplete_time_rejected(self):
        (self.case/'patch.patch.summary').write_text('0.1 2 0.1 10\n')
        self.assertIn('time/cycle coverage', self.evaluate())

    def test_nan_rejected(self):
        self.data[0][4] = math.nan
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_duplicate_cell_rejected(self):
        self.data[0] = self.data[1][:]
        self.assertIn('cell coverage', self.evaluate())

    def test_divergence_rejected(self):
        self.data[0][7] = 0.01
        self.assertIn('divB', self.evaluate())

    def test_wrong_initial_temperature_rejected(self):
        self.data[0][5] += 0.01
        self.assertIn('initial-condition fixture', self.evaluate())

    def test_circular_magnetic_energy(self):
        config = dict(profile='ring', n=64)
        self.assertAlmostEqual(magnetic_energy(config, 0.25, -0.125),
                               0.5*(0.25**2+0.125**2), places=14)
        self.assertEqual(magnetic_energy(config, 0.98, 0.98), 0)

    def test_malformed_table_rejected(self):
        path = self.case/'bad.dat'
        path.write_text('1 2 3\n')
        with self.assertRaises(ValueError):
            rows(path, 8)

    def test_digest_without_python311_file_digest(self):
        for payload in (b'', b'abc', b'abc'*(1024*1024)):
            with self.subTest(size=len(payload)):
                path = self.case/'digest.dat'
                path.write_bytes(payload)
                with mock.patch.object(hashlib, 'file_digest', create=True,
                                       side_effect=AssertionError('Python 3.11 API')):
                    self.assertEqual(run_patch.digest(path),
                                     hashlib.sha256(payload).hexdigest())

    def test_public_host_reaches_setup_without_execution(self):
        class SetupReached(Exception):
            pass

        arguments = ['run_patch.py', '--binary', str(self.case/'unused-athena'),
                     '--root', str(self.case/'uncreated-run')]
        with mock.patch('sys.argv', arguments), \
                mock.patch('socket.gethostname', return_value='public-ci.example'), \
                mock.patch.object(Path, 'mkdir', side_effect=SetupReached), \
                mock.patch.object(run_patch.subprocess, 'run') as launch:
            with self.assertRaises(SetupReached):
                run_patch.main()
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
