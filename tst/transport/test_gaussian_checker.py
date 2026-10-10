"""Synthetic rejection checks: not solver executions or physical validation."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gaussian_oracle import periodic_gaussian, separable_gaussian, tensor
from run_gaussian import configurations, validate


class GaussianCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name)
        self.c = configurations()['vanleer_3d_parallel_n16']
        shape = (16,)*3
        self.initial = separable_gaussian(shape)
        exact = periodic_gaussian(shape, .02, tensor(3, 2, 0, .1, [1, 2, 3]))
        coords = np.stack(np.meshgrid(*([(np.arange(16)+.5)/16]*3),
                                      indexing='ij'), axis=-1).reshape(-1, 3)
        self.data = np.column_stack((coords, np.full(4096, 1/4096), np.ones(4096),
                                     exact.ravel(), self.initial.ravel(),
                                     exact.ravel()+7., np.zeros(4096)))
        (self.case/'gaussian.gaussian.summary').write_text('.02 1\n')
        (self.case/'gaussian.stages').write_text('1 1 1 1.1\n2 1 1 1.1\n')
        (self.case/'gaussian.log').write_text('# no floor events\n')

    def evaluate(self):
        np.savetxt(self.case/'gaussian.gaussian.dat', self.data)
        return validate(self.case, self.c)[1]

    def test_exact_reference_passes(self):
        self.assertEqual(self.evaluate(), [])

    def test_output_order_independent(self):
        self.data = self.data[::-1]
        self.assertEqual(self.evaluate(), [])

    def test_missing_cell(self):
        self.data = self.data[:-1]
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_duplicate_cell(self):
        self.data[0] = self.data[1]
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_nonfinite_cell(self):
        self.data[-1, 5] = np.nan
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_wrong_initial_fixture(self):
        self.data[0, 6] += .01
        self.assertIn('cell-average initial fixture', self.evaluate())

    def test_no_diffusion_detected(self):
        self.data[:, 5] = self.initial.ravel()
        self.data[:, 7] = self.initial.ravel()+7
        self.assertIn('coarse accuracy screening: l1', self.evaluate())

    def test_incomplete_time(self):
        (self.case/'gaussian.gaussian.summary').write_text('.01 1\n')
        self.assertIn('time/cycle coverage', self.evaluate())

    def test_missing_stage(self):
        (self.case/'gaussian.stages').write_text('1 1 1 1.1\n')
        self.assertIn('raw stage coverage', self.evaluate())

    def test_negative_raw_stage(self):
        (self.case/'gaussian.stages').write_text('1 1 -1 1.1\n2 1 1 1.1\n')
        self.assertIn('raw stage extrema', self.evaluate())

    def test_floor_event(self):
        (self.case/'gaussian.log').write_text('# event\n0 0 1 0\n')
        self.assertIn('EOS floor/event log', self.evaluate())

    def test_energy_error(self):
        self.data[0, 7] += .1
        self.assertIn('total energy conservation', self.evaluate())

    def test_heat_error(self):
        self.data[0, 5] += .1
        self.assertIn('excess heat conservation', self.evaluate())

    def test_divergence(self):
        self.data[0, 8] = 1.e-5
        self.assertIn('divB', self.evaluate())

    def test_wrong_volume(self):
        self.data[0, 3] *= 2
        self.assertIn('cell volume', self.evaluate())


if __name__ == '__main__':
    unittest.main()
