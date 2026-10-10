"""Deliberate faults must not pass the portable STS result checker."""
import copy
import unittest

import numpy as np

from sts_checks import FIELDS, compare_fields


class STSChecks(unittest.TestCase):
    def setUp(self):
        self.left = dict(Time=.02, x1v=np.arange(4.), x2v=np.arange(2.),
                         x3v=np.array([0.]))
        self.left.update({k: np.ones((1, 2, 4)) for k in FIELDS})
        self.right = copy.deepcopy(self.left)
        self.metrics = dict(time=.02, mean_temp=1., cosine=.009, total_energy=2.5)
        self.other_metrics = self.metrics.copy()

    def compare(self, kind='restart'):
        return compare_fields(self.left, self.right, self.metrics,
                              self.other_metrics, kind)

    def test_identical(self):
        for kind in ('reference', 'temporal', 'restart'):
            self.assertEqual(self.compare(kind)['status'], 'PASS')

    def test_field_thresholds(self):
        for kind, scale in (('reference', 3.e-6), ('temporal', 3.e-4),
                            ('restart', 1.e-6)):
            for factor, status in ((.5, 'PASS'), (2., 'FAIL')):
                self.right['eint'][:] = self.left['eint'] + factor * scale
                self.assertEqual(self.compare(kind)['status'], status)

    def test_metric_drift_restart_only(self):
        self.other_metrics['mean_temp'] += 1.e-8
        self.assertEqual(self.compare()['status'], 'FAIL')
        self.assertEqual(self.compare('temporal')['status'], 'PASS')

    def test_nonfinite(self):
        for side in (self.left, self.right):
            for key in (*FIELDS, 'x1v', 'x2v', 'x3v'):
                original = side[key].copy()
                for value in (np.nan, np.inf, -np.inf):
                    side[key].flat[0] = value
                    with self.assertRaises(ValueError):
                        self.compare()
                side[key] = original
            side['Time'] = np.nan
            with self.assertRaises(ValueError):
                self.compare()
            side['Time'] = .02

    def test_nonfinite_metrics(self):
        for side in (self.metrics, self.other_metrics):
            for value in (np.nan, np.inf, -np.inf):
                side['mean_temp'] = value
                with self.assertRaises(ValueError):
                    self.compare()
            side['mean_temp'] = 1.

    def test_coordinate_and_time_mismatch(self):
        for key in ('Time', 'x1v', 'x2v', 'x3v'):
            self.right = copy.deepcopy(self.left)
            self.right[key] += .001
            with self.assertRaises(ValueError):
                self.compare()

    def test_empty_and_shape_mismatch(self):
        for key in (*FIELDS, 'x1v'):
            for value in (np.array([]), np.ones(1)):
                self.right = copy.deepcopy(self.left)
                self.right[key] = value
                with self.assertRaises(ValueError):
                    self.compare()

    def test_missing_field(self):
        del self.right['eint']
        with self.assertRaises(ValueError):
            self.compare()

    def test_missing_and_empty_metrics(self):
        del self.other_metrics['time']
        with self.assertRaises(ValueError):
            self.compare()
        self.metrics = self.other_metrics = {}
        with self.assertRaises(ValueError):
            self.compare()

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            self.compare('unsupported')


if __name__ == '__main__':
    unittest.main()
