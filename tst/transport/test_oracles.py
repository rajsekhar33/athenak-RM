"""Run on a permitted test host: python -m unittest discover -s tst/transport."""
import math
from pathlib import Path
import tempfile
import unittest

from oracles import (check_mode, conduction_rate, hyperviscous_rate, read_metrics,
                     sound_roots)


class TransportOracles(unittest.TestCase):
    def test_heat_capacity(self):
        self.assertAlmostEqual(conduction_rate((1, 0, 0), (1, 0, 0), 5/3, 3, 0), 2)

    def test_parallel_perpendicular(self):
        self.assertEqual(conduction_rate((1, 0, 0), (0, 1, 0), 2, 0, 3), 0)
        self.assertAlmostEqual(conduction_rate((1, 0, 0), (1, 0, 0), 2, 0, 3), 3)

    def test_out_of_plane_normalization(self):
        self.assertAlmostEqual(conduction_rate((1, 0, 0), (1, 0, 1), 2, 0, 2), 1)

    def test_reversal_scaling(self):
        expected = conduction_rate((1, 2, 3), (2, 3, 4), 5/3, 0.2, 0.3)
        for scale in (-1, 1.e-12, 1.e12):
            with self.subTest(scale=scale):
                actual = conduction_rate((1, 2, 3), tuple(scale*x for x in (2, 3, 4)),
                                         5/3, 0.2, 0.3)
                self.assertAlmostEqual(actual, expected)

    def test_axes(self):
        values = [conduction_rate(k, b, 2, 0.2, 0.3) for k, b in
                  [((1, 2, 3), (2, 4, 1)), ((3, 1, 2), (1, 2, 4))]]
        self.assertAlmostEqual(*values)

    def test_invalid_inputs(self):
        for coefficient in (-1, math.nan, math.inf):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                conduction_rate((1, 0, 0), (1, 0, 0), 2, coefficient, 0)
        self.assertEqual(conduction_rate((1, 0, 0), (0, 0, 0), 2, 0, 1), 0)

    def test_k_fourth(self):
        self.assertEqual(hyperviscous_rate((2, 0, 0), 0.5) /
                         hyperviscous_rate((1, 0, 0), 0.5), 16)

    def test_hyperviscous_oblique(self):
        self.assertEqual(hyperviscous_rate((3, 4, 0), 2), 1250)

    def test_discrete_symbol_convergence(self):
        exact = hyperviscous_rate((2, 3, 0), 0.1)
        errors = [abs(hyperviscous_rate((2, 3, 0), 0.1, (h, h, h))-exact)
                  for h in (0.02, 0.01)]
        self.assertLess(errors[1]/errors[0], 0.251)

    def test_sound_polynomial(self):
        for damping in (0, 1, 10):
            for root in sound_roots(1, 2, damping):
                self.assertLess(abs(root*root+1j*damping*root-4), 1.e-12)
                self.assertLessEqual(root.imag, 0)

    def test_checker_positive_control(self):
        self.assertEqual(check_mode(self.metrics(), 1, 0.1), [])

    def test_checker_rejects_noop(self):
        data = self.metrics()
        data['cosine'] = 0.01
        self.assertIn('cosine amplitude', check_mode(data, 1, 0.1))

    def test_checker_rejects_phase_energy_time_nan(self):
        for key, value in [('sine', 0.001), ('total_energy', 3), ('time', 0.2),
                           ('mean_temp', math.nan)]:
            with self.subTest(key=key):
                data = self.metrics()
                data[key] = value
                self.assertTrue(check_mode(data, 1, 0.1))

    def test_parser_rejects_bad_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mode.dat'
            for value in ('', '0 1 2', 'nan 1 2 3 4 5 6', '0 1 2 3 4 5 6\n0 1 2 3 4 5 6'):
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    read_metrics(path)

    @staticmethod
    def metrics():
        return dict(time=0.1, mean_temp=1.0, cosine=0.01*math.exp(-0.1), sine=0.0,
                    min_temp=0.991, max_temp=1.009, total_energy=2.0)


if __name__ == '__main__':
    unittest.main()
