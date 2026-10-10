"""Reference self-consistency and deliberately wrong-model discrimination."""
import unittest

import numpy as np

from gaussian_oracle import metrics, periodic_gaussian, separable_gaussian, tensor


class GaussianOracleTests(unittest.TestCase):
    def test_initial_cell_integral(self):
        for shape in ((16, 24), (16, 24, 32)):
            for sigma in (.03, .1, .15):
                a = periodic_gaussian(shape, 0, np.zeros((len(shape),)*2), sigma)
                b = separable_gaussian(shape, sigma)
                np.testing.assert_allclose(a, b, rtol=0, atol=5.e-15)

    def test_isotropic_width(self):
        for shape in ((16, 32), (16, 24, 32)):
            d = len(shape)
            sigma, chi, time = .1, .1, .02
            width = np.sqrt(sigma*sigma+2*chi*time)
            a = periodic_gaussian(shape, time, chi*np.eye(d), sigma)
            b = separable_gaussian(shape, width, .1*(sigma/width)**d)
            np.testing.assert_allclose(a, b, rtol=0, atol=3.e-15)

    def test_cutoff_convergence(self):
        for d in (2, 3):
            diffusion = tensor(d, 5/3, .001, .099, [1, 2, 3])
            a = periodic_gaussian((16,)*d, .02, diffusion, cutoff=16)
            b = periodic_gaussian((16,)*d, .02, diffusion, cutoff=24)
            np.testing.assert_allclose(a, b, rtol=0, atol=1.e-15)

    def test_mean(self):
        for d in (2, 3):
            a = periodic_gaussian((16,)*d, .2, tensor(d, 2, .001, .099, [1, 2, 3]))
            expected = 1+.1*(np.sqrt(2*np.pi)*.1)**d
            self.assertAlmostEqual(np.mean(a), expected, places=14)

    def test_reversal(self):
        np.testing.assert_array_equal(tensor(3, 2, 0, .1, [1, 2, 3]),
                                      tensor(3, 2, 0, .1, [-1, -2, -3]))

    def test_bz_in_2d(self):
        a = tensor(2, 2, 0, .1, [1, 2, 3])
        np.testing.assert_allclose(a, .1*np.array([[1, 2], [2, 4]])/14,
                                   rtol=0, atol=1.e-17)

    def test_zero_field(self):
        np.testing.assert_array_equal(tensor(3, 2, .1, .2, [0, 0, 0]), .1*np.eye(3))

    def test_permutation(self):
        a = periodic_gaussian((16, 24, 32), .02, tensor(3, 2, 0, .1, [1, 2, 3]))
        b = periodic_gaussian((32, 16, 24), .02, tensor(3, 2, 0, .1, [3, 1, 2]))
        np.testing.assert_allclose(a.transpose(2, 0, 1), b, rtol=0, atol=3.e-15)

    def test_wrong_models_are_detectable(self):
        shape = (32, 32, 32)
        a = periodic_gaussian(shape, .02, tensor(3, 2, 0, .1, [1, 2, 3]))
        initial = separable_gaussian(shape)
        wrong = [initial,
                 periodic_gaussian(shape, .02, tensor(3, 5/3, 0, .1, [1, 2, 3])),
                 periodic_gaussian(shape, .02, .1*np.eye(3))]
        for field in wrong:
            self.assertGreater(metrics(field, a, initial)['l1'], .02)

    def test_invalid_tensors(self):
        for kwargs in (dict(gamma=1), dict(iso=-1), dict(aniso=-1),
                       dict(field=[1, float('nan'), 3])):
            args = dict(dimension=3, gamma=2, iso=0, aniso=.1, field=[1, 2, 3])
            args.update(kwargs)
            with self.assertRaises(ValueError):
                tensor(**args)
        with self.assertRaises(ValueError):
            periodic_gaussian((16, 16), .02, -np.eye(2))


if __name__ == '__main__':
    unittest.main()
