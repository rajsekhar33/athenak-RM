"""Synthetic failure injection for heat-budget checks, not physics acceptance."""
import math
from pathlib import Path
import tempfile
import unittest

from run_equilibrium import detected_stage_failure, reference, validate


class EquilibriumCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name)
        self.config = dict(profile='resistance', n=4, ny=4, ratio=10,
                           perturbation=0, time=0.05, aniso=0.1)
        flux = -0.1/(0.5+0.5/10)
        self.data = []
        for x in (-0.375, -0.125, 0.125, 0.375):
            for y in (-0.375, -0.125, 0.125, 0.375):
                rho = 1 if x < 0 else 10
                t = reference(self.config, x, y)
                self.data.append([x, y, 1/16, rho, t, t, rho*t+0.5,
                                  flux, flux, 0, 0, 0])
        energy = math.fsum(r[2]*r[6] for r in self.data)
        self.summary = [0.05, 0, 5, 1.01, 1.99, energy, energy, 0, 0, 0, 0, 0]
        (self.case/'equilibrium.log').write_text('# no events\n')
        (self.case/'equilibrium.stages').write_text('1 5 1.01 1.99\n2 5 1.01 1.99\n')

    def evaluate(self):
        (self.case/'equilibrium.eq.0.dat').write_text(
            '\n'.join(' '.join(str(v) for v in r) for r in self.data)+'\n')
        (self.case/'equilibrium.eq.summary').write_text(
            ' '.join(str(v) for v in self.summary)+'\n')
        return validate(self.case, self.config)[1]

    def test_exact_solution_passes(self):
        self.assertEqual(self.evaluate(), [])

    def test_raw_stage_extremum_rejected(self):
        (self.case/'equilibrium.stages').write_text('1 5 -0.1 1.99\n2 5 1.01 1.99\n')
        self.assertIn('raw stage extrema', self.evaluate())

    def test_missing_stage_rejected(self):
        (self.case/'equilibrium.stages').write_text('1 5 1.01 1.99\n')
        self.assertIn('raw stage coverage', self.evaluate())

    def test_missing_stage_call_rejected(self):
        (self.case/'equilibrium.stages').write_text('1 4 1.01 1.99\n2 5 1.01 1.99\n')
        self.assertIn('raw stage coverage', self.evaluate())

    def test_fault_requires_clean_rejection(self):
        message = 'transport raw stage FAILED: stage=1 min=-1 max=2 invalid=0\n'
        self.assertTrue(detected_stage_failure(1, message))
        self.assertFalse(detected_stage_failure(-11, message))
        self.assertFalse(detected_stage_failure(0, message))
        self.assertFalse(detected_stage_failure(1, message.replace('min=-1', 'min=1')))

    def test_unaccounted_boundary_heat_rejected(self):
        self.summary[8] = 0.01
        self.assertIn('integrated source/boundary energy budget', self.evaluate())

    def test_forged_balanced_source_rejected(self):
        self.summary[7] = self.summary[8] = 0.01
        self.assertIn('RK source integral', self.evaluate())

    def test_wrong_flux_rejected(self):
        self.data[0][7] += 0.01
        self.assertIn('steady interface solution/flux', self.evaluate())

    def test_wrong_density_rejected(self):
        self.data[0][3] = 2
        self.assertIn('fixture/density mismatch', self.evaluate())

    def test_wrong_energy_rejected(self):
        self.data[0][6] += 0.01
        self.assertIn('summary energy mismatch', self.evaluate())

    def test_initial_energy_forgery_rejected(self):
        self.summary[5] += 0.01
        self.assertIn('independent initial energy', self.evaluate())

    def test_partial_time_rejected(self):
        self.summary[0] = 0.025
        self.assertIn('time/RK phase coverage', self.evaluate())

    def test_partial_stage_rejected(self):
        self.summary[11] = 1
        self.assertIn('time/RK phase coverage', self.evaluate())

    def test_duplicate_cells_rejected(self):
        self.data[0] = self.data[1][:]
        self.assertIn('cell/volume coverage', self.evaluate())

    def test_nonfinite_rejected(self):
        self.data[0][4] = math.nan
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_floor_rejected(self):
        (self.case/'equilibrium.log').write_text('# event\n0 0 1\n')
        self.assertIn('EOS floor/event log', self.evaluate())

    def test_nonsteady_temperature_rejected(self):
        self.data[0][4] += 0.01
        self.assertIn('steady interface solution/flux', self.evaluate())

    def test_divergence_rejected(self):
        self.data[0][11] = 0.01
        self.assertIn('divergence', self.evaluate())


if __name__ == '__main__':
    unittest.main()
