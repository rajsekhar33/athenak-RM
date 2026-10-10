"""Failure-injection coverage for additive interpretation of coarse-grid targets."""
import copy
import unittest

from result_policy import annotate_report, assess_case


def sovinec(l1=.08, residual=1.e-5):
    return dict(status='FAIL', returncode=0, configuration=dict(profile='sovinec'),
                metrics=dict(l1=l1, final_residual=residual),
                failures=['Sovinec stationarity/accuracy'])


class AssessmentTests(unittest.TestCase):
    def test_target_miss_not_method_failure(self):
        result = assess_case(sovinec())
        self.assertEqual(result['summary'], 'CHECKS_PASS_TARGET_MISSED')
        self.assertEqual(result['method_verification'], 'NOT_ESTABLISHED_BY_THIS_CASE')

    def test_stationarity_still_required(self):
        result = assess_case(sovinec(residual=1))
        self.assertEqual(result['checks'], 'FAIL')
        self.assertEqual(result['accuracy_targets'], 'MISSED')

    def test_energy_failure_not_hidden(self):
        case = sovinec()
        case['failures'].append('integrated source/boundary energy budget')
        self.assertEqual(assess_case(case)['checks'], 'FAIL')

    def test_crash_not_hidden(self):
        case = sovinec()
        case['returncode'] = -11
        self.assertEqual(assess_case(case)['checks'], 'FAIL')

    def test_missing_metrics(self):
        case = sovinec()
        case['metrics'] = {}
        self.assertEqual(assess_case(case)['checks'], 'FAIL')

    def test_nonfinite(self):
        for value in (float('nan'), float('inf')):
            self.assertEqual(assess_case(sovinec(l1=value))['checks'], 'FAIL')

    def test_unknown_failure(self):
        case = sovinec()
        case['failures'].append('new unknown failure')
        self.assertEqual(assess_case(case)['checks'], 'FAIL')

    def test_incomplete(self):
        case = sovinec()
        case['status'] = 'RUNNING'
        self.assertEqual(assess_case(case)['checks'], 'FAIL')

    def test_radial_target(self):
        case = dict(status='FAIL', returncode=0, configuration=dict(profile='radial'),
                    metrics=dict(normalized_l1=.03),
                    failures=['radial equilibrium drift'])
        self.assertEqual(assess_case(case)['summary'], 'CHECKS_PASS_TARGET_MISSED')

    def test_ring_target(self):
        case = dict(status='FAIL', returncode=0, configuration=dict(profile='ring'),
                    metrics=dict(leakage=.36, moment_error=.01),
                    failures=['ring moment/leakage diagnostic gate'])
        self.assertEqual(assess_case(case)['summary'], 'CHECKS_PASS_TARGET_MISSED')
        case['metrics']['leakage'] = -.01
        self.assertEqual(assess_case(case)['checks'], 'FAIL')

    def test_paper_ring_not_short_ring_target(self):
        case = dict(status='PASS', returncode=0, configuration=dict(kind='ring'),
                    metrics=dict(time=200), failures=[])
        self.assertEqual(assess_case(case)['accuracy_targets'], 'NOT_APPLICABLE')

    def test_rejection_not_physics_pass(self):
        case = dict(status='PASS', expected_rejection=True, returncode=1)
        self.assertEqual(assess_case(case)['checks'], 'EXPECTED_REJECTION_VALIDATED')
        case['returncode'] = -11
        self.assertEqual(assess_case(case)['checks'], 'FAIL')
        case['status'] = 'ERROR'
        self.assertEqual(assess_case(case)['checks'], 'FAIL')

    def test_contradictory_legacy_pass(self):
        case = sovinec()
        case['status'] = 'PASS'
        self.assertEqual(assess_case(case)['checks'], 'FAIL')

    def test_preserve_legacy_and_comparisons(self):
        report = dict(status='FAIL', cases=dict(example=sovinec()),
                      comparisons=dict(device=dict(status='PASS')))
        old = copy.deepcopy(report)
        annotate_report(report)
        self.assertEqual(report['status'], old['status'])
        self.assertEqual(report['comparisons'], old['comparisons'])
        for key, value in old['cases']['example'].items():
            self.assertEqual(report['cases']['example'][key], value)


if __name__ == '__main__':
    unittest.main()
