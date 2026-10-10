"""Separate provisional coarse accuracy targets from other checker failures.

Additive reporting only: preserve legacy status, thresholds, failures and exit codes.
Passing one case never implies method verification or release acceptance.
"""
import math


def assess_case(case):
    result = dict(policy='conduction-assessment-v1', legacy_status=case.get('status'),
                  checks='NOT_ASSESSED', accuracy_targets='NOT_APPLICABLE',
                  target_measurements=[], non_target_failures=[],
                  method_verification='NOT_ESTABLISHED_BY_THIS_CASE',
                  release_acceptance='NOT_ASSESSED')
    problems = result['non_target_failures']
    status = case.get('status')
    if status not in ('PASS', 'FAIL') or case.get('error'):
        problems.append('incomplete/error execution or validation')
    config = case.get('configuration', {})
    expected = case.get('expected_rejection') or config.get('kind') == 'guard'
    if expected:
        if (status == 'PASS' and case.get('returncode') == 1 and
                not case.get('failures') and not problems):
            result.update(checks='EXPECTED_REJECTION_VALIDATED',
                          summary='EXPECTED_REJECTION_VALIDATED')
        else:
            result.update(checks='FAIL', summary='CHECKS_FAILED_OR_INCOMPLETE')
        return result
    if case.get('returncode') != 0:
        problems.append('missing or nonzero Athena exit status')
    metrics = case.get('metrics', {})
    if not metrics or any(isinstance(v, (float, int)) and not math.isfinite(v)
                          for v in metrics.values()):
        problems.append('missing/nonfinite metrics')

    def target(metric, threshold, label):
        value = metrics.get(metric)
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            problems.append('missing/nonfinite '+metric)
            return False
        missed = value > threshold
        result['target_measurements'].append(dict(
            name=label, metric=metric, value=value, threshold=threshold,
            status='MISSED' if missed else 'MET'))
        return missed

    recognized = set()
    failures = case.get('failures', [])
    if config.get('profile') == 'sovinec':
        missed = target('l1', .05, 'absolute temperature L1 target')
        residual = metrics.get('final_residual')
        stationary = (isinstance(residual, (int, float)) and math.isfinite(residual)
                      and residual <= 1.e-3*2*math.pi**2)
        if not stationary:
            problems.append('stationarity not established')
        if missed or not stationary:
            recognized.add('Sovinec stationarity/accuracy')
    if config.get('profile') == 'radial':
        if target('normalized_l1', .01, 'normalized radial drift target'):
            recognized.add('radial equilibrium drift')
    # Late-time paper rings use a different diagnostic and have no 30% gate.
    if config.get('profile') == 'ring' and config.get('kind') != 'ring':
        missed = target('moment_error', .05, 'angular-moment accuracy target')
        missed |= target('leakage', .30, 'short-ring excess-heat leakage target')
        leakage = metrics.get('leakage')
        if isinstance(leakage, (float, int)) and leakage < -1.e-9:
            problems.append('negative excess-heat leakage')
            recognized.add('ring moment/leakage diagnostic gate')
        if missed:
            recognized.add('ring moment/leakage diagnostic gate')
    problems.extend(f for f in failures if f not in recognized)
    if status == 'FAIL' and not failures:
        problems.append('unexplained legacy failure')
    if result['target_measurements']:
        result['accuracy_targets'] = ('MISSED' if any(
            r['status'] == 'MISSED' for r in result['target_measurements']) else 'MET')
    if status == 'PASS' and (failures or result['accuracy_targets'] == 'MISSED'):
        problems.append('legacy PASS contradicts recorded failures/targets')
    result['checks'] = 'FAIL' if problems else 'PASS'
    if problems:
        result['summary'] = 'CHECKS_FAILED_OR_INCOMPLETE'
    elif result['accuracy_targets'] == 'MISSED':
        result['summary'] = 'CHECKS_PASS_TARGET_MISSED'
    else:
        result['summary'] = 'CASE_CHECKS_PASS'
    return result


def annotate_report(report):
    """Attach independent case assessments without altering legacy verdicts."""
    counts = {}
    for case in report['cases'].values():
        assessment = assess_case(case)
        case['assessment'] = assessment
        key = assessment['summary']
        counts[key] = counts.get(key, 0)+1
    report['assessment'] = dict(
        policy='conduction-assessment-v1', case_counts=counts,
        comparison_verdicts_preserved=True,
        method_verification='REQUIRES_MATCHED_REFERENCE_AND_RESOLUTION_STUDY',
        release_acceptance='NOT_ASSESSED',
        exit_code_scope='UNCHANGED_LEGACY_GATES_NOT_NEW_ACCEPTANCE_POLICY')
