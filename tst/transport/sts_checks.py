"""Fail-closed comparisons for the small, fixed-amplitude STS mode fixture."""
import numpy as np


FIELDS = ('dens', 'velx', 'vely', 'velz', 'eint')


def compare_fields(left, right, left_metrics, right_metrics, kind):
    """Retain temporal, reference and restart gates as separate tests.

    The fixture has gamma=5/3 and initial peak-to-peak temperature 0.02.
    The normalization is a fixture-specific error scale, not a general
    conversion of arbitrary density/velocity differences into temperature.
    Invalid data raise ValueError; finite gate violations return FAIL.
    """
    gates = {'reference': 1.e-4, 'temporal': 1.e-2, 'restart': 1.e-6}
    if kind not in gates:
        raise ValueError(f'Unknown comparison kind: {kind}')
    for key in ('Time', 'x1v', 'x2v', 'x3v', *FIELDS):
        if key not in left or key not in right:
            raise ValueError(f'Missing field/coordinate: {key}')
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if a.size == 0 or a.shape != b.shape:
            raise ValueError(f'Empty or incompatible field/coordinate: {key}')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'Nonfinite field/coordinate: {key}')
        if key not in FIELDS and not np.array_equal(a, b):
            raise ValueError(f'Mismatched time/coordinate: {key}')
    if not left_metrics or left_metrics.keys() != right_metrics.keys():
        raise ValueError('Empty or mismatched metric keys')
    deltas = []
    for key in left_metrics:
        a, b = float(left_metrics[key]), float(right_metrics[key])
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError(f'Nonfinite metric: {key}')
        deltas.append(abs(a - b))
    difference = max(float(np.max(np.abs(np.asarray(left[k], dtype=float) -
                                         np.asarray(right[k], dtype=float))))
                     for k in FIELDS)
    normalized = difference * (2 / 3) / .02
    metric_delta = max(deltas)
    error = difference if kind == 'restart' else normalized
    passed = error <= gates[kind] and (kind != 'restart' or metric_delta <= 1.e-10)
    return dict(status='PASS' if passed else 'FAIL', max_field_difference=difference,
                normalized_temperature_difference=normalized,
                metric_difference=metric_delta, gate=gates[kind])
