#!/usr/bin/env python3
"""Self-test of sf_reference.population_check for signed sums (mode = correlation T).

An odd-order correlation sum can vanish exactly by cancellation while its estimate
fluctuates. Synthetic replicate estimates (R = 8) around exact sums of which a quarter
are exactly 0 must give:
  1. signed=True, unbiased: no structural failure; the sampling gate passes in nearly
     every trial (false-alarm rate measured over many trials, <= 1%);
  2. signed=True, the zero bins shifted by 15 standard errors: sampling failure, always
     (the gate is Bonferroni over all bins with 7-dof t-scores, so a shift of 8 is
     detected only part of the time; that rate is printed for information);
  3. signed=True, a zero bin with a zero-spread nonzero estimate: structural failure;
  4. signed=False (nonnegative quantities): a fluctuating estimate of an exact zero is
     still structural, as before.
usage: python3 -I tests/sf/signed_checker_selftest.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf_reference as sr  # noqa: E402

R, NB, NTRIAL = 8, 64, 400
rng = np.random.default_rng(12345)
ref = rng.normal(0.0, 5.0, NB)
ref[::4] = 0.0                       # cancellation bins
zero = ref == 0.0
ok = True


def draw():
    return ref[None, :] + rng.normal(0.0, 1.0, (R, NB))


alarms = 0
for _ in range(NTRIAL):
    r = sr.population_check("T", draw(), ref, R, signed=True)
    if r["exact_bad"]:
        print(f"1. unbiased, signed: structural failure: FAIL ({r})")
        ok = False
        break
    alarms += not r["stat_ok"]
print(f"1. unbiased, signed: sampling false alarms {alarms}/{NTRIAL}"
      f" {'OK' if alarms <= NTRIAL // 100 else 'FAIL'}")
ok &= alarms <= NTRIAL // 100

for shift, need in ((8.0, None), (15.0, NTRIAL)):
    det = 0
    for _ in range(NTRIAL):
        est = draw()
        est[:, zero] += shift / np.sqrt(R)   # true standard error is 1/sqrt(R)
        r = sr.population_check("T", est, ref, R, signed=True)
        det += (not r["stat_ok"]) and r["exact_bad"] == 0
    good = need is None or det == need
    print(f"2. zero bins shifted by {shift:g} se: detected (sampling) {det}/{NTRIAL}"
          f" {'(information)' if need is None else ('OK' if good else 'FAIL')}")
    ok &= good

est = draw()
est[:, 0] = 0.3                      # bin 0 is a zero bin: constant nonzero estimate
r = sr.population_check("T", est, ref, R, signed=True)
print(f"3. zero bin, zero-spread nonzero estimate: exact_bad={r['exact_bad']}"
      f" {'OK' if r['exact_bad'] == 1 else 'FAIL'}")
ok &= r["exact_bad"] == 1

r = sr.population_check("T", draw(), ref, R, signed=False)
print(f"4. unsigned: fluctuating estimates of exact zeros: exact_bad={r['exact_bad']}"
      f" (expected {zero.sum()}) {'OK' if r['exact_bad'] == zero.sum() else 'FAIL'}")
ok &= r["exact_bad"] == zero.sum()

print("SIGNED CHECKER SELFTEST:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
