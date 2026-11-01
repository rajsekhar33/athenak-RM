#!/usr/bin/env python3
"""Compare two sf output files that must hold the same sampled pairs (e.g. the team and
global kernel variants, or CPU and GPU): raw counts n_acc exactly, T/W/C to summation
round-off.  usage: compare_sf.py A.sf B.sf [rtol]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf_reference import read_sf  # noqa: E402


def main():
    fa, fb = sys.argv[1], sys.argv[2]
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
    ma, a = read_sf(fa)
    mb, b = read_sf(fb)
    ok = True
    if not np.array_equal(a["n_acc_er"], b["n_acc_er"]):
        print("n_acc differs: %d entries" % np.sum(a["n_acc_er"] != b["n_acc_er"]))
        ok = False
    keys = ["T_er", "W_er", "C_er"] + (["H_er"] if "H_er" in a else [])
    for k in keys:
        x, y = a[k], b[k]
        scale = np.maximum(np.abs(x), np.abs(y))
        d = np.abs(x - y)
        rel = np.max(np.where(scale > 0, d/np.where(scale > 0, scale, 1), 0.0))
        print("%s max rel diff %.2e" % (k, rel))
        ok &= bool(rel <= rtol)
    for key in ("attempts_e1", "attempts_e2"):
        if ma[key] != mb[key]:
            print("%s differs: %s vs %s" % (key, ma[key], mb[key]))
            ok = False
    print("COMPARE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
