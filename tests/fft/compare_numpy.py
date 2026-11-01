#!/usr/bin/env python3
"""Compare LocalFFT dumps from the fft_unit_test pgen with numpy.fft.

Usage: python3 -I tests/fft/compare_numpy.py <outdir> [tol]

Each <outdir>/fft_N<N>.bin holds (little endian):
  int64 N, int64 nline,
  complex128 x[nline][N], complex128 F[nline][N],
  float64 xr[nline][N], complex128 R[nline][N/2+1]
F must equal numpy.fft.fft(x, axis=1) and R numpy.fft.rfft(xr, axis=1).
The error is max|a - b| / max|b| over the batch.  Exit status 1 on any failure.
"""

import glob
import os
import sys

import numpy as np


def relerr(a, b):
    den = np.max(np.abs(b))
    num = np.max(np.abs(a - b))
    return num / den if den > 0 else num


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    outdir = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0e-12
    files = glob.glob(os.path.join(outdir, "fft_N*.bin"))
    if not files:
        print("FAIL no fft_N*.bin files in", outdir)
        return 1
    files.sort(key=lambda f: int(os.path.basename(f)[5:-4]))
    nfail = 0
    worst = 0.0
    for fname in files:
        raw = open(fname, "rb").read()
        n, nline = np.frombuffer(raw[:16], dtype="<i8")
        n, nline = int(n), int(nline)
        nh = n // 2 + 1
        off = 16
        sizes = [("x", "<c16", nline * n), ("F", "<c16", nline * n),
                 ("xr", "<f8", nline * n), ("R", "<c16", nline * nh)]
        d = {}
        for key, dt, cnt in sizes:
            nbyte = np.dtype(dt).itemsize * cnt
            d[key] = np.frombuffer(raw[off:off + nbyte], dtype=dt)
            off += nbyte
        if off != len(raw):
            print("FAIL N=%d: file size mismatch (%d vs %d)" % (n, off, len(raw)))
            nfail += 1
            continue
        x = d["x"].reshape(nline, n)
        F = d["F"].reshape(nline, n)
        xr = d["xr"].reshape(nline, n)
        R = d["R"].reshape(nline, nh)
        e_fft = relerr(F, np.fft.fft(x, axis=1))
        e_rfft = relerr(R, np.fft.rfft(xr, axis=1))
        e = max(e_fft, e_rfft)
        worst = max(worst, e)
        ok = e <= tol
        nfail += 0 if ok else 1
        print("%s  N=%5d nline=%d  fft_vs_numpy=%.2e  rfft_vs_numpy=%.2e"
              % ("PASS" if ok else "FAIL", n, nline, e_fft, e_rfft))
    print("%s  files=%d failures=%d worst=%.3e (numpy %s)"
          % ("ALL PASS" if nfail == 0 else "SOME FAILED", len(files), nfail, worst,
             np.__version__))
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
