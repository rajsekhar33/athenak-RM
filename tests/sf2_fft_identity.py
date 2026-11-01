#!/usr/bin/env python3
"""SF2/FFT identity: the sf output's S_2 = T/C (order 2, unit weight, full periodic box)
must equal the exact full-population value from AthenaK's own distributed-FFT
coefficients of the same field:

    S_2(l) = (1/N) sum_x (q(x+l) - q(x))^2 = (2/N) sum_k P(k) [1 - cos(2 pi k.l)],
    P = |F|^2 / N,

averaged over the lags of each separation bin (every lag carries N pairs). The FFT
side is exact (no sampling). The sf side is compared with its replicate standard
error (Student t, R-1 dof, Bonferroni over bins).
Usage: python3 -I tests/sf2_fft_identity.py RUN_DIR   (inputs/tests/sf2_identity.athinput)
"""
import glob
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sf"))
from sf_reference import read_sf  # noqa: E402


def main():
    run = sys.argv[1]
    raw = open(os.path.join(run, "fft_test.bin"), "rb").read()
    hdr = np.frombuffer(raw, dtype=np.int32, count=6)
    n, lo = hdr[:3], hdr[3:]
    if np.any(lo != 0):
        print("the identity needs the full periodic box (fft_lo = 0)")
        return 1
    nx, ny, nz = (int(v) for v in n)
    rec = np.frombuffer(raw, dtype=np.float64, offset=24).reshape(-1, 5)
    half = np.zeros((nz, ny, nx // 2 + 1), dtype=complex)
    half[rec[:, 2].astype(int), rec[:, 1].astype(int), rec[:, 0].astype(int)] = \
        rec[:, 3] + 1j * rec[:, 4]
    N = nx * ny * nz
    # autocorrelation c(l) = (1/N) sum_x q(x) q(x+l) = irfftn(|F|^2) / N
    c = np.fft.irfftn(np.abs(half) ** 2, s=(nz, ny, nx), axes=(0, 1, 2)) / N
    S2_lag = 2.0 * (c[0, 0, 0] - c)                       # (z, y, x) lag grid
    ok = True
    files = sorted(glob.glob(os.path.join(run, "sf_*", "*.sf")))
    if not files:
        print("no .sf files")
        return 1
    for f in files:
        meta, s = read_sf(f)
        L = [float(v) for v in meta["domain_length"].split(",")]
        h = [L[0] / nx, L[1] / ny, L[2] / nz]
        # minimum-image lag lengths on the (z, y, x) grid
        lz, ly, lx = np.meshgrid(*[np.where(np.arange(m) > m // 2, np.arange(m) - m,
                                            np.arange(m)) for m in (nz, ny, nx)],
                                 indexing="ij")
        r = np.sqrt((lx * h[0]) ** 2 + (ly * h[1]) ** 2 + (lz * h[2]) ** 2)
        b = np.searchsorted(s["rbins"], r, side="right") - 1
        nb = len(s["rbins"]) - 1
        sel = (r > 0) & (b >= 0) & (b < nb)
        cnt = np.bincount(b[sel], minlength=nb)
        ref = np.bincount(b[sel], weights=S2_lag[sel], minlength=nb) / np.maximum(cnt, 1)
        # sf: per replicate (E1 + E2), order p = 2 (index 1), weight 0, variable 0
        Tt = s["T_er"].sum(axis=0)[:, 0, 0, 1, :]
        Ct = s["C_er"].sum(axis=0)[:, 0, :]
        R = Tt.shape[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            Sr = Tt / Ct
        est = s["T"][0, 0, 1] / s["C"][0]
        se = Sr.std(axis=0, ddof=1) / np.sqrt(R)
        good = cnt > 0
        e, rf, sd = est[good], ref[good], se[good]
        # reference, estimate and uncertainty must all be finite; a bin with zero
        # replicate spread must agree to round-off (it cannot be scored by z)
        finite = bool(good.any() and np.all(np.isfinite(rf)) and np.all(np.isfinite(e))
                      and np.all(np.isfinite(sd)))
        zero = finite and (sd == 0)
        zero_bad = int(np.sum(zero & (np.abs(e - rf)
                                      > 1e-12 * np.maximum(np.abs(rf), 1e-300))))
        if finite:
            pos_ = sd > 0
            z = np.abs(e[pos_] - rf[pos_]) / sd[pos_]
            zmax = z.max(initial=0.0)
            p = min(1.0, 2.0 * stats.t.sf(zmax, R - 1) * max(int(pos_.sum()), 1))
        else:
            zmax, p = np.nan, 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(e / rf - 1.0)
        line_ok = bool(finite and np.isfinite(p) and p > 1e-3 and zero_bad == 0)
        if zero_bad:
            print("  %d zero-spread bins disagree with the reference" % zero_bad)
        ok &= line_ok
        print("%s: bins=%d max|z|=%.2f Bonferroni p=%.2e max rel diff %.2e %s"
              % (os.path.relpath(f, run), good.sum(), zmax, p,
                 np.nanmax(rel, initial=0.0), "OK" if line_ok else "FAIL"))
    print("SF2/FFT IDENTITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
