#!/usr/bin/env python3
"""Estimator-bias test for the sf output: many independent seeds.

Each .sf file is one independent realisation (seed and output number enter the random
keys). For every population (E1, E2), quantity (C, W, T) and bin, the error of the
replicate-averaged estimate against the exhaustive reference is averaged over all files;
its standard error comes from the file-to-file spread, which is valid because files are
independent (unlike bins within one file, which share samples).

Usage: python3 -I tests/sf/seed_scan.py SCAN_DIR [MASK]   (SCAN_DIR/seed*/sf_*/*.sf;
       MASK = problem/mask of the runs, default 0)
"""
import glob
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf_reference as sr  # noqa: E402


MIN_HIT_FILES = 20


def main():
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*", "sf_*", "*.sf")))
    if len(files) < 10:
        print(f"need >= 10 files, found {len(files)}")
        return 1
    meta, s0 = sr.read_sf(files[0])
    meta["test_mask"] = sys.argv[2] if len(sys.argv) > 2 else "0"
    pos, block, fields = sr.build_fields(meta, 1.0)
    variables = meta["variables"].split(",")
    meta["weights"].split(",")
    q = np.stack([fields[v] for v in variables])
    wts = sr.region_weights(meta, pos, fields)
    per = [v == "1" for v in meta["periodic"].split(",")]
    L = [float(v) for v in meta["domain_length"].split(",")]
    P = int(meta["max_order"])
    keep = np.any(np.stack(wts) > 0, axis=0)   # pairs need both endpoints valid
    pos, block, q, wts = pos[:, keep], block[keep], q[:, keep], [w[keep] for w in wts]
    Tx, Wx, Cx = sr.run_exhaustive(s0["rbins"], pos, block, q, wts, per, L,
                                   meta["mode"], P)
    refs = {"C": Cx, "W": Wx, "T": Tx}
    corr = meta["mode"] == "correlation"
    dn, Dx = ("W", Wx) if corr else ("C", Cx)
    errs = {k: [] for k in refs}
    S_err = []
    # nominal sampled pairs per population and bin over all files: attempts times the
    # bin's share of the exact pair count. This is a heuristic scale only: E1 draws from
    # an importance proposal, E2 attempts include rejected pairs and share endpoints, so
    # it neither is an exact expectation nor gives a Poisson no-hit probability
    exp_hits = np.zeros((2,) + Cx.shape[2:])
    share = Cx[:, 0] / np.maximum(Cx[:, 0].sum(axis=-1, keepdims=True), 1e-300)
    nonfinite = 0
    for f in files:
        m_f, s = sr.read_sf(f)
        # every payload array must be finite: a NaN would otherwise drop out of the
        # comparisons below (NaN compares false) instead of failing
        bad = [k for k, v in s.items()
               if v.dtype.kind == "f" and not np.all(np.isfinite(v))]
        if bad:
            print(f"  {os.path.relpath(f, sys.argv[1])}: nonfinite values in {bad} FAIL")
            nonfinite += 1
        exp_hits += np.array([float(m_f["attempts_e1"]),
                              float(m_f["attempts_e2"])])[:, None] \
            * share
        for k in refs:
            est = s[k + "_er"].mean(axis=1)          # (2, ...) replicate mean per pop
            errs[k].append((est - refs[k]) / np.where(refs[k] != 0, np.abs(refs[k]), 1.0))
        # normalised result: T/C (structure functions, relative error) or T/W
        # (correlation, Python cf/weight_sum_bin; absolute error, since it can cross 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            S_est = s["T"] / s[dn][:, None, None, :]
            S_ref = Tx.sum(0) / Dx.sum(0)[:, None, None, :]
            S_err.append(S_est - S_ref if corr else S_est / S_ref - 1.0)
    K = len(files)
    ok = nonfinite == 0
    print(f"{K} independent files, nranks={meta['nranks']}, R={meta['nreplicate']}")
    for k in refs:
        e = np.array(errs[k])                      # (K, 2, ...)
        for pop, name in enumerate(["E1", "E2"]):
            ep = e[:, pop]
            m, se = ep.mean(0), ep.std(0, ddof=1) / np.sqrt(K)
            good = se > 0
            # deterministic entries (no spread over files) must equal the reference, and
            # a nonzero exact entry must be populated in some file
            r = refs[k][pop]
            est_all = (np.array([x[pop] for x in errs[k]])
                       * np.where(r != 0, np.abs(r), 1.0) + r)
            never = (r != 0) & np.all(est_all == 0, axis=0)
            # quota smoke test: a bin with a nominal count >= 10 over the scan must be
            # hit; rarer never-hit bins are reported as unmeasured, not as covered
            # (coverage and precision are established by tests/sf/precision instead)
            eh = np.broadcast_to(exp_hits[pop].reshape((1,) * (never.ndim - 1) + (-1,)),
                                 never.shape)
            never_bad = never & (eh >= 10.0)
            det_bad = (~good) & (np.abs(m) > 1e-9) & ~never
            if never.any():
                print(f"  {name} {k}: never-populated (unmeasured) bins={never.sum()}"
                      f" (of which nominal count >= 10: {never_bad.sum()};"
                      f" max nominal count among them {eh[never].max():.2f})")
            if never_bad.any() or det_bad.any():
                print(f"  {name} {k}: required bins never populated={never_bad.sum()}"
                      f" deterministic mismatches={det_bad.sum()} FAIL")
                ok = False
            # the t-test needs the file mean to be a sum of many nonzero draws: a bin
            # hit in only a few files is a rare-event estimate (mean set by a handful of
            # values, the rest exactly -1) whose z is meaningless; it is reported as
            # sparse and not tested (min-quota scans), unless deterministic (det_bad)
            hits = (np.array([x[pop] for x in errs[k]]) * np.where(r != 0, np.abs(r), 1.0)
                    + r != 0).sum(axis=0)
            sparse = good & (hits < MIN_HIT_FILES)
            if sparse.any():
                print(f"  {name} {k}: sparse bins "
                      f"(hit in < {MIN_HIT_FILES} of {K} files, "
                      f"not z-tested)={sparse.sum()}")
            good = good & ~sparse
            z = np.abs(m[good] / se[good])
            ntest = max(z.size, 1)
            pmin = 2.0 * stats.t.sf(z.max(initial=0), K - 1) * ntest
            line_ok = pmin > 1e-3
            ok &= line_ok
            print(f"  {name} {k}: bins={z.size:4d}"
                  f" max|mean rel err|={np.abs(m[good]).max():.2e}"
                  f" typical single-file rel err={np.median(ep.std(0)[good]):.2e}"
                  f" max|z|={z.max():.2f} Bonferroni p={min(1, pmin):.2e}"
                  f" {'OK' if line_ok else 'FAIL'}")
    S = np.array(S_err)
    m = np.nanmean(S, 0)
    se = np.nanstd(S, 0, ddof=1) / np.sqrt(K)
    good = np.isfinite(m) & (se > 0)
    print(f"  T/{dn}: max|mean {'abs' if corr else 'rel'} err|="
          f"{np.abs(m[good]).max():.2e}"
          f" max|z|={np.abs(m[good] / se[good]).max():.2f} over {good.sum()} entries"
          f" (ratio bias over files, informational)")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
