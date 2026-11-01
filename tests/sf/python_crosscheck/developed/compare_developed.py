#!/usr/bin/env python3
"""Acceptance test: on-the-fly sf output against athena_research on one developed state.

usage: compare_developed.py RUNDIR PYDIR [--alpha 1e-3] [--prec 0.01]

RUNDIR: the AthenaK run (sf_scalar/, sf_longtrans/ files of one cycle). PYDIR: the
python_sf.py results {scalar,longtrans}_sampled_s<seed>.npz on the restart state of the
same cycle. The rules below were fixed before any comparison was run.

Compared quantity, for every mode, weight w, increment q, order p and separation bin b:
  Q = T / W     AthenaK: sum over replicates of T_r / sum of W_r (E1 + E2)
                Python:  sf / weight_sum_bin of one seed
Both estimate the same population ratio sum |w2 q2 - w1 q1|^p / sum (w1 + w2), whatever
the weight (with unit weights Q = S_p / 2), so the two normalisations need no mapping.
Pair counts are compared separately, each against the exact count of the mesh:
  C_exact[b] = N/2 x (number of distinct periodic cell offsets with |l| in bin b)
  AthenaK C (unit weight, replicates pooled) and Python weight_sum_bin / 2 of the unit
  weight (its draws carry importance weights, _random_pair_weight, so the weighted sum
  estimates the population; num_bin_points counts raw draws, which are denser for nearby
  MeshBlocks).
  Revision note: the first version of this script (before any result) normalised
  num_bin_points by the total draws, which assumes uniform pair sampling. The first run
  showed Python's distance-dependent block-pair sampling (up to 46% off in mid-range
  bins), so the Python count now uses the weighted sum. No other rule was changed.

Errors: AthenaK, delete-one-replicate jackknife over all replicates of all files (df =
R - 1); Python, the standard error of the mean over seeds (df = S - 1). Differences use
SE^2 = SE_athenak^2 + SE_python^2 with Welch-Satterthwaite df, and a two-sided t-test.

Acceptance (all must hold):
  1. inventory: identical bin edges (relative 1e-12); in every file and seed, exactly the
     bins with an exact pair have C > 0 / draws > 0 (AthenaK unit and density weights,
     Python each seed); nothing nonfinite;
  2. family-wise test: with M the number of compared values (all Q and both count
     comparisons), no two-sided p-value below alpha / M (Bonferroni, alpha = 1e-3);
     a zero combined error requires an exact match;
  3. precision: per (mode, order), the median over (w, q, b) of the relative combined
     error SE/|Q| is at most --prec (default 1%), so that the test can detect a
     discrepancy of a few percent; otherwise the result is INCONCLUSIVE, not a pass;
  4. AthenaK internal consistency: W = 2 C for the unit weight (relative 1e-12).
Prints a summary per mode and order, the worst values, and DEVELOPED COMPARISON:
PASS / FAIL / INCONCLUSIVE.
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from sf_reference import read_sf  # noqa: E402
from reference_manifest import SEEDS, verify  # noqa: E402


def validate_inventory(mode, metas, arrs, python, filenames):
    """Require every channel and seed in the declared developed-state comparison."""
    velocity = "hydro_w_vx,hydro_w_vy,hydro_w_vz"
    scalar = mode == "scalar"
    nw, nq = (2, 4) if scalar else (1, 2)
    expected = {
        "mode": "scalar" if scalar else "long_trans",
        "variables": velocity + ",hydro_w_d" if scalar else velocity,
        "weights": "ones,hydro_w_d" if scalar else "ones",
        "max_order": "3", "nbin": "100", "nreplicate": "8",
        "mesh_nx": "256,256,256", "meshblock_nx": "64,64,64",
        "periodic": "1,1,1",
    }
    if len(metas) != 1 or len(python) != len(SEEDS):
        raise ValueError(f"{mode}: expected one sf file and {len(SEEDS)} Python seeds")
    for meta, arrays in zip(metas, arrs):
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"{mode}: {key}={meta.get(key)!r}, expected {value!r}")
        if arrays["T"].shape != (nw, nq, 3, 100):
            raise ValueError(f"{mode}: unexpected AthenaK shape {arrays['T'].shape}")
    seen = []
    for result, filename in zip(python, filenames):
        seed = int(result["seed"])
        seen.append(seed)
        if (str(result["mode"]) != mode or str(result["kind"]) != "sampled"
                or os.path.basename(filename) != f"{mode}_sampled_s{seed}.npz"):
            raise ValueError(f"{filename}: mode, kind or seed does not match its name")
        shapes = {"sf": (nw, nq, 3, 100), "weight_sum_bin": (nw, 100),
                  "num_bin_points": (nw, 100), "rbins": (101,)}
        for key, shape in shapes.items():
            if result[key].shape != shape:
                raise ValueError(
                    f"{filename}: {key} shape {result[key].shape} != {shape}")
        if float(result["npairs"]) != float(1024 * 64) ** 2:
            raise ValueError(f"{filename}: sampled pair budget differs from NS1=1024")
    if sorted(seen) != list(SEEDS):
        raise ValueError(f"{mode}: expected seed IDs {list(SEEDS)}, found {sorted(seen)}")


def exact_counts(meta, edges):
    """N/2 x number of distinct offset classes per bin (whole periodic mesh)."""
    n = [int(v) for v in meta["mesh_nx"].split(",")]
    h = [float(v) / m for v, m in zip(meta["domain_length"].split(","), n)]
    ax = []
    for d in range(3):
        off = np.arange(n[d], dtype=np.int64)
        off = np.where(2 * off > n[d], off - n[d], off)    # one offset per class
        ax.append((off.astype(np.float64) * h[d]) ** 2)
    r2 = (ax[2][:, None, None] + ax[1][None, :, None] + ax[0][None, None, :]).ravel()
    r = np.sqrt(r2[r2 > 0])
    b = np.searchsorted(edges, r, side="right") - 1
    nb = len(edges) - 1
    cnt = np.bincount(b[(b >= 0) & (b < nb)], minlength=nb).astype(np.float64)
    N = float(np.prod(n))
    return 0.5 * N * cnt, N


def jackknife(Tr, Wr):
    """ratio of sums and its delete-one jackknife SE; replicate axis 0"""
    Ts, Ws = Tr.sum(0), Wr.sum(0)
    R = Tr.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        Q = Ts / Ws
        Qj = (Ts[None] - Tr) / (Ws[None] - Wr)
    se = np.sqrt((R - 1) / R * ((Qj - Qj.mean(0)) ** 2).sum(0))
    return Q, se, R - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("pydir")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--prec", type=float, default=0.01)
    a = ap.parse_args()
    try:
        verify(a.run, a.pydir)
    except (OSError, ValueError, KeyError) as error:
        print("FAIL: reference provenance:", error)
        print("DEVELOPED COMPARISON: FAIL")
        return 1
    fail, rows, tests = [], [], []   # tests: (label, diff, se, df)
    inconclusive = []
    state = None
    for mode, sub in (("scalar", "sf_scalar"), ("longtrans", "sf_longtrans")):
        files = sorted(glob.glob(os.path.join(a.run, sub, "*.sf")))
        pyf = sorted(glob.glob(os.path.join(a.pydir, f"{mode}_sampled_s*.npz")))
        if not files or len(pyf) < 2:
            fail.append(f"{mode}: {len(files)} sf files, {len(pyf)} python seeds")
            continue
        metas, arrs = zip(*[read_sf(f) for f in files])
        state_keys = ("cycle", "time", "mesh_nx", "domain_length", "meshblock_nx",
                      "periodic", "region")
        for meta in metas:
            current = tuple(meta[key] for key in state_keys)
            if state is not None and current != state:
                fail.append(f"{mode}: output state differs from the other SF files")
            state = current
        cycles = {m["cycle"] for m in metas}
        if len(cycles) != 1:
            fail.append(f"{mode}: sf files of several cycles {sorted(cycles)}")
        edges = arrs[0]["rbins"]
        C_ex, N = exact_counts(metas[0], edges)
        want = C_ex > 0
        # AthenaK replicates (E1 + E2) of all files: [R, w, q, p, b] and [R, w, b]
        Tr = np.concatenate([x["T_er"].sum(0) for x in arrs])
        Wr = np.concatenate([x["W_er"].sum(0) for x in arrs])
        Cr = np.concatenate([x["C_er"].sum(0) for x in arrs])
        for name, v in (("T", Tr), ("W", Wr), ("C", Cr)):
            if not np.all(np.isfinite(v)):
                fail.append(f"{mode}: AthenaK {name} nonfinite")
        for x, f in zip(arrs, files):
            got = x["C"] > 0
            for w in range(got.shape[0]):
                if not np.array_equal(got[w], want):
                    fail.append(f"{mode} {os.path.basename(f)} w{w}: populated bins "
                                f"{np.flatnonzero(got[w] != want).tolist()} differ")
        wnames = metas[0]["weights"].split(",")
        if "ones" in wnames:
            w1 = wnames.index("ones")
            dev = np.max(np.abs(Wr[:, w1] - 2 * Cr[:, w1]) / np.maximum(Wr[:, w1], 1))
            if not dev <= 1e-12:
                fail.append(f"{mode}: unit-weight W != 2C (max rel {dev:.1e})")
        QA, seA, dfA = jackknife(Tr, Wr[:, :, None, None, :])
        # Python seeds
        P = [dict(np.load(f)) for f in pyf]
        for p_ in P:   # weight axis first, also for a single weight
            for name in ("weight_sum_bin", "num_bin_points"):
                if p_[name].ndim == 1:
                    p_[name] = p_[name][None]
        try:
            validate_inventory(mode, metas, arrs, P, pyf)
        except (ValueError, KeyError) as error:
            fail.append(str(error))
            continue
        for p_, f in zip(P, pyf):
            if not np.allclose(p_["rbins"], edges, rtol=1e-12, atol=0):
                fail.append(f"{os.path.basename(f)}: bin edges differ from AthenaK")
            for name in ("sf", "weight_sum_bin", "num_bin_points"):
                if not np.all(np.isfinite(p_[name])):
                    fail.append(f"{os.path.basename(f)}: {name} nonfinite")
            for w in range(p_["num_bin_points"].shape[0]):
                got = p_["num_bin_points"][w] > 0
                if not np.array_equal(got, want):
                    fail.append(f"{os.path.basename(f)} w{w}: populated bins "
                                f"{np.flatnonzero(got != want).tolist()} differ")
        sf = np.stack([p_["sf"] for p_ in P])                     # [S, w, q, p, b]
        ws = np.stack([p_["weight_sum_bin"] for p_ in P])         # [S, w, b]
        np.stack([p_["num_bin_points"] for p_ in P])              # [S, w, b]
        with np.errstate(invalid="ignore", divide="ignore"):
            Qs = sf / ws[:, :, None, None, :]
        S = Qs.shape[0]
        QP, seP, dfP = Qs.mean(0), Qs.std(0, ddof=1) / np.sqrt(S), S - 1
        if QA.shape != QP.shape:
            fail.append(f"{mode}: shapes AthenaK {QA.shape} python {QP.shape}")
            continue
        sel = np.broadcast_to(want, QA.shape)
        se = np.sqrt(seA ** 2 + seP ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            df = se ** 4 / (seA ** 4 / dfA + seP ** 4 / dfP)
        it = np.nditer(QA, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            if not sel[i]:
                continue
            tests.append((f"{mode} Q w{i[0]} q{i[1]} p{i[2] + 1} bin{i[3]}",
                          QA[i] - QP[i], se[i], df[i] if se[i] > 0 else 1.0, QP[i]))
        for p in range(QA.shape[2]):
            rel = (se[:, :, p] / np.abs(QP[:, :, p]))[sel[:, :, p]]
            med = float(np.median(rel))
            z = (np.abs(QA[:, :, p] - QP[:, :, p])[sel[:, :, p]]
                 / se[:, :, p][sel[:, :, p]])
            med_rd = np.median((np.abs(QA[:, :, p] - QP[:, :, p])
                                / np.abs(QP[:, :, p]))[sel[:, :, p]])
            rows.append(f"{mode:9s} p={p + 1}: {sel[:, :, p].sum():5d} values, median "
                        f"rel SE {med:.2%}, median |rel diff| "
                        f"{med_rd:.2%}, "
                        f"max |z| {np.max(z):.2f}")
            if not med <= a.prec:
                inconclusive.append(f"{mode} p={p + 1} median rel SE {med:.2%}"
                                    f" > {a.prec:.0%}")
        # pair counts against the exact count (unit weight of AthenaK; Python per seed)
        if "ones" in wnames:
            # every replicate estimates the full count: mean and its standard error
            Cm = Cr[:, w1]
            R = Cm.shape[0]
            CA = Cm.mean(0)
            seC = Cm.std(0, ddof=1) / np.sqrt(R)
            for b in np.flatnonzero(want):
                tests.append((f"{mode} C athenak bin{b}", CA[b] - C_ex[b], seC[b], R - 1,
                              C_ex[b]))
        CPs = 0.5 * ws[:, 0]                 # unit weight: sum (1 + 1) / 2 per pair
        CP, seCP = CPs.mean(0), CPs.std(0, ddof=1) / np.sqrt(S)
        for b in np.flatnonzero(want):
            tests.append((f"{mode} C python bin{b}", CP[b] - C_ex[b], seCP[b], S - 1,
                          C_ex[b]))
    M = len(tests)
    worst = []
    for lab, d, se, df, ref in tests:
        if not np.isfinite(d) or not np.isfinite(se):
            fail.append(f"{lab}: nonfinite difference or error")
            continue
        if se == 0:
            if d != 0:
                fail.append(f"{lab}: zero error but difference {d:.3e}")
            continue
        pval = 2 * stats.t.sf(abs(d / se), df)
        worst.append((pval, lab, d / se, d / ref if ref else np.nan))
        if pval < a.alpha / max(M, 1):
            fail.append(f"{lab}: z = {d / se:.2f} (p = {pval:.1e} < {a.alpha / M:.1e})")
    worst.sort()
    print("\n".join(rows))
    print(f"{M} compared values; Bonferroni threshold p < {a.alpha / max(M, 1):.2e}")
    for pval, lab, z, rel in worst[:8]:
        print(f"  smallest p: {lab}: z = {z:+.2f}, rel diff {rel:+.2e}, p = {pval:.2e}")
    for f in fail[:30]:
        print("FAIL:", f)
    for f in inconclusive:
        print("INCONCLUSIVE:", f)
    verdict = "FAIL" if fail else ("INCONCLUSIVE" if inconclusive else "PASS")
    print("DEVELOPED COMPARISON:", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
