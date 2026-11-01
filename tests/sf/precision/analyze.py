#!/usr/bin/env python3
"""Statistical precision of the sf output at several sampling budgets (v2).

usage: analyze.py RUNDIR [TMIN] [ORDERS]     e.g. analyze.py run256 2.0 1,2,3

For every sf_<mode>_<budget> directory in RUNDIR and every output with time >= TMIN:
  * the stored estimate S_p = sum_r T_r / sum_r C_r (ratio of replicate sums), for the
    E1 (within-block) and E2 (cross-block) populations separately and combined;
  * its standard error by delete-one-replicate jackknife, which keeps each replicate's
    (T_r, C_r) pair together; RSE = SE / |S_p|;
  * coverage = fraction of (weight, bin) in which every replicate has C > 0.
The table gives the median and 90th percentile of RSE over the populated bins, per
population and order, pooled over components.

Budgets are compared with the largest one, output by output at equal time, by the
PAIRED jackknife: replicate i is deleted from both budgets at once, so draws shared
through the common seed, output number and replicate index are kept as covariance.
Reported per order: rms and max |z| and the fraction of entries with z > 0. A
discrepancy is not a bias estimate: the largest budget is itself sampled.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sf_reference import read_sf  # noqa: E402

run = sys.argv[1]
tmin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
ORDERS = [int(p) for p in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3]
POPS = {"E1": [0], "E2": [1], "all": [0, 1]}


def jack(s, pops):
    """S (nw, nq, P, nb) and its jackknife replicates (R, nw, nq, P, nb)"""
    T = s["T_er"][pops].sum(axis=0)          # (R, nw, nq, P, nb)
    C = s["C_er"][pops].sum(axis=0)          # (R, nw, nb)
    Ts, Cs = T.sum(axis=0), C.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = Ts / Cs[:, None, None, :]
        Sj = (Ts[None] - T) / (Cs[None] - C)[:, :, None, None, :]
    return S, Sj


def jse(Sj):
    R = Sj.shape[0]
    return np.sqrt((R - 1) / R * ((Sj - Sj.mean(axis=0)) ** 2).sum(axis=0))


def load(files):
    out = {}
    for f in files:
        meta, s = read_sf(f)
        if float(meta["time"]) >= tmin:
            out[round(float(meta["time"]), 9)] = (meta, s)   # pair budgets by time
    return out


groups = {}
for d in sorted(glob.glob(os.path.join(run, "sf_*"))):
    mode, budget = os.path.basename(d)[3:].rsplit("_", 1)
    groups.setdefault(mode, {})[budget] = load(sorted(glob.glob(d + "/*.sf")))

for mode, budgets in groups.items():
    budgets = {b: v for b, v in budgets.items() if v}
    print(f"== mode {mode} (outputs with time >= {tmin}; orders {ORDERS})")

    def size(b):   # reference = most E2 pairs, then most E1 draws, then most anchors
        m = next(iter(budgets[b].values()))[0]
        return (int(m["nanchor_global"]) * int(m["nsample_global"]),
                int(m["ndraw_block"]), int(m["nanchor_global"]))
    order = sorted(budgets, key=size)
    ref = order[-1]
    for b in order:
        outs = budgets[b]
        meta, s = next(iter(outs.values()))
        R = s["C_er"].shape[1]
        cov = np.mean([(o[1]["C_er"].sum(axis=0) > 0).all(axis=0).mean()
                       for o in outs.values()])
        te1 = np.mean([float(o[0].get("time_e1", "nan")) for o in outs.values()])
        te2 = np.mean([float(o[0].get("time_e2", "nan")) for o in outs.values()])
        print(f"-- {b}: nreplicate={R} ndraw_block={meta['ndraw_block']} "
              f"nanchor_global={meta['nanchor_global']} "
              f"nsample_global={meta['nsample_global']} seed={meta.get('seed', '?')}; "
              f"{len(outs)} outputs; time_e1 {te1:.2f} s, time_e2 {te2:.2f} s per "
              f"output; coverage {cov:.3f}")
        nw = s["C_er"].shape[2]
        for w in range(nw):
            for pop, idx in POPS.items():
                line = []
                for p in ORDERS:
                    rse = []
                    for _, so in outs.values():
                        S, Sj = jack(so, idx)
                        x, e = S[w, :, p - 1, :], jse(Sj)[w, :, p - 1, :]
                        good = np.isfinite(x) & np.isfinite(e) & (x != 0)
                        rse.append(e[good] / np.abs(x[good]))
                    rse = np.concatenate(rse)
                    line.append(f"p{p}:{np.median(rse):.1e}/{np.percentile(rse, 90):.1e}"
                                if rse.size else f"p{p}:-")
                print(f"   w{w} {pop:3s} RSE median/p90  " + "  ".join(line))
        if b == ref:
            continue
        # paired comparison with the largest budget, matched output by output
        parts = []
        for p in ORDERS:
            zs = []
            for fn, (_, so) in outs.items():
                if fn not in budgets[ref]:
                    continue
                Sa, Sja = jack(so, POPS["all"])
                Sb, Sjb = jack(budgets[ref][fn][1], POPS["all"])
                if Sja.shape[0] != Sjb.shape[0]:
                    raise SystemExit("replicate counts differ; cannot pair")
                with np.errstate(invalid="ignore", divide="ignore"):
                    z = (Sa - Sb) / jse(Sja - Sjb)
                zs.append(z[:, :, p - 1, :][np.isfinite(z[:, :, p - 1, :])])
            z = np.concatenate(zs)
            parts.append(f"p{p}: rms {np.sqrt(np.mean(z ** 2)):.2f}"
                         f" max {np.abs(z).max():.2f}"
                         f" frac>0 {np.mean(z > 0):.2f}")
        print(f"   paired vs {ref} (all pops): " + "; ".join(parts))
