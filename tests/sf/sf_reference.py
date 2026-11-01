#!/usr/bin/env python3
"""Validate AthenaK on-the-fly structure-function output (file_type = sf).

Rebuilds the deterministic fields of src/pgen/sf_unit_test.cpp, computes exhaustive
all-pairs references for the two estimator populations (E1: same MeshBlock, E2:
different MeshBlocks) and compares them with the sampled sums in the .sf files, using
the replicate spread as the standard error.

Usage: python3 -I tests/sf/sf_reference.py RUN_DIR [--noise 1.0] [--zmax 5]
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy import stats

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def mix(x):
    """splitmix64 finaliser on uint64 arrays (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & M64
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & M64
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & M64
        return x ^ (x >> np.uint64(31))


def noise(gi, gj, gk, comp):
    h = mix(np.full(gi.shape, comp, dtype=np.uint64))
    h = mix(h ^ gi.astype(np.uint64))
    h = mix(h ^ gj.astype(np.uint64))
    h = mix(h ^ gk.astype(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0) - 0.5


def read_header(f, fname):
    """Header keys of an open .sf/.spec file, leaving f at the payload. Raises
    ValueError on an empty file, a header cut short or an invalid header offset."""
    meta = {}
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError(f"{fname}: file ends inside the header")
        key, _, val = raw.decode("ascii").rstrip("\n").partition("=")
        meta[key] = val
        if key == "header offset":
            break
    try:
        off = int(meta["header offset"])
    except ValueError:
        raise ValueError(f"{fname}: invalid header offset {meta['header offset']!r}")
    if off != f.tell():
        raise ValueError(f"{fname}: header offset {off} != header length {f.tell()}")
    return meta


def read_sf(fname):
    """Return (meta dict, arrays dict) for one .sf file."""
    with open(fname, "rb") as f:
        meta = read_header(f, fname)
        payload = f.read()
    nw, nq = int(meta["nweight"]), int(meta["nq"])
    P, nb, R = int(meta["max_order"]), int(meta["nbin"]), int(meta["nreplicate"])
    if meta["mode"] == "aniso_mhd":
        rb = [("rbins", (int(meta["nbin_l"]) + 1,), "f8"),
              ("rbins_prll", (int(meta["nbin_prll"]) + 1,), "f8"),
              ("rbins_perp", (int(meta["nbin_perp"]) + 1,), "f8")]
    else:
        rb = [("rbins", (nb + 1,), "f8")]
    shapes = rb + [("T", (nw, nq, P, nb), "f8"),
                   ("W", (nw, nb), "f8"), ("C", (nw, nb), "f8"),
                   ("T_er", (2, R, nw, nq, P, nb), "f8"),
                   ("W_er", (2, R, nw, nb), "f8"),
                   ("C_er", (2, R, nw, nb), "f8"), ("n_acc_er", (2, R, nw, nb), "u8")]
    if meta.get("histogram", "0") == "1":
        nc = int(meta["hist_ncol"])
        shapes += [("hist_edges", (nw, nq, nc - 1), "f8"), ("hist_sigma", (nw, nq), "f8"),
                   ("H", (nw, nq, nb, nc), "f8"), ("H_er", (2, R, nw, nq, nb, nc), "f8")]
    out, off = {}, 0
    for name, shape, dt in shapes:
        n = int(np.prod(shape))
        if off + 8 * n > len(payload):
            raise ValueError(f"{fname}: payload ends inside {name}")
        out[name] = np.frombuffer(payload, dtype=dt, count=n, offset=off).reshape(shape)
        off += 8 * n
    if off != len(payload):
        raise ValueError(f"{fname}: payload is {len(payload)} bytes, layout needs {off}")
    return meta, out


def test_mask(mtype, cx, cy, cz):
    spheres = {1: [(0.30, 0.30, 0.30, 0.16), (0.95, 0.55, 0.02, 0.12)],
               2: [(0.50, 0.50, 0.50, 0.07)],
               4: [(0.25, 0.125, 0.125, 0.07)]}.get(mtype, [])
    m = np.zeros_like(cx)
    for x0, y0, z0, r in spheres:
        dx, dy, dz = cx - x0, cy - y0, cz - z0
        dx = dx - np.floor(dx + 0.5)
        dy = dy - np.floor(dy + 0.5)
        dz = dz - np.floor(dz + 0.5)
        m[dx * dx + dy * dy + dz * dz < r * r] = 1.0
    return m


def build_fields(meta, noise_amp):
    nx = [int(v) for v in meta["mesh_nx"].split(",")]
    xmin = [float(v) for v in meta["mesh_min"].split(",")]
    L = [float(v) for v in meta["domain_length"].split(",")]
    gk, gj, gi = np.meshgrid(np.arange(nx[2]), np.arange(nx[1]), np.arange(nx[0]),
                             indexing="ij")
    gi, gj, gk = gi.ravel(), gj.ravel(), gk.ravel()
    x = xmin[0] + (gi + 0.5) * L[0] / nx[0]
    y = xmin[1] + (gj + 0.5) * L[1] / nx[1]
    z = xmin[2] + (gk + 0.5) * L[2] / nx[2]
    sx, sy, sz = (x - xmin[0]) / L[0], (y - xmin[1]) / L[1], (z - xmin[2]) / L[2]
    tp = 2.0 * np.pi
    d = (1.0 + 0.3 * np.sin(tp * sx) * np.cos(tp * sy)
         + 0.2 * noise_amp * noise(gi, gj, gk, 0))
    vx = np.sin(tp * (sx + 2.0 * sy)) + 0.5 * noise_amp * noise(gi, gj, gk, 1)
    vy = np.cos(tp * (3.0 * sx - sz)) + 0.5 * noise_amp * noise(gi, gj, gk, 2)
    vz = np.sin(tp * sz) * np.cos(tp * sx) + 0.5 * noise_amp * noise(gi, gj, gk, 3)
    nb = [int(v) for v in meta["meshblock_nx"].split(",")]
    block = (gk // nb[2]) * (nx[1] // nb[1]) * (nx[0] // nb[0]) + \
        (gj // nb[1]) * (nx[0] // nb[0]) + gi // nb[0]
    fields = {"hydro_w_d": d, "hydro_w_vx": vx, "hydro_w_vy": vy, "hydro_w_vz": vz,
              "hydro_u_m1": d * vx, "hydro_u_m2": d * vy, "hydro_u_m3": d * vz}
    # eos = ideal (sf_unit_test.cpp):
    # p = 0.6 + 0.25 cos(2 pi (sx - sy)) sin(2 pi sz) + noise
    p = 0.6 + 0.25 * np.cos(tp * (sx - sy)) * np.sin(tp * sz) + \
        0.1 * noise_amp * noise(gi, gj, gk, 4)
    fields["temperature"] = p / d
    fields["pressure"] = p
    # sparse indicator (first passive scalar, problem/mask; TestMask in sf_unit_test.cpp)
    mtype = int(meta.get("test_mask", "0"))
    fields["hydro_w_s"] = (d * test_mask(mtype, (gi + 0.5) / nx[0], (gj + 0.5) / nx[1],
                                         (gk + 0.5) / nx[2])) / d
    for k in list(fields):
        fields["mhd_w_" + k.split("_")[-1]] = fields[k]
    # MHD face fields of sf_unit_test.cpp (functions of the wrapped global face index),
    # cell-centred as AthenaK does: bcc = 0.5*(lower face + upper face)
    n1, n2, n3 = nx
    ci, cj, ck = (gi + 0.5) / n1, (gj + 0.5) / n2, (gk + 0.5) / n3

    def f1(fi):
        return (0.7 + 0.3 * np.sin(tp * (cj + ck))
                + 0.2 * noise_amp * noise(fi, gj, gk, 11))

    def f2(fj):
        return (0.2 * np.cos(tp * (ci + 2.0 * ck))
                + 0.2 * noise_amp * noise(gi, fj, gk, 12))

    def f3(fk):
        return (0.1 + 0.25 * np.sin(tp * ci) * np.cos(tp * cj)
                + 0.2 * noise_amp * noise(gi, gj, fk, 13))

    fields["mhd_bcc1"] = 0.5 * (f1(gi) + f1((gi + 1) % n1))
    fields["mhd_bcc2"] = 0.5 * (f2(gj) + f2((gj + 1) % n2))
    fields["mhd_bcc3"] = 0.5 * (f3(gk) + f3((gk + 1) % n3))
    return np.stack([x, y, z]), block, fields


def selection(meta, fields):
    """select<n>=name,lo,hi header keys: cells with lo <= value <= hi for every n"""
    sel = np.ones(next(iter(fields.values())).shape, dtype=bool)
    for c in range(1, int(meta.get("nselect", "0")) + 1):
        name, lo, hi = meta["select%d" % c].split(",")
        v = fields[name]
        sel &= (v >= float(lo)) & (v <= float(hi))
    return sel


def region_weights(meta, pos, fields):
    """Weight fields with cells outside the output's region, or not selected (select*
    keys), set to 0 (invalid)."""
    reg = [float(v) for v in meta["region"].split(",")]
    inreg = selection(meta, fields)
    for ax in range(3):
        inreg &= (pos[ax] >= reg[2 * ax]) & (pos[ax] <= reg[2 * ax + 1])
    return [np.where(inreg, np.ones(pos.shape[1]) if w == "ones" else fields[w], 0.0)
            for w in meta["weights"].split(",")]


def run_exhaustive(rbins, pos, block, q, wts, per, L, mode, P, chunk=256, hedges=None,
                   hnedge=None):
    """Exact population sums; with hedges (nw, nq, ncol-1) also the increment
    histograms H[e, w, q, bin, col] (col = searchsorted(edges, v, 'right'))."""
    nw, nb = len(wts), len(rbins) - 1
    nq = q.shape[0] if mode == "scalar" else (1 if mode == "correlation" else 2)
    ncol = None if hedges is None else hedges.shape[-1] + 1
    H = None if hedges is None else np.zeros((2, nw, nq, nb, ncol))
    T = np.zeros((2, nw, nq, P, nb))
    W = np.zeros((2, nw, nb))
    C = np.zeros((2, nw, nb))
    N = pos.shape[1]
    for a0 in range(0, N, chunk):
        a = np.arange(a0, min(N, a0 + chunk))
        d = pos[:, None, :] - pos[:, a, None]        # (3, na, N): x_j - x_i
        for ax in range(3):
            if per[ax]:
                big = np.abs(d[ax]) > 0.5 * L[ax]
                d[ax] = np.where(big, d[ax] - np.sign(d[ax]) * L[ax], d[ax])
        r = np.sqrt((d ** 2).sum(axis=0))
        b = np.searchsorted(rbins, r, side="right") - 1
        same = block[a][:, None] == block[None, :]
        notself = a[:, None] != np.arange(N)[None, :]
        inb = (b >= 0) & (b < nb) & notself
        for iw, w in enumerate(wts):
            w1, w2 = w[a][:, None], w[None, :]
            valid = inb & (w1 > 0) & (w2 > 0)
            if mode == "correlation":
                # Python get_cf: w1*w2*((A1 B2)^p + (B1 A2)^p) per unordered pair,
                # W = 2 w1 w2
                ab = q[0][a][:, None] * q[-1][None, :]
                ba = q[-1][a][:, None] * q[0][None, :]
                pwt = w1 * w2
                for e, pop in enumerate([same, ~same]):
                    sel = valid & pop
                    bb = b[sel]
                    W[e, iw] += 0.5 * np.bincount(bb, weights=2 * pwt[sel], minlength=nb)
                    C[e, iw] += 0.5 * np.bincount(bb, minlength=nb)
                    x, y, wt = ab[sel], ba[sel], pwt[sel]
                    for p in range(P):
                        T[e, iw, 0, p] += 0.5 * np.bincount(
                            bb, weights=wt * (x ** (p + 1) + y ** (p + 1)), minlength=nb)
                continue
            if mode == "scalar":
                incs = [np.abs(w2 * q[v][None, :] - w1 * q[v][a][:, None])
                        for v in range(q.shape[0])]
            else:
                u = [w2 * q[c][None, :] - w1 * q[c][a][:, None] for c in range(3)]
                rr = np.where(r > 0, r, 1.0)
                ul = (u[0] * d[0] + u[1] * d[1] + u[2] * d[2]) / rr
                ut = np.sqrt(np.maximum(0.0, u[0] ** 2 + u[1] ** 2 + u[2] ** 2 - ul ** 2))
                incs = [np.abs(ul), ut]
            hvals = incs if mode == "scalar" else [ul, ut]
            for e, pop in enumerate([same, ~same]):
                sel = valid & pop
                bb = b[sel]
                W[e, iw] += 0.5 * np.bincount(bb, weights=(w1 + w2)[sel], minlength=nb)
                C[e, iw] += 0.5 * np.bincount(bb, minlength=nb)
                for iq, inc in enumerate(incs):
                    x = inc[sel]
                    pw = x.copy()
                    for p in range(P):
                        T[e, iw, iq, p] += 0.5 * np.bincount(bb, weights=pw, minlength=nb)
                        pw *= x
                    if H is not None:
                        ed = hedges[iw, iq, :hnedge[iq]]
                        col = np.searchsorted(ed, hvals[iq][sel], side="right")
                        H[e, iw, iq] += 0.5 * np.bincount(
                            bb * ncol + col, minlength=nb * ncol).reshape(nb, ncol)
    if H is not None:
        return T, W, C, H
    return T, W, C


def run_exhaustive_aniso(edges, pos, block, q, wts, per, L, P, cos_prll_min,
                         cos_perp_max, chunk=256):
    """Exact population sums for mode aniso_mhd, following
    structure_function_aniso_mhd_kernel: q = (vx, vy, vz, bx, by, bz); bins of the
    three families side by side (l, l_prll, l_perp)."""
    el, ep, eq = edges
    nbl, nbp, nbq = len(el) - 1, len(ep) - 1, len(eq) - 1
    nb = nbl + nbp + nbq
    nw = len(wts)
    T = np.zeros((2, nw, 4, P, nb))
    W = np.zeros((2, nw, nb))
    C = np.zeros((2, nw, nb))
    N = pos.shape[1]

    def binof(e, x):
        b = np.searchsorted(e, x, side="right") - 1
        return np.where((b >= 0) & (b < len(e) - 1), b, -1)

    for a0 in range(0, N, chunk):
        a = np.arange(a0, min(N, a0 + chunk))
        d = pos[:, None, :] - pos[:, a, None]
        for ax in range(3):
            if per[ax]:
                big = np.abs(d[ax]) > 0.5 * L[ax]
                d[ax] = np.where(big, d[ax] - np.sign(d[ax]) * L[ax], d[ax])
        lsq = (d ** 2).sum(axis=0)
        lmag = np.sqrt(lsq)
        bm = [0.5 * (q[3 + c][a][:, None] + q[3 + c][None, :]) for c in range(3)]
        bmag = np.sqrt(bm[0] ** 2 + bm[1] ** 2 + bm[2] ** 2)
        ok = (bmag >= 1e-30) & (lmag >= 1e-30)
        bh = [np.where(ok, bm[c] / np.where(ok, bmag, 1.0), 0.0) for c in range(3)]
        ldb = d[0] * bh[0] + d[1] * bh[1] + d[2] * bh[2]
        lprll = np.abs(ldb)
        lperp = np.sqrt(np.maximum(0.0, lsq - ldb ** 2))
        cth = np.clip(lprll / np.where(lmag > 0, lmag, 1.0), 0.0, 1.0)
        bl = binof(el, lmag)
        bp = np.where(cth > cos_prll_min, binof(ep, lprll), -1)
        bq = np.where(cth <= cos_perp_max, binof(eq, lperp), -1)
        same = block[a][:, None] == block[None, :]
        for iw, w in enumerate(wts):
            w1, w2 = w[a][:, None], w[None, :]
            valid = ok & (w1 > 0) & (w2 > 0)
            incs = []
            for f in range(2):
                u = [w2 * q[3 * f + c][None, :] - w1 * q[3 * f + c][a][:, None]
                     for c in range(3)]
                usq = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
                ub = u[0] * bh[0] + u[1] * bh[1] + u[2] * bh[2]
                incs += [np.sqrt(usq), np.sqrt(np.maximum(0.0, usq - ub ** 2))]
            for e, pop in enumerate([same, ~same]):
                for fam, (bb_all, off) in enumerate([(bl, 0), (bp, nbl),
                                                     (bq, nbl + nbp)]):
                    sel = valid & pop & (bb_all >= 0)
                    bb = bb_all[sel] + off
                    wsum = np.broadcast_to(w1 + w2, sel.shape)[sel]
                    W[e, iw] += 0.5 * np.bincount(bb, weights=wsum, minlength=nb)
                    C[e, iw] += 0.5 * np.bincount(bb, minlength=nb)
                    for iq, inc in enumerate(incs):
                        x = inc[sel]
                        pw = x.copy()
                        for p in range(P):
                            T[e, iw, iq, p] += 0.5 * np.bincount(bb, weights=pw,
                                                                 minlength=nb)
                            pw *= x
    return T, W, C


def population_check(label, est, ref, R, signed=False):
    """Replicate estimates est (R, ...) of one population against the exact sums ref.

    Structural (exact_bad): an entry with zero replicate spread must equal the exact
    value, and an exact zero must never receive mass. With signed=True (correlation T,
    a sum of signed products) an exact zero can be a cancellation of nonzero terms, so a
    zero exact entry whose estimate fluctuates (spread > 0) is z-tested on its absolute
    error instead; a zero-spread nonzero estimate there stays structural. Sampling
    (stat_ok): Bonferroni over the replicate t-scores, and every required bin populated
    (histogram tails: unsampled exact mass below 1e-3 of the total)."""
    mean = est.mean(axis=0)
    se = est.std(axis=0, ddof=1) / np.sqrt(R)
    mask = (ref != 0) | (mean != 0)
    z = np.zeros_like(mean)
    nz = mask & (se > 0)
    z[nz] = (mean[nz] - ref[nz]) / se[nz]
    unsampled = mask & (se == 0) & (mean == 0) & (ref != 0)
    zero_hit = (ref == 0) & (mean != 0)
    signed_zero = zero_hit & (se > 0) if signed else np.zeros_like(zero_hit)
    exact_bad = (mask & (se == 0) & (mean != 0) &
                 ~np.isclose(mean, ref, rtol=1e-9, atol=0)) | (zero_hit & ~signed_zero)
    if label == "H":
        tail = np.abs(ref[unsampled]).sum() / max(np.abs(ref).sum(), 1e-300)
        tail_ok = tail < 1e-3
    else:
        tail = 0.0
        tail_ok = not unsampled.any()   # every required bin populated
    nzr = mask & (ref != 0)
    rel = np.abs(mean[nzr] - ref[nzr]) / np.abs(ref[nzr])
    zabs = np.abs(z[nz])
    # replicate z-scores follow Student t with R-1 dof
    nu = R - 1
    ntest = max(zabs.size, 1)
    pmin = 2.0 * stats.t.sf(zabs.max(initial=0), nu)
    ez2 = nu / (nu - 2.0) if nu > 2 else np.inf
    mz2 = (zabs ** 2).mean() if zabs.size else 0.0
    # bins within one file share samples (strongly correlated), so <z^2> is only
    # informational here; estimator bias is tested across seeds (seed_scan.py)
    stat_ok = bool((pmin * ntest > 1e-3) and tail_ok)
    return dict(bins=int(mask.sum()), zmax=float(zabs.max(initial=0)),
                p=min(1.0, pmin * ntest), mz2=mz2 / ez2,
                rel=float(np.median(rel)) if rel.size else 0.0,
                exact_bad=int(exact_bad.sum()), unsampled=int(unsampled.sum()),
                tail=float(tail), signed_zero=int(signed_zero.sum()), stat_ok=stat_ok)


def compare(fname, noise_amp, zmax, mask=0):
    meta, s = read_sf(fname)
    meta["test_mask"] = str(mask)
    pos, block, fields = build_fields(meta, noise_amp)
    variables = meta["variables"].split(",")
    meta["weights"].split(",")
    q = np.stack([fields[v] for v in variables])
    wts = region_weights(meta, pos, fields)
    per = [v == "1" for v in meta["periodic"].split(",")]
    L = [float(v) for v in meta["domain_length"].split(",")]
    P, R = int(meta["max_order"]), int(meta["nreplicate"])
    hist = meta.get("histogram", "0") == "1"
    # a pair needs both endpoints valid (weight > 0) for that weight, so the exact sums
    # only involve cells valid for some weight: drop the rest (sparse-weight fixtures)
    keep = np.any(np.stack(wts) > 0, axis=0)
    ok_nsel = True
    if "ncell_selected" in meta:
        reg = [float(v) for v in meta["region"].split(",")]
        nsel = selection(meta, fields)
        for ax in range(3):
            nsel &= (pos[ax] >= reg[2 * ax]) & (pos[ax] <= reg[2 * ax + 1])
        ok_nsel = int(meta["ncell_selected"]) == int(nsel.sum())
        print(f"  ncell_selected {meta['ncell_selected']} vs reference {int(nsel.sum())}"
              f" {'PASS' if ok_nsel else 'FAIL'}")
    pos, block, q = pos[:, keep], block[keep], q[:, keep]
    wts = [w[keep] for w in wts]
    # ok: structural/exact checks (finite payload, exact zero-spread entries, accounting,
    # histogram scales); ok_stat: sampling checks (z-tests, coverage of required bins,
    # unsampled histogram tails). main() exits 1 on a structural failure (or an
    # exception), 3 when only sampling checks fail, 0 otherwise.
    ok, ok_stat = ok_nsel, True
    # nonfinite values are never accepted (the sf output does not filter NaN/Inf input;
    # they propagate into the sums and must be caught here)
    for k, v in s.items():
        if v.dtype.kind == "f" and not np.all(np.isfinite(v)):
            print(f"  nonfinite values in {k}: {np.sum(~np.isfinite(v))} FAIL")
            ok = False
    if hist:
        hnedge = [int(v) for v in meta["hist_nedge"].split(",")]
        Tx, Wx, Cx, Hx = run_exhaustive(s["rbins"], pos, block, q, wts, per, L,
                                        meta["mode"], P, hedges=s["hist_edges"],
                                        hnedge=hnedge)
        # sigma: weighted std over valid cells (masked_increment_scales), 12 digits
        if meta["hist_scale"] == "sigma":
            sig = np.zeros((len(wts), q.shape[0]))
            for iw, w in enumerate(wts):
                ww = np.where(w > 0, w, 0.0)
                for v in range(q.shape[0]):
                    if np.sum(ww) == 0:   # empty selection: no scale (fallback below)
                        sig[iw, v] = np.nan
                        continue
                    m1 = np.sum(ww * q[v]) / np.sum(ww)
                    var = max(np.sum(ww * q[v] ** 2) / np.sum(ww) - m1 * m1, 0.0)
                    sig[iw, v] = float("%.12g" % np.sqrt(var))
            if meta["mode"] != "scalar":
                sig = np.repeat(np.sqrt(np.mean(sig ** 2, axis=1))[:, None], 2, axis=1)
            # BuildHistEdges convention: a final scale that is not finite and > 0
            # (constant field, empty selection) is 1
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0)
            sig_err = np.max(np.abs(sig - s["hist_sigma"]) / sig)
            ok &= bool(sig_err < 1e-10)
            print(f"  hist sigma vs Python formula: max rel err {sig_err:.1e}")
        # every pair lands in exactly one column: row sums equal C
        rowsum = s["H_er"].sum(axis=-1)
        ok_rows = np.allclose(rowsum, s["C_er"][:, :, :, None, :], rtol=1e-9, atol=1e-12)
        ok &= ok_rows
        print(f"  histogram row sums == C: {ok_rows}")
    elif meta["mode"] == "aniso_mhd":
        Tx, Wx, Cx = run_exhaustive_aniso(
            (s["rbins"], s["rbins_prll"], s["rbins_perp"]), pos, block, q, wts, per, L, P,
            np.cos(float(meta["theta_prll_max"])), np.cos(float(meta["theta_perp_min"])))
    else:
        Tx, Wx, Cx = run_exhaustive(s["rbins"], pos, block, q, wts, per, L,
                                    meta["mode"], P)
    print(f"== {fname}  mode={meta['mode']} weights={meta['weights']} R={R}")
    for e, name in enumerate(["E1(same block)", "E2(cross block)"]):
        rows = [("C", s["C_er"][e], Cx[e]), ("W", s["W_er"][e], Wx[e]),
                ("T", s["T_er"][e], Tx[e])]
        if hist:
            rows.append(("H", s["H_er"][e], Hx[e]))
        for label, est, ref in rows:
            # correlation T is a signed sum: an exact zero can come from cancellation
            r = population_check(label, est, ref, R,
                                 signed=(meta["mode"] == "correlation" and label == "T"))
            ok &= r["exact_bad"] == 0
            ok_stat &= r["stat_ok"]
            print(
                f"  {name:16s} {label}: bins={r['bins']:5d} max|z|={r['zmax']:5.2f}"
                f" Bonferroni p={r['p']:.2e} <z^2>/E={r['mz2']:4.2f}"
                f" median rel err={r['rel']:.2e}"
                f" exact-mismatch={r['exact_bad']} unsampled={r['unsampled']}"
                f" (mass {r['tail']:.1e})"
                f"{' signed-zero bins=%d' % r['signed_zero'] if r['signed_zero'] else ''}"
                f" {'OK' if r['stat_ok'] and r['exact_bad'] == 0 else 'FAIL'}")
    # every stored combined array (what a reader uses) equals the mean over replicates
    # of E1+E2
    comb_bad = [k for k in ("T", "W", "C", "H") if k in s and not np.allclose(
        s[k + "_er"].sum(axis=0).mean(axis=0), s[k], rtol=1e-12, atol=0)]
    ok_comb = not comb_bad
    ok &= ok_comb
    # normalised result: S_p = T/C (structure functions), T/W (correlation, Python's
    # cf/weight_sum_bin)
    corr = meta["mode"] == "correlation"
    dn, Dx = ("W", Wx) if corr else ("C", Cx)
    lab = f"T/{dn}"
    Tt, Dt = Tx.sum(axis=0), Dx.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        S_ref = Tt / Dt[:, None, None, :]
        S_est = s["T"] / s[dn][:, None, None, :]
    good = np.isfinite(S_ref) & np.isfinite(S_est)
    # every finite exact ratio needs a finite estimate (denominator > 0 where exact > 0)
    missing = np.isfinite(S_ref) & ~np.isfinite(S_est)
    if missing.any():
        print(f"  {lab}: {missing.sum()} finite exact ratios have no finite "
              f"estimate FAIL")
        ok_stat = False
    if not good.any():
        print(f"  {lab}: no populated bins (empty selection: exact C = 0 everywhere,"
              f" estimate C = {s['C'].sum():g})")
        zero = all(np.all(s[k] == 0) for k in s if k[0] in "TWCHn")
        print(f"  empty selection: every sum and count is zero: {zero}")
        ok &= bool(zero and np.all(Cx == 0))
        return ok, ok_stat
    print(f"  combined T, W, C{', H' if 'H' in s else ''} == replicate mean of E1+E2:"
          f" {ok_comb}{'' if ok_comb else ' FAIL (' + ','.join(comb_bad) + ')'}")
    if corr:
        # ratio of summed numerators and denominators, delete-one-replicate jackknife
        # with each replicate's (T, W) kept together; absolute errors, since the
        # correlation can cross zero
        Tr = s["T_er"].sum(axis=0)                  # (R, nw, nq, P, nb)
        Wr = s["W_er"].sum(axis=0)[:, :, None, None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            jk = (Tr.sum(0) - Tr) / (Wr.sum(0) - Wr)
            se = np.sqrt((R - 1) / R * ((jk - jk.mean(0)) ** 2).sum(0))
            z = (S_est - S_ref) / se
        g = good & (se > 0)
        absd = np.abs(S_est - S_ref)[g]
        print(f"  {lab} vs exact (Python cf/weight_sum_bin): jackknife max|z|="
              f"{np.abs(z[g]).max():.2f} rms z={np.sqrt((z[g] ** 2).mean()):.2f} over"
              f" {g.sum()} entries; max abs err {absd.max():.2e} (max |exact|"
              f" {np.abs(S_ref[g]).max():.2e}) (informational: entries share samples)")
    else:
        relS = np.abs(S_est[good] - S_ref[good]) / np.abs(S_ref[good])
        print(f"  {lab} vs exact: median rel err {np.median(relS):.2e}, "
              f"max {relS.max():.2e}"
              f" (orders 1..{P}; sampling error, informational)")
    return ok, ok_stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--zmax", type=float, default=5.0)
    ap.add_argument("--mask", type=int, default=0, help="problem/mask of the run")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.run_dir, "sf_*", "*.sf")))
    if not files:
        print("no .sf files found")
        return 1
    allok, allstat = True, True
    for f in files:
        o, st = compare(f, args.noise, args.zmax, args.mask)
        allok &= bool(o)
        allstat &= bool(st)
    if allok and allstat:
        print("OVERALL: PASS")
        return 0
    if allok:
        print("OVERALL: FAIL (sampling checks only; structural checks PASS)")
        return 3
    print("OVERALL: FAIL (structural)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
