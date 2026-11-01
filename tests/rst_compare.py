#!/usr/bin/env python3
"""Typed comparison of two AthenaK restart files (double precision, uniform mesh).

usage: rst_compare.py A.rst B.rst [RTOL]   (exit 0 if equivalent within RTOL)

Layout (src/outputs/restart.cpp, WriteOutputFile), every field read with its own type:
  parameter text up to the line "<par_end>"
  int nmb_total, int root_level, RegionSize (9 Real), RegionIndcs mesh and MeshBlock
  (19 int each), Real time, Real dt, int ncycle
  LogicalLocation[nmb] (4 int32), float cost[nmb]
  uint64 marker[2], Real steps[2] (dtold, dt_last_completed), int state[2]
  if <turb_driving> with restart_forcing (default true): RNG_State (3 int64,
    int64 iv[32], int iset, 4 bytes padding, double gset) and int n_turb_updates_yet
  uint64 data_size
  then nmb_total records of data_size bytes in gid order, each
    [hydro u0 (nhydro + nscalars)] [mhd u0 (nmhd + nscalars)] [mhd x1f, x2f, x3f]
    [forcing (6)]
Supported: hydro and/or mhd (ideal or isothermal, passive scalars), optional turbulence
forcing, no mesh refinement, radiation, z4c or adm. Anything else is rejected rather
than parsed by guesswork.

Comparison:
  physical arrays per variable / face component, relative to that array's max |value|
  in A (an all-zero array must match exactly); every value must be finite;
  real-valued metadata (time, dt, dtold, dt_last_completed) finite and within RTOL;
  everything else (integer metadata, mesh geometry, locations, costs, marker, state,
  RNG state, data_size) bitwise identical.
"""
import struct
import sys

import numpy as np

RNG_FMT = "<3q32qi4xd"   # RNG_State, natural alignment (296 bytes)


def params(text):
    blocks, cur = {}, None
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if s.startswith("<") and s.endswith(">"):
            cur = s[1:-1].strip()
            blocks.setdefault(cur, {})
        elif "=" in s and cur is not None:
            k, v = s.split("=", 1)
            blocks[cur][k.strip()] = v.strip()
    return blocks


def layout(p):
    """cell-centred variable names and whether faces/forcing exist, from the params"""
    for b in ("radiation", "z4c", "adm"):
        if b in p:
            raise ValueError("unsupported restart layout: <%s>" % b)
    if p.get("mesh_refinement", {}).get("refinement", "none") != "none":
        raise ValueError("unsupported restart layout: mesh refinement")
    groups = []
    for fluid in ("hydro", "mhd"):
        if fluid in p:
            eos = p[fluid].get("eos")
            if eos not in ("ideal", "isothermal"):
                raise ValueError("unsupported %s eos %r" % (fluid, eos))
            n = (5 if eos == "ideal" else 4) + int(p[fluid].get("nscalars", "0"))
            groups.append((fluid, n))
    if not groups:
        raise ValueError("unsupported restart layout: neither <hydro> nor <mhd>")
    turb = "turb_driving" in p and \
        p["turb_driving"].get("restart_forcing", "true").lower() in ("true", "1")
    return groups, "mhd" in p, turb


def parse(path):
    raw = open(path, "rb").read()
    i = raw.index(b"<par_end>")
    p = params(raw[:i].decode("ascii", "replace"))
    groups, mhd, turb = layout(p)
    hdr = raw.index(b"\n", i) + 1   # file offset of the binary header
    body = raw[hdr:]
    off = 0
    meta = {}

    def take(name, fmt):
        nonlocal off
        v = struct.unpack_from(fmt, body, off)
        meta[name] = (fmt, body[off:off + struct.calcsize(fmt)], v, hdr + off)
        off += struct.calcsize(fmt)
        return v

    nmb, root = take("nmb_root", "<2i")
    take("mesh_size", "<9d")
    take("mesh_indcs", "<19i")
    mbi = take("mb_indcs", "<19i")
    take("time", "<d")
    take("dt", "<d")
    take("ncycle", "<i")
    take("lloc", "<%di" % (4 * nmb))
    take("cost", "<%df" % nmb)
    take("marker", "<2Q")
    take("steps", "<2d")
    take("state", "<2i")
    if turb:
        take("rng", RNG_FMT)
        take("n_turb_updates", "<i")
    (ds,) = take("data_size", "<Q")
    ng, nx1, nx2, nx3 = mbi[:4]
    n1 = nx1 + 2 * ng
    n2 = nx2 + 2 * ng if nx2 > 1 else 1
    n3 = nx3 + 2 * ng if nx3 > 1 else 1
    ncell = n1 * n2 * n3
    names = []   # (name, size)
    for fluid, n in groups:
        names += [("%s_u%d" % (fluid, v), ncell) for v in range(n)]
        if fluid == "mhd":
            names += [("mhd_x1f", n3 * n2 * (n1 + 1)), ("mhd_x2f", n3 * (n2 + 1) * n1),
                      ("mhd_x3f", (n3 + 1) * n2 * n1)]
    if turb:
        names += [("force%d" % v, ncell) for v in range(6)]
    nrec = sum(s for _, s in names)
    if ds != 8 * nrec:
        raise ValueError("%s: data_size %d != 8 x %d expected from the layout"
                         % (path, ds, nrec))
    if off + nmb * ds != len(body):
        raise ValueError("%s: file length %d != header %d + %d records x %d"
                         % (path, len(body), off, nmb, ds))
    data = np.frombuffer(body, dtype="<f8", offset=off).reshape(nmb, nrec)
    arrays, o = {}, 0
    for name, s in names:
        arrays[name] = data[:, o:o + s]
        o += s
    meta["data_offset"] = ("", b"", (), hdr + off)
    return meta, arrays


REAL_META = ("time", "dt", "steps")


def compare(a_path, b_path, rtol):
    ma, A = parse(a_path)
    mb, B = parse(b_path)
    ok, worst, msgs = True, 0.0, []
    if list(ma) != list(mb) or list(A) != list(B):
        return False, ["layouts differ: %s vs %s" % (list(A), list(B))], worst
    nreal = 0
    for k in ma:
        fa, ba, va, _ = ma[k]
        fb, bb, vb, _ = mb[k]
        if fa != fb:
            msgs.append("metadata %s has different sizes" % k)
            ok = False
        elif k in REAL_META:
            for xa, xb in zip(va, vb):
                good = (np.isfinite(xa) and np.isfinite(xb) and
                        abs(xa - xb) <= rtol * max(abs(xa), abs(xb)))
                nreal += xa != xb
                if not good:
                    msgs.append("metadata %s %r vs %r" % (k, xa, xb))
                    ok = False
        elif ba != bb:
            msgs.append("metadata %s differs (must be identical)" % k)
            ok = False
    if ok:
        msgs.append("metadata: typed fields match (%d real values differ within rtol)"
                    % nreal)
    for k in A:
        a, b = A[k], B[k]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            msgs.append("%s nonfinite" % k)
            ok = False
            continue
        scale = np.max(np.abs(a))
        if scale == 0.0:
            if np.any(b != 0.0):
                msgs.append("%s: zero in A, nonzero in B" % k)
                ok = False
            continue
        d = float(np.max(np.abs(a - b)) / scale)
        worst = max(worst, d)
        if not d <= rtol:
            msgs.append("%s max diff / scale %.1e" % (k, d))
            ok = False
    msgs.append("%d arrays (%s), max diff / scale %.1e" % (len(A), ",".join(A), worst))
    return ok, msgs, worst


if __name__ == "__main__":
    try:
        ok, msgs, _ = compare(sys.argv[1], sys.argv[2],
                              float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10)
    except ValueError as e:
        ok, msgs = False, [str(e)]
    print("; ".join(msgs))
    print("RST COMPARE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
