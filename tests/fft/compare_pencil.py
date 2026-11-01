#!/usr/bin/env python3
"""Check the distributed FFT (src/fft/pencil_fft.hpp) against numpy.
Reads RUN_DIR/fft_test.bin written by src/pgen/sf_unit_test.cpp (problem/fft_test = true)
and RUN_DIR/in.athinput, rebuilds the density, crops it and compares every coefficient
with numpy.fft.rfftn (x is the half-length axis).
Usage: python3 -I tests/fft/compare_pencil.py RUN_DIR [--tol 1e-12]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sf"))
from sf_reference import build_fields  # noqa: E402


def read_mesh(path):
    """<mesh> and <meshblock> keys of an AthenaK input file."""
    blocks, cur = {}, None
    for line in open(path):
        line = line.split("#")[0].strip()
        if line.startswith("<"):
            cur = line.strip("<>")
            blocks[cur] = {}
        elif "=" in line and cur:
            k, v = line.split("=", 1)
            blocks[cur][k.strip()] = v.strip()
    m, b = blocks["mesh"], blocks["meshblock"]
    nx = [int(m["nx%d" % d]) for d in (1, 2, 3)]
    lo = [float(m["x%dmin" % d]) for d in (1, 2, 3)]
    hi = [float(m["x%dmax" % d]) for d in (1, 2, 3)]
    nb = [int(b["nx%d" % d]) for d in (1, 2, 3)]
    return {"mesh_nx": ",".join(map(str, nx)), "mesh_min": ",".join(map(str, lo)),
            "domain_length": ",".join(str(h - lw) for lw, h in zip(lo, hi)),
            "meshblock_nx": ",".join(map(str, nb))}, nx, blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tol", type=float, default=1e-12)
    a = ap.parse_args()
    meta, nx, blocks = read_mesh(os.path.join(a.run, "in.athinput"))
    noise_amp = float(blocks.get("problem", {}).get("noise", "1.0"))
    _, _, f = build_fields(meta, noise_amp)
    d = f["hydro_w_d"].reshape(nx[2], nx[1], nx[0])
    raw = open(os.path.join(a.run, "fft_test.bin"), "rb").read()
    hdr = np.frombuffer(raw, dtype=np.int32, count=6)
    n, lo = hdr[:3], hdr[3:]
    rec = np.frombuffer(raw, dtype=np.float64, offset=24).reshape(-1, 5)
    crop = d[lo[2]:lo[2] + n[2], lo[1]:lo[1] + n[1], lo[0]:lo[0] + n[0]]
    ref = np.fft.rfftn(crop)                      # (nz, ny, nx/2+1)
    gx, gy, gz = (rec[:, i].astype(int) for i in range(3))
    got = rec[:, 3] + 1j * rec[:, 4]
    ok = True
    if len(got) != ref.size:
        print("coefficient count %d, expected %d" % (len(got), ref.size))
        ok = False
    seen = np.zeros(ref.shape, dtype=int)
    np.add.at(seen, (gz, gy, gx), 1)
    if not np.all(seen == 1):
        print("coefficients missing or duplicated: %d" % np.sum(seen != 1))
        ok = False
    err = np.max(np.abs(got - ref[gz, gy, gx])) / np.max(np.abs(ref))
    print("shape (x,y,z) = %s crop lo = %s: max |F - numpy| / max |F| = %.2e"
          % (tuple(n), tuple(lo), err))
    ok &= bool(err <= a.tol)
    print("PENCIL FFT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
