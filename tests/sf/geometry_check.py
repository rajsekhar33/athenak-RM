#!/usr/bin/env python3
"""Separation-bin membership of one sf run against the exact enumeration.

usage: geometry_check.py RUNDIR   (RUNDIR/sf_g/*.sf, whole periodic mesh, no region)

The separation of every distinct cell pair follows from its integer cell offset after
the minimum image in whole cells, times the cell width domain_length / mesh_nx (the
convention in docs/structure_function_output.md). A bin is populated if some offset
falls in it. Each file must have C > 0 (summed over populations and replicates) in
every populated bin and exactly C = 0, T = 0 in every other bin. Prints PASS or FAIL
with the offending bins.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf_reference import read_sf  # noqa: E402


def populated(meta, edges):
    n = [int(v) for v in meta["mesh_nx"].split(",")]
    h = [float(v) / m for v, m in zip(meta["domain_length"].split(","), n)]
    per = [v == "1" for v in meta["periodic"].split(",")]
    axes = []
    for d in range(3):
        off = np.arange(-(n[d] - 1), n[d], dtype=np.int64)
        if per[d]:
            off = np.where(2 * np.abs(off) > n[d], off - np.sign(off) * n[d], off)
        axes.append(off.astype(np.float64) * h[d])
    dx, dy, dz = np.meshgrid(*axes, indexing="ij")
    r = np.sqrt(dx * dx + dy * dy + dz * dz).ravel()
    r = r[r > 0]
    b = np.searchsorted(edges, r, side="right") - 1
    nb = len(edges) - 1
    return set(int(v) for v in b[(b >= 0) & (b < nb)]), nb


def main():
    files = sorted(glob.glob(os.path.join(sys.argv[1], "sf_g", "*.sf")))
    if not files:
        print("no sf files: FAIL")
        return 1
    bad = []
    for f in files:
        meta, a = read_sf(f)
        want, nb = populated(meta, a["rbins"])
        C = a["C_er"].sum(axis=(0, 1))          # [nweight, nbin]
        T = a["T_er"].sum(axis=(0, 1))          # [nweight, nq, P, nbin]
        for b in range(nb):
            has = bool(np.all(C[:, b] > 0))
            empty = bool(np.all(C[:, b] == 0) and np.all(T[..., b] == 0))
            if (b in want and not has) or (b not in want and not empty):
                bad.append(f"{os.path.basename(f)} bin {b}: expected "
                           f"{'pairs' if b in want else 'none'}, C = {C[:, b].tolist()}")
    if bad:
        print("; ".join(bad[:4]) + f" ({len(bad)} bins): FAIL")
        return 1
    print(f"{len(files)} files, populated bins {sorted(want)} of {nb}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
