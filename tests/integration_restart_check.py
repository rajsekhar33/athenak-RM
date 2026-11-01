#!/usr/bin/env python3
"""Changed-rank restart check (tests/integration_checks.sh [2]).

usage: integration_restart_check.py CONT_DIR RESTART_DIR SRC RESTART_FILE
  CONT_DIR: continuous run; RESTART_DIR: run restarted from RESTART_FILE (written by
  the continuous run) on a different rank count. Exit 0 on PASS."""
import glob
import os
import struct
import sys
import numpy as np
from scipy import stats
cont, rest, src, rst_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, src + "/tests/sf")
sys.path.insert(0, src + "/tests/spec")
sys.path.insert(0, src + "/tests")
from sf_reference import read_sf  # noqa: E402
from spec_reference import read_spec  # noqa: E402
ok = True
from rst_compare import compare  # noqa: E402
# evolved state: typed comparison of the double-precision restart files written after
# the restart (per variable, relative to its own max |value|; metadata separately)
rst = sorted(glob.glob(rest + "/rst/*.rst"))
for f in rst:
    try:
        good, msgs, _ = compare(cont + "/rst/" + os.path.basename(f), f, 1e-10)
    except (ValueError, OSError, struct.error) as e:
        good, msgs = False, [str(e)]
    print("  state %s: %s %s" % (os.path.basename(f), "; ".join(msgs),
                                 "" if good else "FAIL"))
    ok &= good
if not rst:
    print("  no restart files written after the restart FAIL")
    ok = False
# diagnostic inventory, derived from the restart file and not from the files found:
# every sf/spec block in the restart's parameters must write exactly the continuous
# run's files numbered from that block's stored file_number on (a missing leading
# file, a missing later file or an extra file all fail), and no other sf/spec blocks
# appear
from rst_compare import params  # noqa: E402
rp = params(open(rst_path, "rb").read().split(b"<par_end>")[0]
            .decode("ascii", "replace"))
expected = {}
for blk, kv in rp.items():
    if blk.startswith("output") and kv.get("file_type") in ("sf", "spec"):
        expected["%s_%s" % (kv["file_type"], kv.get("id", blk))] = \
            (kv["file_type"], int(kv["file_number"]))
VOLATILE = ("time_", "timing", "memory_", "nranks", "pencils", "attempts_", "accepted_",
            "bin_scratch", "bin_global")
inv_ok = bool(expected)
if not expected:
    print("  no sf/spec blocks in the restart parameters FAIL")
found = sorted(set(os.path.relpath(os.path.dirname(f), d0) for d0 in (cont, rest)
                   for f in glob.glob(d0 + "/s[fp]*_*/*.s[fp]*")))
if found != sorted(expected):
    print("  output blocks %s differ from the restart's sf/spec blocks %s FAIL"
          % (found, sorted(expected)))
    inv_ok = False
for d, (kind, first) in sorted(expected.items()):
    def num(f):
        return int(f.split(".")[-2])
    rn = sorted(os.path.basename(f) for f in glob.glob(rest + "/%s/*.%s" % (d, kind)))
    cn = sorted(os.path.basename(f) for f in glob.glob(cont + "/%s/*.%s" % (d, kind))
                if num(f) >= first)
    if not cn or num(cn[0]) != first or rn != cn:
        print("  %s: expected files from number %d; continuous %s, restart %s FAIL"
              % (d, first, cn, rn))
        inv_ok = False
print("  inventory: %d sf/spec blocks, first file numbers %s from the restart file %s"
      % (len(expected), {d: v[1] for d, v in expected.items()},
         "" if inv_ok else "FAIL"))
ok &= inv_ok


def schema(meta):
    return {k: v for k, v in meta.items() if not k.startswith(VOLATILE)
            and k not in ("time", "cycle", "file_number", "header offset")}


nchk = 0
for kind, rd in (("spec", read_spec), ("sf", read_sf)):
    for f in sorted(glob.glob(rest + "/%s_*/*.%s" % (kind, kind))):
        g = cont + "/" + os.path.relpath(f, rest)
        if not os.path.exists(g):
            continue
        nchk += 1
        ma, a = rd(g)
        mb, b = rd(f)
        name = os.path.relpath(f, rest)
        # time and cycle compared explicitly (excluded from the schema)
        if ma["cycle"] != mb["cycle"] or ma["file_number"] != mb["file_number"] or \
                not abs(float(ma["time"]) - float(mb["time"])) <= \
                1e-10 * max(abs(float(ma["time"])), 1e-300):
            print("  %s: time/cycle/file_number %s/%s/%s vs %s/%s/%s FAIL" % (name,
                  ma["time"], ma["cycle"], ma["file_number"], mb["time"], mb["cycle"],
                  mb["file_number"]))
            ok = False
        if schema(ma) != schema(mb) or set(a) != set(b):
            print("  %s: header schema or payload arrays differ FAIL" % name)
            ok = False
            continue
        for k in a:
            if a[k].dtype.kind == "f" and not (np.all(np.isfinite(a[k])) and
                                               np.all(np.isfinite(b[k]))):
                print("  %s %s nonfinite FAIL" % (name, k))
                ok = False
        if kind == "spec":
            for k in a:
                sc = np.abs(a[k]).max()
                d = np.max(np.abs(a[k] - b[k])) / sc if sc > 0 else np.max(np.abs(b[k]))
                if not d <= 1e-9:
                    print("  %s %s differs %.1e FAIL" % (name, k, d))
                    ok = False
            continue
        # sf: scalar E1 has geometry-only membership: identical counts, sums to
        # round-off; aniso_mhd E1 and every E2 are tested statistically
        pops = [1]
        if ma["mode"] == "aniso_mhd":
            pops = [0, 1]
            nd = np.sum(a["n_acc_er"][0] != b["n_acc_er"][0])
            print("  %s: aniso E1 count entries differing: %d of %d"
                  % (name, nd, a["n_acc_er"][0].size))
        else:
            if not np.array_equal(a["n_acc_er"][0], b["n_acc_er"][0]):
                print("  %s E1 counts differ FAIL" % name)
                ok = False
            for k in ("T_er", "C_er", "W_er"):
                x, y = a[k][0], b[k][0]
                sc = np.where(np.abs(x) > 0, np.abs(x), 1.0)
                d = np.max(np.abs(x - y) / sc)
                if not d <= 1e-9:
                    print("  %s E1 %s differs %.1e FAIL" % (name, k, d))
                    ok = False
        R = a["C_er"].shape[1]
        for pop in pops:
            for k in ("C_er", "T_er"):
                ea, eb = a[k][pop], b[k][pop]
                ma_, mb_ = ea.mean(0), eb.mean(0)
                se = np.sqrt(ea.var(0, ddof=1) / R + eb.var(0, ddof=1) / R)
                # entries with no replicate spread in either run are deterministic and
                # must agree to round-off; the rest are scored by z
                zero = se == 0
                det = (np.abs(ma_ - mb_)[zero]
                       > 1e-12 * np.maximum(np.abs(ma_[zero]), 1e-300))
                if det.any():
                    print("  %s E%d %s: %d zero-spread entries differ FAIL"
                          % (name, pop + 1, k, det.sum()))
                    ok = False
                z = np.abs(ma_[~zero] - mb_[~zero]) / se[~zero]
                if z.size:
                    p = min(1.0, 2.0 * stats.t.sf(z.max(), 2 * R - 2) * z.size)
                    if not p >= 1e-3:
                        print("  %s E%d %s max|z| %.2f p %.1e FAIL" % (name, pop + 1, k,
                              z.max(), p))
                        ok = False
print("  %d restart-run diagnostic files compared with the continuous run" % nchk)
ok &= nchk > 0
print("[2] restart on a different rank count: state to round-off, spec to round-off, "
      "scalar sf E1 identical counts, aniso E1 and all E2 statistically consistent:",
      "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
