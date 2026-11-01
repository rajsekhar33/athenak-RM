#!/bin/bash
# Restart test for the sf and spec outputs: a continuous 4-cycle run and a run restarted
# at cycle 2 must write identical arrays (draws are keyed by output number, not RNG
# state). Use a serial or single-rank build so floating-point sums are reproducible.
# usage: tests/sf/restart_test.sh ATHENA OUTDIR [LAUNCH_PREFIX]
set -u
BIN=$1; OUT=$2; PRE=${3:-}
SRC=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$OUT"; mkdir -p "$OUT/cont" "$OUT/rest"
python3 - "$OUT" "$SRC" <<'PY'
import re, sys
out, src = sys.argv[1], sys.argv[2]
s = open(src + "/inputs/tests/sf_unit_test.athinput").read()
s = s.replace("nlim       = 0", "nlim       = 4")
s = re.sub(r"dt             = 1.0", "dcycle         = 1", s)
s = s.replace("ndraw_block    = 400000", "ndraw_block    = 40000")
s = s.replace("nanchor_global = 160000", "nanchor_global = 16000")
s += """
<output3>
file_type  = spec
id         = d
mode       = scalar
variables  = hydro_w_d
weights    = ones, hydro_w_d
nbin       = 20
dcycle     = 1

<output4>
file_type  = rst
dcycle     = 2
"""
open(out + "/cont/in.athinput", "w").write(s)
open(out + "/rest/in.athinput", "w").write(s.replace("nlim       = 4", "nlim       = 2"))
PY
(cd "$OUT/cont" && $PRE "$BIN" -i in.athinput > log 2>&1) || { echo "continuous run failed"; exit 1; }
(cd "$OUT/rest" && $PRE "$BIN" -i in.athinput > log1 2>&1 &&
 $PRE "$BIN" -r rst/sfut.00001.rst time/nlim=4 > log2 2>&1) || { echo "restart run failed"; exit 1; }
python3 - "$OUT" "$SRC" <<'PY'
import glob, os, sys
import numpy as np
out, src = sys.argv[1], sys.argv[2]
sys.path.insert(0, src + "/tests/sf")
sys.path.insert(0, src + "/tests/spec")
from sf_reference import read_sf
from spec_reference import read_spec
ok, n = True, 0
for f in sorted(glob.glob(out + "/cont/sf_*/*.sf") + glob.glob(out + "/cont/spec_*/*.spec")):
    g = f.replace("/cont/", "/rest/")
    if not os.path.exists(g):
        print("missing after restart:", os.path.relpath(g, out)); ok = False; continue
    rd = read_sf if f.endswith(".sf") else read_spec
    ma, a = rd(f)
    mb, b = rd(g)
    same = all(np.array_equal(a[k], b[k]) for k in a) and ma["cycle"] == mb["cycle"]
    ok &= same
    n += 1
    if not same:
        print("differs:", os.path.relpath(f, out + "/cont"))
# 3 outputs (sf scalar, sf long_trans, spec) x 6 files (cycles 0-4 plus the final dump)
if n != 18:
    print("expected 18 files, compared %d" % n)
    ok = False
print("RESTART: %s (%d files compared)" % ("PASS" if ok else "FAIL", n))
sys.exit(0 if ok else 1)
PY
