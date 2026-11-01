#!/bin/bash
# Sparse phase-indicator weights (problem/mask: the first passive scalar is a 0/1 mask,
# used as the sf weight hydro_w_s / mhd_w_s), against the exact all-pairs reference
# (tests/sf/sf_reference.py), on a 32^3 mesh with 8^3 MeshBlocks (16^3 for the weight
# and aniso_mhd cases):
#   mask 1: two disconnected spheres, one across the periodic boundary (2.4% of cells)
#   mask 2: one small sphere on a MeshBlock corner (0.17% of cells, 8 blocks)
#   mask 3: empty selection: every sum is 0; histograms stay on (scale fallback 1)
#   mask 4: one small sphere across two blocks of rank 0 (other ranks: no support)
# Each mask runs with the default sampler (draws over region cells) and with
# sparse = true (support cells only: E2 anchors/samples from the support, E1 anchors
# from a 50/50 mixture of support and admissible anchors; Python sparse_data).
# Further sparse cases:
#   distinct supports (weights hydro_w_s, hydro_w_vx) and ones mixed with an indicator,
#   with the header support count checked against the union of positive weights;
#   aniso_mhd with weight mhd_w_s;
#   a support that vanishes and reappears (problem/mask_toggle) across outputs, with a
#   restart in between: counts per output, all-zero outputs while empty, the first
#   output against the exact reference, and the restarted run equal to the continuous
#   one.
# One statistical exception: the default sampler on the rare masks 2 and 4 (56 cells)
# holds a few valid samples per replicate, so the replicate spread underestimates its
# error and a single file can fail a z-test or leave a required bin unsampled. There
# sf_reference.py exit 3 (sampling checks only) is
# accepted; structural failures (exit 1: nonfinite values, accounting, exact
# entries, checker exceptions) still fail (tests/sf/sparse_checker_selftest.sh). A
# seed scan of that case is consistent with the exact raw sums.
# usage: tests/sf/run_sparse.sh ATHENA MPIRUN NP OUTDIR   (PROBLEM=sf_unit_test)
set -u
BIN=$1; M=$2; NP=$3; OUT=$4
SRC=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
IN=$SRC/inputs/tests/sf_sparse_test.athinput
status=0
fail() { echo "$1: FAIL"; status=1; }

# expected support: cells where some weight > 0 (union), from the reference fields
support() {   # mask weights(comma) [mesh n, default 32]
  python3 -I - "$SRC/tests/sf" "$1" "$2" "${3:-32}" <<'PY'
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import sf_reference as sr
n = int(sys.argv[4])
meta = {"mesh_nx": f"{n},{n},{n}", "mesh_min": "0,0,0", "domain_length": "1,1,1",
        "meshblock_nx": "8,8,8", "test_mask": sys.argv[2]}
_, _, f = sr.build_fields(meta, 1.0)
u = np.zeros(n ** 3, dtype=bool)
for w in sys.argv[3].split(","):
    u |= np.ones(n ** 3, dtype=bool) if w == "ones" else f[w] > 0
print(int(u.sum()))
PY
}
ranks_ok() {
  [ "$(grep -ac 'Number of parallel ranks' "$1")" -eq 1 ] &&
    grep -aq "Number of parallel ranks = $NP\$" "$1"
}
hdr() { grep -ah "^$2=" "$1" | head -1 | cut -d= -f2; }

# ---- 1. masks x samplers, scalar (histograms) and long_trans outputs ----
for case in 1 2 3 4 1s 2s 3s 4s; do
  mask=${case%s}
  sp=false; [ "$case" != "$mask" ] && sp=true
  d=$OUT/mask$case
  rm -rf "$d"; mkdir -p "$d"; cp "$IN" "$d/in.athinput"
  (cd "$d" && $M -np "$NP" --oversubscribe "$BIN" -i in.athinput problem/mask=$mask \
     output1/sparse=$sp output2/sparse=$sp > log 2>&1)
  if [ $? -ne 0 ] || ! ranks_ok "$d/log" || [ "$(ls "$d"/sf_*/*.sf 2>/dev/null | wc -l)" -ne 4 ]; then
    fail "mask $mask sparse=$sp: run failed, wrong rank count or missing outputs"; continue
  fi
  python3 -I "$SRC/tests/sf/sf_reference.py" "$d" --mask $mask > "$d/check.txt" 2>&1
  rc=$?
  nsel=$(support $mask hydro_w_s)
  f0=$(ls "$d"/sf_scalar/*.00000.sf)
  info="$nsel cells selected"
  if [ "$sp" = true ]; then
    ns=$(hdr "$f0" ncell_support); nr=$(hdr "$f0" nrank_support)
    info="$info; header ncell_support=$ns nrank_support=$nr"
    [ "$ns" = "$nsel" ] || fail "mask $mask: header support $ns != $nsel"
    if [ $mask -eq 4 ] && [ "$NP" -gt 1 ] && [ "$nr" != 1 ]; then
      fail "mask 4: expected support on exactly 1 of $NP ranks, header says $nr"
    fi
  fi
  echo "mask $mask sparse=$sp ($info): $(tail -1 "$d/check.txt")"
  if [ "$case" = 2 ] || [ "$case" = 4 ]; then
    { [ $rc -eq 0 ] || [ $rc -eq 3 ]; } &&
      echo "  (default sampler on a rare mask: sampling checks informational, structural checks gate)" ||
      { fail "mask $mask default: structural failure (exit $rc)"; grep -a FAIL "$d/check.txt" | head -5; }
  elif [ $rc -ne 0 ]; then
    fail "mask $mask sparse=$sp (exit $rc)"; grep -a FAIL "$d/check.txt" | head -5
  fi
done

# ---- 2. distinct supports; ones mixed with an indicator (sparse; 16^3, 8 blocks, so
#      the exact reference over the near-dense union stays cheap). With ones the
#      support is the whole block, so the support branch of E1 no longer concentrates
#      draws on the indicator: at the default 200000 draws the indicator's largest
#      in-block bin (5 exact pairs) was hit in 1 of 8 replicates (t-test meaningless).
#      E1 is nearly free, so these cases use 4000000 draws per block. ----
for wl in "hydro_w_s,hydro_w_vx" "ones,hydro_w_s"; do
  d=$OUT/weights_${wl//,/_}
  rm -rf "$d"; mkdir -p "$d"; cp "$IN" "$d/in.athinput"
  (cd "$d" && $M -np "$NP" --oversubscribe "$BIN" -i in.athinput problem/mask=1 \
     output1/sparse=true output2/sparse=true output1/weights="$wl" \
     output2/weights="$wl" mesh/nx1=16 mesh/nx2=16 mesh/nx3=16 \
     output1/ndraw_block=4000000 output2/ndraw_block=4000000 > log 2>&1)
  if [ $? -ne 0 ] || ! ranks_ok "$d/log"; then fail "weights $wl: run failed"; continue; fi
  python3 -I "$SRC/tests/sf/sf_reference.py" "$d" --mask 1 > "$d/check.txt" 2>&1
  rc=$?
  exp=$(support 1 "$wl" 16); ns=$(hdr "$(ls "$d"/sf_scalar/*.00000.sf)" ncell_support)
  echo "weights $wl sparse (union of supports $exp, header $ns): $(tail -1 "$d/check.txt")"
  [ "$ns" = "$exp" ] || fail "weights $wl: header support $ns != union $exp"
  [ $rc -eq 0 ] || { fail "weights $wl (exit $rc)"; grep -a FAIL "$d/check.txt" | head -5; }
done

# ---- 3. aniso_mhd with a sparse indicator (16^3, 8 blocks) ----
d=$OUT/aniso_sparse
rm -rf "$d"; mkdir -p "$d"
python3 - "$SRC/inputs/tests/sf_aniso_test.athinput" "$d/in.athinput" <<'PY'
import sys
s = open(sys.argv[1]).read()
for a, b in (("rsolver     = hlle", "rsolver     = hlle\nnscalars    = 1"),
             ("constant  = false", "constant  = false\nmask      = 1"),
             ("weights        = ones, mhd_w_d", "weights        = mhd_w_s\nsparse         = true"),
             ("max_order      = 4", "max_order      = 3")):
    assert a in s, a
    s = s.replace(a, b, 1)
open(sys.argv[2], "w").write(s)
PY
(cd "$d" && $M -np "$NP" --oversubscribe "$BIN" -i in.athinput > log 2>&1)
if [ $? -ne 0 ] || ! ranks_ok "$d/log"; then
  fail "aniso_mhd sparse: run failed"
else
  python3 -I "$SRC/tests/sf/sf_reference.py" "$d" --mask 1 > "$d/check.txt" 2>&1
  rc=$?
  ns=$(hdr "$(ls "$d"/sf_*/*.00000.sf)" ncell_support)
  exp=$(support 1 hydro_w_s 16)
  echo "aniso_mhd sparse (header support $ns, expected $exp): $(tail -1 "$d/check.txt")"
  [ "$ns" = "$exp" ] || fail "aniso_mhd: header support $ns != $exp"
  [ $rc -eq 0 ] || { fail "aniso_mhd sparse (exit $rc)"; grep -a FAIL "$d/check.txt" | head -5; }
fi

# ---- 4. support vanishing and reappearing, with a restart ----
d=$OUT/toggle
rm -rf "$d"; mkdir -p "$d/cont" "$d/rest"
python3 - "$IN" "$d/in.athinput" <<'PY'
import sys
s = open(sys.argv[1]).read()
s = s.replace("nlim       = 0", "nlim       = 8")
s = s.replace("mask      = 1", "mask      = 1\nmask_toggle = 2\nuser_work_in_loop = true")
s = s.replace("sparse         = false", "sparse         = true")
s = s.replace("dt             = 1.0", "dcycle         = 1")
s += "\n<output3>\nfile_type = rst\ndcycle    = 3\n"
open(sys.argv[2], "w").write(s)
PY
(cd "$d/cont" && $M -np "$NP" --oversubscribe "$BIN" -i ../in.athinput > log 2>&1) &&
  ranks_ok "$d/cont/log" || fail "toggle: continuous run failed"
rst=$(ls "$d"/cont/rst/*.00001.rst 2>/dev/null | head -1)
if [ -n "$rst" ]; then
  (cd "$d/rest" && $M -np "$NP" --oversubscribe "$BIN" -r "$rst" > log 2>&1) &&
    ranks_ok "$d/rest/log" || fail "toggle: restart failed"
else
  fail "toggle: no restart file"
fi
mkdir -p "$d/first/sf_scalar" "$d/first/sf_longtrans"
cp "$d"/cont/sf_scalar/*.00000.sf "$d/first/sf_scalar/" 2>/dev/null
cp "$d"/cont/sf_longtrans/*.00000.sf "$d/first/sf_longtrans/" 2>/dev/null
python3 -I "$SRC/tests/sf/sf_reference.py" "$d/first" --mask 1 > "$d/first/check.txt" 2>&1 ||
  fail "toggle: first output vs exact reference"
python3 -I - "$d" "$SRC/tests/sf" "$(support 1 hydro_w_s)" <<'PY' || status=1
import glob, os, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from sf_reference import read_sf
d, nsel = sys.argv[1], sys.argv[3]
ok = True
for sub in ("sf_scalar", "sf_longtrans"):
    cont = sorted(glob.glob(f"{d}/cont/{sub}/*.sf"))
    pat = []
    for f in cont:
        m, s = read_sf(f)
        ns = m["ncell_support"]
        pat.append(ns)
        if ns not in ("0", nsel):
            print(f"  {os.path.basename(f)}: ncell_support {ns} not 0 or {nsel} FAIL"); ok = False
        fin = all(np.all(np.isfinite(v)) for v in s.values() if v.dtype.kind == "f")
        zero = all(np.all(s[k] == 0) for k in s if k[0] in "TWCHn")
        if not fin or (ns == "0") != zero:
            print(f"  {os.path.basename(f)}: finite {fin}, all-zero {zero} with support {ns} FAIL")
            ok = False
    s_ = "".join("1" if p != "0" else "0" for p in pat)
    if "10" not in s_ or "01" not in s_:
        print(f"  {sub}: support pattern {s_} does not vanish and reappear FAIL"); ok = False
    # the restarted run reproduces the continuous run (same rank count)
    rest = sorted(glob.glob(f"{d}/rest/{sub}/*.sf"))
    if not rest:
        print(f"  {sub}: no outputs after the restart FAIL"); ok = False
    for f in rest:
        g = f"{d}/cont/{sub}/{os.path.basename(f)}"
        ma, a = read_sf(g)
        mb, b = read_sf(f)
        same = ma["ncell_support"] == mb["ncell_support"] and ma["cycle"] == mb["cycle"]
        for k in a:
            if a[k].dtype.kind == "f":
                sc = np.where(np.abs(a[k]) > 0, np.abs(a[k]), 1.0)
                same &= bool(np.all(np.abs(a[k] - b[k]) <= 1e-12 * sc))
            else:
                same &= bool(np.array_equal(a[k], b[k]))
        if not same:
            print(f"  {sub}/{os.path.basename(f)}: restarted output differs FAIL"); ok = False
    print(f"  {sub}: support per output {pat}; {len(rest)} outputs after the restart"
          f" compared {'OK' if ok else 'FAIL'}")
print("toggle:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
# the full checkpoint-anchored sequence must exist (continuous 0..nlim+1, restarted
# from the checkpoint onwards), so a missing file cannot shrink the comparison above
python3 -I "$SRC/tests/sf/sf_inventory.py" toggle "$d" || fail "toggle: incomplete output inventory"
echo "toggle first output vs exact: $(tail -1 "$d/first/check.txt")"
echo "SPARSE: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
