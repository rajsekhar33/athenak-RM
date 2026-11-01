#!/bin/bash
# Validation suite for the sf output (exhaustive references, tests/sf/sf_reference.py).
# Usage: tests/sf/run_suite.sh ATHENA_BINARY MPIRUN NRANKS OUTDIR
# Each case overrides inputs/tests/sf_unit_test.athinput on the command line.
BIN=$1; MPIRUN=$2; NP=$3; OUT=$4
SRC=$(cd "$(dirname "$0")/../.." && pwd)
IN=$SRC/inputs/tests/sf_unit_test.athinput
mkdir -p "$OUT"
status=0

run_case() {   # name, then athena overrides (INPUT env var selects another input file)
  local name=$1; shift
  local d="$OUT/$name"; rm -rf "$d"; mkdir -p "$d"
  (cd "$d" && $MPIRUN -np "$NP" --oversubscribe "$BIN" -i "${INPUT:-$IN}" "$@" > log 2>&1)
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[$name] athena exit $rc"; tail -3 "$d/log"; status=1; return; fi
  if [ "$(grep -ac 'Number of parallel ranks' "$d/log")" -ne 1 ] ||
     ! grep -aq "Number of parallel ranks = $NP\$" "$d/log"; then
    echo "[$name] binary did not run on $NP MPI ranks (serial build or wrong mpirun): FAIL"
    status=1; return
  fi
  python3 -I "$SRC/tests/sf/sf_reference.py" "$d" > "$d/check.txt" 2>&1
  local r=$?
  echo "[$name] $(grep OVERALL "$d/check.txt") $(grep -c FAIL "$d/check.txt") failing lines"
  [ $r -ne 0 ] && status=1
}

# smaller E2 budgets keep the suite quick; E1 budgets stay large
B="output1/nanchor_global=40000 output2/nanchor_global=40000"
run_case base $B
# region restriction (both outputs): keys must exist in the input file to be set, so
# write a copy with the bounds added (one key per line)
REG=$OUT/region_input.athinput
awk '{print} /^<output1>/{print "x1_min = 0.2\nx1_max = 0.7\nx3_min = 0.3"}
     /^<output2>/{print "x2_min = 0.1\nx2_max = 0.55"}' "$IN" > "$REG"
INPUT=$REG run_case region $B
# non-periodic x2 (outflow)
run_case outflow_x2 $B mesh/ix2_bc=outflow mesh/ox2_bc=outflow
# 2D mesh, 4x4 blocks of 16x16
run_case twod $B mesh/nx1=64 mesh/nx2=64 mesh/nx3=1 meshblock/nx1=16 meshblock/nx2=16 \
  meshblock/nx3=1
# one MeshBlock spanning the periodic box: E1 needs the minimum image, E2 is empty
if [ "$NP" -eq 1 ]; then
  run_case oneblock $B meshblock/nx1=16 meshblock/nx2=16 meshblock/nx3=16
fi
# non-cubic blocks: many lags have no admissible anchor (A(l)=0 rejections)
run_case noncubic $B meshblock/nx1=8 meshblock/nx2=4 meshblock/nx3=4

# increment histograms (scalar with a density weight; signed long_trans), also with a
# region, where sigma comes from the region's cells only
HIN=$SRC/inputs/tests/sf_histogram_test.athinput
INPUT=$HIN run_case histogram $B
REGH=$OUT/histogram_region_input.athinput
awk '{print} /^<output1>/{print "x1_min = 0.2\nx1_max = 0.7\nx3_min = 0.3"}
     /^<output2>/{print "x2_min = 0.1\nx2_max = 0.55"}' "$HIN" > "$REGH"
INPUT=$REGH run_case histogram_region $B

# aniso_mhd (MHD fields; three bin families)
INPUT=$SRC/inputs/tests/sf_aniso_test.athinput run_case aniso_mhd output1/nanchor_global=40000

# kernel = global draws the same pairs as the default team kernel: counts exact, sums
# to round-off (histogram input, so H is compared too)
GLB=$OUT/global_kernel_input.athinput
awk '{print} /^<output[12]>/{print "kernel = global"}' "$HIN" > "$GLB"
INPUT=$GLB run_case global_kernel $B
for f in "$OUT"/histogram/sf_*/*.sf; do
  g="$OUT/global_kernel/${f#$OUT/histogram/}"
  r=$(python3 -I "$SRC/tests/sf/compare_sf.py" "$f" "$g" | tail -1)
  echo "[team vs global] ${f#$OUT/} $r"
  [[ "$r" == *PASS* ]] || status=1
done

# minimal quotas (one anchor and one sample per replicate and rank): a single file has
# almost no E2 pairs, so unbiasedness and stream independence are checked by a seed scan
MPIRUN=$MPIRUN "$SRC/tests/sf/seed_scan.sh" "$BIN" "$NP" 40 "$OUT/minquota_scan" \
  output1/nsample_global=$((4*NP)) output1/nanchor_global=$((4*NP)) > "$OUT/minquota.txt" 2>&1
r=$?
echo "[minquota seed scan] $(grep OVERALL "$OUT/minquota.txt")"
[ $r -ne 0 ] && status=1

# region restriction across seeds (estimator bias with a partial-block region)
REGS=$OUT/region_scan_input.athinput
awk '{print} /^<output1>/{print "x1_min = 0.2\nx1_max = 0.7\nx3_min = 0.3"}' \
  "$SRC/inputs/tests/sf_seedscan.athinput" > "$REGS"
INPUT=$REGS MPIRUN=$MPIRUN "$SRC/tests/sf/seed_scan.sh" "$BIN" "$NP" 40 \
  "$OUT/region_scan" > "$OUT/region_scan.txt" 2>&1
r=$?
echo "[region seed scan] $(grep OVERALL "$OUT/region_scan.txt")"
[ $r -ne 0 ] && status=1

# replicate normalisation: R = 1 and 2 must also be unbiased (R = 4, 8 covered above)
for r in 1 2; do
  MPIRUN=$MPIRUN SEED0=$((1000*r)) "$SRC/tests/sf/seed_scan.sh" "$BIN" "$NP" 40 \
    "$OUT/rep${r}_scan" output1/nreplicate=$r > "$OUT/rep${r}_scan.txt" 2>&1
  r2=$?
  echo "[R=$r seed scan] $(grep OVERALL "$OUT/rep${r}_scan.txt")"
  [ $r2 -ne 0 ] && status=1
done

# constant fields: every increment is 0. nvcc (and nvhpc on CPU) contract w2*q2 - w1*q1 into an
# fma, which returns the rounding error of w1*q1 (~1e-16 relative) instead of exactly 0
# when a weight multiplies the field, so check the mean |increment| to roundoff.
d="$OUT/constant"; rm -rf "$d"; mkdir -p "$d"
(cd "$d" && $MPIRUN -np "$NP" --oversubscribe "$BIN" -i "$IN" problem/constant=true $B \
   > log 2>&1) || { echo "[constant] athena failed"; status=1; }
python3 -I - "$d" "$SRC/tests/sf" <<'PYEOF'
import glob, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
import sf_reference as sr
files = sorted(glob.glob(sys.argv[1] + "/sf_*/*.sf"))
ok = len(files) == 4          # two sf outputs, initial and final file each
if not ok:
    print(f"  expected 4 .sf files, found {len(files)}")
for f in files:
    _, s = sr.read_sf(f)
    with np.errstate(invalid="ignore", divide="ignore"):
        s1 = s["T"][:, :, 0, :] / s["C"][:, None, :]          # mean |w q increment|
    s1 = np.nan_to_num(s1)
    ok &= bool(np.abs(s1).max() < 1e-14 and s["C"].sum() > 0)
    print(f"  {f}: max mean |increment| = {np.abs(s1).max():.1e}")
print("[constant] mean increment < 1e-14 (fields ~1), C > 0:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYEOF
[ $? -ne 0 ] && status=1

# budget below the minimum quota must be a clear fatal error
d="$OUT/too_small"; rm -rf "$d"; mkdir -p "$d"
(cd "$d" && $MPIRUN -np "$NP" --oversubscribe "$BIN" -i "$IN" output1/nsample_global=3 \
   > log 2>&1)
if grep -q "nsample_global and nanchor_global must each be" "$d/log"; then
  echo "[too_small] fatal error reported as expected"
else
  echo "[too_small] MISSING expected fatal error"; status=1
fi
echo "SUITE: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
