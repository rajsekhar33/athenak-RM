#!/bin/bash
# Separation-bin geometry and input guards of the sf/spec outputs:
#   1. bins do not depend on the MeshBlock layout. A 1D periodic box of length 0.1 with
#      16 cells and linear bins of width one cell (every whole-cell separation lies on a
#      bin edge: bin 0 must stay empty), and a 2D box with anisotropic cells (0.1 x 0.2,
#      16 x 8 cells). For one block, two blocks and two ranks, the populated bins must
#      equal the enumeration from integer cell offsets, and bins without a separation
#      must hold exactly zero pairs.
#   2. invalid input stops with a message naming the problem: names that do not give
#      exactly one field component (hydro_u_e with eos = isothermal next to two passive
#      scalars, sf and spec), spec region bounds, slices and ghost zones, malformed or
#      nonfinite numbers, fractional or huge budgets, a sample too large for one
#      MPI_Allgatherv, an empty region with sparse = true
#   3. valid input still runs: a single passive scalar, spec bounds equal to the mesh
# usage: tests/sf/run_guards.sh ATHENA MPIRUN OUTDIR   (PROBLEM=sf_unit_test, MPI build)
set -u
BIN=$1; M=$2; OUT=$3
SRC=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$OUT"; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
status=0
note() { echo "$1"; [[ "$1" == *FAIL* ]] && status=1; }

# mkin FILE MESH_PY OUTPUT_TEXT: the sf_unit_test input without its output blocks,
# with python edits MESH_PY of its text s, and OUTPUT_TEXT appended
mkin() {
  python3 - "$SRC/inputs/tests/sf_unit_test.athinput" "$1" "$2" "$3" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
s = s[:s.index("<output1>")]
def setkey(blk, key, val):
    global s
    a = s.index("<" + blk + ">")
    b = s.find("\n<", a + 1)
    b = len(s) if b < 0 else b
    body, n = re.subn(r"(?m)^" + key + r"\s*=.*$", f"{key} = {val}", s[a:b])
    if n == 0:
        body = body.rstrip("\n") + f"\n{key} = {val}\n"
    s = s[:a] + body + s[b:]
exec(sys.argv[3])
open(sys.argv[2], "w").write(s.rstrip("\n") + "\n\n" + sys.argv[4])
PY
}
run() {   # dir np input
  mkdir -p "$1"
  (cd "$1" && $M -np "$2" --oversubscribe "$BIN" -i "$3" > log 2>&1)
}

# ---- 1. layout independence of the separation bins ----
SF1D='<output1>
file_type      = sf
id             = g
variables      = hydro_w_vx
max_order      = 2
nbin           = 8
logscale       = false
r_max          = 0.05
ndraw_block    = 200000
nsample_global = 4096
nanchor_global = 4096
nreplicate     = 2
dt             = 1.0
'
MESH1D='
for k, v in (("nx1", 16), ("x1max", 0.1), ("nx2", 1), ("nx3", 1)): setkey("mesh", k, v)
for k, v in (("nx2", 1), ("nx3", 1)): setkey("meshblock", k, v)
'
SF2D="${SF1D/nbin           = 8/nbin           = 7}"
SF2D="${SF2D/r_max          = 0.05/r_max          = 0.09}"
MESH2D='
for k, v in (("nx1", 16), ("x1max", 0.1), ("nx2", 8), ("x2max", 0.2), ("nx3", 1)):
    setkey("mesh", k, v)
setkey("meshblock", "nx3", 1)
'
for geo in 1d 2d; do
  if [ $geo = 1d ]; then MESH=$MESH1D; BLK=$SF1D; else MESH=$MESH2D; BLK=$SF2D; fi
  for lay in one_block:16:1 two_blocks:8:1 two_ranks:8:2; do
    IFS=: read -r name mb np <<< "$lay"
    d=$OUT/layout_${geo}_$name; mkdir -p "$d"
    mbpy="setkey('meshblock', 'nx1', $mb)"
    [ $geo = 2d ] && mbpy="$mbpy; setkey('meshblock', 'nx2', 8)"
    mkin "$d/in.athinput" "$MESH
$mbpy" "$BLK"
    if run "$d" "$np" in.athinput; then
      if geometry_result=$(python3 -I "$SRC/tests/sf/geometry_check.py" "$d" 2>&1); then
        note "[layout $geo $name] $geometry_result"
      else
        note "[layout $geo $name] checker failed: FAIL ($geometry_result)"
      fi
    else
      note "[layout $geo $name] athena failed (see $d/log): FAIL"
    fi
  done
done

# ---- 2. invalid input ----
SFBASE='<output1>
file_type      = sf
id             = g
variables      = hydro_w_vx, hydro_w_vy
max_order      = 2
nbin           = 8
ndraw_block    = 4096
nsample_global = 1024
nanchor_global = 1024
nreplicate     = 2
dt             = 1.0
'
SPECBASE='<output1>
file_type      = spec
id             = g
variables      = hydro_w_vx, hydro_w_vy
nbin           = 8
dt             = 1.0
'
expect_fail() {   # name message mesh_py block
  local d=$OUT/reject_$1
  mkdir -p "$d"; mkin "$d/in.athinput" "$3" "$4"
  if run "$d" 1 in.athinput; then
    note "[reject $1] athena accepted the input: FAIL"
  elif grep -aq -- "$2" "$d/log"; then
    note "[reject $1] stopped with '$2': PASS"
  else
    note "[reject $1] stopped without '$2' (see $d/log): FAIL"
  fi
}
TWOSCAL='setkey("hydro", "nscalars", 2)'
expect_fail sf_cancelling_names "gives 0 field components" "$TWOSCAL" \
  "${SFBASE/hydro_w_vx, hydro_w_vy/hydro_u_e, hydro_w_s}"
expect_fail spec_cancelling_names "gives 0 field components" "$TWOSCAL" \
  "${SPECBASE/hydro_w_vx, hydro_w_vy/hydro_u_e, hydro_w_s}"
expect_fail spec_region "x1_min is not supported" "" "${SPECBASE}x1_min = 0.25
"
expect_fail spec_slice "does not support slice_x1/2/3, gid or ghost_zones" "" "${SPECBASE}slice_x3 = 0.5
"
expect_fail spec_ghost_zones "does not support slice_x1/2/3, gid or ghost_zones" "" "${SPECBASE}ghost_zones = true
"
expect_fail sf_slice "does not support slice_x1/2/3, gid or ghost_zones" "" "${SFBASE}slice_x1 = 0.5
"
expect_fail sf_gid "does not support slice_x1/2/3, gid or ghost_zones" "" "${SFBASE}gid = 1
"
expect_fail sf_fractional_budget "ndraw_block must be a whole number" "" \
  "${SFBASE/ndraw_block    = 4096/ndraw_block    = 4096.5}"
expect_fail sf_huge_budget "nanchor_global must be a whole number" "" \
  "${SFBASE/nanchor_global = 1024/nanchor_global = 1e300}"
expect_fail sf_inf_budget "nsample_global = 'inf' is not a finite number" "" \
  "${SFBASE/nsample_global = 1024/nsample_global = inf}"
expect_fail sf_malformed_budget "ndraw_block = '4096abc' is not a finite number" "" \
  "${SFBASE/ndraw_block    = 4096/ndraw_block    = 4096abc}"
expect_fail sf_inf_rmax "r_max = 'inf' is not a finite number" "" "${SFBASE}r_max = inf
"
expect_fail sf_nan_rmin "r_min = 'nan' is not a finite number" "" "${SFBASE}r_min = nan
"
expect_fail sf_allgatherv_limit "one MPI_Allgatherv" "" \
  "${SFBASE/nsample_global = 1024/nsample_global = 400000000}"
expect_fail sf_sparse_empty_region "the sf region contains no cells" "" "${SFBASE}sparse = true
x1_min = 0.5001
x1_max = 0.5002
"

# ---- 3. valid input ----
expect_ok() {   # name mesh_py block
  local d=$OUT/accept_$1
  mkdir -p "$d"; mkin "$d/in.athinput" "$2" "$3"
  if run "$d" 1 in.athinput && ls "$d"/*_g/*.00000.* > /dev/null 2>&1; then
    note "[accept $1] ran and wrote output: PASS"
  else
    note "[accept $1] failed (see $d/log): FAIL"
  fi
}
expect_ok sf_one_passive_scalar 'setkey("hydro", "nscalars", 1)' \
  "${SFBASE/hydro_w_vx, hydro_w_vy/hydro_w_s, hydro_w_vx}"
expect_ok spec_one_passive_scalar 'setkey("hydro", "nscalars", 1)' \
  "${SPECBASE/hydro_w_vx, hydro_w_vy/hydro_w_s, hydro_w_vx}"
expect_ok spec_bounds_equal_mesh "" "${SPECBASE}x1_min = 0.0
x1_max = 1.0
"
echo "GUARD CHECKS: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
