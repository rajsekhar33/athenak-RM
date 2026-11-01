#!/bin/bash
# sf cell selection (select1_*/select2_*: the Python region masks, e.g. the RadShock
# interface T_lo <= T <= T_hi and the downstream density cut), against the exact
# all-pairs reference (tests/sf/sf_reference.py) on a 32^3 mesh, 8^3 MeshBlocks, eos
# ideal (sf_unit_test sets an analytic pressure, so T = p/rho is known):
#   1. inputs/tests/sf_select_test.athinput as is: scalar output (weights ones and
#      hydro_w_d, temperature band, sparse = true, histograms) and long_trans output
#      (temperature band and hydro_w_d >= 1.1, a field loaded only for the selection,
#      default sampler); header ncell_selected against the reference count
#   2. the same with the samplers swapped
#   3. an empty selection: ncell_selected = 0 and every sum 0
#   3b. the same input as MHD (<mhd>, mhd_w_* fields): temperature from the MHD
#      primitives, with the magnetic energy of the sf_unit_test face fields
#   3c. computed sf variables: scalar variables temperature, pressure, hydro_w_d
#      (weights ones, hydro_w_d) without a selection, on the sub-box [0, 0.5]^3; and
#      MHD with only computed variables (temperature, pressure) and the selection
#   3d. mode = correlation (Python get_cf, raw <A(x) B(x+r)>): cross-correlation
#      hydro_w_vx * hydro_w_d with weights ones, hydro_w_d (sparse, selection) and the
#      temperature autocorrelation (two selections)
#   4. invalid configurations must stop with a message: select1_min > select1_max,
#      select2 without select1, temperature with eos = isothermal, an unknown variable,
#      temperature as a weight, correlation with three variables or a histogram,
#      temperature in a two-fluid run (<hydro> and <mhd>, ion-neutral: ambiguous)
# usage: tests/sf/run_select.sh ATHENA MPIRUN NP OUTDIR   (PROBLEM=sf_unit_test)
set -u
BIN=$1; M=$2; NP=$3; OUT=$4
SRC=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
IN=$SRC/inputs/tests/sf_select_test.athinput
status=0
fail() { echo "$1: FAIL"; status=1; }
ranks_ok() {
  [ "$(grep -ac 'Number of parallel ranks' "$1")" -eq 1 ] &&
    grep -aq "Number of parallel ranks = $NP\$" "$1"
}
hdr() { grep -ah "^$2=" "$1" | head -1 | cut -d= -f2; }

run_case() {   # name, then input overrides
  local name=$1; shift
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  (cd "$d" && "$M" -np "$NP" "$BIN" -i "${CASE_IN:-$IN}" "$@" > log 2>&1)
  local rc=$?
  if [ $rc -ne 0 ] || ! ranks_ok "$d/log"; then
    fail "$name (athena exit $rc or rank count)"; return
  fi
  # both outputs with both files must exist: a missing file must not shrink the check
  if ! python3 -I "$SRC/tests/sf/sf_inventory.py" select "$d" > "$d/inventory.txt" 2>&1; then
    fail "$name (incomplete output inventory, see $d/inventory.txt)"; return
  fi
  python3 -I "$SRC/tests/sf/sf_reference.py" "$d" --noise 1.0 > "$d/ref.txt" 2>&1
  rc=$?
  local nsel
  nsel=$(for f in "$d"/sf_*/sfsel.00000.sf; do hdr "$f" ncell_selected; done | tr '\n' ' ')
  if [ $rc -eq 0 ] && ! grep -q "ncell_selected.*FAIL" "$d/ref.txt"; then
    echo "$name (ncell_selected $nsel): OVERALL: PASS"
  else
    fail "$name (ncell_selected $nsel; reference exit $rc, see $d/ref.txt)"
  fi
}

# ---- 1-3. exact-reference cases ----
run_case select
run_case select_swapped output1/sparse=false output2/sparse=true
run_case select_empty output1/select1_min=100 output1/select1_max=200 \
  output2/select1_min=100 output2/select1_max=200
for f in "$OUT"/select_empty/sf_*/*.sf; do
  [ "$(hdr "$f" ncell_selected)" = "0" ] || fail "select_empty: $f ncell_selected != 0"
done
mkdir -p "$OUT/inputs"
sed -e 's/^<hydro>$/<mhd>/' -e 's/hydro_w_/mhd_w_/g' "$IN" > "$OUT/inputs/mhd.athinput"
CASE_IN=$OUT/inputs/mhd.athinput run_case select_mhd
awk '/^<output1>/{o=1} /^<output2>/{o=0}
     o && /^variables/ {print "variables      = temperature, pressure, hydro_w_d"; next}
     o && /^select1_/ {next}
     o && /^sparse/ {print; print "x1_min = 0.0"; print "x1_max = 0.5"; print "x2_min = 0.0";
                     print "x2_max = 0.5"; print "x3_min = 0.0"; print "x3_max = 0.5"; next}
     {print}' "$IN" > "$OUT/inputs/computed.athinput"
CASE_IN=$OUT/inputs/computed.athinput run_case computed
awk '/^<output1>/{o=1} /^<output2>/{o=0}
     o && /^variables/ {print "variables      = temperature, pressure"; next}
     o && /^weights/ {print "weights        = ones"; next}
     {print}' "$OUT/inputs/mhd.athinput" > "$OUT/inputs/computed_mhd.athinput"
CASE_IN=$OUT/inputs/computed_mhd.athinput run_case computed_mhd
awk '/^<output1>/{o=1} /^<output2>/{o=2}
     o && /^mode/ {print "mode           = correlation"; next}
     o==1 && /^variables/ {print "variables      = hydro_w_vx, hydro_w_d"; next}
     o==2 && /^variables/ {print "variables      = temperature"; next}
     o && /^(histogram|hist_)/ {next}
     {print}' "$IN" > "$OUT/inputs/correlation.athinput"
CASE_IN=$OUT/inputs/correlation.athinput run_case correlation

# ---- 4. invalid configurations ----
bad_case() {   # name, expected message (grep -E), input file
  local d=$OUT/bad_$1
  rm -rf "$d"; mkdir -p "$d"
  (cd "$d" && "$M" -np "$NP" "$BIN" -i "$3" > log 2>&1)
  local rc=$?
  if [ $rc -ne 0 ] && grep -aqE "$2" "$d/log"; then
    echo "bad_$1: rejected as expected"
  else
    fail "bad_$1 (exit $rc; expected message /$2/ missing; see $d/log)"
  fi
}
cp "$IN" "$OUT/inputs/minmax.athinput"
sed -i 's/^select1_max    = 0.7$/select1_max    = 0.5/' "$OUT/inputs/minmax.athinput"
sed -e '0,/^select1_variable = temperature$/{//d}' \
    -e '0,/^select1_min    = 0.6$/{//d}' -e '0,/^select1_max    = 0.7$/{//d}' "$IN" |
  awk '/^<output1>/{o=1} /^<output2>/{o=0} {print} o && /^weights/{print "select2_variable = hydro_w_d"}' \
  > "$OUT/inputs/noselect1.athinput"
sed -e 's/^eos         = ideal$/eos         = isothermal\niso_sound_speed = 1.0/' \
    -e '/^gamma /d' "$IN" > "$OUT/inputs/isothermal.athinput"
sed 's/^select2_variable = hydro_w_d$/select2_variable = hydro_w_nonexistent/' "$IN" \
  > "$OUT/inputs/unknown.athinput"
bad_case minmax "select1_min must be <= select1_max" "$OUT/inputs/minmax.athinput"
bad_case noselect1 "select2_\* needs select1_\*" "$OUT/inputs/noselect1.athinput"
bad_case isothermal "temperature \(computed field\) needs .* eos = ideal" "$OUT/inputs/isothermal.athinput"
sed '0,/^weights        = ones, hydro_w_d$/s//weights        = ones, temperature/' "$IN" \
  > "$OUT/inputs/tweight.athinput"
bad_case unknown "hydro_w_nonexistent" "$OUT/inputs/unknown.athinput"
bad_case tweight "derived variables are not yet supported" "$OUT/inputs/tweight.athinput"
awk '/^<output2>/{o=2} o==2 && /^variables/ {print "variables      = hydro_w_vx, hydro_w_vy, hydro_w_vz"; next}
     {print}' "$OUT/inputs/correlation.athinput" > "$OUT/inputs/corr3.athinput"
bad_case corr3 "mode=correlation needs one variable" "$OUT/inputs/corr3.athinput"
sed -e '0,/^mode           = scalar$/s//mode           = correlation/' \
    -e '0,/^variables      = hydro_w_vx, hydro_w_vy, hydro_w_vz$/s//variables      = hydro_w_vx, hydro_w_d/' \
    "$IN" > "$OUT/inputs/corrhist.athinput"
bad_case corrhist "histogram = true is not available for mode = aniso_mhd or correlation" \
  "$OUT/inputs/corrhist.athinput"
python3 - "$IN" "$OUT/inputs/twofluid.athinput" <<'PY'
import sys
s = open(sys.argv[1]).read()
i = s.index("<hydro>")
j = s.index("\n<", i + 1)
mhd = s[i:j].replace("<hydro>", "<mhd>")
open(sys.argv[2], "w").write(s[:j] + "\n\n" + mhd + "\n\n<ion-neutral>\ndrag_coeff = 1.0\n" + s[j:])
PY
bad_case twofluid "temperature \(computed field\) is ambiguous with both <hydro> and <mhd>" \
  "$OUT/inputs/twofluid.athinput"

if [ $status -eq 0 ]; then echo "SELECT: PASS"; else echo "SELECT: FAIL"; fi
exit $status
