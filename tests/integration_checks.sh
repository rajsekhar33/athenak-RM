#!/bin/bash
# Integration checks (implementation review, step 4).
#   1. State preservation: an isothermal MHD run with turbulence forcing evolves to a
#      bit-identical double-precision state (restart-file bodies, forcing RNG included)
#      with and without every diagnostic output active together (existing pdf, prof and
#      bin outputs plus sf scalar with histograms, sf aniso_mhd and spec scalar and
#      Helmholtz, written every cycle).
#   2. Changed-rank restart: a run on NP1 ranks writes a restart file; restarting on NP2
#      ranks continues the run. The solver with forcing is not bitwise invariant to the
#      rank layout (global reductions; ~1e-12), so: state within 1e-10 relative, spectra
#      within 1e-9, scalar-mode sf E1 counts identical (membership depends on geometry
#      only) with sums within 1e-9, aniso_mhd E1 (direction-dependent membership can
#      flip for a pair on a threshold) and all E2 (draws depend on the layout)
#      statistically consistent.
# usage: tests/integration_checks.sh ATHENA MPIRUN OUTDIR [NP1 NP2]  (PROBLEM=sf_unit_test)
set -u
BIN=$1; M=$2; OUT=$3; NP1=${4:-2}; NP2=${5:-3}
SRC=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$OUT"; mkdir -p "$OUT"
status=0
note() { echo "$1"; [[ "$1" == *FAIL* ]] && status=1; }
run() {   # dir np args...
  local d=$1 np=$2; shift 2
  mkdir -p "$d"
  (cd "$d" && $M -np "$np" --oversubscribe "$BIN" "$@" > log 2>&1) || return
  # one MPI job of np ranks, not np serial copies
  [ "$(grep -ac 'Number of parallel ranks' "$d/log")" -eq 1 ] &&
    grep -aq "Number of parallel ranks = $np\$" "$d/log" ||
    { echo "$d: binary did not run on $np MPI ranks: FAIL"; return 1; }
}

# ---- inputs: MHD + forcing, with and without diagnostics ----
python3 - "$SRC/inputs/tests/sf_aniso_test.athinput" "$OUT" <<'PY'
import sys
s = open(sys.argv[1]).read()
out = sys.argv[2]
base = s[:s.index("<output1>")]
base = base.replace("nlim       = 0", "nlim       = 6")
base = base.replace("noise     = 1.0", "noise     = 0.2")
base += """
<turb_driving>
type  = mhd
tcorr = 0.5
dedt  = 0.1
nlow  = 1
nhigh = 2

<output1>
file_type  = bin
variable   = mhd_w
dcycle     = 3

<output2>
file_type  = rst
dcycle     = 3
"""
diag = """
<output3>
file_type  = pdf
variable   = mhd_w_d
bin_min    = 0.5
bin_max    = 1.5
nbin       = 32
dcycle     = 1

<output4>
file_type  = prof
variable   = mhd_w_vx
coord_axis = x1
bin_min    = 0.0
bin_max    = 1.0
nbin       = 16
logscale   = false
dcycle     = 1

<output5>
file_type      = sf
id             = scal
mode           = scalar
variables      = mhd_w_vx, mhd_w_vy, mhd_w_vz
weights        = ones, mhd_w_d
max_order      = 4
nbin           = 12
histogram      = true
hist_nbin      = 16
hist_min       = 1.0e-2
hist_max       = 1.0e1
ndraw_block    = 40000
nsample_global = 8000
nanchor_global = 16000
nreplicate     = 8
dcycle         = 1

<output6>
file_type      = sf
id             = aniso
mode           = aniso_mhd
variables      = mhd_w_vx, mhd_w_vy, mhd_w_vz, mhd_bcc1, mhd_bcc2, mhd_bcc3
weights        = ones
max_order      = 4
nbin           = 10
nbin_prll      = 8
nbin_perp      = 8
ndraw_block    = 40000
nsample_global = 8000
nanchor_global = 16000
nreplicate     = 8
dcycle         = 1

<output7>
file_type  = spec
id         = d
mode       = scalar
variables  = mhd_w_d
weights    = ones, mhd_w_d
nbin       = 20
dcycle     = 1

<output8>
file_type  = spec
id         = helm
mode       = helmholtz
variables  = mhd_w_vx, mhd_w_vy, mhd_w_vz
nbin       = 20
dcycle     = 1
"""
open(out + "/plain.athinput", "w").write(base)
open(out + "/diag.athinput", "w").write(base + diag)
PY

# ---- 1. state preservation ----
run "$OUT/plain" "$NP1" -i "$OUT/plain.athinput" || note "[1] plain run failed: FAIL"
run "$OUT/diag" "$NP1" -i "$OUT/diag.athinput" || note "[1] diagnostics run failed: FAIL"
# AthenaK .bin = preheader (time, cycle, variables), "header offset=N", N bytes of input
# parameter text (differs: the diagnostics run has more blocks), then the data. Compare
# the preheader and the data.
cat > "$OUT/bincmp.py" <<'PY'
import sys
def parts(path):
    raw = open(path, "rb").read()
    i = raw.index(b"header offset=")
    j = raw.index(b"\n", i) + 1
    n = int(raw[i + len(b"header offset="):j - 1])
    return raw[:i], raw[j + n:]
a, b = parts(sys.argv[1]), parts(sys.argv[2])
sys.exit(0 if (a[0] == b[0] and a[1] == b[1] and len(a[1]) > 0) else 1)
PY
nb=0; same=1
for f in "$OUT"/plain/bin/*.bin; do
  [ -e "$f" ] || continue
  nb=$((nb + 1))
  python3 -I "$OUT/bincmp.py" "$f" "$OUT/diag/bin/$(basename "$f")" ||
    { same=0; echo "  differs: $(basename "$f")"; }
done
nd=$(ls "$OUT"/diag/sf_scal/*.sf "$OUT"/diag/sf_aniso/*.sf "$OUT"/diag/spec_d/*.spec \
        "$OUT"/diag/spec_helm/*.spec "$OUT"/diag/pdf_*/* "$OUT"/diag/prof_*/* 2>/dev/null | wc -l)
# double-precision state: restart-file bodies after the parameter text
nr=0
for f in "$OUT"/plain/rst/*.rst; do
  [ -e "$f" ] || continue
  nr=$((nr + 1))
  python3 -I - "$f" "$OUT/diag/rst/$(basename "$f")" <<'PY' || { same=0; echo "  state differs: $(basename "$f")"; }
import sys
def body(p):
    raw = open(p, "rb").read()
    i = raw.index(b"<par_end>")
    return raw[raw.index(b"\n", i) + 1:]
sys.exit(0 if body(sys.argv[1]) == body(sys.argv[2]) else 1)
PY
done
if [ $nb -ge 3 ] && [ $nr -ge 3 ] && [ $same -eq 1 ] && [ "$nd" -ge 30 ]; then
  note "[1] MHD + forcing, 6 cycles: $nr double-precision states and $nb dumps bit-identical with $nd diagnostic files written: PASS"
else
  note "[1] state preservation: FAIL (rst $nr, bin $nb, identical $same, diagnostic files $nd)"
fi

# ---- 2. changed-rank restart ----
# continuous run on NP1 is $OUT/diag; restart its cycle-3 restart file on NP2
mkdir -p "$OUT/restart"
rst=$(ls "$OUT"/diag/rst/*.00001.rst 2>/dev/null | head -1)
if [ -z "$rst" ]; then
  note "[2] no restart file from the NP1 run: FAIL"
else
  run "$OUT/restart" "$NP2" -r "$rst" || note "[2] restart on $NP2 ranks failed: FAIL"
  python3 -I "$SRC/tests/integration_restart_check.py" "$OUT/diag" "$OUT/restart" "$SRC" "$rst" ||
    status=1
fi

echo "INTEGRATION CHECKS: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
