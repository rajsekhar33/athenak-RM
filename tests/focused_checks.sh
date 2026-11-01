#!/bin/bash
# Focused checks from the implementation review (docs/..._codex_implementation_review):
#   A. sf weight-count bound: 8 weights run (team and global kernels), 9 are rejected
#   B. histogram scales: analytic one-component wave (A/sqrt2 scalar, A/sqrt6 long_trans),
#      the same under a 1e6 velocity boost (centered variance), constant field (fallback
#      1), empty region (fatal error)
#   C. output destinations: two sf and two spec blocks without id get distinct
#      directories; a duplicate explicit id is a fatal error
#   D. spectrum binning across the team-scratch limit (value passes) and beyond it
#      (global-atomics fallback), checked against spectra.py
#   E. aniso_mhd symmetries: B -> -B leaves every array unchanged (same draws); B = 0
#      skips every pair (Python's |mean B| < 1e-30 rule) without failing
# usage: tests/focused_checks.sh ATHENA MPIRUN NRANKS OUTDIR   (PROBLEM=sf_unit_test)
set -u
BIN=$1; MPIRUN=$2; NP=$3; OUT=$4
SRC=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$OUT"; mkdir -p "$OUT"
status=0
note() { echo "$1"; [[ "$1" == *FAIL* ]] && status=1; }
athena() {   # dir input [overrides...]; returns athena's exit status
  local d=$1 in=$2; shift 2
  mkdir -p "$d"
  (cd "$d" && $MPIRUN -np "$NP" --oversubscribe "$BIN" -i "$in" "$@" > log 2>&1) || return
  # a serial build under mpirun runs NP independent copies; refuse that
  if [ "$(grep -ac 'Number of parallel ranks' "$d/log")" -ne 1 ] ||
     ! grep -aq "Number of parallel ranks = $NP\$" "$d/log"; then
    note "[$d] binary did not run on $NP MPI ranks (serial build or wrong mpirun): FAIL"
    return 1
  fi
}
mkin() {   # out-file, python edits of the sf_histogram_test input text s
  python3 - "$SRC/inputs/tests/sf_histogram_test.athinput" "$1" "$2" <<'PY'
import sys
s = open(sys.argv[1]).read()
s = s.replace("ndraw_block    = 400000", "ndraw_block    = 40000")
s = s.replace("nanchor_global = 160000", "nanchor_global = 16000")
exec(sys.argv[3])
open(sys.argv[2], "w").write(s)
PY
}

# ---- A. weight-count bound ----
for k in team global; do
  mkin "$OUT/w8_$k.athinput" "s = s.replace('weights        = ones, hydro_w_d', 'weights        = ones, hydro_w_d, ones, hydro_w_d, ones, hydro_w_d, ones, hydro_w_d', 1); s = s.replace('histogram      = true', 'histogram      = true\nkernel         = $k', 1)"
  if athena "$OUT/w8_$k" "$OUT/w8_$k.athinput" &&
     python3 -I "$SRC/tests/sf/sf_reference.py" "$OUT/w8_$k" > "$OUT/w8_$k/check.txt" 2>&1; then
    note "[A] 8 weights, kernel=$k: run and exhaustive check PASS"
  else
    note "[A] 8 weights, kernel=$k: FAIL (see $OUT/w8_$k)"
  fi
done
mkin "$OUT/w9.athinput" "s = s.replace('weights        = ones, hydro_w_d', 'weights        = ones, ones, ones, ones, ones, ones, ones, ones, ones', 1)"
athena "$OUT/w9" "$OUT/w9.athinput"
if grep -q "at most 8 weights" "$OUT/w9/log"; then
  note "[A] 9 weights rejected with a fatal error: PASS"
else
  note "[A] 9 weights not rejected: FAIL"
fi

# ---- B. histogram scales ----
check_sigma() {   # dir, expected scalar sigma per variable, expected long_trans sigma[, rtol]
  python3 -I - "$1" "$2" "$3" "$SRC/tests/sf" "${4:-1e-9}" <<'PY'
import glob, sys
import numpy as np
sys.path.insert(0, sys.argv[4])
from sf_reference import read_sf
exp_s = [float(v) for v in sys.argv[2].split(",")]
exp_l = float(sys.argv[3])
rtol = float(sys.argv[5])
files = sorted(glob.glob(sys.argv[1] + "/sf_*/*.00000.sf"))
modes = sorted(read_sf(f)[0]["mode"] for f in files)
ok = modes == ["long_trans", "scalar"]   # one initial output of each block, no fewer
if not ok:
    print("  expected the initial scalar and long_trans outputs, found", modes)
for f in files:
    m, a = read_sf(f)
    sig = a["hist_sigma"]
    want = np.array([exp_s] * sig.shape[0]) if m["mode"] == "scalar" else \
        np.full(sig.shape, exp_l)
    # weight 'ones' rows only (row 0); density weight is 1 here as well
    err = np.max(np.abs(sig - want) / want)
    ok &= bool(np.all(np.isfinite(sig)) and err < rtol)
    print("  %s %s sigma=%s want=%s rel err %.1e" % (m["mode"], f.split("/")[-2],
          np.array2string(sig[0], precision=9), np.array2string(want[0], precision=9), err))
sys.exit(0 if ok else 1)
PY
}
A=0.7
SX=$(python3 -c "import math; print($A/math.sqrt(2))")
SL=$(python3 -c "import math; print($A/math.sqrt(6))")
for boost in 0 1e6; do   # (the large-offset, small-amplitude case follows)
  mkin "$OUT/wave_$boost.athinput" "s = s.replace('constant  = false', 'constant  = false\nwave_amp  = $A\nboost     = $boost', 1)"
  if athena "$OUT/wave_$boost" "$OUT/wave_$boost.athinput" &&
     check_sigma "$OUT/wave_$boost" "$SX,1,1" "$SL" > "$OUT/wave_$boost/check.txt" 2>&1; then
    note "[B] one-component wave, boost=$boost: sigma = A/sqrt2 (scalar vx), 1 (zero comps), A/sqrt6 (long_trans): PASS"
  else
    note "[B] one-component wave, boost=$boost: FAIL"; cat "$OUT/wave_$boost/check.txt"
  fi
done
# large offset, small resolved amplitude: q = 1e10 + 1e-3 sin(2 pi x). The stored
# values carry ~2e-6 absolute rounding, so sigma is checked to 1e-3 relative; a
# relative-to-rms constancy threshold would wrongly replace it by 1
A2=1e-3
SX2=$(python3 -c "import math; print($A2/math.sqrt(2))")
SL2=$(python3 -c "import math; print($A2/math.sqrt(6))")
mkin "$OUT/wave_off.athinput" "s = s.replace('constant  = false', 'constant  = false\nwave_amp  = $A2\nboost     = 1e10', 1)"
if athena "$OUT/wave_off" "$OUT/wave_off.athinput" &&
   check_sigma "$OUT/wave_off" "$SX2,1,1" "$SL2" 1e-3 > "$OUT/wave_off/check.txt" 2>&1; then
  note "[B] offset 1e10 with amplitude 1e-3: sigma = A/sqrt2 and A/sqrt6 kept (not replaced by 1): PASS"
else
  note "[B] offset 1e10 with amplitude 1e-3: FAIL"; cat "$OUT/wave_off/check.txt"
fi
# the checker must refuse a directory without the expected outputs
mkdir -p "$OUT/sigma_empty"
if check_sigma "$OUT/sigma_empty" "1,1,1" "1" > "$OUT/sigma_empty.txt" 2>&1; then
  note "[B] sigma checker accepted an empty directory: FAIL"
else
  note "[B] sigma checker rejects an empty directory: PASS"
fi
mkin "$OUT/const.athinput" "s = s.replace('constant  = false', 'constant  = true', 1)"
if athena "$OUT/const" "$OUT/const.athinput" &&
   check_sigma "$OUT/const" "1,1,1" "1" > "$OUT/const/check.txt" 2>&1; then
  note "[B] constant field: every scale falls back to 1: PASS"
else
  note "[B] constant field: FAIL"; cat "$OUT/const/check.txt"
fi
mkin "$OUT/empty.athinput" "s = s.replace('<output1>', '<output1>\nx1_min = 2.0\nx1_max = 3.0', 1)"
athena "$OUT/empty" "$OUT/empty.athinput"
if grep -q "region contains no cells" "$OUT/empty/log"; then
  note "[B] empty selection: fatal error: PASS"
else
  note "[B] empty selection not rejected: FAIL"
fi

# ---- C. output destinations ----
python3 - "$SRC/inputs/tests/sf_unit_test.athinput" "$OUT/dirs.athinput" <<'PY'
import sys
s = open(sys.argv[1]).read()
s = s.replace("id             = scalar\n", "").replace("id             = longtrans\n", "")
s = s.replace("ndraw_block    = 400000", "ndraw_block    = 4000")
s = s.replace("nanchor_global = 160000", "nanchor_global = 16000")
s += """
<output3>
file_type  = spec
variables  = hydro_w_d
nbin       = 10
dt         = 1.0

<output4>
file_type  = spec
variables  = hydro_w_vx
nbin       = 10
dt         = 1.0
"""
open(sys.argv[2], "w").write(s)
PY
if athena "$OUT/dirs" "$OUT/dirs.athinput" && [ -d "$OUT/dirs/sf_output1" ] &&
   [ -d "$OUT/dirs/sf_output2" ] && [ -d "$OUT/dirs/spec_output3" ] &&
   [ -d "$OUT/dirs/spec_output4" ]; then
  note "[C] blocks without id write to sf_output1, sf_output2, spec_output3, spec_output4: PASS"
else
  note "[C] default output directories: FAIL"; ls "$OUT/dirs"
fi
sed 's/^<output2>/<output2>\nid = output1/' "$OUT/dirs.athinput" > "$OUT/dup.athinput"
athena "$OUT/dup" "$OUT/dup.athinput"
if grep -q "which another sf output already uses" "$OUT/dup/log"; then
  note "[C] duplicate explicit id rejected: PASS"
else
  note "[C] duplicate id not rejected: FAIL"
fi

# ---- D. spectrum binning across the scratch limit ----
python3 - "$SRC/inputs/tests/spec_unit_test.athinput" "$OUT/spec_scr.athinput" <<'PY'
import sys
s = open(sys.argv[1]).read()
s = s[:s.index("<output1>")]
s += """
<output1>
file_type  = spec
id         = passes
variables  = hydro_w_vx
weights    = ones, hydro_w_d, hydro_w_vx, hydro_w_vy, hydro_w_vz, hydro_u_m1, hydro_u_m2, hydro_u_m3
nbin       = 1024
logscale   = true
dt         = 1.0

<output2>
file_type  = spec
id         = global
variables  = hydro_w_vy
nbin       = 7000
dt         = 1.0
"""
open(sys.argv[2], "w").write(s)
PY
mkdir -p "$OUT/spec_scr"; cp "$OUT/spec_scr.athinput" "$OUT/spec_scr/in.athinput"
# the header records the path taken: one variable with 8 weights x 1025 slots x 8 B
# (65.6 KB) must need >= 2 scratch passes under the 48 KiB cap; 7000 bins must use the
# global fallback
if athena "$OUT/spec_scr" in.athinput &&
   python3 "$SRC/tests/spec/spec_reference.py" "$OUT/spec_scr" > "$OUT/spec_scr/check.txt" 2>&1 &&
   [ "$(ls "$OUT"/spec_scr/spec_passes/*.spec 2>/dev/null | wc -l)" -ge 1 ] &&
   ! grep -aL "^bin_scratch_passes_max=[2-9]" "$OUT"/spec_scr/spec_passes/*.spec | grep -q . &&
   ! grep -aL "^bin_global_fallback=1" "$OUT"/spec_scr/spec_global/*.spec | grep -q .; then
  note "[D] 1 variable x 8 weights x 1025 slots (header: >= 2 scratch passes) and 7000 bins (header: global fallback) vs spectra.py: PASS"
else
  note "[D] spectrum scratch-limit cases: FAIL"; tail -5 "$OUT/spec_scr/check.txt" "$OUT/spec_scr/log"
fi

# ---- E. aniso_mhd field reversal and zero field ----
for bs in 1 -1 0; do
  python3 - "$SRC/inputs/tests/sf_aniso_test.athinput" "$OUT/aniso_$bs.athinput" "$bs" <<'PY'
import sys
s = open(sys.argv[1]).read()
s = s.replace("constant  = false", "constant  = false\nbscale    = " + sys.argv[3], 1)
s = s.replace("ndraw_block    = 400000", "ndraw_block    = 40000")
s = s.replace("nanchor_global = 160000", "nanchor_global = 16000")
open(sys.argv[2], "w").write(s)
PY
  athena "$OUT/aniso_$bs" "$OUT/aniso_$bs.athinput" || note "[E] aniso run bscale=$bs failed: FAIL"
done
ok=1
for f in "$OUT"/aniso_1/sf_aniso/*.sf; do
  g="$OUT/aniso_-1/sf_aniso/$(basename "$f")"
  python3 -I "$SRC/tests/sf/compare_sf.py" "$f" "$g" > "$OUT/aniso_rev.txt" 2>&1 || ok=0
done
[ -n "$(ls "$OUT"/aniso_1/sf_aniso/*.sf 2>/dev/null)" ] || ok=0
if [ $ok -eq 1 ]; then
  note "[E] B -> -B: identical counts, sums equal to round-off: PASS"
else
  note "[E] B -> -B changes the aniso_mhd output: FAIL"; cat "$OUT/aniso_rev.txt"
fi
if python3 -I - "$OUT/aniso_0" "$SRC/tests/sf" <<'PY'
import glob, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from sf_reference import read_sf
fs = sorted(glob.glob(sys.argv[1] + "/sf_aniso/*.sf"))
ok = len(fs) > 0
for f in fs:
    m, a = read_sf(f)
    ok &= all(np.all(a[k] == 0) for k in ("T_er", "W_er", "C_er", "n_acc_er"))
    ok &= int(m["attempts_e1"]) > 0 and int(m["attempts_e2"]) > 0
sys.exit(0 if ok else 1)
PY
then
  note "[E] B = 0: every pair skipped, empty accumulators, run succeeds: PASS"
else
  note "[E] B = 0 policy: FAIL"
fi

echo "FOCUSED CHECKS: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
