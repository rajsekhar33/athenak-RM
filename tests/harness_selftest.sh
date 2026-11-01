#!/bin/bash
# Deliberate-failure checks for the test harnesses: every case below must be REJECTED
# (nonzero exit). A harness that accepts any of them cannot be trusted.
# usage: tests/harness_selftest.sh ATHENA MPIRUN OUTDIR [SERIAL_ATHENA]
#   SERIAL_ATHENA: a build without MPI (the rank preflight must reject it at np = 2)
#   ATHENA built with -D PROBLEM=sf_unit_test (used for one small sf run and one spec run)
set -u
BIN=$1; MPIRUN=$2; OUT=$3; SERBIN=${4:-}
SRC=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$OUT"; mkdir -p "$OUT"
nbad=0
expect_fail() {   # description, command...
  local what=$1; shift
  if "$@" > "$OUT/last.txt" 2>&1; then
    echo "[NOT REJECTED] $what"; nbad=$((nbad + 1))
  else
    echo "[rejected] $what"
  fi
}

# --- one small valid sf run and one spec run to corrupt ---
mkdir -p "$OUT/sf" "$OUT/spec"
(cd "$OUT/sf" && $MPIRUN -np 1 "$BIN" -i "$SRC/inputs/tests/sf_unit_test.athinput" \
   output1/nanchor_global=40000 output2/nanchor_global=40000 > log 2>&1) ||
  { echo "setup sf run failed"; exit 2; }
cp "$SRC/inputs/tests/spec_unit_test.athinput" "$OUT/spec/in.athinput"
(cd "$OUT/spec" && $MPIRUN -np 1 "$BIN" -i in.athinput > log 2>&1) ||
  { echo "setup spec run failed"; exit 2; }
python3 -I "$SRC/tests/sf/sf_reference.py" "$OUT/sf" > "$OUT/sf_ok.txt" 2>&1 ||
  { echo "setup: valid sf output does not pass; fix that first"; exit 2; }
python3 "$SRC/tests/spec/spec_reference.py" "$OUT/spec" > "$OUT/spec_ok.txt" 2>&1 ||
  { echo "setup: valid spec output does not pass; fix that first"; exit 2; }

# --- launchers that fail ---
expect_fail "pencil suite with a failing launcher" \
  "$SRC/tests/fft/run_pencil_suite.sh" /bin/false "$MPIRUN" "$OUT/pencil_false" 1
expect_fail "restart test with a failing launcher" \
  "$SRC/tests/sf/restart_test.sh" /bin/false "$OUT/restart_false"
# a launcher that starts independent serial copies (what a serial build under mpirun does)
cat > "$OUT/copies_launcher" <<'EOF'
#!/bin/bash
n=1; args=()
while [ $# -gt 0 ]; do
  case $1 in -np) n=$2; shift 2;; --oversubscribe) shift;; *) args+=("$1"); shift;; esac
done
[ "$n" -lt 2 ] && n=2
for ((r = 0; r < n; r++)); do "${args[@]}" & done; wait
EOF
chmod +x "$OUT/copies_launcher"
expect_fail "pencil suite whose launcher starts serial copies" \
  "$SRC/tests/fft/run_pencil_suite.sh" "$BIN" "$OUT/copies_launcher" "$OUT/pencil_copies" 1

# --- missing outputs ---
mkdir -p "$OUT/empty"
expect_fail "sf checker on a directory without .sf files" \
  python3 -I "$SRC/tests/sf/sf_reference.py" "$OUT/empty"
expect_fail "spec checker on a directory without .spec files" \
  python3 "$SRC/tests/spec/spec_reference.py" "$OUT/empty"
expect_fail "pencil checker without fft_test.bin" \
  python3 -I "$SRC/tests/fft/compare_pencil.py" "$OUT/empty"

# --- corrupted payloads (copies of the valid runs) ---
corrupt_sf() {   # dest, python statement applied to arrays a (dict of writable views)
  local dest=$1 stmt=$2
  rm -rf "$dest"; cp -r "$OUT/sf" "$dest"
  python3 -I - "$dest" "$SRC/tests/sf" "$stmt" <<'PY'
import glob, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from sf_reference import read_sf
for f in sorted(glob.glob(sys.argv[1] + "/sf_*/*.sf")):
    meta, s = read_sf(f)
    off = int(meta["header offset"])
    raw = bytearray(open(f, "rb").read())
    pos = off
    a = {}
    for k, v in s.items():
        a[k] = np.frombuffer(raw, dtype=v.dtype, count=v.size, offset=pos).reshape(v.shape)
        pos += v.nbytes
    exec(sys.argv[3])
    open(f, "wb").write(raw)
PY
}
corrupt_sf "$OUT/sf_noE2" 'a["T_er"][1] = 0; a["W_er"][1] = 0; a["C_er"][1] = 0; a["n_acc_er"][1] = 0'
expect_fail "sf output with the E2 population zeroed" \
  python3 -I "$SRC/tests/sf/sf_reference.py" "$OUT/sf_noE2"
corrupt_sf "$OUT/sf_bias" 'a["T_er"][0] *= 1.05'
expect_fail "sf output with E1 moments biased by 5%" \
  python3 -I "$SRC/tests/sf/sf_reference.py" "$OUT/sf_bias"
corrupt_sf "$OUT/sf_nan" 'a["T_er"][1, :, 0, 0, 0, 3] = np.nan'
expect_fail "sf output with a NaN in a required bin" \
  python3 -I "$SRC/tests/sf/sf_reference.py" "$OUT/sf_nan"

rm -rf "$OUT/spec_bad"; cp -r "$OUT/spec" "$OUT/spec_bad"
python3 -I - "$OUT/spec_bad" <<'PY'
import glob, sys
import numpy as np
for f in sorted(glob.glob(sys.argv[1] + "/spec_*/*.spec")):
    raw = bytearray(open(f, "rb").read())
    off = int(raw.split(b"header offset=")[1][:12])
    a = np.frombuffer(raw, dtype=np.float64, offset=off)
    a[-3] *= 1.001      # one E / P_DC / sumsq entry near the end of the payload
    open(f, "wb").write(raw)
PY
expect_fail "spec output with one value changed by 1e-3" \
  python3 "$SRC/tests/spec/spec_reference.py" "$OUT/spec_bad"

# --- seed scan with a NaN in one file (NaN compares false and could drop out) ---
rm -rf "$OUT/scan_nan"
for k in $(seq -w 0 11); do
  mkdir -p "$OUT/scan_nan/seed$k"; cp -r "$OUT/sf/sf_scalar" "$OUT/scan_nan/seed$k/"
done
python3 -I - "$OUT/scan_nan/seed05" "$SRC/tests/sf" <<'PY'
import glob, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from sf_reference import read_sf
f = sorted(glob.glob(sys.argv[1] + "/sf_scalar/*.sf"))[0]
meta, s = read_sf(f)
raw = bytearray(open(f, "rb").read())
a = np.frombuffer(raw, dtype=np.float64, offset=int(meta["header offset"]))
a[s["rbins"].size + 5] = np.nan    # an entry of T
open(f, "wb").write(raw)
PY
seed_scan_nan() {   # "accepted" unless the scan fails AND names the nonfinite file
  python3 -I "$SRC/tests/sf/seed_scan.py" "$OUT/scan_nan" > "$OUT/scan_nan.txt" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || ! grep -q "nonfinite" "$OUT/scan_nan.txt"
}
expect_fail "seed scan with a NaN in one file" seed_scan_nan

# --- an MPI-disabled executable launched on 2 ranks ---
if [ -n "$SERBIN" ]; then
  expect_fail "rank preflight with an MPI-disabled executable at np = 2" \
    "$SRC/tests/check_mpi_ranks.sh" "$SERBIN" "$MPIRUN" "$OUT/ranks_serial"
fi

echo "HARNESS SELFTEST: $([ $nbad -eq 0 ] && echo PASS || echo "FAIL ($nbad not rejected)")"
exit $nbad
