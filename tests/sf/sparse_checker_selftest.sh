#!/bin/bash
# Fault injection for the one statistical exception of run_sparse.sh: default sampler on
# the rare mask (mask 2), where sf_reference.py exit 3 (sampling checks only) is
# accepted. Copies of a saved mask-2 run directory are corrupted; every corruption
# must give exit 1 (structural failure or checker exception), never 0 or 3. The
# unmodified copy must give 0 or 3. No simulation.
# usage: tests/sf/sparse_checker_selftest.sh SAVED_MASK2_DIR WORKDIR
set -u
SAVED=$(cd "$1" && pwd); W=$2
SRC=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$W"; mkdir -p "$W"; W=$(cd "$W" && pwd)
status=0; nrej=0; n=0
fresh() { rm -rf "$W/c"; cp -r "$SAVED" "$W/c"; }
check() { python3 -I "$SRC/tests/sf/sf_reference.py" "$W/c" --mask 2 > "$W/last.txt" 2>&1; }
corrupt() {   # array index how: modify one entry of an array in the first scalar file
  python3 -I - "$SRC/tests/sf" "$W/c" "$1" "$2" "$3" <<'PY'
import glob, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from sf_reference import read_sf
f = sorted(glob.glob(sys.argv[2] + "/sf_scalar/*.sf"))[0]
meta, s = read_sf(f)
raw = bytearray(open(f, "rb").read())
off = int(meta["header offset"])
name, idx, how = sys.argv[3], int(sys.argv[4]), sys.argv[5]
for k, v in s.items():              # arrays in payload order
    if k == name:
        if how == "truncate":
            raw = raw[:off + 8 * idx]
            break
        a = np.frombuffer(raw, dtype=v.dtype, count=v.size, offset=off)
        a.flags.writeable = True
        a[idx] = np.nan if how == "nan" else (a[idx] * 1.5 + 1.0)
        break
    off += v.nbytes
open(f, "wb").write(raw)
PY
}
expect_ok() { fresh; n=$((n + 1)); check; local rc=$?
  if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then echo "control accepted (exit $rc)"
  else echo "control: exit $rc: FAIL"; tail -3 "$W/last.txt"; status=1; fi; }
expect_hard() { local label=$1; shift; fresh; "$@"; n=$((n + 1)); check; local rc=$?
  if [ $rc -eq 1 ]; then nrej=$((nrej + 1)); echo "rejected as structural: $label"
  else echo "NOT a structural failure (exit $rc): $label: FAIL"; status=1; fi; }
expect_ok
expect_hard "NaN in a replicate T entry"        corrupt T_er 7 nan
expect_hard "NaN in a histogram edge"           corrupt hist_edges 3 nan
expect_hard "histogram row no longer sums to C" corrupt H_er 40 scale
expect_hard "combined T != replicate mean"      corrupt T 5 scale
expect_hard "combined W != replicate mean"      corrupt W 5 scale
expect_hard "combined C != replicate mean"      corrupt C 5 scale
expect_hard "combined H != replicate mean"      corrupt H 40 scale
expect_hard "truncated payload (exception)"     corrupt T 2 truncate
expect_hard "hist sigma changed"                corrupt hist_sigma 0 scale
echo "$nrej of $((n - 1)) corruptions rejected as structural"
echo "SPARSE CHECKER SELFTEST: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
