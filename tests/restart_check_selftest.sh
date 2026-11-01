#!/bin/bash
# Fault injection for the changed-rank restart checker (integration_restart_check.py)
# and the typed restart reader (rst_compare.py), on copies of a saved integration run
# (OUTDIR of tests/integration_checks.sh, which holds diag/ and restart/). No simulation.
# The unmodified copy must PASS; every corrupted copy must FAIL.
# usage: tests/restart_check_selftest.sh SAVED_INTEG_DIR WORKDIR
set -u
SAVED=$(cd "$1" && pwd); W=$2
SRC=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$W"; mkdir -p "$W"; W=$(cd "$W" && pwd)
status=0; n=0; nrej=0
fresh() { rm -rf "$W/c"; mkdir -p "$W/c"; cp -r "$SAVED/diag" "$SAVED/restart" "$W/c/"; }
check() { python3 -I "$SRC/tests/integration_restart_check.py" "$W/c/diag" "$W/c/restart" \
            "$SRC" "$W/c/diag/rst/$(basename "$RST")" > "$W/last.txt" 2>&1; }
RST=$(ls "$SAVED"/diag/rst/*.00001.rst | head -1)
R2=restart/rst/$(basename "$(ls "$SAVED"/restart/rst/*.rst | head -1)")
inject() {   # file field how: corrupt a typed field of a restart copy in place
  python3 -I - "$SRC/tests" "$1" "$2" "$3" <<'PY'
import struct, sys
sys.path.insert(0, sys.argv[1])
from rst_compare import parse
path, field, how = sys.argv[2], sys.argv[3], sys.argv[4]
meta, arrays = parse(path)
raw = bytearray(open(path, "rb").read())
if field in meta:
    fmt, _, v, off = meta[field]
    if how == "flipbyte":
        raw[off + struct.calcsize(fmt) - 1] ^= 0x01
    elif how == "scale":
        struct.pack_into("<d", raw, off, v[0] * (1 + 1e-6))
else:   # a physical array: NaN in its first entry of block 0
    o = meta["data_offset"][3]
    names = list(arrays)
    k = 0
    for nm in names[:names.index(field)]:
        k += arrays[nm].shape[1]
    struct.pack_into("<d", raw, o + 8 * k, float("nan"))
open(path, "wb").write(raw)
PY
}
expect() {   # label pass|fail
  n=$((n + 1)); check; local rc=$?
  if [ "$2" = pass ]; then
    [ $rc -eq 0 ] && echo "control accepted: $1" ||
      { echo "control REJECTED: $1: FAIL"; tail -3 "$W/last.txt"; status=1; }
  else
    [ $rc -ne 0 ] && { nrej=$((nrej + 1)); echo "rejected: $1 ($(grep -a FAIL "$W/last.txt" | head -1 | cut -c1-110))"; } ||
      { echo "ACCEPTED: $1: FAIL"; status=1; }
  fi
}
fresh; expect "unmodified copy" pass
fresh; rm "$W/c/restart/sf_scal/"*.00004.sf; expect "leading sf file missing after the restart" fail
fresh; rm "$W/c/restart/spec_helm/"*.00004.spec "$W/c/diag/spec_helm/"*.00004.spec
expect "leading spec file missing in both runs (found-files inventory would pass)" fail
fresh; rm "$W/c/restart/sf_aniso/"*.00007.sf; expect "last aniso file missing after the restart" fail
fresh; rm -r "$W/c/restart/spec_d" "$W/c/diag/spec_d"; expect "a whole spec block missing in both runs" fail
fresh; f=$(ls "$W"/c/restart/sf_scal/*.00005.sf); sed -i '0,/^cycle=5$/s//cycle=6/' "$f"
expect "sf header cycle changed" fail
fresh; inject "$W/c/$R2" rng flipbyte;            expect "forcing RNG state corrupted" fail
fresh; inject "$W/c/$R2" n_turb_updates flipbyte; expect "forcing update counter corrupted" fail
fresh; inject "$W/c/$R2" lloc flipbyte;           expect "block location corrupted" fail
fresh; inject "$W/c/$R2" cost flipbyte;           expect "block cost corrupted" fail
fresh; inject "$W/c/$R2" ncycle flipbyte;         expect "ncycle corrupted" fail
fresh; inject "$W/c/$R2" time scale;              expect "time changed by 1e-6" fail
fresh; inject "$W/c/$R2" mhd_x2f nan;             expect "NaN in a magnetic face array" fail
fresh; inject "$W/c/$R2" force3 nan;              expect "NaN in a forcing array" fail
fresh; f=$W/c/$R2; python3 -I - "$f" <<'PY'
import sys
p = sys.argv[1]; raw = open(p, "rb").read()
open(p, "wb").write(raw.replace(b"<par_end>", b"<z4c>\n<par_end>", 1))
PY
expect "unsupported layout (z4c block) is rejected, not guessed" fail
echo "$nrej of $((n - 1)) corrupted copies rejected"
echo "RESTART CHECK SELFTEST: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
