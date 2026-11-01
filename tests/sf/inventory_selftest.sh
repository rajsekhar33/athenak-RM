#!/bin/bash
# Fault injection for tests/sf/sf_inventory.py on saved runner output (no simulation):
# the unmodified copies must pass; deleting any one expected file (or an output
# directory) must fail.
# usage: tests/sf/inventory_selftest.sh SAVED_SELECT_CASE SAVED_TOGGLE_DIR WORKDIR
#   (e.g. <release>/w_select_cpu/select and <release>/w_sparse_cpu/toggle)
set -u
SEL=$(cd "$1" && pwd); TOG=$(cd "$2" && pwd); W=$3
SRC=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$W"; mkdir -p "$W"; W=$(cd "$W" && pwd)
status=0; n=0; nrej=0
inv() { python3 -I "$SRC/tests/sf/sf_inventory.py" "$1" "$W/c" > "$W/last.txt" 2>&1; }
fresh() { rm -rf "$W/c"; cp -r "$1" "$W/c"; }
control() {   # kind saved
  fresh "$2"; inv "$1" && echo "control $1: accepted" ||
    { echo "control $1: rejected: FAIL"; status=1; }
}
missing() {   # kind saved relative-path
  fresh "$2"; rm -rf "${W:?}/c/$3"; n=$((n + 1))
  if inv "$1"; then echo "NOT detected: $1 without $3: FAIL"; status=1
  else nrej=$((nrej + 1)); echo "detected: $1 without $3"; fi
}
control select "$SEL"
missing select "$SEL" sf_scalar/sfsel.00001.sf
missing select "$SEL" sf_longtrans/sfsel.00000.sf
missing select "$SEL" sf_longtrans
control toggle "$TOG"
missing toggle "$TOG" cont/sf_scalar/sfsp.00009.sf
missing toggle "$TOG" cont/sf_longtrans/sfsp.00005.sf
missing toggle "$TOG" rest/sf_scalar/sfsp.00004.sf
missing toggle "$TOG" rest/sf_longtrans/sfsp.00009.sf
missing toggle "$TOG" rest/sf_scalar
missing toggle "$TOG" cont/rst/sfsp.00001.rst
echo "$nrej of $n deletions detected"
echo "INVENTORY SELFTEST: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
