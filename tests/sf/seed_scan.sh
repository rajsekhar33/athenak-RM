#!/bin/bash
# Usage: [SEED0=offset] [INPUT=file] [MPIRUN=cmd] [MASK=problem/mask] tests/sf/seed_scan.sh ATHENA_BINARY NRANKS NSEEDS OUTDIR [overrides]
# Runs inputs/tests/sf_seedscan.athinput (or $INPUT) with seeds 1..NSEEDS, then tests/sf/seed_scan.py.
set -e
BIN=$1; NP=$2; NS=$3; OUT=$4; shift 4
SRC=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
for s in $(seq $((${SEED0:-0} + 1)) $((${SEED0:-0} + NS))); do
  d="$OUT/seed$s"; rm -rf "$d"; mkdir -p "$d"
  (cd "$d" && ${MPIRUN:-mpirun} -np "$NP" --oversubscribe "$BIN" \
     -i "${INPUT:-$SRC/inputs/tests/sf_seedscan.athinput}" output1/seed=$s "$@" > log 2>&1) ||
    { echo "seed $s: athena failed, see $d/log:"; tail -5 "$d/log"
      echo "OVERALL: FAIL (run for seed $s failed)"; exit 1; }
done
python3 -I "$SRC/tests/sf/seed_scan.py" "$OUT" ${MASK:-0}
