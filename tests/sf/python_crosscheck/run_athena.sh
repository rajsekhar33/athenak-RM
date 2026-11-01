#!/bin/bash
# AthenaK side of the cross-check.
# Usage: ./run_athena.sh ATHENA_BINARY MPIRUN NRANKS [NSEEDS]
#   run/main         : crosscheck.athinput as is (bin snapshot + sf_scalar + sf_longtrans)
#   run/seeds/sK     : same input with the two sf seeds shifted by K (K = 1..NSEEDS-1);
#                      each run writes 2 outputs (00000, 00001) with independent draws,
#                      used to measure the estimator's spread directly.
BIN=$1; MPIRUN=$2; NP=$3; NSEEDS=${4:-32}
HERE=$(cd "$(dirname "$0")" && pwd)
IN=$HERE/crosscheck.athinput
run() {  # dir, overrides...
  local d=$1; shift; rm -rf "$d"; mkdir -p "$d"
  (cd "$d" && $MPIRUN -np "$NP" "$BIN" -i "$IN" "$@" > log 2>&1) || { echo "athena failed in $d"; tail -3 "$d/log"; exit 1; }
  echo "$d: $(grep -c 'sf\[' "$d/log") sf timing lines"
}
run "$HERE/run/main"
for k in $(seq 1 $((NSEEDS-1))); do
  run "$HERE/run/seeds/s$k" output2/seed=$((11+100*k)) output3/seed=$((12+100*k))
done
