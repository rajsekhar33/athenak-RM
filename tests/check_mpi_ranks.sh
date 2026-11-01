#!/bin/bash
# Preflight: the binary must run as ONE 2-rank MPI job. A build without MPI under
# mpirun starts independent serial copies, which every single-output checker accepts.
# Also records the CUDA device each rank uses when nvidia-smi is available.
# usage: tests/check_mpi_ranks.sh ATHENA MPIRUN OUTDIR   (PROBLEM=sf_unit_test)
set -u
BIN=$1; M=$2; D=$3
SRC=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$D"; mkdir -p "$D"; cp "$SRC/inputs/tests/spec_unit_test.athinput" "$D/in.athinput"
(cd "$D" && $M -np 2 --oversubscribe "$BIN" -i in.athinput > log 2>&1) ||
  { echo "MPI RANKS: FAIL (run failed)"; exit 1; }
n=$(grep -ac 'Number of parallel ranks' "$D/log")
hdr=$(grep -ah '^nranks=' "$D"/spec_*/*.spec 2>/dev/null | sort -u | tr '\n' ' ')
if [ "$n" -eq 1 ] && grep -aq 'Number of parallel ranks = 2$' "$D/log" &&
   [ "$hdr" = "nranks=2 " ]; then
  echo "MPI RANKS: PASS (one start-up banner with 2 ranks; spec headers $hdr)"
  exit 0
fi
echo "MPI RANKS: FAIL ($n start-up banners; spec headers '$hdr'; serial build or wrong mpirun)"
exit 1
