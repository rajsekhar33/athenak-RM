#!/bin/bash
# Full selection cross-check on one machine: AthenaK, then Python exact sums, then the
# acceptance test (README.md). usage: run.sh ATHENA MPIRUN NRANKS OUTDIR
# Split across machines: run_athena.sh on the AthenaK machine, copy OUTDIR, then
# run_python.sh OUTDIR on a machine with athena_research + CuPy.
set -u
[ $# -eq 4 ] || { echo "usage: $0 ATHENA MPIRUN NRANKS OUTDIR"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/run_athena.sh" "$1" "$2" "$3" "$4" || { echo "SELECTION CROSSCHECK: FAIL (athena step)"; exit 1; }
"$HERE/run_python.sh" "$4"
