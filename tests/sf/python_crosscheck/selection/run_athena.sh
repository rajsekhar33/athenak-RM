#!/bin/bash
# Step 1 (AthenaK machine, e.g. Frontier or della-vis2): run selection.athinput.
# usage: run_athena.sh ATHENA MPIRUN NRANKS OUTDIR   [extra input overrides...]
# Writes OUTDIR/athena/{log, bin/, sf_scalar/, sf_longtrans/}. OUTDIR must not exist or
# be empty. Only OUTDIR/athena/bin/sel.hydro_w.00001.bin (5.3 MB) and the sf_* directories
# (~0.8 MB) are needed by step 2 (run_python.sh), which may run on another machine.
# MPIRUN may contain arguments (e.g. "srun -N 1 -c 7 --gpus-per-task=1 --gpu-bind=closest");
# it is called as: $MPIRUN -np NRANKS ATHENA ...  unless MPIRUN_NP_FLAG is set (e.g. -n).
# Guards (as the Frontier kit): tests/frontier/check_build.sh ATHENA scaling (MPI on,
# backend REQUIRE_BACKEND, default HIP; the default build with the turbulence problem);
# the launch is bounded (RUN_TIMEOUT, default 1200 s: SIGTERM, SIGKILL 60 s later, and
# with srun a step --time limit); the log must show exactly one banner with NRANKS ranks.
set -u
[ $# -ge 4 ] || { echo "usage: $0 ATHENA MPIRUN NRANKS OUTDIR [overrides]"; exit 2; }
BIN=$1; M=$2; NP=$3; OUT=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "refusing: $OUT exists and is not empty"; exit 2
fi
"$HERE/../../../frontier/check_build.sh" "$BIN" scaling || exit 2
# relative paths (binary, launcher) must still resolve after the cd below
BIN=$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")
read -r m0 mrest <<< "$M"
[ -f "$m0" ] && M="$(cd "$(dirname "$m0")" && pwd)/$(basename "$m0") $mrest"
mkdir -p "$OUT/athena"; OUT=$(cd "$OUT" && pwd)
cp "$HERE/selection.athinput" "$OUT/athena/"
{ echo "bin=$BIN"; sha256sum "$BIN"; echo "launch=$M ${MPIRUN_NP_FLAG:--np} $NP"
  echo "source=$(git -C "$HERE" rev-parse HEAD 2>&1)"; } > "$OUT/athena/run_info.txt"
TO=${RUN_TIMEOUT:-1200}
STEP_LIMIT=""
case $M in srun*) STEP_LIMIT="--time=$(( (TO + 59)/60 + 1 ))" ;; esac
t0=$SECONDS
(cd "$OUT/athena" && timeout --kill-after=60 "$TO" $M $STEP_LIMIT ${MPIRUN_NP_FLAG:--np} "$NP" \
   "$BIN" -i selection.athinput "$@" > log 2>&1)
rc=$?
banner=$(grep -a '^Number of parallel ranks = ' "$OUT/athena/log" | tr '\n' ' ')
echo "athena: exit $rc, $((SECONDS - t0)) s, $NP ranks requested; banner: ${banner:-none}"
[ $rc -eq 0 ] || { tail -5 "$OUT/athena/log"; exit 1; }
[ "$banner" = "Number of parallel ranks = $NP " ] ||
  { echo "rank banner is not exactly one with $NP ranks (serial copies?)"; exit 1; }
for f in bin/sel.hydro_w.00001.bin sf_scalar/sel.00001.sf sf_longtrans/sel.00001.sf; do
  [ -s "$OUT/athena/$f" ] || { echo "missing $OUT/athena/$f"; exit 1; }
done
grep -ah '^ncell_selected=\|^ncell_region=' "$OUT/athena/sf_scalar/sel.00001.sf" | tr '\n' ' '; echo
