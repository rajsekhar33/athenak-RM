#!/bin/bash
# Repeated-output memory check: one launch writing sf and spec outputs every cycle for
# NLIM cycles, while a sampler records the GPU memory in use per device on the first node
# every MEMDT seconds. Growth between an early and a late window of completed output
# events points at buffers that are not released (e.g. the MPICH IPC cache, plan 5).
# usage (inside an allocation): memcheck.sh ATHENA NRANK NX MB NLIM OUTDIR
#   target workload: memcheck.sh <a> 8 512 128 100 <dir>   (256^3 cells per GCD)
# env: VARIANT (default both), TRANSPORT, PERSISTENT=1 (spec persistent_buffers = true),
#      MEMDT (default 2 s), MEMCMD (default: rocm-smi --showmeminfo vram --csv; on della
#      'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'),
#      MEMFILTER (regex; only data lines matching it count, e.g. 'athena' on a shared
#      node; Frontier compute nodes are exclusive, so the default counts every line),
#      MEMMODE (device, default: physical devices, all of MEMEXPECT = card0..card7 must
#      be sampled in both windows; process: per-process rows such as nvidia-smi pids,
#      at least MEMEXPECT = NRANK processes judged), see memcheck_analyze.py,
#      RUN_TIMEOUT (default 3600 s; SIGTERM, SIGKILL 60 s later, and with srun a step
#      --time limit), REQUIRE_BACKEND (check_build.sh), plus make_input.sh's env.
# Result (last line), from the run and the memory together:
#   MEMCHECK: PASS          the run exited 0, reported NRANK ranks, reached NLIM, wrote
#                           every expected output with the layout's size and finite
#                           values, and no device grew more than 5% between the windows;
#   MEMCHECK: GROWTH        valid run, a device grew more than 5%;
#   MEMCHECK: FAIL          the run or its outputs are not valid (memory still reported);
#   MEMCHECK: INCONCLUSIVE  an expected device unsampled, fewer than 3 valid readings in
#                           a window, or too few processes judged.
# Windows (by output events, from the output files' cycle and modification time): early
# = from the completion of event max(2, NLIM/10) over the next NLIM/5 events; late = the
# last NLIM/5 cadence events. Each device or process is judged on its own (first CSV
# column). Peaks are the largest sampled values, not the exact high-water mark.
# OUTDIR must not exist or be empty.
set -u
BIN=$1; NRANK=$2; NX=$3; MB=$4; NLIM=$5; OUT=$6
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2> /dev/null)" ]; then
  echo "MEMCHECK: FAIL ($OUT exists and is not empty; use a new directory)"; exit 1
fi
"$HERE/check_build.sh" "$BIN" scaling || { echo "MEMCHECK: FAIL (build)"; exit 1; }
[ "$NLIM" -ge 20 ] || { echo "MEMCHECK: FAIL (NLIM >= 20 needed for two windows)"; exit 1; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
MEMCMD=${MEMCMD:-rocm-smi --showmeminfo vram --csv}
TO=${RUN_TIMEOUT:-3600}
NRANK=$NRANK "$HERE/make_input.sh" "$NX" "$MB" "${VARIANT:-both}" "$NLIM" 1 > "$OUT/in.athinput" ||
  { echo "MEMCHECK: FAIL (make_input)"; exit 1; }
[ "${PERSISTENT:-0}" = 1 ] && sed -i 's/^dcycle     = 1$/dcycle     = 1\npersistent_buffers = true/' "$OUT/in.athinput"
cd "$OUT"
LAUNCH=${LAUNCH:-srun -n $NRANK}
STEP_LIMIT=""
case $LAUNCH in srun*) STEP_LIMIT="--time=$(( (TO + 59)/60 + 1 ))" ;; esac
SAMPLER="while true; do echo \"## \$(date +%s.%N)\"; $MEMCMD; sleep ${MEMDT:-2}; done"
case $LAUNCH in
  srun*)  # sampler as an overlapping, time-limited step on the first node
    srun --overlap -N 1 -n 1 --time=$(( (TO + 59)/60 + 3 )) bash -c "$SAMPLER" \
      > mem_samples.txt 2>&1 & ;;
  *) bash -c "$SAMPLER" > mem_samples.txt 2>&1 & ;;
esac
SPID=$!
sleep 3
{ echo "bin=$BIN"; sha256sum "$BIN"; echo "launch=$LAUNCH $STEP_LIMIT athena_args=${ATHENA_ARGS:-}"
  echo "nrank=$NRANK nx=$NX mb=$MB nlim=$NLIM variant=${VARIANT:-both} phys=${PHYS:-hydro}" \
       "transport=${TRANSPORT:-device} persistent=${PERSISTENT:-0} budget=${BUDGET:-production}"
  echo "memcmd=$MEMCMD memfilter=${MEMFILTER:-} memdt=${MEMDT:-2}"; } > memcheck_info.txt
timeout --kill-after=60 "$TO" $LAUNCH $STEP_LIMIT "$BIN" ${ATHENA_ARGS:-} \
  -i in.athinput > log 2>&1
rc=$?
sleep 3
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
echo "athena exit $rc" | tee -a memcheck_info.txt
MEMFILTER=${MEMFILTER:-} MEMMODE=${MEMMODE:-device} python3 -I "$HERE/memcheck_analyze.py" \
  "$OUT" "$rc" "$NRANK" "$NLIM" "${VARIANT:-both}" "${PHYS:-hydro}"
