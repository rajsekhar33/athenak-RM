#!/bin/bash
# Distributed FFT test matrix: rank counts, crops, transports, 1D/2D/3D, non-cubic.
# usage: tests/fft/run_pencil_suite.sh ATHENA MPIRUN OUTDIR [MAXNP]
# ATHENA must be built with -D PROBLEM=sf_unit_test.
set -u
BIN=$1; MPIRUN=$2; OUT=$3; MAXNP=${4:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
INP=$HERE/../../inputs/tests/fft_pencil_test.athinput
mkdir -p "$OUT"
fail=0
run() {   # name np overrides...
  local name=$1 np=$2; shift 2
  [ "$np" -gt "$MAXNP" ] && return
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  # write the overrides into the copied input so the checker sees the same mesh
  cp "$INP" "$d/in.athinput"
  for kv in "$@"; do
    local blk=${kv%%/*} rest=${kv#*/}; local key=${rest%%=*} val=${rest#*=}
    awk -v b="<$blk>" -v k="$key" -v v="$val" '
      /^</ {inb = ($1 == b)} inb && $1 == k {print k " = " v; next} {print}' \
      "$d/in.athinput" > "$d/tmp" && mv "$d/tmp" "$d/in.athinput"
  done
  (cd "$d" && $MPIRUN -np "$np" "$BIN" -i in.athinput > log 2>&1)
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$name np=$np] athena exit $rc FAIL"; fail=1; return
  fi
  if [ "$(grep -ac 'Number of parallel ranks' "$d/log")" -ne 1 ] ||
     ! grep -aq "Number of parallel ranks = $np\$" "$d/log"; then
    echo "[$name] binary did not run on $np MPI ranks (serial build or wrong mpirun): FAIL"
    fail=1; return
  fi
  local res
  res=$(python3 -I "$HERE/compare_pencil.py" "$d" 2>&1)
  local crc=$?
  res=$(echo "$res" | tail -1)
  local info; info=$(grep -m1 '^pencil_fft_info' "$d/log" | cut -d' ' -f2-)
  echo "[$name np=$np] $info $res"
  if [ $crc -ne 0 ] || [[ "$res" != *PASS* ]]; then fail=1; fi
  if [[ "$info" != "nranks=$np "* ]]; then
    echo "[$name np=$np] transform reports '$info', not $np ranks: FAIL"; fail=1
  fi
  # with np > 1 some data must cross ranks (a case confined to one rank tests nothing)
  local ch0=${info##*max_chunks_per_message=}
  if [ "$np" -gt 1 ] && ! [ "${ch0:-0}" -gt 0 ] 2>/dev/null; then
    echo "[$name np=$np] no remote MPI message was sent: FAIL"; fail=1
  fi
  # chunk cases: some remote message must need more chunks than the 100 tags it cycles
  if [ -n "${MINCHUNKS:-}" ]; then
    local ch=${info##*max_chunks_per_message=}
    if ! [ "${ch:-0}" -gt "$MINCHUNKS" ] 2>/dev/null; then
      echo "[$name np=$np] max chunks per message ${ch:-?} not above $MINCHUNKS: FAIL"; fail=1
    fi
  fi
}
C="problem/fft_lo1=1 problem/fft_hi1=14 problem/fft_lo2=3 problem/fft_hi2=16 problem/fft_lo3=2 problem/fft_hi3=11"
for np in 1 2 3 4 6 8; do
  run full_np$np $np
  run crop_np$np $np $C
  run host_np$np $np problem/fft_host_stage=true $C
done
NC="mesh/nx1=32 mesh/nx2=12 mesh/nx3=8 meshblock/nx1=8 meshblock/nx2=4 meshblock/nx3=4 problem/fft_hi1=32 problem/fft_hi2=12 problem/fft_hi3=8"
run noncubic_np5 5 $NC
run noncubic_crop_np7 7 $NC problem/fft_hi1=31 problem/fft_lo2=1 problem/fft_hi2=12 problem/fft_hi3=7
TWO="mesh/nx3=1 meshblock/nx3=1 problem/fft_hi3=1"
run twod_np4 4 $TWO
run twod_crop_np3 3 $TWO problem/fft_lo1=2 problem/fft_hi1=15 problem/fft_lo2=1 problem/fft_hi2=10
ONE="mesh/nx2=1 meshblock/nx2=1 mesh/nx3=1 meshblock/nx3=1 problem/fft_hi2=1 problem/fft_hi3=1 mesh/nx1=64 meshblock/nx1=8 problem/fft_hi1=64"
run oned_np4 4 $ONE
run oned_crop_np2 2 $ONE problem/fft_lo1=5 problem/fft_hi1=60
# one-double MPI messages: hundreds of chunks per remote message, so the tag cycles
# (tag_base + k % 100) repeat within one peer pair; device and host staging
MINCHUNKS=100 run chunks_np4 4 problem/fft_max_msg=1
MINCHUNKS=100 run chunks_crop_np2 2 problem/fft_max_msg=1 $C
MINCHUNKS=100 run chunks_host_np3 3 problem/fft_max_msg=1 problem/fft_host_stage=true
# more ranks than useful pencils (Ny = 2 after the crop limits Py and Pz)
run fewlines_np8 8 problem/fft_lo2=7 problem/fft_hi2=9
if [ $fail -eq 0 ]; then echo "PENCIL SUITE: PASS"; else echo "PENCIL SUITE: FAIL"; fi
exit $fail
