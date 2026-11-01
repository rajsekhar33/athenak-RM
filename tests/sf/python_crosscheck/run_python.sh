#!/bin/bash
# Python side of the cross-check: exact (exhaustive) sums plus a sampled estimator
# with NSEED (default 16) seeds, for both modes. Usage: ./run_python.sh [SNAPSHOT] [OUTDIR]
HERE=$(cd "$(dirname "$0")" && pwd)
SNAP=${1:-$HERE/run/main/bin/xchk.hydro_w.00000.bin}
OUT=${2:-$HERE/run/python}
NS1=${NS1:-1024}
NSEED=${NSEED:-16}
mkdir -p "$OUT"
for mode in scalar longtrans; do
  [ -f "$OUT/${mode}_exact.npz" ] || python3 "$HERE/python_sf.py" "$SNAP" --mode $mode \
      --kind exact --out "$OUT/${mode}_exact.npz" 2>&1 | grep "raw pairs"
  for seed in $(seq 0 $((NSEED-1))); do
    python3 "$HERE/python_sf.py" "$SNAP" --mode $mode --kind sampled --seed $seed \
        --ns1 $NS1 --out "$OUT/${mode}_sampled_s${seed}.npz" 2>&1 | grep "raw pairs"
  done
done
