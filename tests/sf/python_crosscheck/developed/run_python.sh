#!/bin/bash
# Python side of the developed comparison: athena_research sampled structure functions on
# the restart state (python_sf.py --rst), NSEED (default 16) independent seeds per mode.
# NS1 (default 1024) is the per-block draw target; each seed reports its timing.
# usage: run_python.sh RUNDIR OUTDIR
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$1; OUT=$2; NS1=${NS1:-1024}; NSEED=${NSEED:-16}
PYREPO=${ATHENA_RESEARCH:-/home/rm3395/analysis_RadShock/athena-research-rm}
python3 "$HERE/reference_manifest.py" prepare "$RUN" "$OUT" --pyrepo "$PYREPO" \
  --ns1 "$NS1" --nseed "$NSEED"
bins=("$RUN"/bin/*.bin); restarts=("$RUN"/rst/*.rst)
BIN=${bins[0]}; RST=${restarts[0]}
mkdir -p "$OUT"
for mode in scalar longtrans; do
  vars=(); if [ "$mode" = scalar ]; then vars=(--vars velx,vely,velz,dens); fi
  for seed in $(seq 0 $((NSEED-1))); do
    f=$OUT/${mode}_sampled_s${seed}.npz
    if [ ! -f "$f" ]; then
      python3 "$HERE/../python_sf.py" "$BIN" --rst "$RST" --mode "$mode" \
        --kind sampled --seed "$seed" --ns1 "$NS1" --nbins 100 --max_order 3 \
        "${vars[@]}" --pyrepo "$PYREPO" --out "$f" > "$f.log" 2>&1
      tail -1 "$f.log"
    fi
  done
done
python3 "$HERE/reference_manifest.py" finish "$RUN" "$OUT" \
  --ns1 "$NS1" --nseed "$NSEED"
