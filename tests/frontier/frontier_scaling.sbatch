#!/bin/bash
#SBATCH -A AST238
#SBATCH -J sf-spec-scaling
#SBATCH -p batch
#SBATCH -t 01:00:00
#SBATCH -o %x-%j.out
# Frontier scaling of the on-the-fly sf/spec outputs: base vs sf vs spec vs both, same
# binary, 8 GCDs per node, one rank per GCD, production launch style.
# usage (submit from this directory):
#   sbatch -N NODES --export=ALL,BIN=/path/athena,NX=1024,MB=128[,TESTBIN=...,NLIM=...] \
#          frontier_scaling.sbatch
# Preflight, before any timed run (the job stops if one fails):
#   0. the environment (frontier_env.sh) and the build (check_build.sh: default build,
#      MPI, HIP, gfx90a);
#   1. device placement of all ranks with BIN (check_binding.sh);
#   2. with 2 or more nodes: TESTBIN (a PROBLEM=sf_unit_test build) is required, and the
#      spectrum reference runs on 16 ranks over 2 nodes, device and host transport
#      (spec_unit_test). A pass writes xnode_accepted.txt (job id, binary hashes).
#      XNODE_PREFLIGHT=0 skips it only with XNODE_ACCEPTED=<an earlier job's
#      xnode_accepted.txt> whose hashes match BIN and TESTBIN.
# Weak scaling at 256^3 per GCD: -N 1 NX=512, -N 8 NX=1024, -N 64 NX=2048, MB=128.
# Strong scaling: constant MB per curve with at least one MeshBlock per rank
# (run_variants.sh refuses fewer) and memory headroom (run_variants.sh refuses a lower
# bound above 90% of a GCD), e.g. NX=1024 MB=128 on 8..64 nodes, MB=64 on 8..128 nodes;
# NX=1024 on one node does not fit.
# Policy (OLCF batch): 1-91 nodes max 2 h, 92-183 nodes max 6 h, 184+ nodes max 12 h;
# four eligible-to-run jobs per user.
# Optional env: VARIANTS, REPEAT (default 2), PHYS=mhd, BUDGET=light,
#   SAMPLING=sqrtp|stress, TRANSPORT=host, SELECT=1, RUN_TIMEOUT,
#   LAUNCH_STYLE=closest (srun -c 7 --gpus-per-task=1 --gpu-bind=closest instead of
#   --kokkos-map-device-id-by=mpi_rank).
set -u
source "$SLURM_SUBMIT_DIR/frontier_env.sh" || { echo "PREFLIGHT FAILED: environment"; exit 1; }
date; echo "ROCM_PATH=${ROCM_PATH:-unset}"; module -t list 2>&1 | tr '\n' ' '; echo
NRANK=$((SLURM_NNODES*8))
LAUNCH="srun -N $SLURM_NNODES -n $NRANK"
if [ "${LAUNCH_STYLE:-production}" = closest ]; then
  export SRUN_EXTRA="-c 7 --gpus-per-task=1 --gpu-bind=closest"
  LAUNCH="$LAUNCH $SRUN_EXTRA"; export ATHENA_ARGS=""
fi
OUT=$SLURM_SUBMIT_DIR/run_N${NX}_n${NRANK}_${SLURM_JOB_ID}
[ -e "$OUT" ] && { echo "PREFLIGHT FAILED: $OUT exists"; exit 1; }
mkdir -p "$OUT"
SRC=$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)
echo "source: $(git -C "$SRC" rev-parse HEAD 2>&1)"
git -C "$SRC" diff --quiet HEAD -- 2> /dev/null ||
  { echo "PREFLIGHT FAILED: tracked files differ from HEAD in $SRC"; exit 1; }
"$SLURM_SUBMIT_DIR/check_build.sh" "$BIN" scaling || { echo "PREFLIGHT FAILED: build"; exit 1; }
XNODE=0
if [ "$SLURM_NNODES" -ge 2 ]; then
  [ -n "${TESTBIN:-}" ] ||
    { echo "PREFLIGHT FAILED: 2 or more nodes need TESTBIN=<sf_unit_test build>"; exit 1; }
  "$SLURM_SUBMIT_DIR/check_build.sh" "$TESTBIN" lanes ||
    { echo "PREFLIGHT FAILED: TESTBIN build"; exit 1; }
  HASHES="bin=$(sha256sum < "$BIN" | cut -d' ' -f1) testbin=$(sha256sum < "$TESTBIN" | cut -d' ' -f1)"
  if [ "${XNODE_PREFLIGHT:-1}" = 0 ]; then
    [ -f "${XNODE_ACCEPTED:-}" ] && grep -qx "PASS $HASHES" "$XNODE_ACCEPTED" ||
      { echo "PREFLIGHT FAILED: XNODE_PREFLIGHT=0 needs XNODE_ACCEPTED=<xnode_accepted.txt> of an earlier pass with these binaries ($HASHES)"; exit 1; }
  else
    XNODE=1
  fi
fi
"$SLURM_SUBMIT_DIR/check_binding.sh" "$NRANK" "$BIN" "$OUT/preflight_binding" |
  tee "$OUT/preflight_binding.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "PREFLIGHT FAILED: placement"; exit 1; }
if [ "$SLURM_NNODES" -ge 2 ] && [ $XNODE = 0 ]; then
  echo "preflight: cross-node spectrum SKIPPED (XNODE_PREFLIGHT=0); accepted earlier in" \
       "$XNODE_ACCEPTED:"; sed 's/^/  /' "$XNODE_ACCEPTED"
elif [ $XNODE = 1 ]; then
  d=$OUT/preflight_spec_np16
  mkdir -p "$d"; cp "$SRC/inputs/tests/spec_unit_test.athinput" "$d/in.athinput"
  (cd "$d" && timeout --kill-after=60 900 srun -N 2 -n 16 --time=17 ${SRUN_EXTRA:-} \
     "$TESTBIN" ${ATHENA_ARGS---kokkos-map-device-id-by=mpi_rank} -i in.athinput > log 2>&1) &&
    grep -q "Number of parallel ranks = 16$" "$d/log" &&
    python3 "$SRC/tests/spec/spec_reference.py" "$d" | tee "$OUT/preflight_spec_np16.txt" |
      tail -1 | grep -q "SPEC REFERENCE: PASS" ||
    { echo "PREFLIGHT FAILED: cross-node spectrum (see $d)"; exit 1; }
  echo "preflight: cross-node spectrum on 2 nodes x 8 ranks PASS (device and host transport)"
  { echo "job $SLURM_JOB_ID $(date -Is) $d"; echo "PASS $HASHES"; } > "$OUT/xnode_accepted.txt"
fi
LAUNCH="$LAUNCH" "$SLURM_SUBMIT_DIR/run_variants.sh" "$BIN" "$NRANK" "$NX" "$MB" \
  "${NLIM:-40}" "${DCYCLE:-10}" "$OUT"
