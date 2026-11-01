#!/bin/bash
# Runs the scaling variants with one launcher, interleaved over repeats, and summarises
# them with collect.py.
# usage: LAUNCH="<launcher with rank count>" run_variants.sh ATHENA NRANK NX MB NLIM DCYCLE OUTDIR
#   e.g. LAUNCH="srun -N 8 -n 64" with ATHENA_ARGS=--kokkos-map-device-id-by=mpi_rank
# env: VARIANTS (default "base sf spec both"); REPEAT (default 2: base, sf, spec, both,
#      base, sf, ... so the baseline scatter is measured next to the feature runs);
#      ATHENA_ARGS (put right after the binary); RUN_TIMEOUT (seconds per launch, default
#      1200; SIGTERM, then SIGKILL 60 s later; with srun also a step --time limit);
#      REQUIRE_BACKEND (check_build.sh, default HIP); GPU_MEM_GIB (memory per rank,
#      default 64: one MI250X GCD); plus make_input.sh's PHYS, BUDGET, SAMPLING,
#      TRANSPORT, SELECT
# Preflight (before any launch): the default (built_in_pgens) MPI build of the required
# backend; NX divisible by MB; at least one MeshBlock per rank; a lower bound on the
# device memory per rank below 90% of GPU_MEM_GIB; no earlier results in OUTDIR.
# The first launch failure or time-out stops the series (the partial summary is kept).
set -u
BIN=$1; NRANK=$2; NX=$3; MB=$4; NLIM=$5; DC=$6; OUT=$7
: "${LAUNCH:?set LAUNCH to the launcher with its rank count}"
HERE=$(cd "$(dirname "$0")" && pwd)
die() { echo "run_variants.sh: $*" >&2; exit 1; }

# ---- preflight: fail before any launch ----
"$HERE/check_build.sh" "$BIN" scaling || die "wrong build (see above)"
[ $((NX % MB)) -eq 0 ] || die "NX=$NX is not a multiple of MB=$MB"
NB=$(( (NX/MB)**3 ))
[ "$NB" -ge "$NRANK" ] || die "$NB MeshBlocks < $NRANK ranks (AthenaK rejects this); use a smaller MB"
[ "$NLIM" -ge "$DC" ] || die "NLIM=$NLIM < DCYCLE=$DC: no output inside the loop"
python3 -c "import numpy" 2>/dev/null || die "python3 with numpy needed for collect.py"
# Lower bound on device memory per rank (not an estimate of the total): hydro
# u0, w0, u1, three fluxes, wl3d, wr3d (8 x 5 components) and forcing force,
# force_tmp1, force_tmp2 (3 x 3), all with nghost = 2, on the busiest rank's blocks; plus,
# when a spectrum runs, 112 bytes per active cell of the rank's share of the mesh (pencils
# 32, exchange 16, local FFT work 32, input 8, Helmholtz coefficients 24). MHD needs more.
MEMCHK=$(python3 - "$NX" "$MB" "$NRANK" "${GPU_MEM_GIB:-64}" "${VARIANTS:-base sf spec both}" <<'PY'
import math, sys
nx, mb, nr, cap = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
spec = any(v in ('spec', 'both') for v in sys.argv[5].split())
bpr = math.ceil((nx // mb) ** 3 / nr)
gib = (bpr * (mb + 4) ** 3 * 49 * 8 + (112 * nx ** 3 / nr if spec else 0)) / 2 ** 30
print('%.1f %d' % (gib, gib <= 0.9 * cap))
PY
)
echo "memory lower bound per rank: ${MEMCHK% *} GiB (limit 90% of ${GPU_MEM_GIB:-64} GiB)"
[ "${MEMCHK#* }" = 1 ] || die "the memory lower bound ${MEMCHK% *} GiB per rank does not fit; use more nodes"
[ -e "$OUT" ] || mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
for f in run_info.txt summary.txt input_both.athinput; do
  [ -e "$OUT/$f" ] && die "$OUT/$f exists: results are never overwritten; use a new OUTDIR"
done
for rep in $(seq 1 "${REPEAT:-2}"); do
  for v in ${VARIANTS:-base sf spec both}; do
    [ -e "$OUT/${v}_r$rep" ] && die "$OUT/${v}_r$rep exists; use a new OUTDIR"
  done
done
NRANK=$NRANK "$HERE/make_input.sh" "$NX" "$MB" both "$NLIM" "$DC" > "$OUT/input_both.athinput" ||
  die "make_input.sh failed"
TO=${RUN_TIMEOUT:-1200}
STEP_LIMIT=""
case $LAUNCH in srun*) STEP_LIMIT="--time=$(( (TO + 59)/60 + 1 ))" ;; esac
{
  echo "date=$(date -Is) host=$(hostname)"
  echo "bin=$BIN"; sha256sum "$BIN"
  cache=$(dirname "$BIN")/../CMakeCache.txt
  if [ -f "$cache" ]; then
    grep -E '^(CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS|CMAKE_BUILD_TYPE|Kokkos_ENABLE_[A-Z]+|Kokkos_ARCH_[A-Z0-9_]+|Athena_ENABLE_MPI|PROBLEM):' "$cache" |
      grep -vi ':BOOL=OFF$' | sed 's/^/cmake: /'
  else
    echo "cmake: $cache not found"
  fi
  echo "nrank=$NRANK nx=$NX mb=$MB meshblocks=$NB blocks_per_rank=$(python3 -c "print($NB/$NRANK)")"
  echo "memory_lower_bound_gib=${MEMCHK% *} gpu_mem_gib=${GPU_MEM_GIB:-64}"
  "$HERE/check_build.sh" "$BIN" scaling | sed 's/^/build_check: /'
  echo "nlim=$NLIM dcycle=$DC repeat=${REPEAT:-2} variants=${VARIANTS:-base sf spec both}"
  echo "phys=${PHYS:-hydro} budget=${BUDGET:-production} sampling=${SAMPLING:-global}" \
       "transport=${TRANSPORT:-device} select=${SELECT:-0}"
  echo "sf budgets: $(grep -E '^(ndraw_block|nanchor_global|nsample_global|nreplicate)' \
       "$OUT/input_both.athinput" | sort -u | tr -s ' ' | tr '\n' ';')"
  echo "launch=$LAUNCH $STEP_LIMIT athena_args=${ATHENA_ARGS:-} run_timeout=$TO"
  command -v module > /dev/null 2>&1 && echo "modules: $(module -t list 2>&1 | tr '\n' ' ')"
  env | grep -E '^(MPICH_|FI_|OMP_|HSA_|HIP_|ROCM_|CUDA_|CRAY_MPICH_VERSION|PE_ENV)' | sort
} > "$OUT/run_info.txt"
failed=0
for rep in $(seq 1 "${REPEAT:-2}"); do
  [ $failed -eq 0 ] || break
  for v in ${VARIANTS:-base sf spec both}; do
    d=$OUT/${v}_r$rep; mkdir -p "$d"
    NRANK=$NRANK "$HERE/make_input.sh" "$NX" "$MB" "$v" "$NLIM" "$DC" > "$d/in.athinput" ||
      die "make_input.sh failed for $v"
    t0=$(date +%s.%N)
    (cd "$d" && timeout --kill-after=60 "$TO" $LAUNCH $STEP_LIMIT "$BIN" ${ATHENA_ARGS:-} \
       -i in.athinput > log 2>&1)
    rc=$?
    t1=$(date +%s.%N)
    echo "exit=$rc wall_launch=$(python3 -c "print($t1 - $t0)")" >> "$d/log"
    [ $rc -eq 124 ] || [ $rc -eq 137 ] && echo "!! $v rep $rep TIMED OUT after $TO s"
    [ $rc -eq 0 ] || { echo "!! $v rep $rep FAILED (exit $rc, see $d/log); stopping the" \
                            "series (resource or launch failures repeat)"; failed=1; break; }
  done
done
python3 -I "$HERE/collect.py" "$OUT" > "$OUT/summary.txt" 2>&1 || failed=1
cat "$OUT/summary.txt"
exit $failed
