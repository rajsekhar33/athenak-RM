#!/bin/bash
# Failure-injection self-test of the Frontier kit: every guard must reject what it is
# meant to reject, before any expensive launch. Cheap; runs without Slurm.
# usage: kit_selftest.sh OUTDIR SCALEBIN LANESBIN SERIALBIN [MPIRUN]
#   SCALEBIN  default (built_in_pgens) MPI GPU build;  LANESBIN  PROBLEM=sf_unit_test
#   MPI GPU build;  SERIALBIN  a non-MPI build, or - to skip that case;  MPIRUN  an
#   mpirun for one small real run_variants series (default: skip it and the
#   collect.py cases that need it). On Frontier, run it on a login node without MPIRUN.
# env: REQUIRE_BACKEND (default HIP; CUDA on della), ATHENA_ARGS for the real run.
# Last line: "KIT SELFTEST: PASS" or "KIT SELFTEST: FAIL (n cases)".
set -u
OUT=$1; SB=$2; LB=$3; XB=$4; MPIRUN=${5:-}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -e "$OUT" ] && { echo "KIT SELFTEST: FAIL ($OUT exists)"; exit 1; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
nfail=0
# expect NAME WANT_RC PATTERN cmd...: WANT_RC 0 or "nz"; PATTERN must appear in the output
expect() {
  local name=$1 want=$2 pat=$3; shift 3
  "$@" > "$OUT/$name.out" 2>&1
  local rc=$? ok=1
  { [ "$want" = 0 ] && [ $rc -ne 0 ]; } && ok=0
  { [ "$want" = nz ] && [ $rc -eq 0 ]; } && ok=0
  grep -q -- "$pat" "$OUT/$name.out" || ok=0
  printf "%-34s rc=%-3s %s\n" "$name" "$rc" "$([ $ok = 1 ] && echo ok || echo "WRONG (want rc $want, '$pat')")"
  [ $ok = 1 ] || nfail=$((nfail + 1))
}

# ---- check_build.sh ----
expect build_scaling_ok        0  "BUILD CHECK: PASS" "$HERE/check_build.sh" "$SB" scaling
expect build_lanes_ok          0  "BUILD CHECK: PASS" "$HERE/check_build.sh" "$LB" lanes
expect build_lanes_as_scaling  nz "scaling needs the default build" "$HERE/check_build.sh" "$LB" scaling
expect build_scaling_as_lanes  nz "lanes need" "$HERE/check_build.sh" "$SB" lanes
[ "$XB" = - ] && echo "(no SERIALBIN: serial-build case skipped)" ||
  expect build_serial          nz "MPI is off" "$HERE/check_build.sh" "$XB" any
mkdir -p "$OUT/nocache/src"; cp "$SB" "$OUT/nocache/src/athena"
expect build_no_cache          nz "no .*CMakeCache.txt" "$HERE/check_build.sh" "$OUT/nocache/src/athena" any
expect build_wrong_backend     nz "is off" env REQUIRE_BACKEND=$([ "${REQUIRE_BACKEND:-HIP}" = HIP ] && echo CUDA || echo HIP) \
  "$HERE/check_build.sh" "$SB" any

# ---- frontier_env.sh with a stand-in module command ----
fake_env() {   # MODE: good | badload | missing
  bash -c 'set -u; MODE=$1
    module() { case "$1" in
      -t) for m in PrgEnv-cray/8.6.0 craype-accel-amd-gfx90a cce/19.0.0 rocm/6.4.2 \
                   cray-mpich/8.1.31 cray-python/3.11.7; do
            [ "$MODE" = missing ] && [ "${m%%/*}" = rocm ] && continue; echo "$m"; done ;;
      load) [ "$MODE" = badload ] && [ "$2" = cmake ] && return 1; return 0 ;;
      *) return 0 ;; esac; }
    source "$2/frontier_env.sh" && echo "ENV OK" || { echo "ENV FAILED"; exit 1; }' _ "$1" "$HERE"
}
expect env_good                0  "ENV OK" fake_env good
expect env_failed_load         nz "module load cmake' failed" fake_env badload
expect env_missing_module      nz "rocm/6.4.2 not loaded" fake_env missing

# ---- run_variants.sh preflight (nothing may launch: LAUNCH would fail loudly) ----
NOLAUNCH=false   # "false ATHENA ..." exits 1: a launch failure
expect rv_wrong_build          nz "wrong build" env LAUNCH="$NOLAUNCH" \
  "$HERE/run_variants.sh" "$LB" 8 512 128 20 10 "$OUT/rv1"
expect rv_too_few_blocks       nz "MeshBlocks <" env LAUNCH="$NOLAUNCH" \
  "$HERE/run_variants.sh" "$SB" 1024 1024 128 20 10 "$OUT/rv2"
expect rv_memory_1024_1node    nz "does not fit" env LAUNCH="$NOLAUNCH" \
  "$HERE/run_variants.sh" "$SB" 8 1024 128 20 10 "$OUT/rv3"
expect rv_memory_1024_8nodes   nz "base rep 1 FAILED" env LAUNCH="$NOLAUNCH" REPEAT=1 VARIANTS=base \
  "$HERE/run_variants.sh" "$SB" 64 1024 128 20 10 "$OUT/rv4"
mkdir -p "$OUT/rv5/base_r1"
expect rv_occupied             nz "exists" env LAUNCH="$NOLAUNCH" \
  "$HERE/run_variants.sh" "$SB" 8 512 128 20 10 "$OUT/rv5"
expect rv_stops_after_failure  nz "stopping the series" env LAUNCH="$NOLAUNCH" \
  "$HERE/run_variants.sh" "$SB" 8 512 128 20 10 "$OUT/rv6"
n=$(ls -d "$OUT"/rv6/*_r* 2> /dev/null | wc -l)
printf "%-34s %s\n" rv_one_launch_only "$([ "$n" = 1 ] && echo ok || { echo "WRONG ($n run dirs)"; })"
[ "$n" = 1 ] || nfail=$((nfail + 1))
expect rv_failed_summary_invalid 0 "SCALING SUMMARY: INVALID" cat "$OUT/rv6/summary.txt"

# ---- other scripts refuse occupied output directories ----
mkdir -p "$OUT/occ"; touch "$OUT/occ/x"
expect bind_occupied           nz "not empty" "$HERE/check_binding.sh" 2 "$SB" "$OUT/occ"
expect mem_occupied            nz "not empty" "$HERE/memcheck.sh" "$SB" 2 64 32 20 "$OUT/occ"
expect lanes_occupied          nz "not empty" "$HERE/frontier_lanes.sh" "$OUT/occ" "$LB"
expect lanes_wrong_build       nz "lanes need" "$HERE/frontier_lanes.sh" "$OUT/lanes_wb" "$SB"
expect mem_failed_run          nz "MEMCHECK: FAIL" env LAUNCH="$NOLAUNCH" MEMCMD="echo dev,used" \
  "$HERE/memcheck.sh" "$SB" 2 64 32 20 "$OUT/mem1"

# ---- frontier_scaling.sbatch: 2 nodes without TESTBIN stop before any launch ----
expect scaling_needs_testbin   nz "need TESTBIN" env SLURM_SUBMIT_DIR="$HERE" SLURM_NNODES=2 \
  SLURM_JOB_ID=selftest$$ BIN="$SB" NX=512 MB=128 \
  bash -c 'module() { case "$1" in -t) printf "%s\n" PrgEnv-cray craype-accel-amd-gfx90a \
    cce/19.0.0 rocm/6.4.2 cray-mpich/8.1.31 cray-python;; esac; return 0; }; export -f module
    srun() { echo "SRUN CALLED"; return 9; }; export -f srun
    git() { [ "$1" = -C ] && shift 2; case "$1" in diff) return 0;; *) echo selftest;; esac; }
    export -f git; bash "$SLURM_SUBMIT_DIR/frontier_scaling.sbatch"'
grep -q "SRUN CALLED" "$OUT/scaling_needs_testbin.out" &&
  { echo "scaling_needs_testbin: srun was called"; nfail=$((nfail + 1)); }
rm -rf "$HERE/run_N512_n16_selftest$$"
# skipping the cross-node preflight needs an earlier pass with the same binaries
scaling() {   # extra env assignments...
  env SLURM_SUBMIT_DIR="$HERE" SLURM_NNODES=2 SLURM_JOB_ID=selftest$$ BIN="$SB" TESTBIN="$LB" \
    NX=512 MB=128 "$@" bash -c 'module() { case "$1" in -t) printf "%s\n" PrgEnv-cray \
      craype-accel-amd-gfx90a cce/19.0.0 rocm/6.4.2 cray-mpich/8.1.31 cray-python;; esac
      return 0; }; export -f module
    srun() { echo "SRUN CALLED"; return 9; }; export -f srun
    git() { [ "$1" = -C ] && shift 2; case "$1" in diff) return 0;; *) echo selftest;; esac; }
    export -f git; bash "$SLURM_SUBMIT_DIR/frontier_scaling.sbatch"'
  local rc=$?
  rm -rf "$HERE/run_N512_n16_selftest$$"
  return $rc
}
H="bin=$(sha256sum < "$SB" | cut -d' ' -f1) testbin=$(sha256sum < "$LB" | cut -d' ' -f1)"
printf 'job 1\nPASS %s\n' "$H" > "$OUT/xnode_ok.txt"
printf 'job 1\nPASS bin=0 testbin=0\n' > "$OUT/xnode_other.txt"
expect xnode_skip_no_record    nz "needs XNODE_ACCEPTED" scaling XNODE_PREFLIGHT=0
expect xnode_skip_other_bins   nz "needs XNODE_ACCEPTED" scaling XNODE_PREFLIGHT=0 \
  XNODE_ACCEPTED="$OUT/xnode_other.txt"
expect xnode_skip_accepted     nz "PREFLIGHT FAILED: placement" scaling XNODE_PREFLIGHT=0 \
  XNODE_ACCEPTED="$OUT/xnode_ok.txt"

# ---- collect.py on a real small series (needs MPIRUN) ----
if [ -n "$MPIRUN" ]; then
  expect rv_real_series        0  "SCALING SUMMARY: VALID" env LAUNCH="$MPIRUN -np 2" REPEAT=1 \
    VARIANTS="base both" BUDGET=light "$HERE/run_variants.sh" "$SB" 2 64 32 10 5 "$OUT/real"
  python3 -I - "$OUT/real" "$OUT" <<'PY'
import glob, os, shutil, sys
src, out = sys.argv[1], sys.argv[2]
def variant(name, mutate):
    d = os.path.join(out, 'col_' + name)
    shutil.copytree(src, d)
    mutate(d)
    return d
def sf_file(d):
    return sorted(glob.glob(os.path.join(d, 'both_r1', 'sf_scalar', '*.sf')))[1]
def header_only(d):
    f = sf_file(d); b = open(f, 'rb').read()
    i = b.index(b'header offset=')
    open(f, 'wb').write(b[:b.index(b'\n', i) + 1])
def truncated(d):
    f = sf_file(d); b = open(f, 'rb').read(); open(f, 'wb').write(b[:-8])
def trailing(d):
    f = sf_file(d); open(f, 'ab').write(b'\0' * 8)
def ranks(d):
    p = os.path.join(d, 'both_r1', 'log'); s = open(p).read()
    open(p, 'w').write(s.replace('Number of parallel ranks = 2', 'Number of parallel ranks = 1'))
for n, m in [('header_only', header_only), ('truncated', truncated), ('trailing', trailing),
             ('rank_banner', ranks)]:
    print(variant(n, m))
PY
  for n in nan inf; do
    cp -a "$OUT/real" "$OUT/col_time_$n"
    sed -i -E "s/(cpu time used\s*=\s*)\S+/\1$n/" "$OUT/col_time_$n/base_r1/log"
    expect "collect_rejects_time_$n" nz "timing unavailable" python3 -I "$HERE/collect.py" \
      "$OUT/col_time_$n"
  done
  # memcheck: one real short run with a stand-in two-card sampler, then the analyzer's
  # fault injection on that run
  expect mem_real_run          0  "MEMCHECK: PASS" env LAUNCH="$MPIRUN -np 2" BUDGET=light \
    MEMCMD="echo device,used; echo card0,1000; echo card1,1000" MEMEXPECT=card0,card1 MEMDT=0.2 \
    "$HERE/memcheck.sh" "$SB" 2 64 32 20 "$OUT/mem_real"
  expect mem_analyzer_faults   0  "MEMCHECK SELFTEST: PASS" python3 -I "$HERE/memcheck_selftest.py" \
    "$OUT/mem_real" "$OUT/mem_faults" 2 20
  for n in header_only truncated trailing rank_banner; do
    expect "collect_rejects_$n"  nz "SCALING SUMMARY: INVALID" python3 -I "$HERE/collect.py" "$OUT/col_$n"
  done
else
  echo "(no MPIRUN: real series and collect.py rejection cases skipped)"
fi
echo "KIT SELFTEST: $([ $nfail = 0 ] && echo PASS || echo "FAIL ($nfail cases)")"
[ $nfail = 0 ]
