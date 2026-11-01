#!/bin/bash
# One validation pass over a frozen build: every suite, with SHA-256 hashes of the
# source, binaries, inputs and checkers written next to the lane results.
# usage: tests/release_validation.sh OUTDIR CPU_BIN GPU_BIN SERIAL_BIN MPIRUN
#   all binaries built with -D PROBLEM=sf_unit_test from the same source tree
set -u
OUT=$1; CPU=$2; GPU=$3; SER=$4; M=$5
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
SRC=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
REC=$OUT/record.txt
# Provenance first; any failure stops before a lane is launched. Every tracked file under
# src/, inputs/tests/ and tests/ is hashed, plus the diff against HEAD, so the record ties
# the binaries to an exact source tree (untracked local scripts are not part of it).
die() { echo "RELEASE VALIDATION: FAIL ($*)" | tee -a "$REC"; exit 1; }
{
  echo "release validation $(date -Iseconds) on $(hostname)"
  echo "git HEAD $(git -C "$SRC" rev-parse HEAD)"
} > "$REC" || die "cannot write $REC"
git -C "$SRC" rev-parse HEAD > /dev/null 2>&1 || die "$SRC is not a git checkout"
for b in "$CPU" "$GPU" "$SER"; do
  [ -x "$b" ] || die "binary $b missing or not executable"
done
DIFF=$(git -C "$SRC" diff HEAD -- src inputs/tests tests) || die "git diff failed"
{
  echo "tracked changes against HEAD (src, inputs/tests, tests):"
  git -C "$SRC" status --short --untracked-files=no -- src inputs/tests tests
  echo "sha256 of that diff: $(printf '%s' "$DIFF" | sha256sum | cut -d' ' -f1)"
  echo "## sha256"
} >> "$REC"
(cd "$SRC" && git ls-files -z src inputs/tests tests | xargs -0 sha256sum) >> "$REC" ||
  die "hashing the source tree failed"
sha256sum "$CPU" "$GPU" "$SER" >> "$REC" || die "hashing the binaries failed"
echo "## lanes" >> "$REC"
status=0
lane() {   # name, command...
  local name=$1; shift
  local t0=$SECONDS
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  local last; last=$(tail -1 "$OUT/$name.log")
  printf "%-28s exit=%d %4ds  %s\n" "$name" $rc $((SECONDS - t0)) "$last" | tee -a "$REC"
  [ $rc -ne 0 ] && status=1
}
ranks_ok() {   # log np: exactly one start-up banner, with np ranks
  [ "$(grep -ac 'Number of parallel ranks' "$1")" -eq 1 ] &&
    grep -aq "Number of parallel ranks = $2\$" "$1" ||
    { echo "binary did not run on $2 MPI ranks: FAIL"; return 1; }
}
spec_lane() {   # bin np dir
  rm -rf "$3"; mkdir -p "$3"; cp "$SRC/inputs/tests/spec_unit_test.athinput" "$3/in.athinput"
  (cd "$3" && $M -np "$2" --oversubscribe "$1" -i in.athinput > log 2>&1) &&
    ranks_ok "$3/log" "$2" && python3 "$SRC/tests/spec/spec_reference.py" "$3"
}
lane mpi_ranks_cpu      "$SRC/tests/check_mpi_ranks.sh" "$CPU" "$M" "$OUT/w_ranks_cpu"
lane mpi_ranks_gpu      "$SRC/tests/check_mpi_ranks.sh" "$GPU" "$M" "$OUT/w_ranks_gpu"
lane harness_selftest   "$SRC/tests/harness_selftest.sh" "$GPU" "$M" "$OUT/w_selftest" "$SER"
lane focused_gpu_np1    "$SRC/tests/focused_checks.sh" "$GPU" "$M" 1 "$OUT/w_focused_gpu"
lane focused_cpu_np3    "$SRC/tests/focused_checks.sh" "$CPU" "$M" 3 "$OUT/w_focused_cpu"
lane focused_gpu_np2    "$SRC/tests/focused_checks.sh" "$GPU" "$M" 2 "$OUT/w_focused_gpu2"
lane sf_suite_gpu_np1   "$SRC/tests/sf/run_suite.sh" "$GPU" "$M" 1 "$OUT/w_sf_gpu"
lane sf_suite_cpu_np3   "$SRC/tests/sf/run_suite.sh" "$CPU" "$M" 3 "$OUT/w_sf_cpu"
lane sf_suite_gpu_np2   "$SRC/tests/sf/run_suite.sh" "$GPU" "$M" 2 "$OUT/w_sf_gpu2"
lane pencil_cpu_np8     "$SRC/tests/fft/run_pencil_suite.sh" "$CPU" "$M" "$OUT/w_pencil_cpu" 8
lane pencil_gpu_np4     "$SRC/tests/fft/run_pencil_suite.sh" "$GPU" "$M" "$OUT/w_pencil_gpu" 4
lane spec_cpu_np1       spec_lane "$CPU" 1 "$OUT/w_spec_cpu1"
lane spec_cpu_np3       spec_lane "$CPU" 3 "$OUT/w_spec_cpu3"
lane spec_gpu_np1       spec_lane "$GPU" 1 "$OUT/w_spec_gpu1"
lane spec_gpu_np3       spec_lane "$GPU" 3 "$OUT/w_spec_gpu3"
lane reader_selftest    python3 -I "$SRC/tests/sf/reader_selftest.py" \
  "$(find "$OUT/w_sf_cpu" -name '*.sf' | sort | head -1)" \
  "$(find "$OUT/w_spec_cpu1" -name '*.spec' | sort | head -1)"
lane restart_serial     "$SRC/tests/sf/restart_test.sh" "$SER" "$OUT/w_restart"
lane sparse_cpu_np3       "$SRC/tests/sf/run_sparse.sh" "$CPU" "$M" 3 "$OUT/w_sparse_cpu"
lane sparse_gpu_np2       "$SRC/tests/sf/run_sparse.sh" "$GPU" "$M" 2 "$OUT/w_sparse_gpu"
lane sparse_checker_selftest "$SRC/tests/sf/sparse_checker_selftest.sh" "$OUT/w_sparse_cpu/mask2" "$OUT/w_scs"
lane select_cpu_np3       "$SRC/tests/sf/run_select.sh" "$CPU" "$M" 3 "$OUT/w_select_cpu"
lane select_gpu_np2       "$SRC/tests/sf/run_select.sh" "$GPU" "$M" 2 "$OUT/w_select_gpu"
lane inventory_selftest   "$SRC/tests/sf/inventory_selftest.sh" "$OUT/w_select_cpu/select" "$OUT/w_sparse_cpu/toggle" "$OUT/w_invst"
lane signed_checker_selftest python3 -I "$SRC/tests/sf/signed_checker_selftest.py"
identity_lane() {   # bin np dir: SF2/FFT identity (sf output vs the FFT of the same field)
  rm -rf "$3"; mkdir -p "$3"; cp "$SRC/inputs/tests/sf2_identity.athinput" "$3/in.athinput"
  (cd "$3" && $M -np "$2" --oversubscribe "$1" -i in.athinput > log 2>&1) &&
    ranks_ok "$3/log" "$2" && python3 -I "$SRC/tests/sf2_fft_identity.py" "$3"
}
lane sf2_identity_cpu_np3 identity_lane "$CPU" 3 "$OUT/w_identity_cpu"
lane sf2_identity_gpu_np1 identity_lane "$GPU" 1 "$OUT/w_identity_gpu"
lane sf2_identity_gpu_np2 identity_lane "$GPU" 2 "$OUT/w_identity_gpu2"
lane integration_cpu_2to3 "$SRC/tests/integration_checks.sh" "$CPU" "$M" "$OUT/w_integ_cpu" 2 3
lane integration_gpu_1to2 "$SRC/tests/integration_checks.sh" "$GPU" "$M" "$OUT/w_integ_gpu" 1 2
lane guards_cpu         "$SRC/tests/sf/run_guards.sh" "$CPU" "$M" "$OUT/w_guards_cpu"
lane guards_gpu         "$SRC/tests/sf/run_guards.sh" "$GPU" "$M" "$OUT/w_guards_gpu"
lane restart_check_selftest "$SRC/tests/restart_check_selftest.sh" "$OUT/w_integ_cpu" "$OUT/w_rcs"
echo "RELEASE VALIDATION: $([ $status -eq 0 ] && echo PASS || echo FAIL)" | tee -a "$REC"
exit $status
