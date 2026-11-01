#!/bin/bash
# HIP correctness lanes for Frontier: the della-vis2 GPU lane list run under srun with
# np = 1, 2 and 8 (one node, one rank per GCD), plus the SF2/FFT identity and a
# changed-rank restart.
# Needs a HIP + MPI build with PROBLEM=sf_unit_test (the test problem generator), and
# python3 with numpy, scipy and h5py (PYENV_BIN is prepended to PATH if set).
# usage (inside a one-node allocation): tests/frontier/frontier_lanes.sh OUTDIR ATHENA
# env: SRUN_EXTRA, ATHENA_ARGS (see srun_np.sh), LAUNCH_TIMEOUT (per launch, default
#      1800 s), REQUIRE_BACKEND (check_build.sh, default HIP), ATHENA_RESEARCH (checkout
#      of athena-research-rm, e.g. a clone of its GitLab main). The reference is pinned
#      by content: spectra.py and the modules it uses must have the SHA-256 hashes
#      della validated (any branch or commit with identical files is accepted)
# OUTDIR must not exist or be empty: results are never overwritten.
# The binding and MPI-rank lanes are fatal: if either fails, no further lane runs.
set -u
OUT=$1; BIN=$2
SRC=$(cd "$(dirname "$0")/../.." && pwd)
M="$SRC/tests/frontier/srun_np.sh"
[ -n "${PYENV_BIN:-}" ] && export PATH=$PYENV_BIN:$PATH
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2> /dev/null)" ]; then
  echo "PREFLIGHT FAILED: $OUT exists and is not empty (use a new directory)"
  echo "FRONTIER LANES: FAIL (preflight)"; exit 1
fi
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
REC=$OUT/record.txt
die() { echo "PREFLIGHT FAILED: $*" | tee -a "$REC"; echo "FRONTIER LANES: FAIL (preflight)" | tee -a "$REC"; exit 1; }

# ---- preflight: stop before any lane if the environment is not what it should be ----
echo "frontier lanes $(date -Iseconds) on $(hostname)" >> "$REC"
[ -x "$BIN" ] || die "binary $BIN missing or not executable"
sha256sum "$BIN" >> "$REC" || die "cannot hash $BIN"
cache=$(dirname "$BIN")/../CMakeCache.txt
[ -f "$cache" ] && grep -E '^(CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS|CMAKE_BUILD_TYPE|Kokkos_ENABLE_[A-Z]+|Kokkos_ARCH_[A-Z0-9_]+|Athena_ENABLE_MPI|PROBLEM):' "$cache" |
  grep -vi ':BOOL=OFF$' | sed 's/^/cmake: /' >> "$REC"
"$SRC/tests/frontier/check_build.sh" "$BIN" lanes >> "$REC" 2>&1 ||
  die "$(tail -1 "$REC")"
echo "source: $(git -C "$SRC" rev-parse HEAD 2>&1)" >> "$REC"
git -C "$SRC" diff --quiet HEAD -- 2> /dev/null ||
  die "tracked files differ from HEAD in $SRC: run the committed kit and source"
git -C "$SRC" submodule status 2>&1 | sed 's/^/submodule: /' >> "$REC"
python3 -c "import sys, numpy, scipy, h5py; print('python', sys.executable, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'h5py', h5py.__version__)" \
  >> "$REC" 2>&1 || die "python3 with numpy, scipy and h5py is needed (set PYENV_BIN)"
# spec_reference.py imports spectra.py from an athena-research-rm checkout
AR=${ATHENA_RESEARCH:-/home/rm3395/analysis_RadShock/athena-research-rm}
python3 -I -c "import sys; sys.path.insert(0, sys.argv[1]); from athena_research.operations import spectra; print('spectra.py imported from', spectra.__file__)" \
  "$AR" >> "$REC" 2>&1 ||
  die "cannot import athena_research.operations.spectra (export ATHENA_RESEARCH=<athena-research-rm checkout>)"
echo "athena-research-rm: $(git -C "$AR" rev-parse HEAD 2>&1)" >> "$REC"
sha256sum "$AR/athena_research/operations/spectra.py" >> "$REC"
# the reference the spectra are compared against, pinned by content
while read -r h f; do
  echo "$h  $AR/$f" | sha256sum --quiet -c - > /dev/null 2>&1 ||
    die "athena-research-rm $f differs from the validated reference (sha256 $h)"
done <<'EOT'
c24e82f4df8dfcd2d92b2999842dc297a1eaf6a82c467588457a0c48b7f9fc8c athena_research/operations/spectra.py
f907b9c76de484c5337dd8e7f07634847203157134f1e0c30b1692df155b0364 athena_research/core/base.py
9e7b4716b8e32058f275720f52b87a88c01247fb3f421906dc1c68c50a47cf3a athena_research/utils/batch_processing.py
EOT
echo "athena-research-rm reference files match the validated hashes" >> "$REC"
command -v srun > /dev/null || die "srun not found (run inside an allocation)"
[ -n "${SLURM_JOB_ID:-}" ] || die "not inside a Slurm allocation"
if command -v module > /dev/null 2>&1; then
  echo "modules: $(module -t list 2>&1 | tr '\n' ' ')" >> "$REC"
fi
command -v hipcc > /dev/null && echo "hipcc: $(hipcc --version 2>&1 | grep -m1 -i 'HIP version')" >> "$REC"
env | grep -E '^(MPICH_|FI_|HSA_|HIP_|ROCM_PATH|CRAY_MPICH_VERSION|PE_ENV)' | sort | sed 's/^/env: /' >> "$REC"
echo "launch: srun_np.sh, SRUN_EXTRA='${SRUN_EXTRA:-}' ATHENA_ARGS='${ATHENA_ARGS---kokkos-map-device-id-by=mpi_rank}' LAUNCH_TIMEOUT=${LAUNCH_TIMEOUT:-1800}" >> "$REC"
echo "## lanes" >> "$REC"

status=0
lane() {
  local name=$1; shift
  local t0=$SECONDS
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  printf "%-22s exit=%d %4ds  %s\n" "$name" $rc $((SECONDS - t0)) "$(tail -1 "$OUT/$name.log")" |
    tee -a "$REC"
  [ $rc -ne 0 ] && status=1
  return $rc
}
fatal() {   # a lane every later result depends on
  lane "$@" && return 0
  echo "FRONTIER LANES: FAIL ($1 failed; later lanes not run)" | tee -a "$REC"; exit 1
}
spec_np() {   # np
  local d=$OUT/w_spec_np$1
  mkdir -p "$d"; cp "$SRC/inputs/tests/spec_unit_test.athinput" "$d/in.athinput"
  (cd "$d" && $M -np "$1" "$BIN" -i in.athinput > log 2>&1) &&
    grep -q "Number of parallel ranks = $1\$" "$d/log" &&
    python3 "$SRC/tests/spec/spec_reference.py" "$d"
}
identity_np() {   # np: SF2/FFT identity (sf output vs the FFT of the same field)
  local d=$OUT/w_identity_np$1
  mkdir -p "$d"; cp "$SRC/inputs/tests/sf2_identity.athinput" "$d/in.athinput"
  (cd "$d" && $M -np "$1" "$BIN" -i in.athinput > log 2>&1) &&
    grep -q "Number of parallel ranks = $1\$" "$d/log" &&
    python3 -I "$SRC/tests/sf2_fft_identity.py" "$d"
}
# device placement first: a wrong placement makes every later timing and result suspect
fatal binding        "$SRC/tests/frontier/check_binding.sh" 8 "$BIN" "$OUT/w_binding"
fatal mpi_ranks      "$SRC/tests/check_mpi_ranks.sh" "$BIN" "$M" "$OUT/w_ranks"
lane pencil_np8      "$SRC/tests/fft/run_pencil_suite.sh" "$BIN" "$M" "$OUT/w_pencil" 8
lane spec_np1        spec_np 1
lane spec_np8        spec_np 8
lane identity_np1    identity_np 1
lane identity_np8    identity_np 8
lane focused_np1     "$SRC/tests/focused_checks.sh" "$BIN" "$M" 1 "$OUT/w_focused1"
lane focused_np2     "$SRC/tests/focused_checks.sh" "$BIN" "$M" 2 "$OUT/w_focused2"
lane sf_suite_np1    "$SRC/tests/sf/run_suite.sh" "$BIN" "$M" 1 "$OUT/w_sf1"
lane sf_suite_np2    "$SRC/tests/sf/run_suite.sh" "$BIN" "$M" 2 "$OUT/w_sf2"
lane sparse_np2      "$SRC/tests/sf/run_sparse.sh" "$BIN" "$M" 2 "$OUT/w_sparse"
lane select_np2      "$SRC/tests/sf/run_select.sh" "$BIN" "$M" 2 "$OUT/w_select"
lane restart_1to2    "$SRC/tests/integration_checks.sh" "$BIN" "$M" "$OUT/w_integ" 1 2
echo "FRONTIER LANES: $([ $status -eq 0 ] && echo PASS || echo FAIL)" | tee -a "$REC"
exit $status
