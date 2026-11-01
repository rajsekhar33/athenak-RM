#!/bin/bash
# Refuses an executable that is not the intended build, before any launch.
# usage: check_build.sh ATHENA KIND      KIND = scaling | lanes | any
#   scaling: the default build (PROBLEM=built_in_pgens: driven turbulence);
#   lanes:   -DPROBLEM=sf_unit_test (the test problem generator);
#   any:     either (the placement check).
# Always required: CMakeCache.txt next to the binary (../CMakeCache.txt), MPI on, and
# the GPU backend REQUIRE_BACKEND (default HIP, which also requires
# Kokkos_ARCH_AMD_GFX90A; CUDA for della tests; none skips the backend check).
# The CMake cache must also be older than the binary it describes.
# Prints the relevant cache lines; exit 1 with "BUILD CHECK: FAIL (...)" otherwise.
set -u
BIN=$1; KIND=$2
fail() { echo "BUILD CHECK: FAIL ($*)"; exit 1; }
[ -x "$BIN" ] || fail "$BIN missing or not executable"
cache=$(cd "$(dirname "$BIN")/.." && pwd)/CMakeCache.txt
[ -f "$cache" ] || fail "no $cache: cannot tell what $BIN was built with"
[ "$cache" -nt "$BIN" ] &&
  fail "$cache is newer than $BIN: the binary may predate its configuration (rebuild)"
v() { sed -n "s/^$1:[A-Z]*=//p" "$cache" | head -1; }
on() { case $(v "$1" | tr '[:lower:]' '[:upper:]') in ON|TRUE|1|YES) return 0 ;; esac; return 1; }
echo "build: $BIN"
echo "build: PROBLEM=$(v PROBLEM) Athena_ENABLE_MPI=$(v Athena_ENABLE_MPI)" \
     "HIP=$(v Kokkos_ENABLE_HIP) CUDA=$(v Kokkos_ENABLE_CUDA) GFX90A=$(v Kokkos_ARCH_AMD_GFX90A)"
on Athena_ENABLE_MPI || fail "MPI is off (a serial build under srun runs independent copies)"
case ${REQUIRE_BACKEND:-HIP} in
  HIP)  on Kokkos_ENABLE_HIP || fail "Kokkos_ENABLE_HIP is off"
        on Kokkos_ARCH_AMD_GFX90A || fail "Kokkos_ARCH_AMD_GFX90A is off" ;;
  CUDA) on Kokkos_ENABLE_CUDA || fail "Kokkos_ENABLE_CUDA is off" ;;
  none) ;;
  *) fail "unknown REQUIRE_BACKEND=${REQUIRE_BACKEND}" ;;
esac
case $KIND in
  scaling) [ "$(v PROBLEM)" = built_in_pgens ] ||
             fail "PROBLEM=$(v PROBLEM): scaling needs the default build (built_in_pgens)" ;;
  lanes)   [ "$(v PROBLEM)" = sf_unit_test ] ||
             fail "PROBLEM=$(v PROBLEM): the lanes need -DPROBLEM=sf_unit_test" ;;
  any) ;;
  *) fail "unknown KIND=$KIND" ;;
esac
echo "BUILD CHECK: PASS ($KIND, backend ${REQUIRE_BACKEND:-HIP})"
