#!/bin/bash
# Adapter so the test scripts' "mpirun -np N [--oversubscribe] [-x VAR] BIN args" lines run
# under srun inside a Frontier allocation (one rank per GCD).
# env: SRUN_EXTRA (extra srun options, default none: the production launch style);
#      ATHENA_ARGS (inserted right after the binary, default
#      --kokkos-map-device-id-by=mpi_rank; set it empty with --gpu-bind=closest);
#      LAUNCH_TIMEOUT (seconds, default 1800): SIGTERM to srun, SIGKILL 60 s later, and a
#      Slurm step --time limit one minute beyond it, so a hung step is always ended
np=1; args=()
while [ $# -gt 0 ]; do
  case $1 in
    -np|-n) np=$2; shift 2 ;;
    --oversubscribe) shift ;;
    -x) shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
to=${LAUNCH_TIMEOUT:-1800}
exec timeout --kill-after=60 "$to" srun --time=$(( (to + 59)/60 + 1 )) -n "$np" ${SRUN_EXTRA:-} \
  "${args[0]}" ${ATHENA_ARGS---kokkos-map-device-id-by=mpi_rank} "${args[@]:1}"
