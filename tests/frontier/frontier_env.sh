# Frontier environment for the sf/spec kit (source it; copied from the production
# turbulence job script). Use the same modules the binary was built with.
# Returns non-zero, with a message, if a required module operation fails or the
# resulting module list lacks a required module; callers use
#   source frontier_env.sh || exit 1
_fe_fail=0
_fe() { "$@" || { echo "frontier_env.sh: '$*' failed" >&2; _fe_fail=1; }; }
_fe module restore
_fe module load PrgEnv-cray
_fe module load craype-accel-amd-gfx90a
_fe module load cmake
_fe module load cray-python
_fe module swap cce cce/19.0.0
_fe module load rocm/6.4.2
_fe module swap cray-mpich cray-mpich/8.1.31
module unload darshan-runtime 2> /dev/null   # optional: absent on some nodes

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_CACHE_MAX_SIZE=1000
export MPICH_MPIIO_HINTS="*:romio_cb_write=disable"
export FI_MR_CACHE_MONITOR=disabled
export MPICH_SMP_SINGLE_COPY_MODE=NONE
# production launch style: every rank sees all 8 GCDs, Kokkos picks one by rank
export ATHENA_ARGS=${ATHENA_ARGS---kokkos-map-device-id-by=mpi_rank}

# the module list must show what was asked for (Lmod can return 0 after a partial swap)
_fe_list=$(module -t list 2>&1)
for _fe_m in PrgEnv-cray craype-accel-amd-gfx90a cce/19.0.0 rocm/6.4.2 cray-mpich/8.1.31 \
             cray-python; do
  grep -q "^${_fe_m}" <<< "$_fe_list" ||
    { echo "frontier_env.sh: module ${_fe_m} not loaded" >&2; _fe_fail=1; }
done
[ "$_fe_fail" -eq 0 ] && echo "frontier_env.sh: modules OK ($(tr '\n' ' ' <<< "$_fe_list"))"
unset -f _fe; unset _fe_list _fe_m
( exit $_fe_fail ) && unset _fe_fail || { unset _fe_fail; false; }
