#!/bin/bash
# AthenaK side of the developed comparison: restart RUNDIR/start.rst on 2 GPUs (della-vis2)
# usage: run_athena.sh ATHENA RUNDIR [NP]
OMPI=/usr/local/openmpi/cuda-12.9/4.1.8/nvhpc255
export PATH=$OMPI/bin:$PATH LD_LIBRARY_PATH=$OMPI/lib64:${LD_LIBRARY_PATH:-}
cd "$2" && timeout 1800 mpirun -np "${3:-2}" -x LD_LIBRARY_PATH "$1" -r start.rst > log 2>&1
echo "exit=$?" >> log
tail -1 log
