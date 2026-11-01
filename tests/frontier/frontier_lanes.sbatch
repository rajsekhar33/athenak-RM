#!/bin/bash
#SBATCH -A AST238
#SBATCH -J sf-spec-lanes
#SBATCH -p service
#SBATCH -q develop
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
# HIP correctness lanes on one node (develop QOS: at most 4 nodes and 2 h, isolated
# nodes). BIN must be a HIP + MPI build with -DPROBLEM=sf_unit_test.
# usage (from this directory): sbatch --export=ALL,BIN=/path/athena,ATHENA_RESEARCH=<checkout> \
#   frontier_lanes.sbatch
set -u
source "$SLURM_SUBMIT_DIR/frontier_env.sh" || { echo "FRONTIER LANES: FAIL (environment)"; exit 1; }
date; module -t list 2>&1 | tr '\n' ' '; echo
"$SLURM_SUBMIT_DIR/frontier_lanes.sh" "$SLURM_SUBMIT_DIR/lanes_${SLURM_JOB_ID}" "$BIN"
