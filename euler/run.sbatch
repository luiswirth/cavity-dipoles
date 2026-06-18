#!/bin/bash
#SBATCH --job-name=cd-array
#SBATCH --account=ls_math
#SBATCH --output=slurm-%A_%a.log
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --constraint=EPYC_7742
#
#   sbatch --array=0-59 euler/run.sbatch epgp-convergence --geometry ellipse
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpu-bind=cores uv run "$@" --index "$SLURM_ARRAY_TASK_ID"
