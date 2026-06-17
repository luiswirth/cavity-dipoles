#!/bin/bash
#SBATCH --job-name=cd-array
#SBATCH --account=ls_math
#SBATCH --output=slurm-%A_%a.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --constraint=EPYC_7742
##SBATCH --exclusive
#
# Run any --index-aware entry point as a SLURM array (one index per task):
#   sbatch --array=0-18 euler/array.sbatch epgp-convergence --geometry ellipse
#   sbatch --array=0-4  euler/array.sbatch epgp-sweep       --geometry ellipse
#   sbatch --array=0-9  euler/array.sbatch epgp-resonance   --geometry ellipse --nchunks 10
# Then merge the per-task fragments (cheap; login node or a small job):
#   uv run <same entrypoint + args> --collect
# euler/submit.sh chains the array + collect automatically. Add --exclusive
# (via SBATCH_EXTRA in submit.sh) for the big regeneration: clean timing/memory.
# EP-GP does not scale past ~8-16 cores; the win is parallelism across the sweep.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
C=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$C OPENBLAS_NUM_THREADS=$C MKL_NUM_THREADS=$C NUMEXPR_NUM_THREADS=$C
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export PYTHONUNBUFFERED=1
mkdir -p epgp_results
# --cpu-bind=cores pins affinity to the allocation so JAX sizes its threadpool
# to $C instead of the whole node (avoids oversubscription).
srun --cpu-bind=cores uv run "$@" --index "$SLURM_ARRAY_TASK_ID"
