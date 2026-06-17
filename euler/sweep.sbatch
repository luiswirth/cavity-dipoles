#!/bin/bash
#SBATCH --job-name=epgp-sweep
#SBATCH --account=ls_math
#SBATCH --output=epgp_results/sweep-%A_%a.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --constraint=EPYC_7742
##SBATCH --exclusive
#
# One n_spectral per array task (19 points in NS_SWEEP). Geometry is the 1st arg.
#   sbatch --array=0-18 euler/sweep.sbatch ellipse
# Then merge the per-ns fragments into manifest.csv:
#   uv run epgp-convergence --geometry ellipse --collect
# euler/submit.sh chains both automatically. For the big regeneration, add
# --exclusive so each task gets a contention-free node (clean timing/memory).
# EP-GP does not scale past ~8-16 cores (medium dense linear algebra); the win
# here is job-level parallelism across the sweep, not more cores per task.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
GEOM=${1:-ellipse}
C=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$C OPENBLAS_NUM_THREADS=$C MKL_NUM_THREADS=$C NUMEXPR_NUM_THREADS=$C
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export PYTHONUNBUFFERED=1
mkdir -p epgp_results
# --cpu-bind=cores pins the step's affinity to its allocated cores so JAX sizes
# its threadpool to $C instead of the whole node (avoids oversubscription).
srun --cpu-bind=cores uv run epgp-convergence --geometry "$GEOM" --index "$SLURM_ARRAY_TASK_ID"
