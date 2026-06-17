#!/bin/bash
#SBATCH --job-name=epgp
#SBATCH --account=ls_math
#SBATCH --output=epgp_results/slurm-%A_%a.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --constraint=EPYC_7742
##SBATCH --exclusive
#
# Set up first on a login node:
#   curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if uv missing
#   uv sync   # needs SSH-agent forwarding to clone the maxwellgp git dep
# Submit one geometry per array index (1=ellipse, 2=sphere):
#   sbatch --array=1-2 euler/run.sbatch
# 16 cores on an EPYC_7742 node matches the BEM reference run for a fair
# same-hardware runtime comparison; the cgroup pins both to 16 cores.
# Override the core count for a scaling scan:  sbatch --cpus-per-task=64 ...
# Final headline timing only: pass --exclusive for a contention-free node
# (or uncomment the ##SBATCH line above):     sbatch --exclusive ...
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export PYTHONUNBUFFERED=1   # stream per-ns progress to the log instead of buffering
ROOT=$(pwd); mkdir -p epgp_results
GEOMS=(ellipse sphere)
geom=${GEOMS[$((SLURM_ARRAY_TASK_ID - 1))]}
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK geom=$geom"
# --cpu-bind=cores pins affinity so JAX sizes its threadpool to the allocation.
srun --cpu-bind=cores uv run epgp-convergence --geometry "$geom"
cp -r "out/${geom}_epgp" "epgp_results/${geom}_epgp"
echo "done $geom"
