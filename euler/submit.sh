#!/bin/bash
# Submit any --index/--collect study as a SLURM array, then merge the per-task
# fragments once it finishes (afterok dependency).
#
#   euler/submit.sh 0-18 epgp-convergence --geometry ellipse
#   euler/submit.sh 0-4  epgp-sweep       --geometry ellipse
#   euler/submit.sh 0-9  epgp-resonance   --geometry ellipse --nchunks 10
#
# Extra sbatch flags (exclusive, throttling, ...) via SBATCH_EXTRA:
#   SBATCH_EXTRA="--exclusive" euler/submit.sh 0-18 epgp-convergence --geometry ellipse
#
# Aggregation (which also needs the BEM reference) stays a local post-step.
set -euo pipefail
RANGE=$1; shift
jid=$(sbatch --parsable ${SBATCH_EXTRA:-} --array="$RANGE" euler/array.sbatch "$@")
echo "array job $jid: $* [$RANGE]"
sbatch --dependency=afterok:"$jid" --account=ls_math --job-name=collect \
       --time=00:15:00 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=4G \
       --output="collect-%j.log" \
       --wrap "export PATH=\$HOME/.local/bin:\$PATH; cd ~/cavity-dipoles; uv run $* --collect"
echo "collect queued (afterok:$jid)"
