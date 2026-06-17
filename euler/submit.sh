#!/bin/bash
# Submit the EP-GP convergence sweep as a SLURM array, then collect the per-ns
# fragments into manifest.csv once the array finishes (afterok dependency).
#
#   euler/submit.sh ellipse                 # default 16 cores/task
#   euler/submit.sh sphere --exclusive      # extra args forwarded to the array
#
# Aggregation (which needs the BEM reference too) stays a local post-step: pull
# epgp_results/ back, arrange BEM+EP-GP under cavity-dipoles/out/, run aggregate.
set -euo pipefail
GEOM=${1:-ellipse}; shift || true
jid=$(sbatch --parsable --array=0-18 "$@" euler/sweep.sbatch "$GEOM")
echo "sweep array: $jid"
sbatch --dependency=afterok:"$jid" --account=ls_math --job-name=epgp-collect \
       --time=00:15:00 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=4G \
       --output="epgp_results/collect-%j.log" \
       --wrap "export PATH=\$HOME/.local/bin:\$PATH; cd ~/cavity-dipoles; uv run epgp-convergence --geometry $GEOM --collect"
echo "collect queued (afterok:$jid)"
