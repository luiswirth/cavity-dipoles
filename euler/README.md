# Running on Euler

EPGP convergence sweeps on ETH Euler, for a same-hardware runtime comparison
against the BEM reference (`cavity-bem/euler`). Both run on 16 cores of an
EPYC_7742 node.

Jobs run under the `ls_math` shareholder account (`#SBATCH --account=ls_math`).
Omitting it falls back to the `public` guest account, capped at 48 cores /
128 GiB.

## Setup (login node, once)

SSH-agent forwarding must reach GitHub (for cloning this repo and the
`maxwellgp` dependency): check with `ssh -T git@github.com`.

    git clone git@github.com:luiswirth/cavity-dipoles.git   # or git pull
    curl -LsSf https://astral.sh/uv/install.sh | sh         # if uv is missing
    cd ~/cavity-dipoles
    uv sync

For the GPU backend (optional, throughput only): `uv sync --extra gpu` on a GPU
node and submit to a GPU partition; JAX uses the GPU automatically. The
BEM-matched fairness comparison stays on CPU (Bembel is CPU-only).

## Run

Preferred: one task per sweep point (job-level parallelism). EP-GP does not
scale past ~8-16 cores, so this, not more cores per task, is what makes a study
finish fast. `array.sbatch` runs any `--index`/`--collect`-aware entry point;
`submit.sh` chains the array + the collect step (afterok):

    euler/submit.sh 0-18 epgp-convergence --geometry ellipse
    euler/submit.sh 0-4  epgp-sweep       --geometry ellipse
    euler/submit.sh 0-9  epgp-resonance   --geometry ellipse --nchunks 10

Add --exclusive for the big regeneration (clean timing/memory per task):

    SBATCH_EXTRA="--exclusive" euler/submit.sh 0-18 epgp-convergence --geometry sphere

Or manually: `sbatch --array=0-18 euler/array.sbatch epgp-convergence --geometry
ellipse`, then `uv run epgp-convergence --geometry ellipse --collect`.

Whole-sweep-in-one-job (simpler, slower): `sbatch --array=1-2 euler/run.sbatch`
(1=ellipse, 2=sphere).

Threads are pinned to the allocation via `srun --cpu-bind=cores` (else JAX
oversubscribes the node's full core count). Add `--exclusive` for the big
regeneration so each task gets a contention-free node (clean timing/memory).

Results land in `epgp_results/<geom>_epgp/` (per-`n_spectral` operators +
`manifest.csv` with wall-time, cond, recip, norm, maxrss + `provenance.json`).
Pull them back; aggregation (which also needs the BEM reference) is a local step.
