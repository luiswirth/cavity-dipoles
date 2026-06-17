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

    euler/submit.sh 0-41 epgp-convergence --geometry ellipse   # 7 n_spectral x 6 n_boundary
    euler/submit.sh 0-4  epgp-sweep       --geometry ellipse
    euler/submit.sh 0-9  epgp-resonance   --geometry ellipse --nchunks 10

## Three independent measurements

These measure different things and have different node requirements. The key
fact: peak RSS is per-process and contention-independent, so memory is valid
from any run and is recorded everywhere; only wall-time is polluted by sharing a
node, so only the runtime measurement needs --exclusive.

1. 2D convergence grid (accuracy + memory). EP-GP (n_spectral x n_boundary,
   42 points, `0-41`), BEM (p x m, 20 points). Shared nodes -- accuracy
   (selfconv, err, recip, cond) and memory (maxrss) do not depend on contention.
   The grid's `secs` is informational only; the authoritative timing is (3). No
   --warmup here.

       euler/submit.sh 0-41 epgp-convergence --geometry ellipse
       euler/submit.sh 0-41 epgp-convergence --geometry sphere
       # BEM: sbatch --array=1-20 euler/run.sbatch

2. Single high-fidelity operator (the reported result + cross-validation).
   The high corner of each grid: EP-GP (max n_spectral, max n_boundary),
   BEM (p5/m4). Already produced as the corner point of (1) -- no separate run
   needed; it is just the operator we report and cross-validate. (The BEM
   *reference* for EP-GP's err is the finer p6/m4, run via euler/bem_ref.txt,
   distinct from the BEM high-fidelity grid point p5/m4.)

3. Runtime comparison (fair wall-time only). Re-run the configuration being
   timed on a contention-free node, with compile excluded. --exclusive (clean
   timing) + --warmup (secs excludes JAX/XLA compile). Matched 16 cores on the
   same EPYC_7742 node for both solvers. Time the single high-fidelity config of
   each (what the benchmark actually delivers):

       SBATCH_EXTRA="--exclusive" euler/submit.sh 41-41 epgp-convergence \
           --geometry ellipse --warmup
       # BEM: sbatch --exclusive --array=20 euler/run.sbatch   # p5/m4

Or manually: `sbatch --array=0-41 euler/array.sbatch epgp-convergence --geometry
ellipse`, then `uv run epgp-convergence --geometry ellipse --collect`.

Whole-sweep-in-one-job (simpler, slower): `sbatch --array=1-2 euler/run.sbatch`
(1=ellipse, 2=sphere). Note: this runs the whole grid in one process, so the
per-point maxrss is not meaningful (only the global peak). Use the array path
above for memory.

Threads are pinned to the allocation via `srun --cpu-bind=cores` (else JAX
oversubscribes the node's full core count).

Results land in `out/<geom>_epgp/` (per-grid-point operators T_epgp_*.npy +
`manifest.csv` + `provenance.json`). The manifest records only what cannot be
reconstructed from the saved operators: dofs, secs, cond, maxrss (log_noise is a
fixed input -> provenance.json, not a per-row value).
Derived quantities (norm, recip, selfconv, err) are computed in post-processing
by results.aggregate from the saved T's. Pull the directory back; aggregation
(which also needs the BEM reference) is a local step.
