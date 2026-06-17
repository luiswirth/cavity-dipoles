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

## Run

    sbatch --array=1-2 euler/run.sbatch                    # 1=ellipse, 2=sphere
    sbatch --array=1-2 --cpus-per-task=64 euler/run.sbatch # core-count scan point
    sbatch --array=1-2 --exclusive euler/run.sbatch        # contention-free final timing

Each task runs `epgp-convergence` for one geometry and copies the per-`n_spectral`
operators and `manifest.csv` (wall-time, cond, recip, norm) into
`epgp_results/<geom>_epgp/`. Pull those back and feed them to `results.aggregate`.
