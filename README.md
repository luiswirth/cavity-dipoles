# cavity-dipoles

Orchestration + shared physics + EP-GP driver for the cavity reaction-operator
benchmark. Generates the shared config consumed by both solvers, owns the
analytic physics, drives the EP-GP, and aggregates/plots results. uv-managed;
always `uv run`.

## Studies (entry points)

- `gen-config` -- write the shared config (Lambda points, tangent frames, k,
  semi-axes) for a geometry. Single source of truth for both solvers.
- `epgp-operator` -- assemble one EP-GP reaction operator, or a `field` slice.
- `epgp-convergence` -- the main accuracy study: a 2D grid over
  `(n_spectral, n_boundary)`, mirroring the BEM `(p, m)` grid. The best operator
  is the high corner of both axes. Records recip, self-convergence vs the corner,
  err vs reference, cond, wall-time, peak memory.
- `epgp-sweep` -- wavenumber x n_spectral (band-limit / ksweep diagnostic).
- `epgp-resonance` -- subspace-angle resonance locator over k (certifies k=2 is
  clear of interior PEC resonances).
- `aggregate` -- combine manifests + saved operators into `results.csv`
  (BEM reference for the ellipse, analytic multipole for the sphere).
- `make-figures` -- aggregate + all convergence/field figures.

## Two geometries

- `ellipse` (4,4,6): no closed form; EP-GP cross-validated against the BEM
  reference (`cavity-bem`).
- `sphere` (4,4,4): exact analytic multipole reference (`sphere.py`); EP-GP
  measured directly against it.

## Convergence axes

- BEM: poly-degree `p` (spectral) and refinement `m` (mesh). 2D `(p,m)` grid.
- EP-GP: `n_spectral` (plane-wave richness) and `n_boundary` (collocation
  density). 2D grid. The n_boundary axis behaves like a boundary-quadrature
  error (~1/n_boundary), which motivates the Lebedev/spherical-design future work.

## Conventions

- EP-GP noise is fixed (`opt_noise=False`, `log_noise=-12`); no per-run tuning.
- Each `*-convergence`/`-sweep`/`-resonance` study is SLURM-array friendly:
  `--index` runs one grid point, `--collect` merges per-task fragments.
- Metrics recorded: accuracy (recip / self-conv / err), cost (wall-time, peak
  memory), plus `provenance.json` (commit, params, versions).

## Running

All heavy runs go on ETH Euler now, not locally. See `euler/README.md` for the
array workflow (`euler/run.sbatch`), thread pinning, the
`ls_math` account, and GPU. Pull artifacts back; aggregation + plotting are local.

## Data flow

`gen-config` -> shared config -> {`cavity-bem` (BEM), `epgp-convergence` (EP-GP)}
each compute operators -> `aggregate` compares -> `make-figures` -> figures/CSVs
copied into `epgp-thesis/res/` (manual; thesis tables/figures read them there).
