# cavity-benchmark

Cross-validation harness for the cavity reaction-operator benchmark. It owns the
shared cavity config, the analytic sphere reference operator, and the
aggregation and plotting of both solvers' outputs (and the EPGP uncertainty). It
does no heavy solving and does not run on a cluster; it consumes the operators
produced by [cavity-bem](https://github.com/luiswirth/cavity-bem) and
[cavity-epgp](https://github.com/luiswirth/cavity-epgp).

Two geometries:

- `ellipse` (semi-axes 4, 4, 6): EPGP cross-validated against the BEM reference.
- `sphere` (4, 4, 4): EPGP validated against the exact analytic operator.

## Requirements

- [uv](https://docs.astral.sh/uv/)

`uv` pulls `cavity-epgp` (and transitively `maxwellgp`) from GitHub over https.

## Workflow

```bash
uv run gen-config              # write res/config_{shape}.txt (rarely needed; committed)

# 1. produce operators with each solver (see their READMEs), locally or on a cluster
# 2. collect them into out/{solver}/{mode}/{shape}/:
./pull-euler.sh                # from ETH Euler (default)
REMOTE=.. ./pull-euler.sh      # from local sibling checkouts in semproj/

uv run make-figures            # aggregate CSVs + generate all figures into out/figs/
```

`make-figures` accepts `--skip-anim` and `--skip-field` to skip the slow field
plots. `uv run aggregate --geometry {ellipse,sphere}` runs aggregation alone.

The figures and result CSVs are pulled into the thesis with `epgp-thesis/pull-results.sh`.

## Collecting results

`pull-euler.sh` rsyncs the solver `out/{mode}/{shape}/` dirs (bem: `grid`;
epgp: `grid`, `field`, `ksweep`, `noise`) from `$REMOTE` into
`out/{bem,epgp}/{mode}/{shape}/`. `REMOTE` is any rsync source root holding the two
solver repos: the Euler login node by default, a local parent directory for a fully
local run, or another host.

## Layout consumed

- `out/bem/grid/{shape}/`, `out/epgp/grid/{shape}/` convergence grids (with `manifest.csv`)
- the ellipse BEM reference operator is the most refined grid run, `out/bem/grid/ellipse/T_p5_m4.dat`
- `out/epgp/{field,noise,ksweep}/{shape}/` field slice, noise sweep, conditioning sweep
