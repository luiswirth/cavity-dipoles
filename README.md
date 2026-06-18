# cavity-benchmark

Cross-validation harness for the cavity reaction-operator benchmark.
Generates shared config, aggregates solver outputs, produces figures.

## Workflow

    uv run gen-config               # write config -> res/ (rarely needed)
    # run solvers: euler/submit.sh in each solver repo on Euler
    ./pull-euler.sh                 # rsync results from Euler
    uv run make-figures             # aggregate + all figures
    # in epgp-thesis:
    ./pull-results.sh               # copy figures/CSVs into thesis res/

## Geometries

- `ellipse` (4,4,6): EP-GP cross-validated against BEM reference.
- `sphere` (4,4,4): EP-GP validated against exact analytic operator.
