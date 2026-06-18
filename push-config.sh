#!/bin/bash
# Copy shared config from cavity-benchmark/res/ into the solver repos.
# Run from the cavity-benchmark root, then commit+push both solver repos.
set -euo pipefail
cp res/config_ellipse.txt ../cavity-epgp/res/config_ellipse.txt
cp res/config_sphere.txt  ../cavity-epgp/res/config_sphere.txt
cp res/config_ellipse.txt ../cavity-bem/res/config_ellipse.txt
