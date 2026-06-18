#!/bin/bash
# Pull solver outputs from Euler into cavity-benchmark/out/.
# Run from the cavity-benchmark root.
set -euo pipefail
EULER=euler
REMOTE=$EULER:~/semproj

rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-bem/out/ellipse/"  out/bem/ellipse/
rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-epgp/out/ellipse/" out/epgp/ellipse/
rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-epgp/out/sphere/"  out/epgp/sphere/
