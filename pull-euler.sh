#!/bin/bash
# Pull solver outputs from Euler into cavity-benchmark/out/.
# Run from the cavity-benchmark root.
set -euo pipefail
EULER=euler
REMOTE=$EULER:~/semproj

rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-bem/out/grid/ellipse/"  out/bem/grid/ellipse/
rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-bem/out/grid/sphere/"   out/bem/grid/sphere/
rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-bem/out/ref/ellipse/"   out/bem/ref/ellipse/
rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-bem/out/ref/sphere/"    out/bem/ref/sphere/
rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-epgp/out/grid/ellipse/" out/epgp/grid/ellipse/
rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-epgp/out/grid/sphere/"  out/epgp/grid/sphere/
rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-epgp/out/ref/ellipse/"  out/epgp/ref/ellipse/
rsync -av --exclude=work/ --exclude=logs/ "$REMOTE/cavity-epgp/out/ref/sphere/"   out/epgp/ref/sphere/
