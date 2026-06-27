#!/bin/bash
# Collect solver outputs into cavity-benchmark/out/. Run from the repo root.
# Missing/failed sources are warned about and skipped, so one absent run does
# not abort the whole pull.
#
# REMOTE is the rsync source root holding the solver repos. Defaults to ETH
# Euler; override to gather results produced anywhere, e.g.:
#   REMOTE=..              ./pull-euler.sh   # local sibling checkouts (semproj/)
#   REMOTE=other:~/semproj ./pull-euler.sh   # another cluster/host
set -uo pipefail
REMOTE="${REMOTE:-euler:~/semproj}"
rc=0

pull() {  # pull <remote-subpath> <local-dest>
  if rsync -a --mkpath --exclude=work/ --exclude=logs/ "$REMOTE/$1/" "$2/"; then
    echo "ok   $1"
  else
    echo "SKIP $1 (missing or failed)" >&2
    rc=1
  fi
}

for geom in ellipse sphere; do
  for mode in grid ksweep; do
    pull "cavity-bem/out/$mode/$geom" "out/bem/$mode/$geom"
  done
  for mode in grid field ksweep noise; do
    pull "cavity-epgp/out/$mode/$geom" "out/epgp/$mode/$geom"
  done
done

exit $rc
