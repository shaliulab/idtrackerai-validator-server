#!/usr/bin/env bash
# Packs the built web interface (idtrackerai_validator_server/frontend, made by build_frontend.sh) as a release
# asset: dist/idtrackerai-validator-frontend-<version>-<commit>.tar.gz. Attached to a GitHub release, the
# app's "Check for updates" (updater.py) installs it instead of building the client with npm on the server.
#
#   ./build_frontend.sh && scripts/make_frontend_asset.sh
#   gh release upload v<version> dist/idtrackerai-validator-frontend-*.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
test -f idtrackerai_validator_server/frontend/index.html || { echo "build the frontend first (./build_frontend.sh)" >&2; exit 1; }
version=$(sed -n "s/^ *version='\([^']*\)'.*/\1/p" setup.py)
commit=$(git rev-parse --short HEAD)
mkdir -p dist
out="dist/idtrackerai-validator-frontend-${version}-${commit}.tar.gz"
tar -C idtrackerai_validator_server/frontend -czf "$out" --exclude CLIENT_COMMIT .
ls -l "$out"
