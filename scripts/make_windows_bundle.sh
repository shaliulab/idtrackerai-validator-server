#!/usr/bin/env bash
# Builds the Windows install bundle: dist/idtrackerai-validator-windows-<version>-<commit>.zip
# (install.bat, install.ps1, start-validator.*, TESTING.md, wheels\ with the lab's packages and the
# validator with its web interface). Needs Docker with BuildKit; no GitHub account or ssh key.
#
#   scripts/make_windows_bundle.sh
#
# Give the zip to the Windows user: unzip, double click install.bat (scripts/windows/TESTING.md).
set -euo pipefail
cd "$(dirname "$0")/.."

commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet HEAD -- 2>/dev/null || commit="${commit}-dirty"
out=dist/windows-bundle
rm -rf "$out"
DOCKER_BUILDKIT=1 docker build --target windows-bundle --output "$out" .

version=$(ls "$out"/wheels/idtrackerai_validator_server-*.whl | sed 's/.*idtrackerai_validator_server-\([^-]*\)-.*/\1/')
echo "idtrackerai-validator ${version} (${commit}), built $(date -u +%Y-%m-%dT%H:%MZ)" > "$out/VERSION.txt"
zip_name="idtrackerai-validator-windows-${version}-${commit}.zip"
rm -f "dist/$zip_name"
(cd "$out" && zip -qr "../$zip_name" .)
ls -l "dist/$zip_name"
