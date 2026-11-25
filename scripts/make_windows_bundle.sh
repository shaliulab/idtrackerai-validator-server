#!/usr/bin/env bash
# Builds the Windows install bundle: dist/idtrackerai-validator-windows-<version>-<commit>.zip
# (install.bat, install.ps1, start-validator.*, TESTING.md, wheels\ with the lab's packages and the
# validator with its web interface). Needs Docker with BuildKit; no GitHub account or ssh key.
#
#   scripts/make_windows_bundle.sh
#   FLYHOSTEL_SRC=/path/to/flyhostel scripts/make_windows_bundle.sh
#   CLIENT_SRC=idtrackerai-validator-client scripts/make_windows_bundle.sh
#
# CLIENT_SRC: build the web interface from the commit setup.py pins in a local client checkout (its
# `git archive`) instead of GitHub, e.g. a client commit not pushed yet.
#
# FLYHOSTEL_SRC: build the lab's wheels from a local flyhostel checkout (libraries/ethoscopy,
# idtrackerai and imgstore checked out) instead of GitHub, e.g. to test commits not pushed yet. Its
# HEAD and libraries/imgstore must be the commits setup.py pins, without uncommitted changes.
#
# Give the zip to the Windows user: unzip, double click install.bat (scripts/windows/TESTING.md).
set -euo pipefail
cd "$(dirname "$0")/.."

commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet HEAD -- 2>/dev/null || commit="${commit}-dirty"
out=dist/windows-bundle
rm -rf "$out"
context=()
if [ -n "${FLYHOSTEL_SRC:-}" ]; then
    pin() { sed -n "s/^ *\"$1\": \"\([0-9a-f]\{40\}\)\".*/\1/p" setup.py; }
    for check in "flyhostel:$FLYHOSTEL_SRC" "flyhostel\/libraries\/imgstore:$FLYHOSTEL_SRC/libraries/imgstore"; do
        name=${check%%:*} path=${check#*:}
        want=$(pin "$name") have=$(git -C "$path" rev-parse HEAD)
        [ "$want" = "$have" ] || { echo "$path is at $have, setup.py pins $want" >&2; exit 1; }
        git -C "$path" diff --quiet HEAD -- || { echo "$path has uncommitted changes" >&2; exit 1; }
    done
    context=(--build-context "flyhostel-src=$FLYHOSTEL_SRC")
fi
if [ -n "${CLIENT_SRC:-}" ]; then
    want=$(sed -n 's/^ *"idtrackerai-validator-client": "\([0-9a-f]\{40\}\)".*/\1/p' setup.py)
    export_dir=$(mktemp -d)
    trap 'rm -rf "$export_dir"' EXIT
    git -C "$CLIENT_SRC" archive "$want" | tar -x -C "$export_dir"
    echo "$want" > "$export_dir/CLIENT_COMMIT"
    context+=(--build-context "client-src=$export_dir")
fi
DOCKER_BUILDKIT=1 docker build "${context[@]}" --target windows-bundle --output "$out" .

version=$(ls "$out"/wheels/idtrackerai_validator_server-*.whl | sed 's/.*idtrackerai_validator_server-\([^-]*\)-.*/\1/')
echo "idtrackerai-validator ${version} (${commit}), built $(date -u +%Y-%m-%dT%H:%MZ)" > "$out/VERSION.txt"
zip_name="idtrackerai-validator-windows-${version}-${commit}.zip"
rm -f "dist/$zip_name"
(cd "$out" && zip -qr "../$zip_name" .)
ls -l "dist/$zip_name"
