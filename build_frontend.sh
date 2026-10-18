#!/usr/bin/env bash
set -euo pipefail

CLIENT_DIR="idtrackerai-validator-client"
DEST="idtrackerai_validator_server/frontend"

git submodule update --init --recursive

test -f "$CLIENT_DIR/package.json" \
  || { echo "submodule $CLIENT_DIR not initialized"; exit 1; }

# npm ci wipes node_modules: only run it when the lockfile changed.
STAMP="node_modules/.package-lock.sha256"
if ! (cd "$CLIENT_DIR" && sha256sum --status -c "$STAMP" 2>/dev/null); then
  npm --prefix "$CLIENT_DIR" ci --no-audit --no-fund
  (cd "$CLIENT_DIR" && sha256sum package-lock.json > "$STAMP")
fi
npm --prefix "$CLIENT_DIR" run build

rm -rf "$DEST"
cp -r "$CLIENT_DIR/build" "$DEST"

test -f "$DEST/index.html" || { echo "build produced no index.html"; exit 1; }
echo "frontend built: $(du -sh "$DEST" | cut -f1)"
