#!/usr/bin/env bash
set -euo pipefail

CLIENT_DIR="idtrackerai-validator-client"
DEST="idtrackerai_validator_server/frontend"

git submodule update --init --recursive

test -f "$CLIENT_DIR/package.json" \
  || { echo "submodule $CLIENT_DIR not initialized"; exit 1; }

npm --prefix "$CLIENT_DIR" ci
npm --prefix "$CLIENT_DIR" run build

rm -rf "$DEST"
cp -r "$CLIENT_DIR/build" "$DEST"

test -f "$DEST/index.html" || { echo "build produced no index.html"; exit 1; }
echo "frontend built: $(du -sh "$DEST" | cut -f1)"