#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    echo "Installing Electron dependencies..."
    npm install
fi

echo "Starting Mac Watermark Remover (development Electron)..."
npm run electron:dev -- "$@"
