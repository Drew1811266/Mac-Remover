#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[clean] Removing local-only directories..."
rm -rf \
  .omx \
  .trae \
  .vscode \
  logs \
  venv \
  frontend/node_modules \
  models/.download_staging \
  models/florence2-base/.cache

echo "[clean] Removing macOS metadata and Python caches..."
find . -name '.DS_Store' -type f -delete
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +
find . -name '.mypy_cache' -type d -prune -exec rm -rf {} +
find . -name '.ruff_cache' -type d -prune -exec rm -rf {} +
find . -name '.cache' -type d -prune -exec rm -rf {} +

echo "[clean] Done."
