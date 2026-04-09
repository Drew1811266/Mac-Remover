#!/usr/bin/env bash
# 前端构建脚本：
# 1) 进入 frontend 目录
# 2) 缺依赖时安装 node_modules
# 3) 执行生产构建，把产物输出到 templates/dist
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"
DIST_DIR="${ROOT_DIR}/src/gui/templates/dist"

if ! command -v npm >/dev/null 2>&1; then
  # 明确提示依赖缺失，避免后续报错不直观。
  echo "npm is required but not found." >&2
  exit 1
fi

cd "${FRONTEND_DIR}"

echo "[build_frontend] Installing dependencies (if needed)..."
if [[ ! -d node_modules ]]; then
  # 仅首次安装，减少重复构建时间。
  npm install \
    --no-audit \
    --no-fund \
    --prefer-offline \
    --fetch-retries=1 \
    --fetch-retry-mintimeout=2000 \
    --fetch-retry-maxtimeout=10000 \
    --fetch-timeout=20000
fi

echo "[build_frontend] Building React + Semi frontend..."
npm run build

echo "[build_frontend] Build complete. Dist: ${DIST_DIR}"
