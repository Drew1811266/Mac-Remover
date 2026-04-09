#!/bin/bash
set -euo pipefail

# 启动脚本：
# 1) 切到项目根目录
# 2) 自动创建并激活根 venv
# 3) 缺依赖时按 requirements.txt 自举安装
# 4) 启动 Python 主程序
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEPS_MARKER="venv/.wmr_requirements_installed"

if [ ! -d "venv" ]; then
    # 首次运行时自动准备虚拟环境。
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f "${DEPS_MARKER}" ] || [ "requirements.txt" -nt "${DEPS_MARKER}" ]; then
    echo "Installing Python dependencies..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    date +"%Y-%m-%dT%H:%M:%S%z" > "${DEPS_MARKER}"
fi

echo "Starting Mac Watermark Remover..."
# 把外部传入参数原样转发给 `src.main`（例如调试参数）。
python -m src.main "$@"
