# Mac Watermark Remover

Mac/Windows 桌面端视频去水印工具。当前产品路径正在切换为 Electron + React + TypeScript + Node 本地服务，发布包不再依赖 Python/pywebview。

## Repository Shape

- Electron 主进程、preload、本地服务位于 `frontend/electron/`
- React 前端源码位于 `frontend/src/`
- 运行时加载 `src/gui/templates/dist/` 中的已构建前端
- 离线模型与运行时资产位于 `models/`
- 内置 FFmpeg 位于 `vendor/ffmpeg/`

保留在仓库中的重资产是有意设计：

- `vendor/ffmpeg/`: 内置 FFmpeg/FFprobe，保证开箱可用
- `src/gui/templates/dist/`: 已构建前端，避免首次运行前必须手动构建
- `models/big-lama/`: 本地 LaMa 资产
- `models/florence2-base/`: 本地 Florence-2 资产，当前作为正式离线模型目录保留
- `models/runtime/`: 旧 Python 版 SeedVR2 / Real-ESRGAN 资产，仅作为迁移参考，不进入 Electron 发布包

例外说明：

- `models/runtime/seedvr-py312/models/SEEDVR2/seedvr2_ema_3b-Q8_0.gguf` 不再提交到 GitHub
- 原因是 GitHub LFS 单文件上限为 2 GiB，而该文件约 3.4 GiB
- 仓库保留默认推荐的 `Q4_K_M` 离线模型；`Q8_0` 改为按需在线下载或本地私有保存

本仓库不应再提交以下内容：

- 本机虚拟环境和依赖缓存
- `node_modules`
- 日志、IDE 配置、代理/自动化状态目录
- `.DS_Store`、`__pycache__` 等临时文件

详细资产约束见 `docs/repository-assets.md`。

## Git LFS

仓库中的大模型、权重和部分离线二进制通过 Git LFS 管理。克隆或推送前先安装并初始化 Git LFS：

```bash
git lfs install
git lfs pull
```

如果未安装 Git LFS，GitHub 无法正常接收超过 100MB 的模型文件。

## Local Development

启动开发版 Electron 桌面应用：

```bash
./run.sh
```

首次运行会在 `frontend/` 下安装 Node/Electron 依赖，然后构建并启动开发版 Electron。日常功能开发使用这条路径，不需要启动 `frontend/release/` 下的打包 `.app`。该路径不会创建 Python `venv`。

构建前端与 Electron 主进程：

```bash
cd frontend
npm run build
npm run electron:build
```

功能回归测试：

```bash
cd frontend
npm run functional:test
```

功能测试说明见 `docs/functional-testing.md`。

发布前打包 Electron：

```bash
cd frontend
npm run electron:pack -- --mac
npm run electron:verify-release
```

清理本机开发垃圾文件：

```bash
./scripts/clean_local_dev_artifacts.sh
```

## Notes

- `models/runtime/*/venv` 不进入 Electron 发布包。
- `seedvr2_ema_3b-Q8_0.gguf` 如需使用，请在本机单独下载到 `models/runtime/seedvr-py312/models/SEEDVR2/`。
- Electron 版去水印处理仅内置 LaMa-ROI；Real-ESRGAN、SeedVR2 放大能力在可信原生资产缺失时会显示为阻塞状态。
