# Repository Assets Policy

## Tracked Source Code

- `src/`: Python 应用源码
- `frontend/`: React + TypeScript 前端源码
- `tests/`: Python 测试
- `scripts/`: 构建、下载和仓库维护脚本

## Tracked Offline Runtime Assets

- `src/gui/templates/dist/`
  - 已构建前端产物
  - 保留原因：桌面应用默认直接加载该目录
- `vendor/ffmpeg/`
  - 内置 FFmpeg/FFprobe
  - 保留原因：不依赖系统 PATH 即可运行
- `models/big-lama/`
  - LaMa 模型目录
- `models/florence2-base/`
  - Florence-2 模型目录
  - 保留原因：本轮明确作为正式离线模型资产保留
- `models/runtime/seedvr-py312/`
  - SeedVR2 离线运行时、权重和独立 Python 环境
- `models/runtime/realesrgan-py312/`
  - Real-ESRGAN 离线运行时、权重和独立 Python 环境
- `models/third_party/ProPainter/`
  - 仍在主流程中使用的第三方源码与权重

## Removed / Not Tracked

以下内容不应提交到仓库：

- `.omx/`, `.trae/`, `.vscode/`
- `logs/`
- 根 `venv/`
- `frontend/node_modules/`
- `models/.download_staging/`
- 所有 `.DS_Store`、`__pycache__`、`.cache/`

以下历史第三方目录已从主仓库清理：

- `models/third_party/STTN/`
- `models/third_party/RealBasicVSR/`
- `models/third_party/Real-ESRGAN/`

## Git LFS Scope

以下大文件类型默认通过 Git LFS 管理：

- `*.bin`
- `*.ckpt`
- `*.dylib`
- `*.gguf`
- `*.pth`
- `*.safetensors`
- `*.so`
- `vendor/ffmpeg/*/ffmpeg`

这不是“可选优化”，而是 GitHub 上传的必要条件。当前仓库存在多个超过 100MB 的离线模型文件。

