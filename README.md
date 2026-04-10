# Mac Watermark Remover

Mac 桌面端视频去水印工具。当前仓库定位为“可上传到 GitHub、可离线运行”的产品仓库，而不是单机开发目录快照。

## Repository Shape

- Python 主程序位于 `src/`
- React 前端源码位于 `frontend/`
- 运行时优先加载 `src/gui/templates/dist/` 中的已构建前端
- 离线模型与运行时资产位于 `models/`
- 内置 FFmpeg 位于 `vendor/ffmpeg/`

保留在仓库中的重资产是有意设计：

- `vendor/ffmpeg/`: 内置 FFmpeg/FFprobe，保证开箱可用
- `src/gui/templates/dist/`: 已构建前端，避免首次运行前必须手动构建
- `models/big-lama/`: 本地 LaMa 资产
- `models/florence2-base/`: 本地 Florence-2 资产，当前作为正式离线模型目录保留
- `models/runtime/`: SeedVR2 / Real-ESRGAN 离线运行时与权重
- `models/third_party/ProPainter/`: 主流程仍在使用的第三方源码与权重

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

启动桌面应用：

```bash
./run.sh
```

首次运行会自动创建根 `venv/` 并安装 `requirements.txt` 中的 Python 依赖。该根环境是本机可重建产物，不纳入版本库。

构建前端：

```bash
./scripts/build_frontend.sh
```

下载基础去水印模型：

```bash
python3 scripts/download_models.py --all
```

清理本机开发垃圾文件：

```bash
./scripts/clean_local_dev_artifacts.sh
```

## Notes

- `models/runtime/*/venv` 当前作为正式离线资产保留。本轮没有改成在线重建模式。
- `seedvr2_ema_3b-Q8_0.gguf` 如需使用，请在本机单独下载到 `models/runtime/seedvr-py312/models/SEEDVR2/`。
- 如果后续要进一步瘦身仓库，建议单独做“离线 bootstrap / wheelhouse”重构，而不是在当前结构上半删半留。
