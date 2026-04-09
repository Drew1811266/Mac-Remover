"""
Real-ESRGAN 运行时与模型清单定义。

说明：
- 仅包含静态元数据，不执行下载或推理逻辑；
- 运行时/下载器/处理器统一读取该清单，避免前后不一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"

REALESRGAN_RUNTIME_ROOT = MODELS_ROOT / "runtime" / "realesrgan-py312"
REALESRGAN_VENV_DIR = REALESRGAN_RUNTIME_ROOT / "venv"
REALESRGAN_MODEL_DIR = REALESRGAN_RUNTIME_ROOT / "weights"
REALESRGAN_STATE_FILE = REALESRGAN_RUNTIME_ROOT / "install_state.json"
REALESRGAN_DEPS_MARKER = REALESRGAN_RUNTIME_ROOT / ".deps_ready"

# Real-ESRGAN worker 位于项目源码中，避免动态改第三方仓库文件。
REALESRGAN_WORKER_SCRIPT = PROJECT_ROOT / "src" / "core" / "realesrgan_worker.py"

REALESRGAN_ENGINE_ID = "realesrgan"
REALESRGAN_UPSCALE_ENGINES: Tuple[str, ...] = (REALESRGAN_ENGINE_ID,)
REALESRGAN_UPSCALE_MODELS: Tuple[str, ...] = (
    "realesrgan_general_x4v3",
    "realesrgan_x2plus",
)
REALESRGAN_DEFAULT_MODEL_ID = "realesrgan_general_x4v3"


@dataclass(frozen=True)
class RealESRGANWeightSpec:
    filename: str
    source_url: str
    sha256: str = ""


@dataclass(frozen=True)
class RealESRGANModelSpec:
    model_id: str
    display_name: str
    install_hint: str
    min_memory_gb: float
    preferred_tile: int
    weights: Tuple[RealESRGANWeightSpec, ...]


REALESRGAN_MODEL_SPECS: Dict[str, RealESRGANModelSpec] = {
    "realesrgan_general_x4v3": RealESRGANModelSpec(
        model_id="realesrgan_general_x4v3",
        display_name="Real-ESRGAN General x4v3",
        install_hint="轻量通用模型（默认），速度更快，适合日常视频放大。",
        min_memory_gb=8.0,
        preferred_tile=512,
        weights=(
            RealESRGANWeightSpec(
                filename="realesr-general-x4v3.pth",
                source_url=(
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
                    "realesr-general-x4v3.pth"
                ),
                sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
            ),
            RealESRGANWeightSpec(
                filename="realesr-general-wdn-x4v3.pth",
                source_url=(
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
                    "realesr-general-wdn-x4v3.pth"
                ),
                sha256="1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
            ),
        ),
    ),
    "realesrgan_x2plus": RealESRGANModelSpec(
        model_id="realesrgan_x2plus",
        display_name="Real-ESRGAN x2plus",
        install_hint="质量优先的 x2 模型，耗时和内存占用高于默认模型。",
        min_memory_gb=10.0,
        preferred_tile=384,
        weights=(
            RealESRGANWeightSpec(
                filename="RealESRGAN_x2plus.pth",
                source_url=(
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
                    "RealESRGAN_x2plus.pth"
                ),
                sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
            ),
        ),
    ),
}


def get_realesrgan_model_spec(model_id: str) -> RealESRGANModelSpec:
    normalized = str(model_id or "").strip()
    if normalized not in REALESRGAN_MODEL_SPECS:
        raise KeyError(f"Unknown Real-ESRGAN model id: {normalized}")
    return REALESRGAN_MODEL_SPECS[normalized]


__all__ = [
    "REALESRGAN_DEPS_MARKER",
    "REALESRGAN_DEFAULT_MODEL_ID",
    "REALESRGAN_ENGINE_ID",
    "REALESRGAN_MODEL_DIR",
    "REALESRGAN_MODEL_SPECS",
    "REALESRGAN_RUNTIME_ROOT",
    "REALESRGAN_STATE_FILE",
    "REALESRGAN_UPSCALE_ENGINES",
    "REALESRGAN_UPSCALE_MODELS",
    "REALESRGAN_VENV_DIR",
    "REALESRGAN_WORKER_SCRIPT",
    "RealESRGANModelSpec",
    "RealESRGANWeightSpec",
    "get_realesrgan_model_spec",
]
