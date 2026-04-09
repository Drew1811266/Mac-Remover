"""
SeedVR2 运行时与模型清单定义。

说明：
- 该文件只定义静态元数据，不执行下载或推理逻辑。
- 下载器与运行时管理器通过此清单实现一致的版本与路径约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"

# SeedVR2 独立运行时根目录（Python 3.12 虚拟环境 + 推理代码 + 模型文件）。
SEEDVR_RUNTIME_ROOT = MODELS_ROOT / "runtime" / "seedvr-py312"
SEEDVR_REPO_DIR = SEEDVR_RUNTIME_ROOT / "seedvr2_repo"
SEEDVR_VENV_DIR = SEEDVR_RUNTIME_ROOT / "venv"
SEEDVR_MODEL_DIR = SEEDVR_RUNTIME_ROOT / "models" / "SEEDVR2"
SEEDVR_STATE_FILE = SEEDVR_RUNTIME_ROOT / "install_state.json"
SEEDVR_REPO_REF_FILE = SEEDVR_RUNTIME_ROOT / ".repo_ref"

# 推理脚本与依赖文件（来自 numz 官方 ComfyUI SeedVR2 仓库）。
# 固定到已验证的上游提交，避免 main 漂移导致行为不可复现。
SEEDVR_REPO_REF = "4490bd1f482e026674543386bb2a4d176da245b9"
SEEDVR_REPO_ARCHIVE_URL = (
    "https://codeload.github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/tar.gz/"
    f"{SEEDVR_REPO_REF}"
)
SEEDVR_INFERENCE_SCRIPT = SEEDVR_REPO_DIR / "inference_cli.py"
SEEDVR_REQUIREMENTS_FILE = SEEDVR_REPO_DIR / "requirements.txt"

SEEDVR_ENGINE_ID = "seedvr2"
SEEDVR_UPSCALE_ENGINES: Tuple[str, ...] = (SEEDVR_ENGINE_ID,)
SEEDVR_UPSCALE_MODELS: Tuple[str, ...] = (
    "seedvr2_3b_q8_0_gguf",
    "seedvr2_3b_q4_k_m_gguf",
)
SEEDVR_DEFAULT_MODEL_ID = "seedvr2_3b_q4_k_m_gguf"
LEGACY_REMOVED_MODELS = {"seedvr2_3b_fp8", "seedvr2_7b_fp8"}


@dataclass(frozen=True)
class SeedVRModelSpec:
    model_id: str
    display_name: str
    install_hint: str
    dit_model_name: str
    source_url: str
    min_memory_gb: float
    sha256: str = ""


SEEDVR_MODEL_SPECS: Dict[str, SeedVRModelSpec] = {
    "seedvr2_3b_q8_0_gguf": SeedVRModelSpec(
        model_id="seedvr2_3b_q8_0_gguf",
        display_name="SeedVR2 3B Q8_0 (GGUF)",
        install_hint="画质更高但更重，建议 24GB+ 统一内存设备使用。",
        dit_model_name="seedvr2_ema_3b-Q8_0.gguf",
        source_url=(
            "https://huggingface.co/AInVFX/SeedVR2_comfyUI/resolve/main/"
            "seedvr2_ema_3b-Q8_0.gguf"
        ),
        min_memory_gb=16.0,
        sha256="be0d60083a2051a265eb4b77f28edf494e6db67ffc250216f32b72292e5cbd96",
    ),
    "seedvr2_3b_q4_k_m_gguf": SeedVRModelSpec(
        model_id="seedvr2_3b_q4_k_m_gguf",
        display_name="SeedVR2 3B Q4_K_M (GGUF)",
        install_hint="默认推荐：更省内存，适合 16GB/24GB Apple 芯片日常使用。",
        dit_model_name="seedvr2_ema_3b-Q4_K_M.gguf",
        source_url=(
            "https://huggingface.co/AInVFX/SeedVR2_comfyUI/resolve/main/"
            "seedvr2_ema_3b-Q4_K_M.gguf"
        ),
        min_memory_gb=12.0,
        sha256="e665e3909de1a8c88a69c609bca9d43ff5a134647face2ce4497640cc3597f0e",
    ),
}


def get_seedvr_model_spec(model_id: str) -> SeedVRModelSpec:
    normalized = str(model_id or "").strip()
    if normalized not in SEEDVR_MODEL_SPECS:
        raise KeyError(f"Unknown SeedVR model id: {normalized}")
    return SEEDVR_MODEL_SPECS[normalized]


__all__ = [
    "SEEDVR_ENGINE_ID",
    "SEEDVR_DEFAULT_MODEL_ID",
    "SEEDVR_INFERENCE_SCRIPT",
    "LEGACY_REMOVED_MODELS",
    "SEEDVR_MODEL_DIR",
    "SEEDVR_MODEL_SPECS",
    "SEEDVR_REPO_ARCHIVE_URL",
    "SEEDVR_REPO_REF",
    "SEEDVR_REPO_REF_FILE",
    "SEEDVR_REPO_DIR",
    "SEEDVR_REQUIREMENTS_FILE",
    "SEEDVR_RUNTIME_ROOT",
    "SEEDVR_STATE_FILE",
    "SEEDVR_UPSCALE_ENGINES",
    "SEEDVR_UPSCALE_MODELS",
    "SEEDVR_VENV_DIR",
    "SeedVRModelSpec",
    "get_seedvr_model_spec",
]
