"""
FFmpeg 运行时定位模块。

这个文件做两件事：
1. 在 `vendor/ffmpeg/<平台-架构>/` 里优先找内置 ffmpeg/ffprobe。
2. 内置不存在时回退到系统 PATH 里的可执行文件。
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INFO_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def _normalize_arch(machine: str) -> str:
    """把不同写法的架构名归一化，方便拼接目录名。"""
    value = str(machine or "").strip().lower()
    if value in {"arm64", "aarch64"}:
        return "arm64"
    if value in {"x86_64", "amd64", "x64"}:
        return "x86_64"
    return value or "unknown"


def _platform_key() -> str:
    """生成类似 `darwin-arm64` 的平台键。"""
    system = platform.system().strip().lower() or "unknown"
    arch = _normalize_arch(platform.machine())
    return f"{system}-{arch}"


def _vendor_root() -> Path:
    """
    返回内置 ffmpeg 根目录。

    支持通过环境变量 `WMR_FFMPEG_VENDOR_DIR` 覆盖默认目录，
    方便调试或自定义打包路径。
    """
    custom = os.getenv("WMR_FFMPEG_VENDOR_DIR", "").strip()
    if custom:
        return Path(custom).expanduser()
    return PROJECT_ROOT / "vendor" / "ffmpeg"


def ensure_executable(path: Path) -> bool:
    """
    确保目标文件具有可执行权限。

    返回值表示是否已可执行，不抛异常给上层。
    """
    try:
        current_mode = path.stat().st_mode
    except OSError:
        return False

    if current_mode & stat.S_IXUSR:
        return True

    try:
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return True
    except OSError:
        return False


def _resolve_embedded(tool: str) -> Optional[Path]:
    """尝试解析内置工具路径，找到后校正权限并返回。"""
    candidate = _vendor_root() / _platform_key() / tool
    if not candidate.exists() or not candidate.is_file():
        return None

    ensure_executable(candidate)
    if os.access(str(candidate), os.X_OK):
        return candidate
    return None


def _resolve_system(tool: str) -> Optional[Path]:
    """从系统 PATH 中查找工具路径。"""
    system_path = shutil.which(tool)
    if not system_path:
        return None
    return Path(system_path)


def _resolve_tool(tool: str) -> Tuple[Optional[Path], str]:
    """
    统一工具解析入口。

    返回：
    - 路径（可能为空）
    - 来源标记：`embedded` / `system` / `missing`
    """
    embedded = _resolve_embedded(tool)
    if embedded is not None:
        return embedded, "embedded"

    system_path = _resolve_system(tool)
    if system_path is not None:
        return system_path, "system"

    return None, "missing"


def resolve_ffmpeg_path() -> Optional[str]:
    """仅返回 ffmpeg 路径字符串，供业务代码快速调用。"""
    path, _ = _resolve_tool("ffmpeg")
    return str(path) if path else None


def resolve_ffprobe_path() -> Optional[str]:
    """仅返回 ffprobe 路径字符串，供业务代码快速调用。"""
    path, _ = _resolve_tool("ffprobe")
    return str(path) if path else None


def _read_version(executable: Optional[Path]) -> str:
    """
    读取执行文件的版本首行文本。

    失败时返回空串，避免让日志采集影响主流程。
    """
    if executable is None:
        return ""
    try:
        result = subprocess.run(
            [str(executable), "-version"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        first_line = (result.stdout or result.stderr or "").splitlines()
        return first_line[0].strip() if first_line else ""
    except Exception:
        return ""


def runtime_ffmpeg_info() -> Dict[str, Dict[str, str]]:
    """
    返回 ffmpeg / ffprobe 的完整运行时信息，并做进程内缓存。

    缓存意义：
    - 避免每次都执行 `-version` 带来的额外开销。
    - 日志和状态查询可以复用同一份结果。
    """
    global _INFO_CACHE
    if _INFO_CACHE is not None:
        return {
            "ffmpeg": dict(_INFO_CACHE["ffmpeg"]),
            "ffprobe": dict(_INFO_CACHE["ffprobe"]),
        }

    ffmpeg_path, ffmpeg_source = _resolve_tool("ffmpeg")
    ffprobe_path, ffprobe_source = _resolve_tool("ffprobe")
    ffmpeg_version = _read_version(ffmpeg_path)
    ffprobe_version = _read_version(ffprobe_path)

    _INFO_CACHE = {
        "ffmpeg": {
            "source": ffmpeg_source,
            "path": str(ffmpeg_path) if ffmpeg_path else "",
            "version": ffmpeg_version,
        },
        "ffprobe": {
            "source": ffprobe_source,
            "path": str(ffprobe_path) if ffprobe_path else "",
            "version": ffprobe_version,
        },
    }
    return {
        "ffmpeg": dict(_INFO_CACHE["ffmpeg"]),
        "ffprobe": dict(_INFO_CACHE["ffprobe"]),
    }


def clear_ffmpeg_runtime_cache() -> None:
    """清空运行时缓存，通常用于测试或运行时切换后重查。"""
    global _INFO_CACHE
    _INFO_CACHE = None
