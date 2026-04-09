"""
统一内存回收工具。

目标：
1. 在任务收尾时主动触发 Python + Torch 缓存回收；
2. 尽量兼容 CPU/CUDA/MPS 环境；
3. 回收失败不抛异常，只记录日志并返回结构化结果。
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List

try:  # pragma: no cover - 运行时依赖分支
    import torch
except Exception:  # pragma: no cover - 运行时依赖分支
    torch = None  # type: ignore[assignment]

from .logger import logger


def _run_step(step_name: str, action) -> Dict[str, Any]:
    try:
        payload = action()
        return {
            "step": step_name,
            "ok": True,
            "detail": payload,
        }
    except Exception as exc:  # pragma: no cover - 运行时兜底
        return {
            "step": step_name,
            "ok": False,
            "error": str(exc),
        }


def release_unified_memory(reason: str) -> Dict[str, Any]:
    """
    触发统一内存回收（容错，不抛异常）。

    固定顺序：
    1) gc.collect()
    2) torch.mps.empty_cache()
    3) torch.cuda.empty_cache()
    4) torch.cuda.ipc_collect()
    """
    safe_reason = str(reason or "unknown")
    steps: List[Dict[str, Any]] = []
    steps.append(_run_step("gc.collect", lambda: {"collected": int(gc.collect())}))

    if torch is None:
        steps.append(
            {
                "step": "torch",
                "ok": True,
                "detail": "torch_unavailable",
            }
        )
    else:
        # MPS 缓存释放（Apple Silicon 统一内存路径）
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            try:
                if bool(torch.mps.is_available()) and hasattr(torch.mps, "empty_cache"):
                    steps.append(_run_step("torch.mps.empty_cache", lambda: torch.mps.empty_cache()))
                else:
                    steps.append(
                        {
                            "step": "torch.mps.empty_cache",
                            "ok": True,
                            "detail": "skipped_unavailable",
                        }
                    )
            except Exception as exc:  # pragma: no cover - 运行时兜底
                steps.append(
                    {
                        "step": "torch.mps.empty_cache",
                        "ok": False,
                        "error": str(exc),
                    }
                )
        else:
            steps.append(
                {
                    "step": "torch.mps.empty_cache",
                    "ok": True,
                    "detail": "skipped_not_supported",
                }
            )

        # CUDA 路径（兼容非 macOS 测试环境）
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available"):
            try:
                if bool(torch.cuda.is_available()):
                    steps.append(_run_step("torch.cuda.empty_cache", lambda: torch.cuda.empty_cache()))
                    if hasattr(torch.cuda, "ipc_collect"):
                        steps.append(_run_step("torch.cuda.ipc_collect", lambda: torch.cuda.ipc_collect()))
                    else:
                        steps.append(
                            {
                                "step": "torch.cuda.ipc_collect",
                                "ok": True,
                                "detail": "skipped_not_supported",
                            }
                        )
                else:
                    steps.append(
                        {
                            "step": "torch.cuda.empty_cache",
                            "ok": True,
                            "detail": "skipped_unavailable",
                        }
                    )
                    steps.append(
                        {
                            "step": "torch.cuda.ipc_collect",
                            "ok": True,
                            "detail": "skipped_unavailable",
                        }
                    )
            except Exception as exc:  # pragma: no cover - 运行时兜底
                steps.append(
                    {
                        "step": "torch.cuda.empty_cache",
                        "ok": False,
                        "error": str(exc),
                    }
                )
                steps.append(
                    {
                        "step": "torch.cuda.ipc_collect",
                        "ok": False,
                        "error": str(exc),
                    }
                )
        else:
            steps.append(
                {
                    "step": "torch.cuda.empty_cache",
                    "ok": True,
                    "detail": "skipped_not_supported",
                }
            )
            steps.append(
                {
                    "step": "torch.cuda.ipc_collect",
                    "ok": True,
                    "detail": "skipped_not_supported",
                }
            )

    success = all(bool(item.get("ok")) for item in steps)
    errors = [str(item.get("error") or "") for item in steps if not item.get("ok")]
    result = {
        "reason": safe_reason,
        "success": success,
        "steps": steps,
        "errors": [item for item in errors if item],
    }

    if success:
        logger.info("[memory-cleanup] reason=%s success=true", safe_reason)
    else:
        logger.warning(
            "[memory-cleanup] reason=%s success=false errors=%s",
            safe_reason,
            "; ".join(result["errors"]) if result["errors"] else "unknown",
        )

    return result


__all__ = ["release_unified_memory"]
