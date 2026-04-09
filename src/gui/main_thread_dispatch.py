"""
主线程调度工具。

用途：
1. 在 macOS 上把 AppKit 相关调用同步派发到主线程；
2. 非 macOS 环境直接执行，避免影响现有测试与运行路径；
3. 调度失败时给出确定性异常，避免静默失败。
"""

from __future__ import annotations

import platform
import threading
from typing import Any, Callable, Dict, TypeVar


T = TypeVar("T")

_IS_DARWIN = platform.system() == "Darwin"
_NSThread = None
_AppHelper = None

if _IS_DARWIN:  # pragma: no cover - 运行时分支
    try:
        from Foundation import NSThread as _NSThread  # type: ignore[assignment]
        from PyObjCTools import AppHelper as _AppHelper  # type: ignore[assignment]
    except Exception:
        _NSThread = None
        _AppHelper = None


class MainThreadDispatchTimeoutError(RuntimeError):
    """主线程调度超时。"""


def is_main_thread() -> bool:
    """
    判断当前线程是否主线程。

    优先使用 Cocoa 线程信息；不可用时退回 Python 主线程判断。
    """
    if _NSThread is not None:  # pragma: no branch - 运行时分支
        try:
            return bool(_NSThread.isMainThread())
        except Exception:
            pass
    return threading.current_thread() is threading.main_thread()


def run_on_main_sync(
    fn: Callable[..., T],
    *args: Any,
    timeout_sec: float = 10.0,
    **kwargs: Any,
) -> T:
    """
    同步在主线程执行函数并返回结果。

    - Darwin + 可用桥接：后台线程通过 AppHelper.callAfter 派发到主线程。
    - 其他环境：直接执行。
    """
    if not _IS_DARWIN or _NSThread is None or _AppHelper is None:
        return fn(*args, **kwargs)

    if is_main_thread():
        return fn(*args, **kwargs)

    event = threading.Event()
    box: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["result"] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - 运行时兜底
            box["error"] = exc
        finally:
            event.set()

    _AppHelper.callAfter(_runner)
    if not event.wait(timeout=max(0.1, float(timeout_sec))):
        raise MainThreadDispatchTimeoutError(
            f"Main-thread dispatch timeout after {float(timeout_sec):.1f}s"
        )
    if "error" in box:
        raise box["error"]
    return box.get("result")


__all__ = [
    "MainThreadDispatchTimeoutError",
    "is_main_thread",
    "run_on_main_sync",
]
