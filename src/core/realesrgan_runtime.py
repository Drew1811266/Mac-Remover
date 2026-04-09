"""
Real-ESRGAN 独立运行时管理器。

职责：
1) 发现 Python 3.12 并维护隔离 venv；
2) 安装 Real-ESRGAN 推理依赖；
3) 执行 worker 进程并桥接进度、取消、超时与失败摘要。
"""

from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

from .realesrgan_manifest import (
    REALESRGAN_DEPS_MARKER,
    REALESRGAN_MODEL_DIR,
    REALESRGAN_RUNTIME_ROOT,
    REALESRGAN_STATE_FILE,
    REALESRGAN_VENV_DIR,
    REALESRGAN_WORKER_SCRIPT,
)
from .seedvr_runtime import resolve_python312


class RealESRGANRuntimeError(RuntimeError):
    """Real-ESRGAN 运行时错误。"""


def _safe_emit(
    callback: Optional[Callable[[Dict[str, Any]], None]],
    payload: Dict[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        return


class RealESRGANRuntime:
    """Real-ESRGAN 独立运行时封装。"""

    def __init__(self) -> None:
        self.runtime_root = REALESRGAN_RUNTIME_ROOT
        self.venv_dir = REALESRGAN_VENV_DIR
        self.venv_python = self.venv_dir / "bin" / "python"

    def get_status(self) -> Dict[str, Any]:
        python312 = resolve_python312()
        venv_ok = self.venv_python.exists()
        worker_ok = REALESRGAN_WORKER_SCRIPT.exists()
        model_dir_ok = REALESRGAN_MODEL_DIR.exists()
        deps_ok = REALESRGAN_DEPS_MARKER.exists()
        ready = bool(python312 and venv_ok and worker_ok and model_dir_ok and deps_ok)

        reason = ""
        if not ready:
            if not python312:
                reason = "Python 3.12 not found. Install python3.12 or set SEEDVR_PYTHON."
            elif not worker_ok:
                reason = "Real-ESRGAN worker script missing."
            elif not venv_ok or not deps_ok:
                reason = "Real-ESRGAN runtime dependencies are not ready. Download model first."
            elif not model_dir_ok:
                reason = "Real-ESRGAN model directory is missing."
            else:
                reason = "Real-ESRGAN runtime unavailable."
        return {
            "ready": ready,
            "reason": reason,
            "python312": python312,
            "venv_python": str(self.venv_python),
            "worker_script": str(REALESRGAN_WORKER_SCRIPT),
            "model_dir": str(REALESRGAN_MODEL_DIR),
        }

    def ensure_runtime(
        self,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        python312 = resolve_python312()
        if not python312:
            raise RealESRGANRuntimeError(
                "Python 3.12 not found. Please install python3.12 or set SEEDVR_PYTHON."
            )

        self.runtime_root.mkdir(parents=True, exist_ok=True)
        REALESRGAN_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if cancel_event is not None and cancel_event.is_set():
            raise RealESRGANRuntimeError("Real-ESRGAN runtime preparation cancelled")

        _safe_emit(progress_callback, {"message": "Preparing Real-ESRGAN Python environment..."})
        if not self.venv_python.exists():
            self._run_command(
                [python312, "-m", "venv", str(self.venv_dir)],
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                message="Creating Real-ESRGAN virtual environment...",
                timeout_sec=240,
                parse_progress=False,
            )

        if not REALESRGAN_DEPS_MARKER.exists():
            self._run_command(
                [str(self.venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                message="Installing Real-ESRGAN base Python packages...",
                timeout_sec=900,
                parse_progress=False,
            )
            self._run_command(
                [
                    str(self.venv_python),
                    "-m",
                    "pip",
                    "install",
                    "numpy",
                    "opencv-python",
                    "pillow",
                    "tqdm",
                    "torch",
                    "torchvision",
                    "basicsr>=1.4.2",
                    "realesrgan>=0.3.0",
                ],
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                message="Installing Real-ESRGAN runtime dependencies...",
                timeout_sec=3600,
                parse_progress=False,
            )
            REALESRGAN_DEPS_MARKER.write_text(str(int(time.time())), encoding="utf-8")

        state = {
            "prepared_at": int(time.time()),
            "python312": python312,
            "venv_python": str(self.venv_python),
            "worker_script": str(REALESRGAN_WORKER_SCRIPT),
            "model_dir": str(REALESRGAN_MODEL_DIR),
        }
        try:
            REALESRGAN_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return state

    def run_inference(
        self,
        *,
        input_path: str,
        output_dir: str,
        model_id: str,
        outscale: float,
        denoise_strength: float,
        tile: int,
        tile_pad: int,
        pre_pad: int,
        cancel_event: Optional[Any] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout_sec: float = 5400.0,
    ) -> str:
        status = self.get_status()
        if not status.get("ready"):
            self.ensure_runtime(progress_callback=progress_callback, cancel_event=cancel_event)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "realesrgan_output.mp4"
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass

        cmd = [
            str(self.venv_python),
            "-u",
            str(REALESRGAN_WORKER_SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model-id",
            str(model_id),
            "--weights-dir",
            str(REALESRGAN_MODEL_DIR),
            "--outscale",
            f"{float(outscale):.6f}",
            "--denoise-strength",
            f"{max(0.0, min(1.0, float(denoise_strength))):.3f}",
            "--tile",
            str(max(0, int(tile))),
            "--tile-pad",
            str(max(0, int(tile_pad))),
            "--pre-pad",
            str(max(0, int(pre_pad))),
        ]
        self._run_command(
            cmd,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            message="Running Real-ESRGAN inference (device=mps, backend=pytorch)...",
            timeout_sec=float(timeout_sec),
            parse_progress=True,
            output_watch_file=output_path,
        )
        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RealESRGANRuntimeError("Real-ESRGAN inference completed but output file is missing")
        return str(output_path)

    @staticmethod
    def _build_env() -> Dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.82")
        env.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.68")
        if psutil is not None:
            try:
                logical = max(1, int(os.cpu_count() or 1))
            except Exception:
                logical = 1
            thread_cap = str(max(1, int(logical * 0.8)))
            env["OMP_NUM_THREADS"] = thread_cap
            env["MKL_NUM_THREADS"] = thread_cap
            env["OPENBLAS_NUM_THREADS"] = thread_cap
            env["VECLIB_MAXIMUM_THREADS"] = thread_cap
            env["NUMEXPR_NUM_THREADS"] = thread_cap
        return env

    def _run_command(
        self,
        cmd: list[str],
        *,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        message: str,
        timeout_sec: float,
        parse_progress: bool,
        output_watch_file: Optional[Path] = None,
    ) -> None:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.runtime_root),
            env=self._build_env(),
        )
        if proc.stdout is None:
            raise RealESRGANRuntimeError("Failed to capture Real-ESRGAN process output")

        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        started = time.time()
        last_emit = 0.0
        tail_lines: list[str] = []
        progress_regex = re.compile(r"^PROGRESS\s+frame=(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
        warmup_regex = re.compile(r"PROGRESS\s+stage=warmup\s+step=([a-zA-Z0-9_\-]+)", re.IGNORECASE)
        parsed_progress = 0.0
        last_activity_at = started
        last_forward_at = started
        warmup_mode = bool(parse_progress)
        warmup_step = "bootstrap"
        file_mtime = self._safe_mtime(output_watch_file)

        try:
            while True:
                now = time.time()
                if (now - started) > timeout_sec:
                    proc.kill()
                    raise RealESRGANRuntimeError(f"Real-ESRGAN command timed out after {int(timeout_sec)}s")

                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                    raise RealESRGANRuntimeError("Real-ESRGAN command cancelled")

                events = selector.select(timeout=0.35)
                for key, _ in events:
                    line = key.fileobj.readline()  # type: ignore[attr-defined]
                    if not line:
                        continue
                    text = str(line).strip()
                    if not text:
                        continue
                    last_activity_at = now
                    tail_lines.append(text)
                    if len(tail_lines) > 20:
                        tail_lines.pop(0)
                    if parse_progress:
                        warmup_match = warmup_regex.search(text)
                        if warmup_match:
                            warmup_step = str(warmup_match.group(1) or "").strip().lower() or warmup_step
                        match = progress_regex.search(text)
                        if match:
                            done = int(match.group(1) or 0)
                            total = int(match.group(2) or 0)
                            if total > 0:
                                frac = min(1.0, max(0.0, float(done) / float(total)))
                                if frac > parsed_progress:
                                    parsed_progress = frac
                                    warmup_mode = False
                                    last_forward_at = now

                next_mtime = self._safe_mtime(output_watch_file)
                if next_mtime > file_mtime:
                    file_mtime = next_mtime
                    last_activity_at = now
                    if parsed_progress <= 0.0:
                        parsed_progress = max(parsed_progress, 0.01)
                    warmup_mode = False
                    last_forward_at = now

                if parse_progress:
                    if warmup_mode and (now - last_activity_at) >= 240.0:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        raise RealESRGANRuntimeError("Warmup stalled (no activity for 240s)")
                    if (not warmup_mode) and (now - last_forward_at) >= 90.0:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        raise RealESRGANRuntimeError("Inference stalled (no forward progress for 90s)")

                if progress_callback and (now - last_emit) > 0.55:
                    phase = "model_warmup" if warmup_mode else "chunk_infer"
                    heartbeat = f"{message} phase={phase}"
                    if warmup_mode and warmup_step:
                        heartbeat = f"{heartbeat} stage=warmup step={warmup_step}"
                    _safe_emit(
                        progress_callback,
                        {
                            "message": heartbeat,
                            "progress": parsed_progress if parse_progress else 0.0,
                        },
                    )
                    last_emit = now

                ret = proc.poll()
                if ret is not None:
                    if ret != 0:
                        tail = "\n".join(tail_lines[-10:]) if tail_lines else "unknown runtime error"
                        raise RealESRGANRuntimeError(
                            f"Real-ESRGAN command failed (code {ret}): {' '.join(cmd)}\n{tail}"
                        )
                    break
        finally:
            try:
                selector.unregister(proc.stdout)
            except Exception:
                pass
            selector.close()
            proc.stdout.close()

    @staticmethod
    def _safe_mtime(path: Optional[Path]) -> float:
        if path is None:
            return 0.0
        target = Path(path)
        if not target.exists():
            return 0.0
        try:
            return float(target.stat().st_mtime)
        except Exception:
            return 0.0


__all__ = ["RealESRGANRuntime", "RealESRGANRuntimeError"]

