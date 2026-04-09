"""
SeedVR2 独立运行时管理器。

职责：
1) 发现可用 Python 3.12；
2) 在 models/runtime/seedvr-py312 下维护隔离 venv；
3) 提供 SeedVR2 推理命令执行与取消控制；
4) 输出稳定的错误信息，供 API 与能力探测使用。
"""

from __future__ import annotations

import json
import signal
import os
import re
import selectors
import shutil
import subprocess
import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import psutil
except Exception:  # pragma: no cover - 运行时兜底
    psutil = None  # type: ignore[assignment]

from .seedvr_manifest import (
    SEEDVR_INFERENCE_SCRIPT,
    SEEDVR_MODEL_DIR,
    SEEDVR_REQUIREMENTS_FILE,
    SEEDVR_REPO_DIR,
    SEEDVR_RUNTIME_ROOT,
    SEEDVR_STATE_FILE,
    SEEDVR_VENV_DIR,
)


SEEDVR_DEPS_MARKER = SEEDVR_RUNTIME_ROOT / ".deps_ready"
SCENE_SPLIT_RUNTIME_DEPS: tuple[tuple[str, str], ...] = (
    ("scenedetect", "scenedetect"),
    ("transnetv2_pytorch", "transnetv2-pytorch"),
)

# 负载治理常量（固定 80% 策略，不对外暴露）。
LOAD_CAP_ENABLED = True
LOAD_CAP_CPU_PERCENT = 88.0
LOAD_CAP_MIN_STREAK = 8
LOAD_CAP_PAUSE_SEC = 0.08
LOAD_CAP_MAX_PAUSES_PER_MINUTE = 8
LOAD_CAP_WARMUP_GRACE_SEC = 60.0
WARMUP_STALL_TIMEOUT_SEC = 240.0
RUN_STALL_TIMEOUT_SEC = 90.0
WARMUP_NO_PROGRESS_TIMEOUT_SEC = 120.0
WARMUP_NO_OUTPUT_TIMEOUT_SEC = 150.0
RUN_PARSED_PROGRESS_TIMEOUT_SEC = 120.0
DEFAULT_VIDEO_BACKEND = "ffmpeg"
FALLBACK_VIDEO_BACKEND = "opencv"

logger = logging.getLogger("MacWatermarkRemover")


class SeedVRRuntimeError(RuntimeError):
    """SeedVR 运行时错误。"""


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


def _python_version_ok(python_exe: str) -> bool:
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
        )
    except Exception:
        return False
    version_text = str(proc.stdout or "").strip()
    try:
        major, minor = version_text.split(".", 1)
        return int(major) == 3 and int(minor) >= 12
    except Exception:
        return False


def resolve_python312() -> str:
    """按优先顺序解析 Python 3.12 可执行文件。"""
    home = str(Path.home())
    candidates = [
        str(os.environ.get("SEEDVR_PYTHON") or "").strip(),
        shutil.which("python3.12") or "",
        str(Path(home) / ".homebrew" / "opt" / "python@3.12" / "bin" / "python3.12"),
        str(Path(home) / "homebrew" / "opt" / "python@3.12" / "bin" / "python3.12"),
        "/opt/homebrew/bin/python3.12",
        "/usr/local/bin/python3.12",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = ""
        if Path(candidate).exists():
            resolved = str(Path(candidate))
        else:
            resolved = shutil.which(candidate) or ""
        if not resolved:
            continue
        if _python_version_ok(resolved):
            return resolved
    return ""


class SeedVRRuntime:
    """SeedVR 独立运行时封装。"""

    def __init__(self) -> None:
        self.runtime_root = SEEDVR_RUNTIME_ROOT
        self.venv_dir = SEEDVR_VENV_DIR
        self.venv_python = self.venv_dir / "bin" / "python"

    def get_status(self) -> Dict[str, Any]:
        python312 = resolve_python312()
        script_ok = SEEDVR_INFERENCE_SCRIPT.exists()
        model_dir_ok = SEEDVR_MODEL_DIR.exists()
        venv_ok = self.venv_python.exists()
        deps_ok = SEEDVR_DEPS_MARKER.exists()
        scene_split_deps_ready = bool(venv_ok and deps_ok)

        ready = bool(script_ok and model_dir_ok and venv_ok and deps_ok)
        reason = ""
        if not ready:
            if not python312:
                reason = "Python 3.12 not found. Install python3.12 or set SEEDVR_PYTHON."
            elif not script_ok:
                reason = "SeedVR runtime source is missing. Download a SeedVR model first."
            elif not venv_ok or not deps_ok:
                reason = "SeedVR runtime dependencies are not ready. Download a SeedVR model first."
            elif not model_dir_ok:
                reason = "SeedVR model directory is missing."
            else:
                reason = "SeedVR runtime is unavailable."
        return {
            "ready": ready,
            "reason": reason,
            "python312": python312,
            "venv_python": str(self.venv_python),
            "inference_script": str(SEEDVR_INFERENCE_SCRIPT),
            "model_dir": str(SEEDVR_MODEL_DIR),
            "scene_split_deps_ready": scene_split_deps_ready,
        }

    def ensure_runtime(
        self,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """准备 SeedVR 独立运行时（仅 venv + pip 依赖）。"""
        python312 = resolve_python312()
        if not python312:
            raise SeedVRRuntimeError(
                "Python 3.12 not found. Please install python3.12 or set SEEDVR_PYTHON."
            )

        self.runtime_root.mkdir(parents=True, exist_ok=True)
        SEEDVR_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        _safe_emit(progress_callback, {"message": "Preparing SeedVR Python environment..."})
        if cancel_event is not None and cancel_event.is_set():
            raise SeedVRRuntimeError("SeedVR runtime preparation cancelled")

        if not self.venv_python.exists():
            self._run_command(
                [python312, "-m", "venv", str(self.venv_dir)],
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                message="Creating SeedVR virtual environment...",
                timeout_sec=240,
            )

        if not SEEDVR_REQUIREMENTS_FILE.exists():
            raise SeedVRRuntimeError(
                f"SeedVR requirements file missing: {SEEDVR_REQUIREMENTS_FILE}. "
                "Download SeedVR runtime source first."
            )

        if not SEEDVR_DEPS_MARKER.exists():
            self._run_command(
                [str(self.venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                message="Installing SeedVR base Python packages...",
                timeout_sec=600,
            )
            self._run_command(
                [str(self.venv_python), "-m", "pip", "install", "-r", str(SEEDVR_REQUIREMENTS_FILE)],
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                message="Installing SeedVR runtime dependencies...",
                timeout_sec=1800,
            )

        missing_scene_split_deps = self._missing_scene_split_packages()
        if missing_scene_split_deps:
            _safe_emit(progress_callback, {"message": "Installing scene split dependencies..."})
            self._run_command(
                [str(self.venv_python), "-m", "pip", "install", *missing_scene_split_deps],
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                message="Installing scene split runtime dependencies...",
                timeout_sec=1200,
            )

        SEEDVR_DEPS_MARKER.write_text(str(int(time.time())), encoding="utf-8")

        state = {
            "prepared_at": int(time.time()),
            "python312": python312,
            "venv_python": str(self.venv_python),
            "inference_script": str(SEEDVR_INFERENCE_SCRIPT),
            "model_dir": str(SEEDVR_MODEL_DIR),
        }
        try:
            SEEDVR_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return state

    def run_inference(
        self,
        *,
        input_path: str,
        output_dir: str,
        dit_model_name: str,
        target_short_resolution: int,
        denoise_strength: float,
        same_res_strength: str,
        batch_size: int = 1,
        chunk_size: int = 10,
        temporal_overlap: int = 1,
        max_resolution: int = 1080,
        vae_encode_tiled: bool = True,
        vae_decode_tiled: bool = True,
        vae_tile_size: int = 768,
        vae_tile_overlap: int = 128,
        dit_offload_device: str = "none",
        vae_offload_device: str = "none",
        tensor_offload_device: str = "cpu",
        cache_dit: bool = False,
        cache_vae: bool = False,
        mps_high_watermark_ratio: Optional[float] = None,
        mps_low_watermark_ratio: Optional[float] = None,
        memory_guard_min_available_gb: Optional[float] = None,
        memory_guard_max_process_rss_gb: Optional[float] = None,
        cancel_event: Optional[Any] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        video_backend_preference: str = DEFAULT_VIDEO_BACKEND,
        ffmpeg_bin: str = "",
        timeout_sec: float = 7200.0,
    ) -> str:
        """
        调用 SeedVR2 CLI 完成推理，返回生成视频路径。
        """
        _ = same_res_strength
        status = self.get_status()
        if not status.get("ready"):
            self.ensure_runtime(progress_callback=progress_callback, cancel_event=cancel_event)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # SeedVR 参数映射：保持与现有 UI 字段兼容。
        denoise_strength = max(0.0, min(1.0, float(denoise_strength)))
        normalized_batch_size = int(batch_size) if int(batch_size) > 0 else self._recommend_batch_size(dit_model_name)
        if ((normalized_batch_size - 1) % 4) != 0:
            # SeedVR CLI 要求 batch_size 满足 4n+1，自动矫正到最近合法值。
            normalized_batch_size = 1 + (max(1, normalized_batch_size - 1) // 4) * 4
            if normalized_batch_size < 1:
                normalized_batch_size = 1
        input_noise_scale = denoise_strength * 0.2
        latent_noise_scale = denoise_strength * 0.3
        normalized_chunk_size = max(0, int(chunk_size))
        normalized_temporal_overlap = max(0, int(temporal_overlap))
        normalized_max_resolution = max(0, int(max_resolution))
        normalized_vae_tile_size = max(256, int(vae_tile_size))
        normalized_vae_tile_overlap = max(16, int(vae_tile_overlap))
        normalized_dit_offload = str(dit_offload_device or "cpu").strip() or "cpu"
        normalized_vae_offload = str(vae_offload_device or "cpu").strip() or "cpu"
        normalized_tensor_offload = str(tensor_offload_device or "cpu").strip() or "cpu"

        env_overrides: Dict[str, str] = {}
        if mps_high_watermark_ratio is not None:
            env_overrides["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{float(mps_high_watermark_ratio):.2f}"
        if mps_low_watermark_ratio is not None:
            env_overrides["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{float(mps_low_watermark_ratio):.2f}"
        preferred_backend = self._normalize_video_backend(video_backend_preference)
        backend_candidates = [preferred_backend]
        if preferred_backend != FALLBACK_VIDEO_BACKEND:
            backend_candidates.append(FALLBACK_VIDEO_BACKEND)

        memory_guard_min = (
            float(memory_guard_min_available_gb)
            if memory_guard_min_available_gb is not None
            else 1.6
        )
        memory_guard_max = (
            float(memory_guard_max_process_rss_gb)
            if memory_guard_max_process_rss_gb is not None
            else None
        )

        last_error: Optional[SeedVRRuntimeError] = None
        for backend in backend_candidates:
            cmd = self._build_inference_command(
                input_path=input_path,
                output_dir=out_dir,
                dit_model_name=dit_model_name,
                target_short_resolution=target_short_resolution,
                batch_size=normalized_batch_size,
                chunk_size=normalized_chunk_size,
                temporal_overlap=normalized_temporal_overlap,
                max_resolution=normalized_max_resolution,
                input_noise_scale=input_noise_scale,
                latent_noise_scale=latent_noise_scale,
                backend=backend,
                dit_offload_device=normalized_dit_offload,
                vae_offload_device=normalized_vae_offload,
                tensor_offload_device=normalized_tensor_offload,
                cache_dit=cache_dit,
                cache_vae=cache_vae,
                vae_encode_tiled=vae_encode_tiled,
                vae_decode_tiled=vae_decode_tiled,
                vae_tile_size=normalized_vae_tile_size,
                vae_tile_overlap=normalized_vae_tile_overlap,
            )
            current_env = dict(env_overrides)
            if backend == "ffmpeg":
                ffmpeg_exec = str(ffmpeg_bin or "").strip()
                if ffmpeg_exec:
                    ffmpeg_dir = str(Path(ffmpeg_exec).expanduser().resolve().parent)
                    current_path = str(current_env.get("PATH") or os.environ.get("PATH") or "")
                    current_env["PATH"] = f"{ffmpeg_dir}:{current_path}" if current_path else ffmpeg_dir

            try:
                self._run_command(
                    cmd,
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                    message=self._build_runtime_message(
                        backend=backend,
                        dit_offload=normalized_dit_offload,
                        vae_offload=normalized_vae_offload,
                    ),
                    timeout_sec=timeout_sec,
                    parse_progress=True,
                    env_overrides=current_env,
                    memory_guard_min_available_gb=memory_guard_min,
                    memory_guard_max_process_rss_gb=memory_guard_max,
                    output_watch_dir=out_dir,
                )
                generated = self._find_latest_video(out_dir)
                if generated is None:
                    raise SeedVRRuntimeError(
                        f"SeedVR inference completed but no output video found in {out_dir}"
                    )
                return str(generated)
            except SeedVRRuntimeError as exc:
                last_error = exc
                if cancel_event is not None and cancel_event.is_set():
                    raise
                if backend != "ffmpeg":
                    raise
                if not self._is_ffmpeg_backend_fallback_error(str(exc)):
                    raise
                self._clear_output_directory(out_dir)
                _safe_emit(
                    progress_callback,
                    {
                        "message": "FFmpeg backend failed, retrying with OpenCV backend...",
                        "progress": 0.0,
                    },
                )
                continue

        if last_error is not None:
            raise last_error
        raise SeedVRRuntimeError("SeedVR inference failed: no backend candidates available")

    @staticmethod
    def _build_runtime_message(*, backend: str, dit_offload: str, vae_offload: str) -> str:
        normalized_backend = str(backend or FALLBACK_VIDEO_BACKEND).strip().lower()
        dit = str(dit_offload or "none").strip().lower()
        vae = str(vae_offload or "none").strip().lower()
        if dit == "none" and vae == "none":
            return (
                "Running SeedVR2 inference "
                f"(device=mps, backend={normalized_backend}, offload=none)..."
            )
        if dit == "cpu" and vae == "cpu":
            return (
                "Running SeedVR2 inference "
                f"(device=mps, backend={normalized_backend}, offload=cpu)..."
            )
        return (
            "Running SeedVR2 inference "
            f"(device=mps, backend={normalized_backend}, offload=mixed)..."
        )

    @staticmethod
    def _normalize_video_backend(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"ffmpeg", "opencv"}:
            return normalized
        return DEFAULT_VIDEO_BACKEND

    @staticmethod
    def _is_ffmpeg_backend_fallback_error(text: str) -> bool:
        lower = str(text or "").lower()
        if "cancelled" in lower:
            return False
        if (
            "memory guard triggered" in lower
            or "out of memory" in lower
            or "inference stalled" in lower
            or "warmup stalled" in lower
            or "no forward progress" in lower
        ):
            return False
        if "ffmpeg backend produced no progress output" in lower:
            return True
        if "ffmpeg backend timed out before first progress" in lower:
            return True
        if "--video_backend ffmpeg" in lower or "video_backend ffmpeg" in lower:
            return True
        if "ffmpeg" in lower and (
            "unknown encoder" in lower
            or "error while opening" in lower
            or "error opening output" in lower
            or "no such file or directory" in lower
        ):
            return True
        return False

    @staticmethod
    def _clear_output_directory(directory: Path) -> None:
        target = Path(directory)
        if not target.exists():
            return
        try:
            for path in target.iterdir():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        except Exception:
            return

    def _build_inference_command(
        self,
        *,
        input_path: str,
        output_dir: Path,
        dit_model_name: str,
        target_short_resolution: int,
        batch_size: int,
        chunk_size: int,
        temporal_overlap: int,
        max_resolution: int,
        input_noise_scale: float,
        latent_noise_scale: float,
        backend: str,
        dit_offload_device: str,
        vae_offload_device: str,
        tensor_offload_device: str,
        cache_dit: bool,
        cache_vae: bool,
        vae_encode_tiled: bool,
        vae_decode_tiled: bool,
        vae_tile_size: int,
        vae_tile_overlap: int,
    ) -> list[str]:
        cmd = [
            str(self.venv_python),
            "-u",
            str(SEEDVR_INFERENCE_SCRIPT),
            "--output",
            str(output_dir),
            "--model_dir",
            str(SEEDVR_MODEL_DIR),
            "--dit_model",
            str(dit_model_name),
            "--resolution",
            str(max(256, int(target_short_resolution))),
            "--batch_size",
            str(max(1, int(batch_size))),
            "--chunk_size",
            str(max(0, int(chunk_size))),
            "--temporal_overlap",
            str(max(0, int(temporal_overlap))),
            "--max_resolution",
            str(max(0, int(max_resolution))),
            "--input_noise_scale",
            f"{float(input_noise_scale):.3f}",
            "--latent_noise_scale",
            f"{float(latent_noise_scale):.3f}",
            "--video_backend",
            str(self._normalize_video_backend(backend)),
            "--dit_offload_device",
            str(dit_offload_device),
            "--vae_offload_device",
            str(vae_offload_device),
            "--tensor_offload_device",
            str(tensor_offload_device),
        ]
        if cache_dit:
            cmd.append("--cache_dit")
        if cache_vae:
            cmd.append("--cache_vae")
        if vae_encode_tiled:
            cmd.append("--vae_encode_tiled")
        if vae_decode_tiled:
            cmd.append("--vae_decode_tiled")
        if vae_encode_tiled or vae_decode_tiled:
            cmd.extend(
                [
                    "--vae_encode_tile_size",
                    str(max(256, int(vae_tile_size))),
                    "--vae_encode_tile_overlap",
                    str(max(16, int(vae_tile_overlap))),
                    "--vae_decode_tile_size",
                    str(max(256, int(vae_tile_size))),
                    "--vae_decode_tile_overlap",
                    str(max(16, int(vae_tile_overlap))),
                ]
            )
        cmd.append(str(input_path))
        return cmd

    @staticmethod
    def _recommend_batch_size(dit_model_name: str) -> int:
        _ = dit_model_name
        # 稳定优先：默认固定 1，避免批次放大导致内存峰值。
        return 1

    def _has_python_module(self, module_name: str) -> bool:
        if not self.venv_python.exists():
            return False
        try:
            result = subprocess.run(
                [str(self.venv_python), "-c", f"import {module_name}"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return int(result.returncode or 1) == 0
        except Exception:
            return False

    def _missing_scene_split_packages(self) -> list[str]:
        missing: list[str] = []
        for module_name, package_name in SCENE_SPLIT_RUNTIME_DEPS:
            if not self._has_python_module(module_name):
                missing.append(package_name)
        return missing

    @staticmethod
    def _find_latest_video(directory: Path) -> Optional[Path]:
        candidates = []
        for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"):
            candidates.extend(directory.rglob(ext))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    @staticmethod
    def _resolve_logical_cpu_count() -> int:
        try:
            return max(1, int(os.cpu_count() or 1))
        except Exception:
            return 1

    @staticmethod
    def _resolve_thread_cap() -> int:
        logical = SeedVRRuntime._resolve_logical_cpu_count()
        return max(1, int(logical * 0.8))

    @staticmethod
    def _normalize_cpu_percent(process_cpu_percent: float, logical_cpu_count: int) -> float:
        cpus = max(1, int(logical_cpu_count or 1))
        raw = max(0.0, float(process_cpu_percent))
        return max(0.0, min(100.0, raw / float(cpus)))

    @staticmethod
    def _next_cpu_overload_streak(current_streak: int, normalized_cpu_percent: float) -> int:
        if float(normalized_cpu_percent) > float(LOAD_CAP_CPU_PERCENT):
            return int(current_streak) + 1
        return 0

    @staticmethod
    def _load_governor_supported() -> bool:
        return bool(
            LOAD_CAP_ENABLED
            and psutil is not None
            and hasattr(signal, "SIGSTOP")
            and hasattr(signal, "SIGCONT")
        )

    @staticmethod
    def _throttle_process(proc: subprocess.Popen[str], pause_sec: float) -> float:
        pause = max(0.0, float(pause_sec))
        if pause <= 0.0:
            return 0.0
        started = time.time()
        try:
            os.kill(int(proc.pid), signal.SIGSTOP)
        except Exception:
            return 0.0
        try:
            time.sleep(pause)
        finally:
            try:
                os.kill(int(proc.pid), signal.SIGCONT)
            except Exception:
                pass
        return max(0.0, time.time() - started)

    def _run_command(
        self,
        cmd: list[str],
        *,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        message: str,
        timeout_sec: float,
        parse_progress: bool = False,
        env_overrides: Optional[Dict[str, str]] = None,
        memory_guard_min_available_gb: Optional[float] = None,
        memory_guard_max_process_rss_gb: Optional[float] = None,
        output_watch_dir: Optional[Path] = None,
    ) -> None:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(SEEDVR_REPO_DIR if SEEDVR_REPO_DIR.exists() else self.runtime_root),
            env=self._build_env(env_overrides=env_overrides),
        )
        if proc.stdout is None:
            raise SeedVRRuntimeError("Failed to capture SeedVR process output")

        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        started = time.time()
        last_emit = 0.0
        tail_lines: list[str] = []
        progress_regex = re.compile(r"^PROGRESS\s+chunk=(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
        warmup_stage_regex = re.compile(r"PROGRESS\s+stage=warmup\s+step=([a-zA-Z0-9_\-]+)", re.IGNORECASE)
        parsed_progress = 0.0
        low_memory_streak = 0
        high_rss_streak = 0
        cpu_overload_streak = 0
        governor_active_until = 0.0
        pause_window_started = started
        pause_count = 0
        logical_cpu_count = self._resolve_logical_cpu_count()
        load_governor_enabled = bool(parse_progress and self._load_governor_supported())
        last_activity_at = started
        last_forward_progress_at = started
        output_mtime = self._latest_mtime(output_watch_dir)
        output_progress_observed = False
        output_progress_started_at = started
        last_parsed_progress_at = started
        phase_tag = "model_warmup"
        warmup_mode = bool(parse_progress)
        warmup_step_token = "bootstrap"
        chunk_loop_started_at: Optional[float] = None
        message_lower = str(message or "").lower()
        is_ffmpeg_runtime = "backend=ffmpeg" in message_lower
        last_cpu_total = 0.0
        process_handle: Optional[Any] = None
        if psutil is not None and (
            memory_guard_max_process_rss_gb is not None or load_governor_enabled
        ):
            try:
                process_handle = psutil.Process(proc.pid)
            except Exception:
                process_handle = None
        if process_handle is not None and load_governor_enabled:
            try:
                process_handle.cpu_percent(interval=None)
            except Exception:
                pass
        if process_handle is not None and parse_progress:
            try:
                cpu_times = process_handle.cpu_times()
                last_cpu_total = float(cpu_times.user) + float(cpu_times.system)
            except Exception:
                last_cpu_total = 0.0
        runtime_log_interval_sec = 15.0
        last_runtime_log_at = started

        try:
            while True:
                now = time.time()
                if (now - started) > timeout_sec:
                    proc.kill()
                    if (
                        parse_progress
                        and is_ffmpeg_runtime
                        and parsed_progress <= 0.0
                        and not output_progress_observed
                    ):
                        raise SeedVRRuntimeError(
                            f"FFmpeg backend timed out before first progress ({int(timeout_sec)}s)"
                        )
                    raise SeedVRRuntimeError(f"SeedVR command timed out after {int(timeout_sec)}s")

                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                    raise SeedVRRuntimeError("SeedVR command cancelled")

                if memory_guard_min_available_gb is not None:
                    available_gb = self._available_memory_gb()
                    if available_gb > 0 and available_gb < float(memory_guard_min_available_gb):
                        low_memory_streak += 1
                    else:
                        low_memory_streak = 0
                    if low_memory_streak >= 3:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        raise SeedVRRuntimeError(
                            f"Memory guard triggered: available system memory dropped below "
                            f"{float(memory_guard_min_available_gb):.1f}GB"
                        )

                if (
                    process_handle is not None
                    and memory_guard_max_process_rss_gb is not None
                    and memory_guard_max_process_rss_gb > 0
                ):
                    try:
                        rss_gb = float(process_handle.memory_info().rss) / (1024 ** 3)
                    except Exception:
                        rss_gb = 0.0
                    if rss_gb > float(memory_guard_max_process_rss_gb):
                        high_rss_streak += 1
                    else:
                        high_rss_streak = 0
                    if high_rss_streak >= 3:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        raise SeedVRRuntimeError(
                            f"Memory guard triggered: SeedVR RSS exceeded "
                            f"{float(memory_guard_max_process_rss_gb):.1f}GB"
                        )

                if process_handle is not None and parse_progress:
                    try:
                        cpu_times = process_handle.cpu_times()
                        current_cpu_total = float(cpu_times.user) + float(cpu_times.system)
                    except Exception:
                        current_cpu_total = last_cpu_total
                    last_cpu_total = max(last_cpu_total, current_cpu_total)

                if load_governor_enabled and process_handle is not None:
                    should_throttle = False
                    warmup_throttle_blocked = bool(
                        parse_progress
                        and warmup_mode
                        and (now - started) < float(LOAD_CAP_WARMUP_GRACE_SEC)
                    )
                    if not warmup_throttle_blocked:
                        try:
                            process_cpu = float(process_handle.cpu_percent(interval=None))
                        except Exception:
                            process_cpu = 0.0
                        normalized_cpu = self._normalize_cpu_percent(process_cpu, logical_cpu_count)
                        cpu_overload_streak = self._next_cpu_overload_streak(
                            cpu_overload_streak, normalized_cpu
                        )
                        if cpu_overload_streak >= int(LOAD_CAP_MIN_STREAK):
                            should_throttle = True
                            cpu_overload_streak = 0
                    else:
                        cpu_overload_streak = 0

                    if should_throttle:
                        if (now - pause_window_started) >= 60.0:
                            pause_window_started = now
                            pause_count = 0
                        if pause_count < int(LOAD_CAP_MAX_PAUSES_PER_MINUTE):
                            paused_sec = self._throttle_process(proc, LOAD_CAP_PAUSE_SEC)
                            if paused_sec > 0:
                                pause_count += 1
                                governor_active_until = time.time() + 3.0

                events = selector.select(timeout=0.35)
                for key, _ in events:
                    line = key.fileobj.readline()  # type: ignore[attr-defined]
                    if not line:
                        continue
                    text = str(line or "").strip()
                    if text:
                        last_activity_at = now
                        tail_lines.append(text)
                        if len(tail_lines) > 20:
                            tail_lines.pop(0)

                        if parse_progress:
                            lower_text = text.lower()
                            warmup_match = warmup_stage_regex.search(text)
                            if warmup_match:
                                warmup_step_token = str(warmup_match.group(1) or "").strip().lower() or warmup_step_token
                                phase_tag = "model_warmup"
                                if warmup_step_token == "chunk_loop_start" and chunk_loop_started_at is None:
                                    chunk_loop_started_at = now
                            match = progress_regex.search(text)
                            if match:
                                done = int(match.group(1) or 0)
                                total = int(match.group(2) or 0)
                                if total > 0 and done <= total:
                                    frac = min(1.0, max(0.0, float(done) / float(total)))
                                    if frac > parsed_progress:
                                        parsed_progress = frac
                                        last_forward_progress_at = now
                                        last_parsed_progress_at = now
                                        output_progress_started_at = now
                                        phase_tag = "chunk_infer"
                                        warmup_mode = False
                            elif "streaming complete" in lower_text or "saving output" in lower_text:
                                phase_tag = "flush_output"
                                warmup_mode = False
                                last_forward_progress_at = now

                next_mtime = self._latest_mtime(output_watch_dir)
                if next_mtime > output_mtime:
                    output_mtime = next_mtime
                    output_progress_observed = True
                    last_activity_at = now
                    last_forward_progress_at = now
                    last_parsed_progress_at = now
                    output_progress_started_at = now
                    warmup_mode = False
                    if parse_progress and phase_tag == "model_warmup":
                        phase_tag = "chunk_infer"

                if parse_progress:
                    if (
                        parsed_progress <= 0.0
                        and not output_progress_observed
                        and (now - output_progress_started_at) >= float(WARMUP_NO_OUTPUT_TIMEOUT_SEC)
                    ):
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        raise SeedVRRuntimeError(
                            f"Warmup stalled (no output progress for {int(WARMUP_NO_OUTPUT_TIMEOUT_SEC)}s)"
                        )
                    if (
                        warmup_mode
                        and is_ffmpeg_runtime
                        and chunk_loop_started_at is not None
                        and parsed_progress <= 0.0
                        and not output_progress_observed
                        and (now - chunk_loop_started_at) >= float(WARMUP_NO_PROGRESS_TIMEOUT_SEC)
                    ):
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        raise SeedVRRuntimeError(
                            f"FFmpeg backend produced no progress output after warmup "
                            f"({int(WARMUP_NO_PROGRESS_TIMEOUT_SEC)}s)"
                        )
                    if (
                        not warmup_mode
                        and (now - last_parsed_progress_at) >= float(RUN_PARSED_PROGRESS_TIMEOUT_SEC)
                    ):
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        raise SeedVRRuntimeError(
                            f"Inference stalled (no parsed progress for {int(RUN_PARSED_PROGRESS_TIMEOUT_SEC)}s)"
                        )
                    if warmup_mode:
                        stalled = (now - last_activity_at) >= float(WARMUP_STALL_TIMEOUT_SEC)
                    else:
                        run_anchor = last_forward_progress_at
                        stalled = (now - run_anchor) >= float(RUN_STALL_TIMEOUT_SEC)
                    if stalled:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        if warmup_mode:
                            raise SeedVRRuntimeError("Warmup stalled (no activity for 240s)")
                        raise SeedVRRuntimeError("Inference stalled (no forward progress for 90s)")

                # 运行时心跳，避免前端长时间无状态变化。
                if progress_callback and (now - last_emit) > 0.55:
                    heartbeat_message = f"{message} phase={phase_tag}"
                    if warmup_mode:
                        heartbeat_message = f"{heartbeat_message} stage=warmup"
                        if warmup_step_token:
                            heartbeat_message = f"{heartbeat_message} step={warmup_step_token}"
                    if now <= governor_active_until:
                        heartbeat_message = (
                            f"{heartbeat_message} Load governor active: throttling to 80% target."
                        )
                    _safe_emit(
                        progress_callback,
                        {
                            "message": heartbeat_message,
                            "progress": parsed_progress if parse_progress else 0.0,
                        },
                    )
                    if (now - last_runtime_log_at) >= runtime_log_interval_sec:
                        try:
                            logger.info(
                                "[seedvr-runtime] %s progress=%.3f",
                                heartbeat_message,
                                parsed_progress if parse_progress else 0.0,
                            )
                        except Exception:
                            pass
                        last_runtime_log_at = now
                    last_emit = now

                ret = proc.poll()
                if ret is not None:
                    if ret != 0:
                        tail = "\n".join(tail_lines[-10:]) if tail_lines else "unknown runtime error"
                        raise SeedVRRuntimeError(
                            f"SeedVR command failed (code {ret}): {' '.join(cmd)}\n{tail}"
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
    def _available_memory_gb() -> float:
        if psutil is None:  # pragma: no cover - 运行时兜底
            return 0.0
        try:
            return float(psutil.virtual_memory().available) / (1024 ** 3)
        except Exception:  # pragma: no cover - 运行时兜底
            return 0.0

    @staticmethod
    def _build_env(*, env_overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = dict(os.environ)
        # 避免 tokenizer 并行导致噪声日志或锁竞争。
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONUNBUFFERED"] = "1"
        # Apple 芯片场景固定 MPS fallback，并且默认注入保守水位。
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.72"
        env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.56"
        thread_cap = str(SeedVRRuntime._resolve_thread_cap())
        env["OMP_NUM_THREADS"] = thread_cap
        env["MKL_NUM_THREADS"] = thread_cap
        env["OPENBLAS_NUM_THREADS"] = thread_cap
        env["VECLIB_MAXIMUM_THREADS"] = thread_cap
        env["NUMEXPR_NUM_THREADS"] = thread_cap
        if env_overrides:
            env.update({k: str(v) for k, v in env_overrides.items()})
        return env

    @staticmethod
    def _latest_mtime(directory: Optional[Path]) -> float:
        if directory is None:
            return 0.0
        target = Path(directory)
        if not target.exists():
            return 0.0
        try:
            latest = float(target.stat().st_mtime)
        except Exception:
            latest = 0.0
        try:
            for path in target.rglob("*"):
                try:
                    mtime = float(path.stat().st_mtime)
                except Exception:
                    continue
                if mtime > latest:
                    latest = mtime
        except Exception:
            pass
        return latest


__all__ = [
    "SeedVRRuntime",
    "SeedVRRuntimeError",
    "resolve_python312",
]
