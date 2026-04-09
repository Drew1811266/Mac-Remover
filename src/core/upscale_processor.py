"""
视频 AI 放大处理器（双引擎：Real-ESRGAN + SeedVR2）。

能力：
1) Real-ESRGAN / SeedVR2 推理（Apple Silicon / MPS 优先）；
2) 两种模式：提升分辨率 / 同分辨率清晰增强；
3) 进度回调、任务取消、后处理封装（含音频回灌）。
"""

from __future__ import annotations

import os
import selectors
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2

from .realesrgan_manifest import (
    REALESRGAN_DEFAULT_MODEL_ID,
    REALESRGAN_ENGINE_ID,
    REALESRGAN_MODEL_SPECS,
    REALESRGAN_UPSCALE_MODELS,
    get_realesrgan_model_spec,
)
from .realesrgan_runtime import RealESRGANRuntime, RealESRGANRuntimeError
from ..utils.ffmpeg_runtime import resolve_ffmpeg_path
from ..utils.memory_cleanup import release_unified_memory
from .scene_splitter import SceneSegment, SceneSplitCancelled, SceneSplitResult, SceneSplitter
from .seedvr_memory_policy import (
    build_emergency_seedvr_profile,
    build_stall_recovery_seedvr_profile,
    build_seedvr_memory_profile,
    detect_system_memory_gb,
    normalize_same_res_strength,
)
from .seedvr_manifest import (
    SEEDVR_DEFAULT_MODEL_ID,
    SEEDVR_ENGINE_ID,
    SEEDVR_MODEL_SPECS,
    SEEDVR_UPSCALE_MODELS,
    get_seedvr_model_spec,
)
from .seedvr_runtime import SeedVRRuntime, SeedVRRuntimeError
from .upscale_model_downloader import is_upscale_model_installed


UPSCALE_MODES = ("upscale_resolution", "enhance_same_resolution")
UPSCALE_ENGINES = (REALESRGAN_ENGINE_ID, SEEDVR_ENGINE_ID)
UPSCALE_TARGET_PRESETS = ("1080p",)
UPSCALE_SAME_RES_STRENGTHS = ("x2_then_downscale",)
SEEDVR_MODELS = SEEDVR_UPSCALE_MODELS
REALESRGAN_MODELS = REALESRGAN_UPSCALE_MODELS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHORT_VIDEO_SCENE_SPLIT_BYPASS_SEC = 20.0


class UpscaleCancelled(Exception):
    """当用户取消放大任务时抛出。"""


class UpscaleProcessor:
    """视频放大处理入口。"""

    def __init__(self) -> None:
        self._capability_cache: Optional[Dict[str, Any]] = None
        self._capability_cache_fingerprint: Optional[Tuple[Any, ...]] = None
        self._seedvr_runtime = SeedVRRuntime()
        # 兼容旧测试/旧调用方：_runtime 仍指向 SeedVR runtime。
        self._runtime = self._seedvr_runtime
        self._realesrgan_runtime = RealESRGANRuntime()

    def invalidate_capabilities_cache(self) -> None:
        self._capability_cache = None
        self._capability_cache_fingerprint = None

    @staticmethod
    def _round_even(value: float) -> int:
        rounded = int(round(float(value)))
        if rounded < 2:
            rounded = 2
        if rounded % 2 != 0:
            rounded += 1
        return rounded

    @staticmethod
    def _video_meta(path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions: {path}")
        if fps <= 0:
            fps = 24.0
        duration_sec = float(frame_count / fps) if frame_count > 0 else 0.0

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": max(0.0, duration_sec),
        }

    @staticmethod
    def _build_output_name(input_path: str, mode: str, suffix: str) -> str:
        stem = Path(input_path).stem
        safe_suffix = suffix.replace("/", "_").replace(" ", "_")
        return f"{stem}_upscaled_{mode}_{safe_suffix}.mp4"

    @staticmethod
    def _segment_label(index: int, total: int, start: float, end: float) -> str:
        return f"segment {index}/{total} ({start:.1f}s-{end:.1f}s)"

    def _compute_target_resolution(
        self,
        source_w: int,
        source_h: int,
        mode: str,
        target_preset: Optional[str],
    ) -> Tuple[int, int]:
        if mode == "enhance_same_resolution":
            return source_w, source_h

        preset_to_height = {"1080p": 1080}
        target_h = int(preset_to_height.get(str(target_preset), 1080))
        if source_h >= target_h:
            return source_w, source_h
        scale = float(target_h) / float(source_h)
        return self._round_even(source_w * scale), target_h

    @staticmethod
    def _estimate_duration(duration_sec: float, target_w: int, target_h: int, source_w: int, source_h: int) -> float:
        if duration_sec <= 0:
            duration_sec = 30.0
        source_pixels = max(1.0, float(source_w * source_h))
        target_pixels = max(1.0, float(target_w * target_h))
        pixel_ratio = target_pixels / source_pixels
        # SeedVR2 执行成本显著高于传统超分，给更保守 ETA。
        return max(45.0, duration_sec * max(3.0, pixel_ratio * 2.2))

    @staticmethod
    def _safe_eta(started_at: float, estimated_total: float, progress_frac: float) -> Optional[float]:
        if estimated_total <= 0:
            return None
        elapsed = max(0.0, time.time() - started_at)
        expected_elapsed = estimated_total * max(0.0, min(progress_frac, 1.0))
        remaining = max(0.0, estimated_total - max(elapsed, expected_elapsed))
        return remaining

    @staticmethod
    def _is_memory_pressure_error(text: str) -> bool:
        lowered = str(text or "").lower()
        keywords = (
            "memory guard triggered",
            "out of memory",
            "mps backend out of memory",
            "not enough memory",
            "allocator",
            "std::bad_alloc",
            "insufficient memory",
        )
        return any(token in lowered for token in keywords)

    @staticmethod
    def _is_stall_error(text: str) -> bool:
        lowered = str(text or "").lower()
        keywords = (
            "inference stalled",
            "no forward progress",
            "stalled",
            "timed out",
            "no progress output",
        )
        return any(token in lowered for token in keywords)

    @staticmethod
    def _check_filter_available(ffmpeg_bin: str, filter_name: str) -> bool:
        try:
            result = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-filters"],
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
        except Exception:
            return False
        content = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        return filter_name.lower() in content

    def get_capabilities(self, *, force_refresh: bool = False) -> Dict[str, Any]:
        ffmpeg_bin = resolve_ffmpeg_path()
        has_ffmpeg = bool(ffmpeg_bin)
        libplacebo_ok = bool(ffmpeg_bin and self._check_filter_available(ffmpeg_bin, "libplacebo"))

        seedvr_runtime = self._seedvr_runtime.get_status()
        seedvr_runtime_ready = bool(seedvr_runtime.get("ready"))
        seedvr_runtime_reason = str(seedvr_runtime.get("reason") or "")
        realesrgan_runtime = self._realesrgan_runtime.get_status()
        realesrgan_runtime_ready = bool(realesrgan_runtime.get("ready"))
        realesrgan_runtime_reason = str(realesrgan_runtime.get("reason") or "")

        seedvr_installed_map = {
            model_id: is_upscale_model_installed(model_id) for model_id in SEEDVR_MODELS
        }
        realesrgan_installed_map = {
            model_id: is_upscale_model_installed(model_id) for model_id in REALESRGAN_MODELS
        }
        has_seedvr_model = any(seedvr_installed_map.values())
        has_realesrgan_model = any(realesrgan_installed_map.values())
        model_installed_map = {**seedvr_installed_map, **realesrgan_installed_map}

        fingerprint: Tuple[Any, ...] = (
            str(ffmpeg_bin or ""),
            has_ffmpeg,
            libplacebo_ok,
            seedvr_runtime_ready,
            seedvr_runtime_reason,
            realesrgan_runtime_ready,
            realesrgan_runtime_reason,
            tuple(sorted(model_installed_map.items())),
        )
        if (
            not force_refresh
            and self._capability_cache is not None
            and self._capability_cache_fingerprint == fingerprint
        ):
            return dict(self._capability_cache)

        seedvr_reason_parts = []
        realesrgan_reason_parts = []
        if not has_ffmpeg:
            seedvr_reason_parts.append("FFmpeg runtime unavailable")
            realesrgan_reason_parts.append("FFmpeg runtime unavailable")
        if not seedvr_runtime_ready:
            seedvr_reason_parts.append(seedvr_runtime_reason or "SeedVR runtime unavailable")
        if seedvr_runtime_ready and not has_seedvr_model:
            seedvr_reason_parts.append("SeedVR model not installed")
        if not realesrgan_runtime_ready:
            realesrgan_reason_parts.append(
                realesrgan_runtime_reason or "Real-ESRGAN runtime unavailable"
            )
        if realesrgan_runtime_ready and not has_realesrgan_model:
            realesrgan_reason_parts.append("Real-ESRGAN model not installed")

        seedvr_available = bool(has_ffmpeg and seedvr_runtime_ready and has_seedvr_model)
        realesrgan_available = bool(has_ffmpeg and realesrgan_runtime_ready and has_realesrgan_model)
        default_engine = REALESRGAN_ENGINE_ID
        default_model = REALESRGAN_DEFAULT_MODEL_ID
        if not realesrgan_available and seedvr_available:
            default_engine = SEEDVR_ENGINE_ID
            default_model = SEEDVR_DEFAULT_MODEL_ID

        capabilities = {
            "success": True,
            "engines": [
                {
                    "engine": REALESRGAN_ENGINE_ID,
                    "display_name": "Real-ESRGAN",
                    "available": realesrgan_available,
                    "reason": "" if realesrgan_available else "; ".join(realesrgan_reason_parts),
                    "runtime_hint": "" if realesrgan_runtime_ready else (
                        realesrgan_runtime_reason or "Python 3.12 runtime required"
                    ),
                },
                {
                    "engine": SEEDVR_ENGINE_ID,
                    "display_name": "SeedVR2",
                    "available": seedvr_available,
                    "reason": "" if seedvr_available else "; ".join(seedvr_reason_parts),
                    "runtime_hint": "" if seedvr_runtime_ready else (
                        seedvr_runtime_reason or "Python 3.12 runtime required"
                    ),
                },
            ],
            "models": (
                [
                    {
                        "engine": REALESRGAN_ENGINE_ID,
                        "model_id": model_id,
                        "display_name": spec.display_name,
                        "installed": bool(model_installed_map.get(model_id)),
                    }
                    for model_id, spec in REALESRGAN_MODEL_SPECS.items()
                ]
                + [
                    {
                        "engine": SEEDVR_ENGINE_ID,
                        "model_id": model_id,
                        "display_name": spec.display_name,
                        "installed": bool(model_installed_map.get(model_id)),
                    }
                    for model_id, spec in SEEDVR_MODEL_SPECS.items()
                ]
            ),
            "modes": list(UPSCALE_MODES),
            "target_presets": list(UPSCALE_TARGET_PRESETS),
            "same_res_strengths": list(UPSCALE_SAME_RES_STRENGTHS),
            "defaults": {
                "engine": default_engine,
                "mode": "upscale_resolution",
                "model_id": default_model,
                "target_preset": "1080p",
                "same_res_strength": "x2_then_downscale",
                "denoise_strength": 0.35,
                "keep_audio": True,
            },
            "ffmpeg": {
                "available": has_ffmpeg,
                "libplacebo_available": libplacebo_ok,
            },
            "runtime": realesrgan_runtime,
            "runtime_by_engine": {
                REALESRGAN_ENGINE_ID: realesrgan_runtime,
                SEEDVR_ENGINE_ID: seedvr_runtime,
            },
        }
        # 向后兼容：历史前端只读取 runtime 字段时仍可得到默认引擎状态。
        capabilities["runtime"] = realesrgan_runtime
        self._capability_cache = dict(capabilities)
        self._capability_cache_fingerprint = fingerprint
        return dict(capabilities)

    def _run_ffmpeg_with_progress(
        self,
        *,
        cmd: list[str],
        duration_sec: float,
        estimated_total_sec: float,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        progress_start: float = 0.20,
        progress_span: float = 0.72,
        phase: str = "infer",
        message: str = "Processing video...",
    ) -> None:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if proc.stderr is None:
            raise RuntimeError("Failed to capture ffmpeg progress stream")

        selector = selectors.DefaultSelector()
        selector.register(proc.stderr, selectors.EVENT_READ)

        infer_progress = 0.0
        started_at = time.time()
        last_progress_emit = 0.0
        tail_lines: list[str] = []

        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                    raise UpscaleCancelled("Upscale task cancelled")

                events = selector.select(timeout=0.35)
                for key, _ in events:
                    line = key.fileobj.readline()  # type: ignore[attr-defined]
                    if not line:
                        continue
                    text = line.strip()
                    if text:
                        tail_lines.append(text)
                        if len(tail_lines) > 12:
                            tail_lines.pop(0)
                    if text.startswith("out_time_ms="):
                        try:
                            out_time_ms = int(text.split("=", 1)[1] or "0")
                        except ValueError:
                            out_time_ms = 0
                        if duration_sec > 0:
                            infer_progress = min(1.0, max(0.0, (out_time_ms / 1_000_000.0) / duration_sec))

                if progress_callback and (time.time() - last_progress_emit) > 0.45:
                    global_progress = progress_start + infer_progress * progress_span
                    eta = self._safe_eta(started_at, estimated_total_sec, global_progress)
                    progress_callback(
                        {
                            "progress": global_progress,
                            "phase": phase,
                            "message": message,
                            "eta_seconds": eta,
                        }
                    )
                    last_progress_emit = time.time()

                ret = proc.poll()
                if ret is not None:
                    if ret != 0:
                        tail = "\n".join(tail_lines[-6:]) if tail_lines else "unknown ffmpeg error"
                        raise RuntimeError(f"FFmpeg command failed: {tail}")
                    break
        finally:
            try:
                selector.unregister(proc.stderr)
            except Exception:
                pass
            selector.close()
            proc.stderr.close()

    @staticmethod
    def _resolve_seedvr_short_side(
        *,
        source_w: int,
        source_h: int,
        target_w: int,
        target_h: int,
        mode: str,
        same_res_strength: str,
    ) -> int:
        _ = (source_w, source_h, target_w, target_h, mode, same_res_strength)
        # 固定稳定策略：内部推理短边统一限制为 1080。
        return 1080

    @staticmethod
    def _is_720p_baseline(width: int, height: int) -> bool:
        short_side = min(int(width), int(height))
        long_side = max(int(width), int(height))
        return short_side == 720 and long_side == 1280

    @staticmethod
    def _resolve_ffmpeg_thread_cap() -> int:
        try:
            logical = max(1, int(os.cpu_count() or 1))
        except Exception:
            logical = 1
        return max(1, int(logical * 0.8))

    def _prepare_seedvr_input_720p(
        self,
        *,
        ffmpeg_bin: str,
        input_path: str,
        source_w: int,
        source_h: int,
        duration_sec: float,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Tuple[str, Optional[Path], list[str]]:
        if self._is_720p_baseline(source_w, source_h):
            return input_path, None, []

        work_dir = Path(tempfile.mkdtemp(prefix="wmr-seedvr-input-"))
        preprocessed_path = work_dir / "seedvr_input_720p.mp4"
        if source_w >= source_h:
            scale_expr = "scale=-2:720:flags=lanczos"
        else:
            scale_expr = "scale=720:-2:flags=lanczos"
        vf = f"{scale_expr},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
        thread_cap = str(self._resolve_ffmpeg_thread_cap())
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            input_path,
            "-progress",
            "pipe:2",
            "-stats_period",
            "0.5",
            "-nostats",
            "-v",
            "error",
            "-threads",
            thread_cap,
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-an",
            str(preprocessed_path),
        ]

        self._run_ffmpeg_with_progress(
            cmd=cmd,
            duration_sec=max(1.0, float(duration_sec)),
            estimated_total_sec=max(4.0, float(duration_sec) * 0.12),
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            progress_start=0.04,
            progress_span=0.08,
            phase="prepare",
            message="Preparing 720p input for stable upscale...",
        )
        if not preprocessed_path.exists():
            raise RuntimeError("Failed to prepare 720p input for SeedVR")

        warnings = [
            "Auto-preprocessed input to 720p for memory safety.",
        ]
        return str(preprocessed_path), work_dir, warnings

    def _run_seedvr2_inference(
        self,
        *,
        ffmpeg_bin: str,
        input_path: str,
        model_id: str,
        mode: str,
        same_res_strength: str,
        denoise_strength: float,
        source_w: int,
        source_h: int,
        target_w: int,
        target_h: int,
        duration_sec: float,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        estimated_total_sec: float,
        progress_start: float = 0.14,
        progress_span: float = 0.70,
        phase: str = "infer",
        message_prefix: str = "Running SeedVR2 inference...",
        segment_index: Optional[int] = None,
        segment_total: Optional[int] = None,
        scene_split_mode: str = "rule",
    ) -> Tuple[str, Path, list[str]]:
        spec = get_seedvr_model_spec(model_id)
        requested_target_short = self._resolve_seedvr_short_side(
            source_w=source_w,
            source_h=source_h,
            target_w=target_w,
            target_h=target_h,
            mode=mode,
            same_res_strength=same_res_strength,
        )
        total_gb, available_gb = detect_system_memory_gb()
        if available_gb > 0.0 and available_gb < 4.0:
            raise RuntimeError(
                f"Available system memory is too low ({available_gb:.1f}GB). "
                "Close other heavy apps and retry AI upscale."
            )
        memory_profile = build_seedvr_memory_profile(
            mode=mode,
            same_res_strength=same_res_strength,
            requested_short_resolution=requested_target_short,
            duration_sec=duration_sec,
            total_memory_gb=total_gb,
            available_memory_gb=available_gb,
            model_id=model_id,
        )
        warnings: list[str] = list(memory_profile.warnings)
        profiles = [memory_profile]
        retried = False

        for attempt_index, profile in enumerate(profiles):
            work_dir = Path(tempfile.mkdtemp(prefix="wmr-seedvr2-"))
            runtime_output_dir = work_dir / "runtime_output"
            runtime_output_dir.mkdir(parents=True, exist_ok=True)
            runtime_started_at = time.time()
            floor_step_sec = 8.0
            floor_step_progress = 0.003  # 每 8 秒至少推进 0.3%（全局进度）
            floor_cap_progress = min(
                float(progress_start) + float(progress_span) * 0.25,
                float(progress_start) + float(progress_span) - 0.01,
            )
            warmup_floor_cap_progress = min(
                float(progress_start) + float(progress_span) * 0.10,
                float(progress_start) + float(progress_span) - 0.02,
            )

            # 将 runtime 内部进度映射到指定全局区间，避免固定 84% 停滞体感。
            def runtime_progress(payload: Dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                frac = max(0.0, min(1.0, float(payload.get("progress") or 0.0)))
                runtime_message = str(payload.get("message") or "Running SeedVR2 inference...")
                message = runtime_message
                if runtime_message.lower().startswith("running seedvr2 inference"):
                    message = f"{message_prefix} | {runtime_message}"
                global_progress = float(progress_start) + frac * float(progress_span)
                elapsed = max(0.0, time.time() - runtime_started_at)
                floor_steps = int(elapsed // floor_step_sec)
                lower_message = runtime_message.lower()
                is_warmup_phase = (
                    "phase=model_warmup" in lower_message
                    or "stage=warmup" in lower_message
                )
                active_floor_cap = warmup_floor_cap_progress if is_warmup_phase else floor_cap_progress
                floor_progress = min(
                    active_floor_cap,
                    float(progress_start) + floor_steps * floor_step_progress,
                )
                global_progress = max(global_progress, floor_progress)
                eta = self._safe_eta(runtime_started_at, estimated_total_sec, global_progress)
                payload_extra: Dict[str, Any] = {
                    "progress": global_progress,
                    "phase": phase,
                    "message": message,
                    "eta_seconds": eta,
                    "scene_split_mode": scene_split_mode,
                }
                if segment_index is not None:
                    payload_extra["segment_index"] = int(segment_index)
                if segment_total is not None:
                    payload_extra["segment_total"] = int(segment_total)
                progress_callback(payload_extra)

            try:
                segment_min_timeout_sec = (
                    360.0 if segment_index is None or int(segment_index) <= 1 else 120.0
                )
                segment_timeout_sec = max(
                    segment_min_timeout_sec,
                    min(
                        1200.0,
                        max(float(duration_sec) * 24.0, float(estimated_total_sec) * 2.0),
                    ),
                )
                generated_path = self._seedvr_runtime.run_inference(
                    input_path=input_path,
                    output_dir=str(runtime_output_dir),
                    dit_model_name=spec.dit_model_name,
                    target_short_resolution=profile.target_short_resolution,
                    denoise_strength=denoise_strength,
                    same_res_strength=same_res_strength,
                    batch_size=profile.batch_size,
                    chunk_size=profile.chunk_size,
                    temporal_overlap=profile.temporal_overlap,
                    max_resolution=profile.max_resolution,
                    vae_encode_tiled=profile.vae_encode_tiled,
                    vae_decode_tiled=profile.vae_decode_tiled,
                    vae_tile_size=profile.vae_tile_size,
                    vae_tile_overlap=profile.vae_tile_overlap,
                    dit_offload_device=profile.dit_offload_device,
                    vae_offload_device=profile.vae_offload_device,
                    tensor_offload_device=profile.tensor_offload_device,
                    cache_dit=profile.cache_dit,
                    cache_vae=profile.cache_vae,
                    mps_high_watermark_ratio=profile.mps_high_watermark_ratio,
                    mps_low_watermark_ratio=profile.mps_low_watermark_ratio,
                    memory_guard_min_available_gb=profile.memory_guard_min_available_gb,
                    memory_guard_max_process_rss_gb=profile.memory_guard_max_process_rss_gb,
                    cancel_event=cancel_event,
                    progress_callback=runtime_progress,
                    video_backend_preference="ffmpeg",
                    ffmpeg_bin=ffmpeg_bin,
                    timeout_sec=segment_timeout_sec,
                )
                return generated_path, work_dir, warnings
            except SeedVRRuntimeError as exc:
                # 失败的临时目录立即清理，避免累积缓存。
                shutil.rmtree(work_dir, ignore_errors=True)

                if cancel_event is not None and cancel_event.is_set():
                    raise UpscaleCancelled("Upscale task cancelled")
                text = str(exc)
                if "cancelled" in text.lower():
                    raise UpscaleCancelled("Upscale task cancelled")

                retry_memory = self._is_memory_pressure_error(text)
                retry_stall = self._is_stall_error(text)
                should_retry = attempt_index == 0 and (retry_memory or retry_stall) and not retried
                if should_retry:
                    retried = True
                    if retry_memory:
                        recovery_profile = build_emergency_seedvr_profile(
                            requested_short_resolution=requested_target_short
                        )
                    else:
                        recovery_profile = build_stall_recovery_seedvr_profile(
                            requested_short_resolution=requested_target_short
                        )
                    warnings.extend(list(recovery_profile.warnings))
                    profiles.append(recovery_profile)
                    if progress_callback:
                        if retry_stall:
                            if "warmup stalled" in text.lower():
                                retry_message = "Warmup timed out, retrying with stall-recovery profile..."
                            else:
                                retry_message = "Inference stalled, retrying with stall-recovery profile..."
                        else:
                            retry_message = "Memory pressure detected, retrying with stricter profile..."
                        retry_payload: Dict[str, Any] = {
                            "progress": max(0.0, float(progress_start)),
                            "phase": phase,
                            "message": retry_message,
                            "scene_split_mode": scene_split_mode,
                        }
                        if segment_index is not None:
                            retry_payload["segment_index"] = int(segment_index)
                        if segment_total is not None:
                            retry_payload["segment_total"] = int(segment_total)
                        progress_callback(retry_payload)
                    continue

                raise RuntimeError(text)

        raise RuntimeError("SeedVR inference failed after emergency retry")

    def _run_realesrgan_inference(
        self,
        *,
        input_path: str,
        model_id: str,
        denoise_strength: float,
        source_w: int,
        source_h: int,
        target_w: int,
        target_h: int,
        duration_sec: float,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        estimated_total_sec: float,
        progress_start: float = 0.14,
        progress_span: float = 0.70,
        phase: str = "infer",
        message_prefix: str = "Running Real-ESRGAN inference...",
        segment_index: Optional[int] = None,
        segment_total: Optional[int] = None,
        scene_split_mode: str = "rule",
    ) -> Tuple[str, Path, list[str]]:
        spec = get_realesrgan_model_spec(model_id)
        source_short = max(1, min(int(source_w), int(source_h)))
        target_short = max(1, min(int(target_w), int(target_h)))
        outscale = max(1.0, float(target_short) / float(source_short))
        total_gb, available_gb = detect_system_memory_gb()

        base_tile = max(128, int(spec.preferred_tile))
        if available_gb > 0.0 and available_gb < 6.0:
            first_tile = min(base_tile, 256)
        elif available_gb > 0.0 and available_gb < 10.0:
            first_tile = min(base_tile, 384)
        else:
            first_tile = base_tile

        tile_candidates: list[int] = [first_tile]
        for fallback_tile in (256, 192):
            if fallback_tile < first_tile and fallback_tile not in tile_candidates:
                tile_candidates.append(fallback_tile)

        warnings: list[str] = [
            "Applied Real-ESRGAN MPS-first execution policy.",
            f"Applied Real-ESRGAN profile: tile={first_tile}, outscale={outscale:.3f}.",
        ]
        if total_gb > 0:
            warnings.append(
                f"Real-ESRGAN device memory snapshot: total={total_gb:.1f}GB, available={available_gb:.1f}GB."
            )

        for attempt_index, tile_size in enumerate(tile_candidates):
            work_dir = Path(tempfile.mkdtemp(prefix="wmr-realesrgan-"))
            runtime_output_dir = work_dir / "runtime_output"
            runtime_output_dir.mkdir(parents=True, exist_ok=True)
            runtime_started_at = time.time()
            floor_step_sec = 6.0
            floor_step_progress = 0.003
            floor_cap_progress = min(
                float(progress_start) + float(progress_span) * 0.28,
                float(progress_start) + float(progress_span) - 0.01,
            )

            def runtime_progress(payload: Dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                frac = max(0.0, min(1.0, float(payload.get("progress") or 0.0)))
                runtime_message = str(payload.get("message") or "Running Real-ESRGAN inference...")
                message = runtime_message
                if runtime_message.lower().startswith("running real-esrgan inference"):
                    message = f"{message_prefix} | {runtime_message}"

                global_progress = float(progress_start) + frac * float(progress_span)
                elapsed = max(0.0, time.time() - runtime_started_at)
                floor_steps = int(elapsed // floor_step_sec)
                floor_progress = min(
                    floor_cap_progress,
                    float(progress_start) + floor_steps * floor_step_progress,
                )
                global_progress = max(global_progress, floor_progress)
                eta = self._safe_eta(runtime_started_at, estimated_total_sec, global_progress)
                payload_extra: Dict[str, Any] = {
                    "progress": global_progress,
                    "phase": phase,
                    "message": message,
                    "eta_seconds": eta,
                    "scene_split_mode": scene_split_mode,
                }
                if segment_index is not None:
                    payload_extra["segment_index"] = int(segment_index)
                if segment_total is not None:
                    payload_extra["segment_total"] = int(segment_total)
                progress_callback(payload_extra)

            try:
                segment_min_timeout_sec = 300.0 if segment_index is None or int(segment_index) <= 1 else 120.0
                segment_timeout_sec = max(
                    segment_min_timeout_sec,
                    min(
                        1800.0,
                        max(float(duration_sec) * 12.0, float(estimated_total_sec) * 1.8),
                    ),
                )
                generated_path = self._realesrgan_runtime.run_inference(
                    input_path=input_path,
                    output_dir=str(runtime_output_dir),
                    model_id=model_id,
                    outscale=outscale,
                    denoise_strength=denoise_strength,
                    tile=tile_size,
                    tile_pad=10,
                    pre_pad=0,
                    cancel_event=cancel_event,
                    progress_callback=runtime_progress,
                    timeout_sec=segment_timeout_sec,
                )
                return generated_path, work_dir, warnings
            except RealESRGANRuntimeError as exc:
                shutil.rmtree(work_dir, ignore_errors=True)
                if cancel_event is not None and cancel_event.is_set():
                    raise UpscaleCancelled("Upscale task cancelled")

                text = str(exc or "")
                lower = text.lower()
                is_retryable = bool(
                    self._is_memory_pressure_error(text)
                    or self._is_stall_error(text)
                    or "timed out" in lower
                )
                if is_retryable and attempt_index + 1 < len(tile_candidates):
                    next_tile = tile_candidates[attempt_index + 1]
                    warnings.append(
                        f"Real-ESRGAN retry: reducing tile from {tile_size} to {next_tile} for stability."
                    )
                    if progress_callback:
                        retry_payload: Dict[str, Any] = {
                            "progress": max(0.0, float(progress_start)),
                            "phase": phase,
                            "message": f"Real-ESRGAN retry with smaller tile ({next_tile})...",
                            "scene_split_mode": scene_split_mode,
                        }
                        if segment_index is not None:
                            retry_payload["segment_index"] = int(segment_index)
                        if segment_total is not None:
                            retry_payload["segment_total"] = int(segment_total)
                        progress_callback(retry_payload)
                    continue
                raise RuntimeError(text)

        raise RuntimeError("Real-ESRGAN inference failed after retry")

    def _build_filter_chain(
        self,
        *,
        source_w: int,
        source_h: int,
        target_w: int,
        target_h: int,
        mode: str,
        same_res_strength: str,
        denoise_strength: float,
        prefer_libplacebo: bool,
    ) -> str:
        _ = (source_w, source_h, same_res_strength)
        denoise_strength = max(0.0, min(1.0, float(denoise_strength)))
        denoise_chain = ""
        if denoise_strength > 0.001:
            luma = 1.0 + denoise_strength * 3.0
            chroma = 0.8 + denoise_strength * 2.2
            denoise_chain = f"hqdn3d={luma:.2f}:{chroma:.2f}:{luma * 1.4:.2f}:{chroma * 1.4:.2f},"

        if mode == "enhance_same_resolution" and prefer_libplacebo:
            scale_chain = f"libplacebo=w={target_w}:h={target_h}"
        else:
            scale_chain = f"scale={target_w}:{target_h}:flags=lanczos"
        return f"{scale_chain},{denoise_chain}unsharp=5:5:0.55:3:3:0.12"

    def _finalize_with_ffmpeg(
        self,
        *,
        ffmpeg_bin: str,
        ai_output_path: str,
        source_path: Optional[str],
        output_path: str,
        mode: str,
        source_w: int,
        source_h: int,
        target_w: int,
        target_h: int,
        same_res_strength: str,
        denoise_strength: float,
        keep_audio: bool,
        duration_sec: float,
        estimated_total_sec: float,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        prefer_libplacebo: bool,
        progress_start: float = 0.86,
        progress_span: float = 0.12,
        phase: str = "compose",
        message: str = "Composing final output...",
        segment_index: Optional[int] = None,
        segment_total: Optional[int] = None,
        scene_split_mode: str = "rule",
    ) -> None:
        filter_chain = self._build_filter_chain(
            source_w=source_w,
            source_h=source_h,
            target_w=target_w,
            target_h=target_h,
            mode=mode,
            same_res_strength=same_res_strength,
            denoise_strength=denoise_strength,
            prefer_libplacebo=prefer_libplacebo,
        )
        vf_arg = f"{filter_chain},format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2"

        thread_cap = str(self._resolve_ffmpeg_thread_cap())
        cmd = [ffmpeg_bin, "-y", "-i", ai_output_path]
        if keep_audio:
            if not source_path:
                raise RuntimeError("source_path is required when keep_audio=True")
            cmd += ["-i", source_path]
        cmd += [
            "-progress",
            "pipe:2",
            "-stats_period",
            "0.5",
            "-nostats",
            "-v",
            "error",
            "-threads",
            thread_cap,
            "-vf",
            vf_arg,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
        ]
        if keep_audio:
            cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            cmd += ["-an"]
        cmd.append(output_path)

        cb = progress_callback
        if progress_callback and (segment_index is not None or segment_total is not None):
            def wrapped(payload: Dict[str, Any]) -> None:
                next_payload = dict(payload)
                next_payload["scene_split_mode"] = scene_split_mode
                if segment_index is not None:
                    next_payload["segment_index"] = int(segment_index)
                if segment_total is not None:
                    next_payload["segment_total"] = int(segment_total)
                progress_callback(next_payload)
            cb = wrapped

        self._run_ffmpeg_with_progress(
            cmd=cmd,
            duration_sec=duration_sec,
            estimated_total_sec=max(8.0, estimated_total_sec * 0.22),
            cancel_event=cancel_event,
            progress_callback=cb,
            progress_start=progress_start,
            progress_span=progress_span,
            phase=phase,
            message=message,
        )

    @staticmethod
    def _assert_not_cancelled(cancel_event: Optional[Any]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise UpscaleCancelled("Upscale task cancelled")

    def _split_video_scenes(
        self,
        *,
        ffmpeg_bin: str,
        input_path: str,
        duration_sec: float,
        fps: float,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> SceneSplitResult:
        splitter = SceneSplitter(ffmpeg_bin=ffmpeg_bin)

        def callback_bridge(payload: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            sub = max(0.0, min(1.0, float(payload.get("progress") or 0.0)))
            ratio = sub / 0.15 if sub <= 0.15 else sub
            ratio = max(0.0, min(1.0, ratio))
            message = str(payload.get("message") or "Scene split in progress...")
            progress_callback(
                {
                    "progress": 0.04 + 0.11 * ratio,
                    "phase": "prepare",
                    "message": message,
                    "scene_split_mode": "rule",
                }
            )

        try:
            return splitter.split(
                input_path=input_path,
                duration_sec=duration_sec,
                fps=fps,
                cancel_event=cancel_event,
                progress_callback=callback_bridge,
            )
        except SceneSplitCancelled as exc:
            raise UpscaleCancelled("Upscale task cancelled") from exc

    @staticmethod
    def _bypass_scene_split_result(*, duration_sec: float) -> SceneSplitResult:
        total = max(0.1, float(duration_sec))
        return SceneSplitResult(
            segments=(SceneSegment(idx=1, start=0.0, end=total, duration=total),),
            split_mode="bypass_short_video",
            warnings=(
                f"Scene split bypassed for short input (<= {SHORT_VIDEO_SCENE_SPLIT_BYPASS_SEC:.0f}s) to reduce warmup overhead.",
            ),
            cuts=(),
            stats={"bypass_reason": "short_video", "duration_sec": total},
        )

    def _extract_segment_without_audio(
        self,
        *,
        ffmpeg_bin: str,
        source_path: str,
        output_path: str,
        start_sec: float,
        end_sec: float,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        progress_start: float,
        progress_span: float,
        message: str,
        segment_index: int,
        segment_total: int,
        scene_split_mode: str,
    ) -> None:
        clip_duration = max(0.1, float(end_sec - start_sec))
        thread_cap = str(self._resolve_ffmpeg_thread_cap())
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{max(0.0, float(start_sec)):.3f}",
            "-to",
            f"{max(float(end_sec), float(start_sec) + 0.05):.3f}",
            "-i",
            source_path,
            "-progress",
            "pipe:2",
            "-stats_period",
            "0.5",
            "-nostats",
            "-v",
            "error",
            "-threads",
            thread_cap,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]

        if progress_callback is None:
            wrapped = None
        else:
            def wrapped(payload: Dict[str, Any]) -> None:
                next_payload = dict(payload)
                next_payload["scene_split_mode"] = scene_split_mode
                next_payload["segment_index"] = int(segment_index)
                next_payload["segment_total"] = int(segment_total)
                progress_callback(next_payload)

        self._run_ffmpeg_with_progress(
            cmd=cmd,
            duration_sec=clip_duration,
            estimated_total_sec=max(1.0, clip_duration * 0.4),
            cancel_event=cancel_event,
            progress_callback=wrapped,
            progress_start=progress_start,
            progress_span=progress_span,
            phase="extract",
            message=message,
        )

    def _concat_segments(
        self,
        *,
        ffmpeg_bin: str,
        segment_paths: Sequence[str],
        output_path: str,
        work_dir: Path,
        duration_sec: float,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        progress_start: float,
        progress_span: float,
        scene_split_mode: str,
    ) -> List[str]:
        warnings: List[str] = []
        list_file = work_dir / "concat_list.txt"
        escaped_lines = []
        for item in segment_paths:
            safe = str(item).replace("'", "'\\''")
            escaped_lines.append(f"file '{safe}'")
        list_file.write_text("\n".join(escaped_lines) + "\n", encoding="utf-8")
        thread_cap = str(self._resolve_ffmpeg_thread_cap())

        wrapped = progress_callback
        if progress_callback:
            def wrapped(payload: Dict[str, Any]) -> None:
                next_payload = dict(payload)
                next_payload["scene_split_mode"] = scene_split_mode
                progress_callback(next_payload)

        copy_cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-progress",
            "pipe:2",
            "-stats_period",
            "0.5",
            "-nostats",
            "-v",
            "error",
            "-threads",
            thread_cap,
            "-c",
            "copy",
            output_path,
        ]
        try:
            self._run_ffmpeg_with_progress(
                cmd=copy_cmd,
                duration_sec=max(1.0, float(duration_sec)),
                estimated_total_sec=max(1.0, float(duration_sec) * 0.2),
                cancel_event=cancel_event,
                progress_callback=wrapped,
                progress_start=progress_start,
                progress_span=progress_span,
                phase="compose",
                message=f"Merging {len(segment_paths)} segments...",
            )
            return warnings
        except Exception:
            warnings.append("Concat copy failed; fallback to re-encode concat.")

        fallback_cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-progress",
            "pipe:2",
            "-stats_period",
            "0.5",
            "-nostats",
            "-v",
            "error",
            "-threads",
            thread_cap,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-an",
            output_path,
        ]
        self._run_ffmpeg_with_progress(
            cmd=fallback_cmd,
            duration_sec=max(1.0, float(duration_sec)),
            estimated_total_sec=max(1.0, float(duration_sec) * 0.26),
            cancel_event=cancel_event,
            progress_callback=wrapped,
            progress_start=progress_start,
            progress_span=progress_span,
            phase="compose",
            message=f"Merging {len(segment_paths)} segments (re-encode fallback)...",
        )
        return warnings

    def _mux_audio(
        self,
        *,
        ffmpeg_bin: str,
        video_path: str,
        source_path: str,
        output_path: str,
        keep_audio: bool,
        duration_sec: float,
        cancel_event: Optional[Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        scene_split_mode: str,
    ) -> None:
        thread_cap = str(self._resolve_ffmpeg_thread_cap())
        cmd = [ffmpeg_bin, "-y", "-i", video_path]
        if keep_audio:
            cmd += ["-i", source_path]
        cmd += [
            "-progress",
            "pipe:2",
            "-stats_period",
            "0.5",
            "-nostats",
            "-v",
            "error",
            "-threads",
            thread_cap,
            "-map",
            "0:v:0",
        ]
        if keep_audio:
            cmd += ["-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            cmd += ["-c:v", "copy", "-an"]
        cmd += ["-movflags", "+faststart", output_path]

        wrapped = progress_callback
        if progress_callback:
            def wrapped(payload: Dict[str, Any]) -> None:
                next_payload = dict(payload)
                next_payload["scene_split_mode"] = scene_split_mode
                progress_callback(next_payload)

        self._run_ffmpeg_with_progress(
            cmd=cmd,
            duration_sec=max(1.0, float(duration_sec)),
            estimated_total_sec=max(1.0, float(duration_sec) * 0.16),
            cancel_event=cancel_event,
            progress_callback=wrapped,
            progress_start=0.96,
            progress_span=0.03,
            phase="compose",
            message="Muxing original audio...",
        )

    def upscale_video(
        self,
        *,
        input_path: str,
        output_dir: str,
        mode: str,
        engine: str,
        model_id: str,
        target_preset: Optional[str],
        same_res_strength: str,
        denoise_strength: float,
        keep_audio: bool,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        ffmpeg_bin = resolve_ffmpeg_path()
        if not ffmpeg_bin:
            raise RuntimeError("FFmpeg runtime not found")

        normalized_same_res_strength, normalize_warnings = normalize_same_res_strength(same_res_strength)
        same_res_strength = normalized_same_res_strength

        if mode not in UPSCALE_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        if engine not in UPSCALE_ENGINES:
            raise ValueError(f"Unsupported engine: {engine}")
        if mode == "upscale_resolution" and target_preset not in UPSCALE_TARGET_PRESETS:
            raise ValueError(f"Unsupported target_preset: {target_preset}")
        if mode == "enhance_same_resolution" and same_res_strength not in UPSCALE_SAME_RES_STRENGTHS:
            raise ValueError(f"Unsupported same_res_strength: {same_res_strength}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        meta = self._video_meta(input_path)
        source_w = int(meta["width"])
        source_h = int(meta["height"])
        source_fps = float(meta["fps"] or 24.0)
        duration_sec = float(meta["duration_sec"])

        warnings: list[str] = list(normalize_warnings)
        prepared_input_path = input_path
        preprocess_dir: Optional[Path] = None

        if progress_callback:
            progress_callback(
                {
                    "progress": 0.02,
                    "phase": "prepare",
                    "message": "Preparing AI upscale pipeline...",
                }
            )

        if engine == SEEDVR_ENGINE_ID:
            prepared_path, preprocess_dir, preprocess_warnings = self._prepare_seedvr_input_720p(
                ffmpeg_bin=ffmpeg_bin,
                input_path=input_path,
                source_w=source_w,
                source_h=source_h,
                duration_sec=duration_sec,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
            )
            prepared_input_path = prepared_path
            warnings.extend(preprocess_warnings)

            if preprocess_dir is not None:
                prepared_meta = self._video_meta(prepared_input_path)
                source_w = int(prepared_meta["width"])
                source_h = int(prepared_meta["height"])
                source_fps = float(prepared_meta["fps"] or source_fps)
                duration_sec = float(prepared_meta["duration_sec"])

        target_w, target_h = self._compute_target_resolution(source_w, source_h, mode, target_preset)
        estimated_total = self._estimate_duration(duration_sec, target_w, target_h, source_w, source_h)

        if progress_callback:
            progress_callback(
                {
                    "progress": 0.03,
                    "phase": "prepare",
                    "message": "Preparing AI upscale pipeline...",
                    "eta_seconds": estimated_total,
                }
            )

        capabilities = self.get_capabilities()
        selected_engine_payload = None
        for item in capabilities.get("engines") or []:
            if str(item.get("engine") or "") == engine:
                selected_engine_payload = item
                break
        if not selected_engine_payload or not bool(selected_engine_payload.get("available")):
            reason = str((selected_engine_payload or {}).get("reason") or f"{engine} engine unavailable")
            raise RuntimeError(reason)

        if engine == SEEDVR_ENGINE_ID:
            supported_models = set(SEEDVR_MODELS)
            default_model_id = SEEDVR_DEFAULT_MODEL_ID
        elif engine == REALESRGAN_ENGINE_ID:
            supported_models = set(REALESRGAN_MODELS)
            default_model_id = REALESRGAN_DEFAULT_MODEL_ID
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        effective_model = model_id if model_id in supported_models else default_model_id
        if effective_model != model_id:
            raise ValueError(
                f"Invalid model_id: {model_id}. "
                f"Supported values: {', '.join(sorted(supported_models))}"
            )
        if not is_upscale_model_installed(effective_model):
            raise RuntimeError(f"Upscale model not installed: {effective_model}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        suffix = target_preset if mode == "upscale_resolution" else same_res_strength
        output_path = str(Path(output_dir) / self._build_output_name(input_path, mode, str(suffix)))
        work_dirs: list[Path] = []

        try:
            prefer_libplacebo = bool(capabilities.get("ffmpeg", {}).get("libplacebo_available"))
            self._assert_not_cancelled(cancel_event)

            if duration_sec <= float(SHORT_VIDEO_SCENE_SPLIT_BYPASS_SEC):
                scene_result = self._bypass_scene_split_result(duration_sec=duration_sec)
            else:
                scene_result = self._split_video_scenes(
                    ffmpeg_bin=ffmpeg_bin,
                    input_path=prepared_input_path,
                    duration_sec=duration_sec,
                    fps=source_fps,
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                )
            split_mode = str(scene_result.split_mode or "rule")
            warnings.extend(list(scene_result.warnings))
            segments = list(scene_result.segments)
            segment_total = max(1, len(segments))
            if segment_total > 1:
                warnings.append(
                    "Enabled segment-level memory cleanup between scene chunks."
                )

            if progress_callback:
                progress_callback(
                    {
                        "progress": 0.15,
                        "phase": "prepare",
                        "message": f"Scene split ready: {segment_total} segment(s).",
                        "scene_split_mode": split_mode,
                        "segment_total": segment_total,
                    }
                )

            pipeline_dir = Path(tempfile.mkdtemp(prefix="wmr-seedvr-pipeline-"))
            work_dirs.append(pipeline_dir)

            segment_outputs: List[str] = []
            processed_duration = 0.0
            total_duration_for_weight = max(0.01, sum(max(seg.duration, 0.01) for seg in segments))

            for idx, segment in enumerate(segments, start=1):
                self._assert_not_cancelled(cancel_event)
                segment_label = self._segment_label(
                    idx,
                    segment_total,
                    float(segment.start),
                    float(segment.end),
                )
                segment_root = pipeline_dir / f"segment_{idx:03d}"
                segment_root.mkdir(parents=True, exist_ok=True)
                segment_input_path = str(segment_root / "segment_input.mp4")
                segment_post_path = str(segment_root / "segment_post.mp4")

                seg_weight = max(0.01, float(segment.duration)) / total_duration_for_weight
                seg_start_progress = 0.15 + 0.75 * (processed_duration / total_duration_for_weight)
                seg_span = 0.75 * seg_weight
                seg_extract_span = max(0.01, seg_span * 0.12)
                seg_infer_span = max(0.01, seg_span * 0.70)
                seg_post_span = max(0.005, seg_span - seg_extract_span - seg_infer_span)

                self._extract_segment_without_audio(
                    ffmpeg_bin=ffmpeg_bin,
                    source_path=prepared_input_path,
                    output_path=segment_input_path,
                    start_sec=float(segment.start),
                    end_sec=float(segment.end),
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                    progress_start=seg_start_progress,
                    progress_span=seg_extract_span,
                    message=f"Scene split: {segment_label} - extract",
                    segment_index=idx,
                    segment_total=segment_total,
                    scene_split_mode=split_mode,
                )
                self._assert_not_cancelled(cancel_event)

                segment_meta = self._video_meta(segment_input_path)
                infer_kwargs = dict(
                    input_path=segment_input_path,
                    model_id=effective_model,
                    denoise_strength=denoise_strength,
                    source_w=int(segment_meta["width"]),
                    source_h=int(segment_meta["height"]),
                    target_w=target_w,
                    target_h=target_h,
                    duration_sec=float(segment_meta["duration_sec"]),
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                    estimated_total_sec=max(45.0, estimated_total * seg_weight),
                    progress_start=seg_start_progress + seg_extract_span,
                    progress_span=seg_infer_span,
                    phase="infer",
                    message_prefix=f"Scene split: {segment_label} - infer",
                    segment_index=idx,
                    segment_total=segment_total,
                    scene_split_mode=split_mode,
                )
                if engine == SEEDVR_ENGINE_ID:
                    ai_output, runtime_dir, runtime_warnings = self._run_seedvr2_inference(
                        ffmpeg_bin=ffmpeg_bin,
                        mode=mode,
                        same_res_strength=same_res_strength,
                        **infer_kwargs,
                    )
                else:
                    ai_output, runtime_dir, runtime_warnings = self._run_realesrgan_inference(
                        **infer_kwargs,
                    )
                warnings.extend(runtime_warnings)
                work_dirs.append(runtime_dir)

                self._finalize_with_ffmpeg(
                    ffmpeg_bin=ffmpeg_bin,
                    ai_output_path=ai_output,
                    source_path=None,
                    output_path=segment_post_path,
                    mode=mode,
                    source_w=int(segment_meta["width"]),
                    source_h=int(segment_meta["height"]),
                    target_w=target_w,
                    target_h=target_h,
                    same_res_strength=same_res_strength,
                    denoise_strength=denoise_strength,
                    keep_audio=False,
                    duration_sec=max(0.1, float(segment_meta["duration_sec"])),
                    estimated_total_sec=max(6.0, estimated_total * seg_weight * 0.6),
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                    prefer_libplacebo=prefer_libplacebo,
                    progress_start=seg_start_progress + seg_extract_span + seg_infer_span,
                    progress_span=seg_post_span,
                    phase="compose",
                    message=f"Scene split: {segment_label} - post",
                    segment_index=idx,
                    segment_total=segment_total,
                    scene_split_mode=split_mode,
                )

                segment_outputs.append(segment_post_path)
                processed_duration += max(0.01, float(segment.duration))
                if progress_callback:
                    progress_callback(
                        {
                            "progress": min(0.90, seg_start_progress + seg_span),
                            "phase": "compose",
                            "message": f"Scene split: {segment_label} - done",
                            "scene_split_mode": split_mode,
                            "segment_index": idx,
                            "segment_total": segment_total,
                        }
                    )
                # 段间主动释放推理缓存，降低长任务统一内存持续高压。
                if segment_total > 1:
                    try:
                        release_unified_memory(
                            f"upscale_segment_finalize:{idx}/{segment_total}"
                        )
                    except Exception:
                        pass

            if not segment_outputs:
                raise RuntimeError("Scene split generated no valid segment outputs")

            merged_path = str(pipeline_dir / "merged_segments.mp4")
            concat_warnings = self._concat_segments(
                ffmpeg_bin=ffmpeg_bin,
                segment_paths=segment_outputs,
                output_path=merged_path,
                work_dir=pipeline_dir,
                duration_sec=duration_sec,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                progress_start=0.90,
                progress_span=0.06,
                scene_split_mode=split_mode,
            )
            warnings.extend(concat_warnings)

            self._assert_not_cancelled(cancel_event)
            self._mux_audio(
                ffmpeg_bin=ffmpeg_bin,
                video_path=merged_path,
                source_path=input_path,
                output_path=output_path,
                keep_audio=keep_audio,
                duration_sec=duration_sec,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                scene_split_mode=split_mode,
            )

            if not os.path.exists(output_path):
                raise RuntimeError("Upscaled output not generated")
            if progress_callback:
                progress_callback(
                    {
                        "progress": 1.0,
                        "phase": "finalize",
                        "message": "Upscale completed",
                        "eta_seconds": 0.0,
                        "scene_split_mode": split_mode,
                        "segment_total": segment_total,
                    }
                )

            return {
                "output_path": output_path,
                "effective_engine": engine,
                "model_id": effective_model,
                "mode": mode,
                "target_width": int(target_w),
                "target_height": int(target_h),
                "scene_split_mode": split_mode,
                "segment_total": segment_total,
                "warning": "; ".join([item for item in warnings if item]),
                "denoise_strength": float(max(0.0, min(1.0, denoise_strength))),
            }
        finally:
            if preprocess_dir is not None:
                try:
                    shutil.rmtree(preprocess_dir, ignore_errors=True)
                except Exception:
                    pass
            for item in work_dirs:
                try:
                    shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass
            try:
                release_unified_memory("upscale_processor_finalize")
            except Exception:
                pass


__all__ = [
    "REALESRGAN_MODELS",
    "SEEDVR_MODELS",
    "UPSCALE_ENGINES",
    "UPSCALE_MODES",
    "UPSCALE_SAME_RES_STRENGTHS",
    "UPSCALE_TARGET_PRESETS",
    "UpscaleCancelled",
    "UpscaleProcessor",
]
