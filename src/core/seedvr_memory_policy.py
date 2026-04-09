"""
SeedVR2 内存策略与参数降档规则。

目标：
1) 在 Apple Silicon 上优先保证稳定性，避免系统级内存崩溃；
2) 在 720p -> 1080p 固定场景下采用保守推理参数；
3) 提供与业务层解耦的纯策略输出，便于测试和复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:
    import psutil
except Exception:  # pragma: no cover - 运行时兜底
    psutil = None  # type: ignore[assignment]


SUPPORTED_SAME_RES_STRENGTH = "x2_then_downscale"
LEGACY_SAME_RES_X4 = "x4_then_downscale"


@dataclass(frozen=True)
class SeedVRMemoryProfile:
    """一次推理的内存配置档位。"""

    risk_level: str
    target_short_resolution: int
    batch_size: int
    chunk_size: int
    temporal_overlap: int
    max_resolution: int
    vae_encode_tiled: bool
    vae_decode_tiled: bool
    vae_tile_size: int
    vae_tile_overlap: int
    mps_high_watermark_ratio: float
    mps_low_watermark_ratio: float
    memory_guard_min_available_gb: float
    memory_guard_max_process_rss_gb: float
    dit_offload_device: str
    vae_offload_device: str
    tensor_offload_device: str
    cache_dit: bool
    cache_vae: bool
    warnings: Tuple[str, ...] = ()


def normalize_same_res_strength(value: str) -> Tuple[str, List[str]]:
    """
    兼容旧值，并统一到当前唯一支持档位。
    """
    raw = str(value or "").strip()
    warnings: List[str] = []
    if raw == LEGACY_SAME_RES_X4:
        warnings.append("Same-resolution x4 is disabled for memory safety. Automatically switched to x2.")
        return SUPPORTED_SAME_RES_STRENGTH, warnings
    if raw == SUPPORTED_SAME_RES_STRENGTH:
        return raw, warnings
    return raw, warnings


def detect_system_memory_gb() -> Tuple[float, float]:
    """
    返回 (total_gb, available_gb)。
    """
    if psutil is None:  # pragma: no cover - 运行时兜底
        return 0.0, 0.0
    try:
        vm = psutil.virtual_memory()
        return float(vm.total) / (1024 ** 3), float(vm.available) / (1024 ** 3)
    except Exception:  # pragma: no cover - 运行时兜底
        return 0.0, 0.0


def _rss_guard_cap(*, total_memory_gb: float, ratio: float) -> float:
    total = max(0.0, float(total_memory_gb))
    if total <= 0:
        return 6.0
    # 保留系统与其他进程缓冲，避免 SeedVR 子进程独占统一内存。
    capped = total * max(0.18, min(0.45, float(ratio)))
    return max(4.5, min(10.0, capped))


def _risk_level(*, available_memory_gb: float, duration_sec: float, total_memory_gb: float) -> str:
    available = max(0.0, float(available_memory_gb))
    duration = max(0.0, float(duration_sec))
    total = max(0.0, float(total_memory_gb))

    is_16g_class = total > 0.0 and total <= 18.0
    critical_available = 9.5 if is_16g_class else 8.0
    guarded_available = 13.0 if is_16g_class else 12.0
    critical_duration = 75.0 if is_16g_class else 90.0
    guarded_duration = 35.0 if is_16g_class else 45.0

    if available < critical_available or duration > critical_duration:
        return "critical"
    if available < guarded_available or duration > guarded_duration:
        return "guarded"
    return "safe"


def build_seedvr_memory_profile(
    *,
    mode: str,
    same_res_strength: str,
    requested_short_resolution: int,
    duration_sec: float,
    total_memory_gb: float,
    available_memory_gb: float,
    model_id: str = "",
) -> SeedVRMemoryProfile:
    """
    根据当前输入构建 SeedVR 推理内存档位。
    """
    del mode, same_res_strength  # 当前策略暂不按这两个维度分叉。

    risk = _risk_level(
        available_memory_gb=available_memory_gb,
        duration_sec=duration_sec,
        total_memory_gb=total_memory_gb,
    )

    if risk == "safe":
        batch_size = 5
        chunk_size = 24
        temporal_overlap = 2
        max_resolution = 1080
        vae_encode_tiled = True
        vae_decode_tiled = True
        vae_tile_size = 896
        vae_tile_overlap = 128
        high_watermark = 0.78
        low_watermark = 0.62
        min_available_guard = 2.2
        rss_guard = _rss_guard_cap(total_memory_gb=total_memory_gb, ratio=0.34)
        dit_offload_device = "none"
        vae_offload_device = "none"
        tensor_offload_device = "cpu"
        cache_dit = True
        cache_vae = True
    elif risk == "guarded":
        batch_size = 5
        chunk_size = 12
        temporal_overlap = 1
        max_resolution = 1080
        vae_encode_tiled = True
        vae_decode_tiled = True
        vae_tile_size = 768
        vae_tile_overlap = 128
        high_watermark = 0.72
        low_watermark = 0.56
        min_available_guard = 2.6
        rss_guard = _rss_guard_cap(total_memory_gb=total_memory_gb, ratio=0.30)
        dit_offload_device = "none"
        vae_offload_device = "cpu"
        tensor_offload_device = "cpu"
        cache_dit = False
        cache_vae = False
    else:
        batch_size = 1
        chunk_size = 8
        temporal_overlap = 0
        max_resolution = 1080
        vae_encode_tiled = True
        vae_decode_tiled = True
        vae_tile_size = 640
        vae_tile_overlap = 128
        high_watermark = 0.68
        low_watermark = 0.52
        min_available_guard = 3.0
        rss_guard = _rss_guard_cap(total_memory_gb=total_memory_gb, ratio=0.26)
        dit_offload_device = "cpu"
        vae_offload_device = "cpu"
        tensor_offload_device = "cpu"
        cache_dit = False
        cache_vae = False

    requested = max(256, int(requested_short_resolution))
    target_short = min(requested, max_resolution)
    normalized_model_id = str(model_id or "").strip().lower()
    q4_conservative_override = normalized_model_id == "seedvr2_3b_q4_k_m_gguf"

    # Q4 模型以稳定优先：默认关闭缓存并降低首段负载，减少“长时间同步等待”概率。
    if q4_conservative_override:
        cache_dit = False
        cache_vae = False
        if risk == "safe":
            batch_size = 1
            chunk_size = min(chunk_size, 12)
            temporal_overlap = 1
            vae_tile_size = min(vae_tile_size, 768)
            high_watermark = min(high_watermark, 0.72)
            low_watermark = min(low_watermark, 0.56)
            min_available_guard = max(min_available_guard, 2.8)
            rss_guard = min(
                rss_guard,
                _rss_guard_cap(total_memory_gb=total_memory_gb, ratio=0.30),
            )
            dit_offload_device = "none"
            vae_offload_device = "none"
            tensor_offload_device = "cpu"
        elif risk == "guarded":
            batch_size = 1
            chunk_size = min(chunk_size, 8)
            temporal_overlap = 0
            vae_tile_size = min(vae_tile_size, 640)
            dit_offload_device = "none"
            vae_offload_device = "none"
            tensor_offload_device = "cpu"

    warnings: List[str] = []
    warnings.append("Applied MPS-first execution policy.")
    warnings.append("Video backend preference: ffmpeg.")
    warnings.append(
        "Applied streaming profile: "
        f"{risk} (batch={batch_size}, chunk={chunk_size}, tile={vae_tile_size}, "
        f"cache={'on' if cache_dit and cache_vae else 'off'})."
    )
    if cache_dit and cache_vae:
        warnings.append("Model caching enabled for streaming chunks (cache_dit/cache_vae).")
    if q4_conservative_override:
        warnings.append("Q4 conservative startup profile applied: cache disabled and lower batch/chunk.")
    if risk != "safe":
        warnings.append(
            f"Applied low-memory profile: {risk} (batch={batch_size}, chunk={chunk_size}, tile={vae_tile_size})."
        )
    if target_short < requested:
        warnings.append(
            f"Capped internal target short side from {requested} to {target_short} for memory safety."
        )

    return SeedVRMemoryProfile(
        risk_level=risk,
        target_short_resolution=target_short,
        batch_size=batch_size,
        chunk_size=chunk_size,
        temporal_overlap=temporal_overlap,
        max_resolution=max_resolution,
        vae_encode_tiled=vae_encode_tiled,
        vae_decode_tiled=vae_decode_tiled,
        vae_tile_size=vae_tile_size,
        vae_tile_overlap=vae_tile_overlap,
        mps_high_watermark_ratio=high_watermark,
        mps_low_watermark_ratio=low_watermark,
        memory_guard_min_available_gb=min_available_guard,
        memory_guard_max_process_rss_gb=rss_guard,
        dit_offload_device=dit_offload_device,
        vae_offload_device=vae_offload_device,
        tensor_offload_device=tensor_offload_device,
        cache_dit=cache_dit,
        cache_vae=cache_vae,
        warnings=tuple(warnings),
    )


def build_emergency_seedvr_profile(*, requested_short_resolution: int) -> SeedVRMemoryProfile:
    """
    内存异常时的紧急重试档位。
    """
    requested = max(256, int(requested_short_resolution))
    target_short = min(requested, 1080)
    warnings: List[str] = [
        "Applied MPS-first execution policy.",
        "Video backend preference: ffmpeg.",
        "Escalated to CPU offload due memory/stall.",
        "Memory pressure detected. Retrying with emergency low-memory profile.",
        "Applied streaming profile: emergency (batch=1, chunk=4, tile=512, cache=off).",
    ]
    if target_short < requested:
        warnings.append(
            f"Emergency profile capped target short side from {requested} to {target_short}."
        )

    return SeedVRMemoryProfile(
        risk_level="emergency",
        target_short_resolution=target_short,
        batch_size=1,
        chunk_size=4,
        temporal_overlap=0,
        max_resolution=1080,
        vae_encode_tiled=True,
        vae_decode_tiled=True,
        vae_tile_size=512,
        vae_tile_overlap=128,
        mps_high_watermark_ratio=0.64,
        mps_low_watermark_ratio=0.50,
        memory_guard_min_available_gb=3.2,
        memory_guard_max_process_rss_gb=5.5,
        dit_offload_device="cpu",
        vae_offload_device="cpu",
        tensor_offload_device="cpu",
        cache_dit=False,
        cache_vae=False,
        warnings=tuple(warnings),
    )


def build_stall_recovery_seedvr_profile(*, requested_short_resolution: int) -> SeedVRMemoryProfile:
    """
    推理卡住（无前进）时的恢复档位：
    - 保持 MPS-first（避免直接全量 CPU offload）；
    - 通过更小 chunk/tile 缩短单次推进窗口；
    - 仅在仍失败时再由上层进入 emergency。
    """
    requested = max(256, int(requested_short_resolution))
    target_short = min(1080, requested)
    warnings: List[str] = [
        "No forward progress detected. Retrying with stall-recovery profile.",
        "Applied streaming profile: stall_recovery (batch=1, chunk=4, tile=512, cache=off, offload=none/none/cpu).",
        "Applied MPS-first execution policy.",
    ]
    if target_short < requested:
        warnings.append(
            f"Stall recovery profile capped target short side from {requested} to {target_short}."
        )

    return SeedVRMemoryProfile(
        risk_level="stall_recovery",
        target_short_resolution=target_short,
        batch_size=1,
        chunk_size=4,
        temporal_overlap=0,
        max_resolution=1080,
        vae_encode_tiled=True,
        vae_decode_tiled=True,
        vae_tile_size=512,
        vae_tile_overlap=128,
        mps_high_watermark_ratio=0.66,
        mps_low_watermark_ratio=0.50,
        memory_guard_min_available_gb=2.8,
        memory_guard_max_process_rss_gb=max(4.8, min(8.0, _rss_guard_cap(total_memory_gb=16.0, ratio=0.28))),
        dit_offload_device="none",
        vae_offload_device="none",
        tensor_offload_device="cpu",
        cache_dit=False,
        cache_vae=False,
        warnings=tuple(warnings),
    )


__all__ = [
    "LEGACY_SAME_RES_X4",
    "SUPPORTED_SAME_RES_STRENGTH",
    "SeedVRMemoryProfile",
    "build_emergency_seedvr_profile",
    "build_stall_recovery_seedvr_profile",
    "build_seedvr_memory_profile",
    "detect_system_memory_gb",
    "normalize_same_res_strength",
]
