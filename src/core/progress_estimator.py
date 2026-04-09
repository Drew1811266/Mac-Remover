"""
处理进度与 ETA（预计剩余时间）估算器。

设计思路：
- 把任务分成多个阶段（prepare/load/extract/infer/...）。
- 用阶段权重合成全局进度，避免单一计数导致跳变。
- 用 EMA（指数滑动平均）平滑速率，提升 ETA 稳定性。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

PHASES: Tuple[str, ...] = (
    "prepare",
    "load_models",
    "extract",
    "infer",
    "compose",
    "finalize",
)

DEFAULT_PHASE_WEIGHTS: Dict[str, float] = {
    "prepare": 0.01,
    "load_models": 0.07,
    "extract": 0.10,
    "infer": 0.72,
    "compose": 0.07,
    "finalize": 0.03,
}


@dataclass
class _Ema:
    """简单 EMA 容器：保存平滑值并提供偏差修正。"""
    alpha: float
    value: float = 0.0
    count: int = 0

    def update(self, sample: float) -> None:
        """喂入新样本并更新 EMA。"""
        sample = max(0.0, float(sample))
        if self.count == 0:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        self.count += 1

    def corrected(self) -> float:
        """返回偏差修正后的 EMA 值（早期样本更可靠）。"""
        if self.count <= 0:
            return 0.0
        # Warm-up period: keep raw EMA to avoid aggressive early over-correction.
        if self.count <= 1:
            return self.value
        # Bias correction makes early EMA values less pessimistic.
        bias = 1.0 - ((1.0 - self.alpha) ** self.count)
        if bias <= 1e-9:
            return self.value
        return self.value / bias


class ProgressEstimator:
    """对外使用的进度估算器。"""
    def __init__(
        self,
        total_frames: int = 0,
        *,
        alpha: float = 0.25,
        sample_window: float = 0.8,
        min_rate: float = 1e-6,
    ) -> None:
        # 初始化阶段进度、帧计数与速率采样状态。
        self._weights = dict(DEFAULT_PHASE_WEIGHTS)
        self._phase_progress: Dict[str, float] = {phase: 0.0 for phase in PHASES}
        self._active_phase = "prepare"
        self._opaque_infer = False

        self._total_frames = max(0, int(total_frames))
        self._processed_frames = 0

        self._progress_ema = _Ema(alpha=max(0.01, min(0.95, float(alpha))))
        self._frame_ema = _Ema(alpha=max(0.01, min(0.95, float(alpha))))
        self._sample_window = max(0.1, float(sample_window))
        self._min_rate = max(float(min_rate), 1e-9)

        now = time.time()
        self._last_progress_sample = now
        self._last_progress_value = 0.0
        self._last_frame_sample = now
        self._last_frame_value = 0
        self._last_snapshot_ts = now
        self._snapshot_cache: Optional[Dict[str, object]] = None

    @staticmethod
    def _clamp01(value: float) -> float:
        """把数值夹到 [0, 1]，并处理 NaN/Inf。"""
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return min(max(value, 0.0), 1.0)

    def transition_to(self, phase: str) -> None:
        """
        切换阶段。

        进入新阶段时，会把之前阶段至少标记到 100%，并重置速率基线。
        """
        if phase not in PHASES:
            return
        idx = PHASES.index(phase)
        for previous in PHASES[:idx]:
            self._phase_progress[previous] = max(self._phase_progress[previous], 1.0)
        self._active_phase = phase
        # Phase transitions are coarse milestones; reset rate baseline so ETA
        # depends on subsequent incremental work rather than transition jumps.
        now = time.time()
        self._last_progress_sample = now
        self._last_progress_value = self._global_progress()
        self._snapshot_cache = None

    def set_opaque_infer(self, enabled: bool) -> None:
        """标记推理阶段是否“不可见进度”（用于 UI 提示）。"""
        self._opaque_infer = bool(enabled)
        self._snapshot_cache = None

    def complete_all(self) -> None:
        """强制将所有阶段置为完成。"""
        for phase in PHASES:
            self._phase_progress[phase] = 1.0
        self._active_phase = "finalize"
        self._snapshot_cache = None

    def update_phase_progress(self, phase: str, progress: float) -> None:
        """更新某个阶段进度（单调不回退）。"""
        if phase not in PHASES:
            return
        self._active_phase = phase
        clamped = self._clamp01(progress)
        if clamped < self._phase_progress[phase]:
            clamped = self._phase_progress[phase]
        self._phase_progress[phase] = clamped
        self._snapshot_cache = None

    def update_phase_step(self, phase: str, step: int, total: int) -> None:
        """通过 step/total 更新阶段进度。"""
        safe_total = max(1, int(total))
        safe_step = min(max(0, int(step)), safe_total)
        self.update_phase_progress(phase, float(safe_step) / float(safe_total))

    def update_processed_frames(self, processed_frames: int, total_frames: Optional[int] = None) -> None:
        """
        更新已处理帧数，并同步估算吞吐率与 infer 阶段进度。
        """
        if total_frames is not None:
            self._total_frames = max(0, int(total_frames))

        current = max(0, int(processed_frames))
        if current < self._processed_frames:
            current = self._processed_frames

        now = time.time()
        delta_frames = current - self._last_frame_value
        delta_t = now - self._last_frame_sample
        if delta_frames > 0 and delta_t >= self._sample_window * 0.3:
            self._frame_ema.update(float(delta_frames) / max(delta_t, 1e-6))
            self._last_frame_sample = now
            self._last_frame_value = current
        elif delta_frames > 0 and delta_t > 0:
            self._last_frame_value = current

        self._processed_frames = current
        if self._total_frames > 0:
            ratio = float(current) / float(max(1, self._total_frames))
            self.update_phase_progress("infer", ratio)
        self._snapshot_cache = None

    def _global_progress(self) -> float:
        """按阶段权重计算全局进度。"""
        weighted = 0.0
        for phase in PHASES:
            weighted += self._weights.get(phase, 0.0) * self._phase_progress.get(phase, 0.0)
        return self._clamp01(weighted)

    @staticmethod
    def _format_eta(seconds: Optional[float]) -> str:
        """把秒数格式化为 `HH:MM:SS` 或 `MM:SS`。"""
        if seconds is None or seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
            return "--:--"
        total_seconds = int(round(seconds))
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def snapshot(self, *, force_recompute: bool = False) -> Dict[str, object]:
        """
        生成当前快照（进度、ETA、FPS 等）。

        为避免高频调用抖动，内部有短时间缓存。
        """
        now = time.time()
        if self._snapshot_cache is not None and not force_recompute:
            if (now - self._last_snapshot_ts) < 0.05:
                return dict(self._snapshot_cache)

        progress = self._global_progress()
        delta_progress = max(0.0, progress - self._last_progress_value)
        delta_t = now - self._last_progress_sample
        if delta_progress > 0 and delta_t >= self._sample_window * 0.3:
            self._progress_ema.update(delta_progress / max(delta_t, 1e-6))
            self._last_progress_sample = now
            self._last_progress_value = progress
        elif delta_progress > 0 and delta_t > 0:
            self._last_progress_value = progress

        smooth_rate = self._progress_ema.corrected()
        if progress >= 0.999999:
            eta_seconds = 0.0
        elif smooth_rate <= self._min_rate:
            eta_seconds: Optional[float] = None
        else:
            eta_seconds = max(0.0, (1.0 - progress) / smooth_rate)

        throughput_fps = self._frame_ema.corrected()
        if throughput_fps <= 0:
            throughput_fps = 0.0

        snapshot: Dict[str, object] = {
            "progress": progress,
            "phase": self._active_phase,
            "opaque_infer": self._opaque_infer,
            "eta_seconds": eta_seconds,
            "estimated_time": self._format_eta(eta_seconds),
            "throughput_fps": throughput_fps,
            "processed_frames": int(self._processed_frames),
            "total_frames": int(self._total_frames),
        }
        self._last_snapshot_ts = now
        self._snapshot_cache = dict(snapshot)
        return snapshot
