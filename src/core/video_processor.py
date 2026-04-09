"""
手工标注模式的视频处理主流水线。

整体流程（给初学者）：
1) 读取视频并按帧遍历；
2) 根据“标记段”决定哪些帧需要修复；
3) 对命中段按模型做 ROI 修复（含多模型回退）；
4) 融合回原帧并写出无音频视频；
5) 最后用 FFmpeg 输出标准 H.264 MP4，并尽量保留音频。
"""

import cv2
import numpy as np
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

from ..config import get_config
from ..utils.ffmpeg_runtime import resolve_ffmpeg_path, resolve_ffprobe_path
from ..utils.logger import logger
from .model_registry import ModelRegistry, SUPPORTED_MODEL_IDS
from .remover import WatermarkRemover


@dataclass
class VideoInfo:
    """视频基础信息对象（供流程和 UI 共同使用）。"""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    has_audio: bool
    codec: str

    @property
    def resolution(self) -> Tuple[int, int]:
        """返回 `(width, height)`。"""
        return (self.width, self.height)


class VideoProcessor:
    """手工标注视频处理器。"""
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    SUPPORTED_MODEL_IDS = SUPPORTED_MODEL_IDS

    def __init__(self, remover: Optional[WatermarkRemover] = None):
        # 注入修复器实例，并初始化处理状态。
        self.remover = remover
        self.config = get_config()
        self._is_processing = False
        self._should_stop = False
        self._model_registry: Optional[ModelRegistry] = None

    def _get_model_registry(self) -> ModelRegistry:
        """懒加载模型注册器，确保与当前 remover 绑定。"""
        if self.remover is None:
            raise RuntimeError('WatermarkRemover is not initialized')
        if self._model_registry is None:
            self._model_registry = ModelRegistry(self.remover)
        return self._model_registry

    @staticmethod
    def _format_eta(seconds: Optional[float]) -> str:
        """把秒数格式化成 `HH:MM:SS` / `MM:SS`。"""
        if seconds is None or seconds < 0:
            return '--:--'

        total_seconds = int(round(seconds))
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)

        if hours > 0:
            return f'{hours:02d}:{minutes:02d}:{secs:02d}'
        return f'{minutes:02d}:{secs:02d}'

    @staticmethod
    def _emit_progress(
        progress_callback: Optional[Callable],
        progress: float,
        message: str,
        processed_frames: Optional[int] = None,
        total_frames: Optional[int] = None,
        estimated_time: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        统一进度回调出口。

        兼容两类回调签名：
        - 新签名：带多字段；
        - 老签名：仅 `(progress, message)`。
        """
        if not progress_callback:
            return

        safe_progress = min(max(progress, 0.0), 1.0)

        try:
            progress_callback(
                safe_progress,
                message,
                processed_frames,
                total_frames,
                estimated_time,
                extra,
            )
        except TypeError:
            try:
                progress_callback(safe_progress, message)
            except Exception as e:
                logger.warning(f'Progress callback failed: {e}')
        except Exception as e:
            logger.warning(f'Progress callback failed: {e}')

    @staticmethod
    def _resolve_active_annotation_segment(
        frame_idx: int,
        segments: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Resolve segment by frame with "last-added wins" priority."""
        active: Optional[Dict[str, Any]] = None
        for seg in segments:
            if seg['start_frame'] <= frame_idx <= seg['end_frame']:
                active = seg
        return active

    @staticmethod
    def _resolve_active_annotation_segments(
        frame_idx: int,
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Resolve all active segments for the frame.

        Priority:
        1) larger annotation area first
        2) earlier creation order first (stable tie-break)
        """
        active_items: List[Tuple[int, int, int, Dict[str, Any]]] = []
        for idx, seg in enumerate(segments):
            if not (seg['start_frame'] <= frame_idx <= seg['end_frame']):
                continue
            rect = seg.get('rect', {}) if isinstance(seg.get('rect'), dict) else {}
            area = seg.get('_area')
            if area is None:
                area = int(max(1, int(rect.get('width', 0))) * max(1, int(rect.get('height', 0))))
            order = int(seg.get('_order', idx))
            active_items.append((-int(area), order, idx, seg))
        active_items.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in active_items]

    @staticmethod
    def _compute_lama_adaptive_expand_feather(
        rect_w: int,
        rect_h: int,
        expand_px: int,
        feather_px: int,
    ) -> Tuple[int, int]:
        """
        按标注框尺寸自适应放大/羽化参数。

        目的：小框不过度扩张，大框有足够过渡带，降低边缘突兀感。
        """
        max_side = max(1, int(max(rect_w, rect_h)))
        effective_expand = max(int(expand_px), int(round(max_side * 0.18)))
        effective_feather = max(int(feather_px), int(round(max_side * 0.10)))
        effective_expand = int(np.clip(effective_expand, 6, 48))
        effective_feather = int(np.clip(effective_feather, 4, 24))
        return effective_expand, effective_feather

    @staticmethod
    def _compute_propainter_adaptive_expand_feather(
        rect_w: int,
        rect_h: int,
        expand_px: int,
        feather_px: int,
    ) -> Tuple[int, int]:
        """
        ProPainter 专用自适应几何参数。

        相比 LaMa，ProPainter 在 ROI 小区域上更依赖上下文，因此默认扩张更大。
        """
        max_side = max(1, int(max(rect_w, rect_h)))
        effective_expand = max(int(expand_px), int(round(max_side * 0.20)))
        effective_feather = max(int(feather_px), int(round(max_side * 0.12)))
        effective_expand = int(np.clip(effective_expand, 8, 56))
        effective_feather = int(np.clip(effective_feather, 5, 26))
        return effective_expand, effective_feather

    @staticmethod
    def _resolve_segment_geometry(
        rect: Dict[str, Any],
        expand_px: int,
        feather_px: int,
        frame_width: int,
        frame_height: int,
        use_lama_adaptive: bool,
        use_propainter_adaptive: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        将标注段转换为可执行几何信息。

        输出包括：
        - ROI 裁剪边界；
        - ROI 内掩码框位置；
        - 实际生效的 expand/feather 参数。
        """
        sx = int(rect.get('x', 0))
        sy = int(rect.get('y', 0))
        sw = int(rect.get('width', 0))
        sh = int(rect.get('height', 0))
        if sw <= 0 or sh <= 0:
            return None

        raw_expand = max(0, int(expand_px))
        raw_feather = max(0, int(feather_px))
        if use_propainter_adaptive:
            effective_expand, effective_feather = VideoProcessor._compute_propainter_adaptive_expand_feather(
                rect_w=sw,
                rect_h=sh,
                expand_px=raw_expand,
                feather_px=raw_feather,
            )
        elif use_lama_adaptive:
            effective_expand, effective_feather = VideoProcessor._compute_lama_adaptive_expand_feather(
                rect_w=sw,
                rect_h=sh,
                expand_px=raw_expand,
                feather_px=raw_feather,
            )
        else:
            effective_expand = raw_expand
            effective_feather = raw_feather

        x1 = max(0, sx - effective_expand)
        y1 = max(0, sy - effective_expand)
        x2 = min(frame_width, sx + sw + effective_expand)
        y2 = min(frame_height, sy + sh + effective_expand)
        if x2 <= x1 or y2 <= y1:
            return None

        mask_x1 = max(0, sx - x1)
        mask_y1 = max(0, sy - y1)
        mask_x2 = min(x2 - x1, sx + sw - x1)
        mask_y2 = min(y2 - y1, sy + sh - y1)
        if mask_x2 <= mask_x1 or mask_y2 <= mask_y1:
            return None

        return {
            'expand_px': int(effective_expand),
            'feather_px': int(effective_feather),
            'roi_bounds': (int(x1), int(y1), int(x2), int(y2)),
            'mask_box': (int(mask_x1), int(mask_y1), int(mask_x2), int(mask_y2)),
        }

    @staticmethod
    def _build_soft_alpha_mask(mask: np.ndarray, feather_px: int) -> np.ndarray:
        """把硬掩码转成软 Alpha，用于平滑融合。"""
        mask_bin = (np.asarray(mask) > 0).astype(np.uint8)
        if mask_bin.size == 0 or int(mask_bin.max()) == 0:
            return np.zeros((*mask_bin.shape, 1), dtype=np.float32)

        if feather_px <= 0:
            return mask_bin.astype(np.float32)[:, :, np.newaxis]

        distance = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3)
        alpha = np.clip(distance / float(max(1, int(feather_px))), 0.0, 1.0)
        kernel_size = max(3, int(feather_px) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
        alpha = np.clip(alpha, 0.0, 1.0) * mask_bin.astype(np.float32)
        return alpha[:, :, np.newaxis]

    @staticmethod
    def _build_lama_inpaint_mask(
        core_mask: np.ndarray,
        feather_px: int,
        rect_w: int,
        rect_h: int,
    ) -> np.ndarray:
        """LaMa 第一阶段掩码：在核心区域外做一次扩张。"""
        core_bin = (np.asarray(core_mask) > 0).astype(np.uint8)
        if core_bin.size == 0 or int(core_bin.max()) == 0:
            return np.zeros_like(core_bin, dtype=np.uint8)

        max_side = max(1, int(max(rect_w, rect_h)))
        dilate_px = max(
            1,
            int(round(max(int(feather_px) * 0.9, max_side * 0.12))),
        )
        dilate_px = int(np.clip(dilate_px, 1, 18))
        kernel_size = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(core_bin, kernel, iterations=1)
        return (expanded * 255).astype(np.uint8)

    @staticmethod
    def _build_lama_inpaint_mask_stage2(
        stage1_mask: np.ndarray,
        feather_px: int,
        rect_w: int,
        rect_h: int,
    ) -> np.ndarray:
        """LaMa 第二阶段掩码：在 stage1 基础上再次外扩。"""
        stage1_bin = (np.asarray(stage1_mask) > 0).astype(np.uint8)
        if stage1_bin.size == 0 or int(stage1_bin.max()) == 0:
            return np.zeros_like(stage1_bin, dtype=np.uint8)

        rect_max = max(1, int(max(rect_w, rect_h)))
        r2 = int(round(max(rect_max * 0.10, float(feather_px) * 0.8)))
        r2 = int(np.clip(r2, 2, 14))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r2 * 2 + 1, r2 * 2 + 1))
        expanded = cv2.dilate(stage1_bin, kernel, iterations=1)
        return (expanded * 255).astype(np.uint8)

    @staticmethod
    def _build_lama_blend_alpha(
        core_mask: np.ndarray,
        inpaint_mask: np.ndarray,
        feather_px: int,
        rect_w: int,
        rect_h: int,
    ) -> np.ndarray:
        """
        生成 LaMa 旧版混合 Alpha。

        规则：
        - 核心替换区强制 alpha=1；
        - 过渡区按羽化逐渐衰减。
        """
        core_bin = (np.asarray(core_mask) > 0).astype(np.uint8)
        inpaint_bin = (np.asarray(inpaint_mask) > 0).astype(np.uint8)
        if inpaint_bin.size == 0 or int(inpaint_bin.max()) == 0:
            return np.zeros((*inpaint_bin.shape, 1), dtype=np.float32)

        ring_feather = max(1, int(round(max(1, int(feather_px)) * 0.75)))
        alpha = VideoProcessor._build_soft_alpha_mask(inpaint_bin * 255, ring_feather)[:, :, 0]
        rect_max = max(1, int(max(rect_w, rect_h)))
        hard_replace_radius = int(round(max(float(feather_px) * 0.55, rect_max * 0.05)))
        hard_replace_radius = int(np.clip(hard_replace_radius, 1, 8))
        hard_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (hard_replace_radius * 2 + 1, hard_replace_radius * 2 + 1),
        )
        hard_replace_mask = cv2.dilate(core_bin, hard_kernel, iterations=1)
        # Core/hard-replace zone is always fully replaced to avoid watermark residue.
        alpha[hard_replace_mask > 0] = 1.0
        alpha = np.clip(alpha, 0.0, 1.0) * inpaint_bin.astype(np.float32)
        return alpha[:, :, np.newaxis]

    @staticmethod
    def _build_lama_transition_masks(
        core_mask: np.ndarray,
        inpaint_mask: np.ndarray,
        feather_px: int,
        rect_w: int,
        rect_h: int,
    ) -> Dict[str, Any]:
        """
        构建 LaMa v2 融合所需的多类掩码：
        - core_replace / transition / context / rounded。
        """
        core_bin = (np.asarray(core_mask) > 0).astype(np.uint8)
        inpaint_bin = (np.asarray(inpaint_mask) > 0).astype(np.uint8)
        h, w = inpaint_bin.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        if inpaint_bin.size == 0 or int(inpaint_bin.max()) == 0:
            return {
                'rounded_mask': empty,
                'core_replace_mask': empty,
                'transition_mask': empty,
                'context_mask': empty,
                'transition_band_width': 0.0,
            }

        rect_max = max(1, int(max(rect_w, rect_h)))
        close_radius = int(np.clip(round(max(float(feather_px) * 0.8, rect_max * 0.06)), 1, 10))
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_radius * 2 + 1, close_radius * 2 + 1),
        )
        rounded = cv2.morphologyEx(inpaint_bin, cv2.MORPH_CLOSE, close_kernel)
        blur_radius = int(np.clip(close_radius + 1, 2, 12))
        blur_kernel = blur_radius * 2 + 1
        rounded_float = cv2.GaussianBlur(rounded.astype(np.float32), (blur_kernel, blur_kernel), 0)
        rounded_bin = (rounded_float > 0.15).astype(np.uint8)
        rounded_bin = np.maximum(rounded_bin, inpaint_bin)

        hard_replace_radius = int(round(max(float(feather_px) * 0.55, rect_max * 0.05)))
        hard_replace_radius = int(np.clip(hard_replace_radius, 1, 8))
        hard_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (hard_replace_radius * 2 + 1, hard_replace_radius * 2 + 1),
        )
        core_replace = cv2.dilate(core_bin, hard_kernel, iterations=1)
        core_replace = np.minimum(core_replace, rounded_bin)

        transition_width = int(np.clip(round(max(float(feather_px) * 1.3, rect_max * 0.12)), 3, 20))
        outer_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (transition_width * 2 + 1, transition_width * 2 + 1),
        )
        inner_radius = int(np.clip(round(transition_width * 0.45), 1, transition_width))
        inner_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (inner_radius * 2 + 1, inner_radius * 2 + 1),
        )
        transition_outer = cv2.dilate(core_replace, outer_kernel, iterations=1)
        transition_inner = cv2.erode(core_replace, inner_kernel, iterations=1)
        transition = ((transition_outer > 0) & ~(transition_inner > 0)).astype(np.uint8)
        transition = ((transition > 0) & (rounded_bin > 0) & ~(core_replace > 0)).astype(np.uint8)
        if int(np.count_nonzero(transition)) < 24:
            transition = ((rounded_bin > 0) & ~(core_replace > 0)).astype(np.uint8)

        context_radius = int(np.clip(round(transition_width * 1.35), 4, 28))
        context_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (context_radius * 2 + 1, context_radius * 2 + 1),
        )
        context_outer = cv2.dilate(rounded_bin, context_kernel, iterations=1)
        context_mask = ((context_outer > 0) & ~(rounded_bin > 0)).astype(np.uint8)
        if int(np.count_nonzero(context_mask)) < 24:
            context_mask = ((transition_outer > 0) & ~(core_replace > 0)).astype(np.uint8)
            context_mask = np.clip(context_mask - transition, 0, 1).astype(np.uint8)

        return {
            'rounded_mask': (rounded_bin * 255).astype(np.uint8),
            'core_replace_mask': (core_replace * 255).astype(np.uint8),
            'transition_mask': (transition * 255).astype(np.uint8),
            'context_mask': (context_mask * 255).astype(np.uint8),
            'transition_band_width': float(transition_width),
        }

    @staticmethod
    def _build_propainter_transition_masks_v3(
        core_mask: np.ndarray,
        inpaint_mask: np.ndarray,
        feather_px: int,
        rect_w: int,
        rect_h: int,
    ) -> Dict[str, Any]:
        """
        ProPainter 边界融合 V3 掩码构建。

        在 LaMa v2 掩码基础上增加“内环 + 外环”以强化去矩形化和环带融合。
        """
        base_masks = VideoProcessor._build_lama_transition_masks(
            core_mask=core_mask,
            inpaint_mask=inpaint_mask,
            feather_px=feather_px,
            rect_w=rect_w,
            rect_h=rect_h,
        )
        rounded_bin = (np.asarray(base_masks.get("rounded_mask")) > 0).astype(np.uint8)
        core_replace_bin = (np.asarray(base_masks.get("core_replace_mask")) > 0).astype(np.uint8)
        rect_max = max(1, int(max(rect_w, rect_h)))
        if int(np.count_nonzero(rounded_bin)) == 0:
            empty = np.zeros_like(rounded_bin, dtype=np.uint8)
            out = dict(base_masks)
            out["transition_inner_mask"] = empty
            out["transition_outer_mask"] = empty
            return out

        inner_width = int(np.clip(round(max(float(feather_px) * 0.85, rect_max * 0.08)), 2, 12))
        outer_width = int(np.clip(round(max(float(feather_px) * 1.8, rect_max * 0.18)), 4, 28))
        inner_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (inner_width * 2 + 1, inner_width * 2 + 1),
        )
        outer_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (outer_width * 2 + 1, outer_width * 2 + 1),
        )
        inner_outer = cv2.dilate(core_replace_bin, inner_kernel, iterations=1)
        outer_outer = cv2.dilate(core_replace_bin, outer_kernel, iterations=1)
        transition_inner = ((inner_outer > 0) & ~(core_replace_bin > 0) & (rounded_bin > 0)).astype(np.uint8)
        transition_outer = ((outer_outer > 0) & ~(inner_outer > 0) & (rounded_bin > 0)).astype(np.uint8)
        if int(np.count_nonzero(transition_outer)) < 8:
            base_transition = (np.asarray(base_masks.get("transition_mask")) > 0).astype(np.uint8)
            transition_outer = ((base_transition > 0) & ~(transition_inner > 0)).astype(np.uint8)
            if int(np.count_nonzero(transition_outer)) < 8:
                transition_outer = base_transition
        transition_union = ((transition_inner > 0) | (transition_outer > 0)).astype(np.uint8)
        if int(np.count_nonzero(transition_union)) >= 16:
            base_masks["transition_mask"] = (transition_union * 255).astype(np.uint8)
        base_masks["transition_inner_mask"] = (transition_inner * 255).astype(np.uint8)
        base_masks["transition_outer_mask"] = (transition_outer * 255).astype(np.uint8)
        return base_masks

    @staticmethod
    def _build_edge_aware_alpha(
        core_replace_mask: np.ndarray,
        transition_mask: np.ndarray,
        reference_roi: np.ndarray,
        feather_px: int,
    ) -> np.ndarray:
        """
        边缘感知 Alpha：
        - 距离核心越远，替换权重越低；
        - 梯度越强（边缘越明显），越保守地融合。
        """
        core_bin = (np.asarray(core_replace_mask) > 0).astype(np.uint8)
        transition_bin = (np.asarray(transition_mask) > 0).astype(np.uint8)
        h, w = core_bin.shape[:2]
        alpha = np.zeros((h, w), dtype=np.float32)
        if core_bin.size == 0:
            return alpha[:, :, np.newaxis]

        alpha[core_bin > 0] = 1.0
        if int(np.count_nonzero(transition_bin)) == 0:
            return alpha[:, :, np.newaxis]

        inv_core = (1 - core_bin).astype(np.uint8)
        dist_to_core = cv2.distanceTransform(inv_core, cv2.DIST_L2, 3)
        transition_dist = dist_to_core[transition_bin > 0]
        dist_min = float(transition_dist.min()) if transition_dist.size > 0 else 0.0
        dist_max = float(transition_dist.max()) if transition_dist.size > 0 else 1.0
        dist_norm = np.zeros_like(dist_to_core, dtype=np.float32)
        dist_norm[transition_bin > 0] = (
            (dist_to_core[transition_bin > 0] - dist_min) / max(1e-6, dist_max - dist_min)
        )
        dist_norm = np.clip(dist_norm, 0.0, 1.0)
        base_alpha = 1.0 - dist_norm
        base_alpha = base_alpha * base_alpha * (3.0 - 2.0 * base_alpha)

        gray = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_scale = float(np.percentile(grad_mag, 95)) if grad_mag.size > 0 else 1.0
        grad_norm = np.clip(grad_mag / max(1.0, grad_scale), 0.0, 1.0)
        edge_suppress = 1.0 - 0.42 * grad_norm
        edge_suppress = np.clip(edge_suppress, 0.58, 1.0)

        transition_alpha = np.clip(base_alpha * edge_suppress, 0.04, 0.98)
        alpha[transition_bin > 0] = transition_alpha[transition_bin > 0]
        alpha = np.clip(alpha, 0.0, 1.0)
        return alpha[:, :, np.newaxis]

    @staticmethod
    def _build_edge_aware_alpha_v3(
        core_replace_mask: np.ndarray,
        transition_inner_mask: np.ndarray,
        transition_outer_mask: np.ndarray,
        reference_roi: np.ndarray,
    ) -> np.ndarray:
        """
        ProPainter V3 边缘感知 Alpha。

        inner 环带允许更高替换强度，outer 环带更保守以压低方形边界。
        """
        core_bin = (np.asarray(core_replace_mask) > 0).astype(np.uint8)
        inner_bin = (np.asarray(transition_inner_mask) > 0).astype(np.uint8)
        outer_bin = (np.asarray(transition_outer_mask) > 0).astype(np.uint8)
        h, w = core_bin.shape[:2]
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[core_bin > 0] = 1.0
        transition_bin = ((inner_bin > 0) | (outer_bin > 0)).astype(np.uint8)
        if int(np.count_nonzero(transition_bin)) == 0:
            return alpha[:, :, np.newaxis]

        inv_core = (1 - core_bin).astype(np.uint8)
        dist_to_core = cv2.distanceTransform(inv_core, cv2.DIST_L2, 3)
        transition_dist = dist_to_core[transition_bin > 0]
        dist_min = float(transition_dist.min()) if transition_dist.size > 0 else 0.0
        dist_max = float(transition_dist.max()) if transition_dist.size > 0 else 1.0
        dist_norm = np.zeros_like(dist_to_core, dtype=np.float32)
        if dist_max - dist_min > 1e-6:
            dist_norm[transition_bin > 0] = (
                (dist_to_core[transition_bin > 0] - dist_min) / max(1e-6, dist_max - dist_min)
            )
        dist_norm = np.clip(dist_norm, 0.0, 1.0)
        smooth = 1.0 - dist_norm
        smooth = smooth * smooth * (3.0 - 2.0 * smooth)

        gray = cv2.cvtColor(VideoProcessor._as_uint8_bgr(reference_roi), cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_scale = float(np.percentile(grad_mag, 95)) if grad_mag.size > 0 else 1.0
        grad_norm = np.clip(grad_mag / max(1.0, grad_scale), 0.0, 1.0)
        edge_suppress = 1.0 - 0.48 * grad_norm
        edge_suppress = np.clip(edge_suppress, 0.52, 1.0)

        inner_alpha = np.clip(0.62 + 0.38 * smooth, 0.55, 0.98) * edge_suppress
        outer_alpha = np.clip(0.28 + 0.45 * smooth, 0.14, 0.78) * edge_suppress
        alpha[inner_bin > 0] = np.clip(inner_alpha[inner_bin > 0], 0.16, 0.98)
        alpha[outer_bin > 0] = np.clip(outer_alpha[outer_bin > 0], 0.08, 0.82)
        alpha = np.clip(alpha, 0.0, 1.0)
        return alpha[:, :, np.newaxis]

    @staticmethod
    def _laplacian_blend_roi(
        roi_original: np.ndarray,
        roi_inpainted: np.ndarray,
        alpha: np.ndarray,
        levels: int = 4,
    ) -> np.ndarray:
        """使用拉普拉斯金字塔做多尺度融合，减少接缝感。"""
        original = VideoProcessor._as_uint8_bgr(roi_original).astype(np.float32)
        inpainted = VideoProcessor._as_uint8_bgr(roi_inpainted).astype(np.float32)
        alpha_arr = np.asarray(alpha, dtype=np.float32)
        if alpha_arr.ndim == 2:
            alpha_arr = alpha_arr[:, :, np.newaxis]
        alpha_arr = np.clip(alpha_arr, 0.0, 1.0)
        h, w = original.shape[:2]
        min_side = max(1, min(h, w))
        max_levels = max(2, int(np.floor(np.log2(min_side))) - 1)
        pyr_levels = int(np.clip(levels, 2, max_levels))

        gp_mask = [alpha_arr]
        gp_inpaint = [inpainted]
        gp_original = [original]
        for _ in range(1, pyr_levels):
            gp_mask.append(cv2.pyrDown(gp_mask[-1]))
            gp_inpaint.append(cv2.pyrDown(gp_inpaint[-1]))
            gp_original.append(cv2.pyrDown(gp_original[-1]))

        lp_inpaint: List[np.ndarray] = [gp_inpaint[-1]]
        lp_original: List[np.ndarray] = [gp_original[-1]]
        for level_idx in range(pyr_levels - 1, 0, -1):
            target_size = (gp_inpaint[level_idx - 1].shape[1], gp_inpaint[level_idx - 1].shape[0])
            inpaint_up = cv2.pyrUp(gp_inpaint[level_idx], dstsize=target_size)
            original_up = cv2.pyrUp(gp_original[level_idx], dstsize=target_size)
            lp_inpaint.append(gp_inpaint[level_idx - 1] - inpaint_up)
            lp_original.append(gp_original[level_idx - 1] - original_up)

        blended_pyramid: List[np.ndarray] = []
        for idx in range(pyr_levels):
            mask_level = gp_mask[pyr_levels - 1 - idx]
            if mask_level.ndim == 2:
                mask_level = mask_level[:, :, np.newaxis]
            blended_level = lp_inpaint[idx] * mask_level + lp_original[idx] * (1.0 - mask_level)
            blended_pyramid.append(blended_level)

        blended = blended_pyramid[0]
        for idx in range(1, len(blended_pyramid)):
            target_size = (blended_pyramid[idx].shape[1], blended_pyramid[idx].shape[0])
            blended = cv2.pyrUp(blended, dstsize=target_size)
            blended = blended + blended_pyramid[idx]

        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _compute_seam_delta(
        roi_original: np.ndarray,
        roi_candidate: np.ndarray,
        seam_mask: np.ndarray,
    ) -> float:
        """计算候选 ROI 在接缝区域相对原图的颜色偏差。"""
        seam_bool = np.asarray(seam_mask) > 0
        if seam_bool.ndim != 2 or int(np.count_nonzero(seam_bool)) < 8:
            return 0.0

        original_lab = cv2.cvtColor(VideoProcessor._as_uint8_bgr(roi_original), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        candidate_lab = cv2.cvtColor(
            VideoProcessor._as_uint8_bgr(roi_candidate), cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        delta = np.abs(candidate_lab - original_lab).mean(axis=2)
        return float(delta[seam_bool].mean() / 255.0)

    @staticmethod
    def _evaluate_lama_frame_quality(
        roi_original: np.ndarray,
        roi_candidate: np.ndarray,
        core_mask: np.ndarray,
        transition_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        对单帧候选结果打质量分并给出告警标志。

        输出用于后续“候选选择 + 回退 + 救援”决策。
        """
        original = VideoProcessor._as_uint8_bgr(roi_original)
        candidate = VideoProcessor._as_uint8_bgr(roi_candidate)
        core_bool = np.asarray(core_mask) > 0
        transition_bool = np.asarray(transition_mask) > 0

        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
        candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)

        core_luma_shift = 0.0
        core_texture_ratio = 1.0
        if core_bool.ndim == 2 and int(np.count_nonzero(core_bool)) >= 8:
            original_core = original_gray[core_bool]
            candidate_core = candidate_gray[core_bool]
            core_luma_shift = float((candidate_core.mean() - original_core.mean()) / 255.0)
            core_texture_ratio = float(np.std(candidate_core) / max(np.std(original_core), 1.0))

        seam_delta = 0.0
        transition_edge_ratio = 1.0
        if transition_bool.ndim == 2 and int(np.count_nonzero(transition_bool)) >= 8:
            seam_delta = VideoProcessor._compute_seam_delta(
                roi_original=original,
                roi_candidate=candidate,
                seam_mask=transition_mask,
            )
            original_lap = np.abs(cv2.Laplacian(original_gray, cv2.CV_32F, ksize=3))
            candidate_lap = np.abs(cv2.Laplacian(candidate_gray, cv2.CV_32F, ksize=3))
            transition_edge_ratio = float(
                candidate_lap[transition_bool].mean() / max(original_lap[transition_bool].mean(), 1.0)
            )

        dark_block_flag = (core_luma_shift < -0.07) and (core_texture_ratio < 0.72)
        seam_bad_flag = seam_delta > 0.060
        seam_extreme_flag = seam_delta > 0.090
        edge_collapse_flag = transition_edge_ratio < 0.62
        score = (
            seam_delta * 1.00
            + max(0.0, -core_luma_shift - 0.02) * 0.90
            + max(0.0, 0.70 - core_texture_ratio) * 0.70
            + max(0.0, 0.75 - transition_edge_ratio) * 0.50
        )

        return {
            'seam_delta_transition': float(seam_delta),
            'core_luma_shift': float(core_luma_shift),
            'core_texture_ratio': float(core_texture_ratio),
            'transition_edge_ratio': float(transition_edge_ratio),
            'dark_block_flag': bool(dark_block_flag),
            'seam_bad_flag': bool(seam_bad_flag),
            'seam_extreme_flag': bool(seam_extreme_flag),
            'edge_collapse_flag': bool(edge_collapse_flag),
            'score': float(score),
        }

    @staticmethod
    def _select_lama_frame_candidate(
        evaluations: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, Dict[str, int]]:
        """在 stage2_v2 / stage1_v2 / legacy 三个候选中选最优。"""
        eligible_names = [
            name
            for name in ('stage2_v2', 'stage1_v2', 'legacy')
            if not evaluations[name]['dark_block_flag']
            and not evaluations[name].get('seam_extreme_flag', False)
        ]
        if eligible_names:
            selected_name = min(
                eligible_names,
                key=lambda name: float(evaluations[name]['score']),
            )
        else:
            selected_name = min(
                ('stage2_v2', 'stage1_v2', 'legacy'),
                key=lambda name: float(evaluations[name]['score']),
            )

        legacy_score = float(evaluations['legacy']['score'])
        if (
            selected_name != 'legacy'
            and float(evaluations[selected_name]['score']) > legacy_score * 1.08
        ):
            selected_name = 'legacy'

        dark_rejects = 0
        seam_rejects = 0
        for candidate_name in ('stage2_v2', 'stage1_v2'):
            if candidate_name == selected_name:
                continue
            if evaluations[candidate_name]['dark_block_flag']:
                dark_rejects += 1
            if evaluations[candidate_name]['seam_bad_flag']:
                seam_rejects += 1

        return selected_name, {
            'dark_rejects': int(dark_rejects),
            'seam_rejects': int(seam_rejects),
        }

    @staticmethod
    def _apply_frame_guard_hysteresis(
        selected_name: str,
        evaluations: Dict[str, Dict[str, Any]],
        previous_name: Optional[str],
    ) -> Tuple[str, bool]:
        """候选切换滞回：避免相邻帧在方案间频繁抖动。"""
        if (
            previous_name not in ('stage2_v2', 'stage1_v2', 'legacy')
            or selected_name == previous_name
        ):
            return selected_name, False

        prev_eval = evaluations.get(previous_name)
        selected_eval = evaluations.get(selected_name)
        if prev_eval is None or selected_eval is None:
            return selected_name, False

        if prev_eval.get('dark_block_flag', False) or prev_eval.get('seam_extreme_flag', False):
            return selected_name, False

        if float(selected_eval.get('score', 0.0)) >= float(prev_eval.get('score', 0.0)) * 0.88:
            return previous_name, True
        return selected_name, False

    @staticmethod
    def _should_accept_lama_rescue(
        selected_quality: Dict[str, Any],
        rescue_quality: Dict[str, Any],
    ) -> bool:
        """判断 seamlessClone“救援融合”结果是否值得采用。"""
        return (
            float(rescue_quality.get('score', 0.0))
            < float(selected_quality.get('score', 0.0)) * 0.95
            and not bool(rescue_quality.get('dark_block_flag', False))
        )

    @staticmethod
    def _apply_final_micro_smoothing(
        current_roi: np.ndarray,
        previous_roi: Optional[np.ndarray],
        smoothing_mask: np.ndarray,
        force_reset: bool,
    ) -> Tuple[np.ndarray, bool]:
        """
        最后一层轻微时序平滑。

        仅在低运动场景启用，避免把动态细节抹掉。
        """
        if force_reset or previous_roi is None:
            return VideoProcessor._as_uint8_bgr(current_roi), False

        current_u8 = VideoProcessor._as_uint8_bgr(current_roi)
        previous_u8 = VideoProcessor._as_uint8_bgr(previous_roi)
        if previous_u8.shape[:2] != current_u8.shape[:2]:
            previous_u8 = cv2.resize(
                previous_u8,
                (current_u8.shape[1], current_u8.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        mask_bool = np.asarray(smoothing_mask) > 0
        if mask_bool.ndim != 2 or int(np.count_nonzero(mask_bool)) < 24:
            return current_u8, False

        prev_gray = cv2.cvtColor(previous_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        current_gray = cv2.cvtColor(current_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shift_xy, _ = cv2.phaseCorrelate(prev_gray, current_gray)
        shift_x = float(shift_xy[0]) if np.isfinite(shift_xy[0]) else 0.0
        shift_y = float(shift_xy[1]) if np.isfinite(shift_xy[1]) else 0.0
        shift_x = float(np.clip(shift_x, -4.0, 4.0))
        shift_y = float(np.clip(shift_y, -4.0, 4.0))
        warp = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
        aligned_prev = cv2.warpAffine(
            previous_u8,
            warp,
            (current_u8.shape[1], current_u8.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        aligned_prev_gray = cv2.cvtColor(aligned_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion_level = float(np.mean(np.abs(current_gray[mask_bool] - aligned_prev_gray[mask_bool])) / 255.0)
        if motion_level >= 0.10:
            return current_u8, False

        weight = float(np.clip(0.28 - motion_level * 2.2, 0.06, 0.24))
        out = current_u8.astype(np.float32)
        prev_f = aligned_prev.astype(np.float32)
        out[mask_bool] = out[mask_bool] * (1.0 - weight) + prev_f[mask_bool] * weight
        return np.clip(out, 0.0, 255.0).astype(np.uint8), True

    @staticmethod
    def _rescue_blend_with_seamless_clone(
        roi_original: np.ndarray,
        roi_candidate: np.ndarray,
        rounded_mask: np.ndarray,
    ) -> np.ndarray:
        """使用 OpenCV seamlessClone 做一次“缝合救援”。"""
        src = VideoProcessor._as_uint8_bgr(roi_candidate)
        dst = VideoProcessor._as_uint8_bgr(roi_original)
        mask = ((np.asarray(rounded_mask) > 0).astype(np.uint8) * 255)
        if int(np.count_nonzero(mask)) < 16:
            return src

        x, y, w, h = cv2.boundingRect(mask)
        center = (int(x + w / 2), int(y + h / 2))
        return cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)

    @staticmethod
    def _harmonize_transition_seam(
        roi_original: np.ndarray,
        blended_roi: np.ndarray,
        core_replace_mask: np.ndarray,
        transition_mask: np.ndarray,
    ) -> np.ndarray:
        """对过渡带做二次协调，进一步压低边缘可见度。"""
        transition_bool = np.asarray(transition_mask) > 0
        if transition_bool.ndim != 2 or int(np.count_nonzero(transition_bool)) == 0:
            return blended_roi

        core_bin = (np.asarray(core_replace_mask) > 0).astype(np.uint8)
        inv_core = (1 - core_bin).astype(np.uint8)
        dist_to_core = cv2.distanceTransform(inv_core, cv2.DIST_L2, 3)
        transition_dist = dist_to_core[transition_bool]
        dist_min = float(transition_dist.min()) if transition_dist.size > 0 else 0.0
        dist_max = float(transition_dist.max()) if transition_dist.size > 0 else 1.0
        seam_weight = np.zeros_like(dist_to_core, dtype=np.float32)
        if dist_max - dist_min > 1e-6:
            seam_weight[transition_bool] = (
                (dist_to_core[transition_bool] - dist_min) / max(1e-6, dist_max - dist_min)
            )
        seam_weight = np.clip(0.55 + 0.40 * seam_weight, 0.0, 0.95)

        out = VideoProcessor._as_uint8_bgr(blended_roi).astype(np.float32)
        orig = VideoProcessor._as_uint8_bgr(roi_original).astype(np.float32)
        w = seam_weight[transition_bool][:, np.newaxis]
        out[transition_bool] = out[transition_bool] * (1.0 - w) + orig[transition_bool] * w
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _apply_boundary_color_correction_weighted(
        roi_original: np.ndarray,
        inpainted_roi: np.ndarray,
        core_replace_mask: np.ndarray,
        transition_mask: np.ndarray,
        context_mask: np.ndarray,
        feather_px: int,
    ) -> np.ndarray:
        """
        加权边界色彩校正。

        用上下文区域统计量矫正过渡带颜色，减少“色块感”。
        """
        transition_bool = np.asarray(transition_mask) > 0
        context_bool = np.asarray(context_mask) > 0
        core_bool = np.asarray(core_replace_mask) > 0
        if int(np.count_nonzero(transition_bool)) < 24 or int(np.count_nonzero(context_bool)) < 24:
            return inpainted_roi

        source_lab = cv2.cvtColor(VideoProcessor._as_uint8_bgr(roi_original), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        target_lab = cv2.cvtColor(VideoProcessor._as_uint8_bgr(inpainted_roi), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        source_pixels = source_lab[context_bool]
        target_pixels = target_lab[transition_bool]
        if source_pixels.size == 0 or target_pixels.size == 0:
            return inpainted_roi

        source_mean = source_pixels.mean(axis=0)
        source_std = source_pixels.std(axis=0)
        target_mean = target_pixels.mean(axis=0)
        target_std = target_pixels.std(axis=0)
        gain = source_std / np.maximum(target_std, 1.0)
        gain = np.clip(gain, 0.86, 1.14)
        bias = source_mean - target_mean * gain
        bias = np.clip(bias, -14.0, 14.0)

        inv_core = (1 - core_bool.astype(np.uint8)).astype(np.uint8)
        dist_to_core = cv2.distanceTransform(inv_core, cv2.DIST_L2, 3)
        transition_dist = dist_to_core[transition_bool]
        dist_min = float(transition_dist.min()) if transition_dist.size > 0 else 0.0
        dist_max = float(transition_dist.max()) if transition_dist.size > 0 else 1.0
        weight = np.zeros_like(dist_to_core, dtype=np.float32)
        if dist_max - dist_min > 1e-6:
            weight[transition_bool] = (
                (dist_to_core[transition_bool] - dist_min) / max(1e-6, dist_max - dist_min)
            )
        weight = np.clip(0.30 + 0.70 * weight, 0.0, 1.0)

        corrected = target_lab.copy()
        corrected_transition = corrected[transition_bool] * gain + bias
        corrected_transition = np.clip(corrected_transition, 0.0, 255.0)
        w = weight[transition_bool][:, np.newaxis]
        corrected[transition_bool] = corrected[transition_bool] * (1.0 - w) + corrected_transition * w
        return cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _is_lama_pass2_acceptable(
        stage1_rois: List[np.ndarray],
        stage2_rois: List[np.ndarray],
        core_mask: np.ndarray,
    ) -> Tuple[bool, int]:
        """检查 LaMa 第二阶段是否出现明显“变暗/糊化/偏移”。"""
        if len(stage1_rois) != len(stage2_rois):
            return False, max(len(stage1_rois), len(stage2_rois))

        core_bool = (np.asarray(core_mask) > 0)
        if core_bool.ndim != 2 or int(np.count_nonzero(core_bool)) == 0:
            return False, len(stage2_rois)

        rejected_frames = 0
        for idx in range(len(stage1_rois)):
            s1 = VideoProcessor._as_uint8_bgr(stage1_rois[idx])
            s2 = VideoProcessor._as_uint8_bgr(stage2_rois[idx])
            if s2.shape[:2] != s1.shape[:2]:
                s2 = cv2.resize(s2, (s1.shape[1], s1.shape[0]), interpolation=cv2.INTER_CUBIC)

            s1_gray = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY).astype(np.float32)
            s2_gray = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY).astype(np.float32)
            s1_core = s1_gray[core_bool]
            s2_core = s2_gray[core_bool]
            if s1_core.size == 0 or s2_core.size == 0:
                rejected_frames += 1
                continue

            mean_shift = float(np.abs(s2_core.mean() - s1_core.mean()) / 255.0)
            std1 = float(np.std(s1_core))
            std2 = float(np.std(s2_core))
            texture_ratio = std2 / max(std1, 1.0)
            dark_collapse = (float(s2_core.mean()) < float(s1_core.mean()) - 25.0) and (texture_ratio < 0.55)
            severe_flatten = std1 > 10.0 and texture_ratio < 0.35
            severe_shift = mean_shift > 0.18
            if dark_collapse or severe_flatten or severe_shift:
                rejected_frames += 1

        max_reject = max(1, int(round(len(stage1_rois) * 0.15)))
        return rejected_frames <= max_reject, rejected_frames

    @staticmethod
    def _detect_hard_cuts_in_run(
        segment_frames: List[np.ndarray],
        roi_bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Set[int]:
        """检测段内硬切帧索引（用于时序平滑重置）。"""
        if len(segment_frames) < 2:
            return set()

        forced_reset_indices: Set[int] = set()
        prev_small_gray: Optional[np.ndarray] = None
        prev_hist: Optional[np.ndarray] = None
        prev_edge: Optional[np.ndarray] = None
        prev_roi_hist: Optional[np.ndarray] = None
        prev_roi_edge: Optional[np.ndarray] = None

        roi_x1 = roi_y1 = roi_x2 = roi_y2 = 0
        use_roi_signal = False
        if roi_bounds is not None and len(roi_bounds) == 4:
            roi_x1, roi_y1, roi_x2, roi_y2 = [int(v) for v in roi_bounds]
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                use_roi_signal = True

        for idx, frame in enumerate(segment_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            small_gray_f = small_gray.astype(np.float32)
            hist = cv2.calcHist([small_gray], [0], None, [32], [0, 256]).astype(np.float32)
            hist = hist / max(1e-6, float(hist.sum()))
            edge = np.abs(cv2.Laplacian(small_gray_f, cv2.CV_32F, ksize=3))
            roi_hist = None
            roi_edge = None
            if use_roi_signal:
                frame_h, frame_w = gray.shape[:2]
                rx1 = int(np.clip(roi_x1, 0, frame_w - 1))
                ry1 = int(np.clip(roi_y1, 0, frame_h - 1))
                rx2 = int(np.clip(roi_x2, rx1 + 1, frame_w))
                ry2 = int(np.clip(roi_y2, ry1 + 1, frame_h))
                roi_gray = gray[ry1:ry2, rx1:rx2]
                if roi_gray.size >= 16:
                    roi_small = cv2.resize(roi_gray, (80, 48), interpolation=cv2.INTER_AREA)
                    roi_hist = cv2.calcHist([roi_small], [0], None, [24], [0, 256]).astype(np.float32)
                    roi_hist = roi_hist / max(1e-6, float(roi_hist.sum()))
                    roi_edge = np.abs(cv2.Laplacian(roi_small.astype(np.float32), cv2.CV_32F, ksize=3))

            if prev_small_gray is None or prev_hist is None or prev_edge is None:
                prev_small_gray = small_gray_f
                prev_hist = hist
                prev_edge = edge
                prev_roi_hist = roi_hist
                prev_roi_edge = roi_edge
                continue

            luma_delta = float(abs(float(small_gray_f.mean()) - float(prev_small_gray.mean())) / 255.0)
            hist_chisq = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR))
            hist_delta = float(np.clip(hist_chisq / 2.0, 0.0, 1.0))
            edge_delta = float(np.mean(np.abs(edge - prev_edge)) / 255.0)
            edge_delta = float(np.clip(edge_delta, 0.0, 1.0))
            cut_score = 0.45 * hist_delta + 0.35 * luma_delta + 0.20 * edge_delta

            roi_hist_delta = 0.0
            roi_edge_delta = 0.0
            roi_cut_score = 0.0
            if (
                use_roi_signal
                and roi_hist is not None
                and roi_edge is not None
                and prev_roi_hist is not None
                and prev_roi_edge is not None
            ):
                roi_hist_chisq = float(cv2.compareHist(prev_roi_hist, roi_hist, cv2.HISTCMP_CHISQR))
                roi_hist_delta = float(np.clip(roi_hist_chisq / 2.0, 0.0, 1.0))
                roi_edge_delta = float(np.mean(np.abs(roi_edge - prev_roi_edge)) / 255.0)
                roi_edge_delta = float(np.clip(roi_edge_delta, 0.0, 1.0))
                roi_cut_score = 0.62 * roi_hist_delta + 0.38 * roi_edge_delta

            if (
                cut_score >= 0.56
                or (cut_score >= 0.48 and roi_cut_score >= 0.42)
                or (hist_delta >= 0.75 and luma_delta >= 0.10)
                or (hist_delta >= 0.45 and luma_delta >= 0.22)
            ):
                forced_reset_indices.add(idx)

            prev_small_gray = small_gray_f
            prev_hist = hist
            prev_edge = edge
            prev_roi_hist = roi_hist
            prev_roi_edge = roi_edge

        return forced_reset_indices

    @staticmethod
    def _compute_cut_quarantine_indices(
        total_length: int,
        cut_indices: Set[int],
        before: int = 1,
        after: int = 3,
    ) -> Set[int]:
        """基于硬切点生成隔离窗口索引，默认窗口 [cut-1, cut+3]。"""
        quarantine: Set[int] = set()
        if total_length <= 0 or not cut_indices:
            return quarantine
        for cut_idx in cut_indices:
            for offset in range(-int(before), int(after) + 1):
                idx = int(cut_idx) + offset
                if 0 <= idx < total_length:
                    quarantine.add(idx)
        return quarantine

    @staticmethod
    def _as_uint8_bgr(image: np.ndarray) -> np.ndarray:
        """把输入统一成 `uint8 BGR` 三通道格式。"""
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f'Unsupported image shape for BGR conversion: {arr.shape}')
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _apply_boundary_color_correction(
        roi_original: np.ndarray,
        inpainted_roi: np.ndarray,
        roi_mask: np.ndarray,
        feather_px: int,
    ) -> np.ndarray:
        """旧版边界色彩校正（保留给非 v2 路径使用）。"""
        mask_bin = (np.asarray(roi_mask) > 0).astype(np.uint8)
        if mask_bin.size == 0 or int(mask_bin.max()) == 0:
            return inpainted_roi

        ring_radius = int(np.clip(round(max(2, feather_px) * 1.25), 2, 12))
        outer_kernel_size = ring_radius * 2 + 1
        inner_kernel_size = max(3, (max(1, ring_radius // 2) * 2 + 1))
        outer_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (outer_kernel_size, outer_kernel_size),
        )
        inner_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (inner_kernel_size, inner_kernel_size),
        )

        dilated_mask = cv2.dilate(mask_bin, outer_kernel, iterations=1)
        eroded_mask = cv2.erode(mask_bin, inner_kernel, iterations=1)
        inner_ring = (mask_bin > 0) & ~(eroded_mask > 0)
        outer_ring = (dilated_mask > 0) & ~(mask_bin > 0)
        if int(np.count_nonzero(inner_ring)) < 32 or int(np.count_nonzero(outer_ring)) < 32:
            return inpainted_roi

        source_lab = cv2.cvtColor(roi_original, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(inpainted_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        source_pixels = source_lab[outer_ring]
        target_pixels = target_lab[inner_ring]
        if source_pixels.size == 0 or target_pixels.size == 0:
            return inpainted_roi

        source_mean = source_pixels.mean(axis=0)
        source_std = source_pixels.std(axis=0)
        target_mean = target_pixels.mean(axis=0)
        target_std = target_pixels.std(axis=0)

        gain = source_std / np.maximum(target_std, 1.0)
        gain = np.clip(gain, 0.82, 1.18)
        bias = source_mean - target_mean * gain
        bias = np.clip(bias, -18.0, 18.0)

        corrected = target_lab.copy()
        mask_bool = mask_bin > 0
        corrected_pixels = corrected[mask_bool]
        corrected_pixels = corrected_pixels * gain + bias
        corrected[mask_bool] = np.clip(corrected_pixels, 0.0, 255.0)
        return cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _compute_lama_temporal_strength(
        segment_total_frames: int,
        frame_width: int,
        frame_height: int,
    ) -> float:
        """按分辨率与段长度估计时序平滑强度。"""
        strength = 1.0
        if segment_total_frames > 1200:
            strength *= 0.9
        if segment_total_frames > 2400:
            strength *= 0.8

        megapixels = (max(1, frame_width) * max(1, frame_height)) / 1_000_000.0
        if megapixels > 2.1:
            strength *= 0.85
        if megapixels > 4.0:
            strength *= 0.75
        return float(np.clip(strength, 0.45, 1.0))

    @staticmethod
    def _compute_propainter_temporal_strength(
        segment_total_frames: int,
        frame_width: int,
        frame_height: int,
    ) -> float:
        """按片段长度与分辨率估算 ProPainter 时序平滑强度。"""
        strength = 1.0
        if segment_total_frames > 180:
            strength *= 0.92
        if segment_total_frames > 360:
            strength *= 0.84

        megapixels = (max(1, frame_width) * max(1, frame_height)) / 1_000_000.0
        if megapixels > 2.1:
            strength *= 0.84
        if megapixels > 4.0:
            strength *= 0.72
        return float(np.clip(strength, 0.40, 1.0))

    @staticmethod
    def _compute_propainter_infer_options(
        rect_w: int,
        rect_h: int,
        feather_px: int,
        segment_total_frames: int,
        fps: float,
    ) -> Dict[str, int]:
        """生成 ProPainter 内部推理参数（自适应，外部接口无变化）。"""
        rect_max = max(1, int(max(rect_w, rect_h)))
        mask_dilation = int(
            np.clip(max(float(feather_px) * 1.5, float(rect_max) * 0.10), 4, 14)
        )
        neighbor_length = 14 if segment_total_frames >= 120 else 12
        ref_stride = 6 if segment_total_frames >= 120 else 8
        subvideo_length = 80 if segment_total_frames >= 300 else 100
        save_fps = int(np.clip(round(float(fps) if fps and fps > 0 else 24.0), 1, 120))
        return {
            "mask_dilation": int(mask_dilation),
            "neighbor_length": int(neighbor_length),
            "ref_stride": int(ref_stride),
            "subvideo_length": int(subvideo_length),
            "raft_iter": 24,
            "save_fps": int(save_fps),
        }

    @staticmethod
    def _compute_propainter_rerun_options(
        base_options: Dict[str, int],
        rect_w: int,
        rect_h: int,
    ) -> Dict[str, int]:
        """为失败段生成更激进的一次性重跑参数。"""
        rect_max = max(1, int(max(rect_w, rect_h)))
        dilation_boost = max(2, int(round(rect_max * 0.04)))
        options = dict(base_options)
        options["mask_dilation"] = int(np.clip(int(options.get("mask_dilation", 4)) + dilation_boost, 6, 20))
        options["neighbor_length"] = int(np.clip(int(options.get("neighbor_length", 12)) + 4, 12, 20))
        options["ref_stride"] = int(np.clip(int(options.get("ref_stride", 8)) - 2, 4, 10))
        options["subvideo_length"] = int(min(int(options.get("subvideo_length", 100)), 80))
        options["raft_iter"] = int(np.clip(int(options.get("raft_iter", 24)) + 8, 24, 36))
        return options

    @staticmethod
    def _should_rerun_propainter_segment(
        legacy_ratio: float,
        median_remove_ratio: float,
        reappear_count: int,
        frame_count: int,
        median_residual_hf_corr: float = 0.0,
        under_remove_rate: float = 0.0,
        burst_count: int = 0,
    ) -> bool:
        """判定段级失败是否触发一次性重跑。"""
        total = max(1, int(frame_count))
        if float(legacy_ratio) > 0.25:
            return True
        if float(median_remove_ratio) < 0.60:
            return True
        if int(reappear_count) > max(2, int(round(total * 0.03))):
            return True
        if float(median_residual_hf_corr) > 0.68:
            return True
        if float(under_remove_rate) > 0.18:
            return True
        if int(burst_count) > max(2, int(round(total * 0.03))):
            return True
        return False

    @staticmethod
    def _should_accept_propainter_rerun(
        pass1_median_remove_ratio: float,
        pass1_legacy_ratio: float,
        pass2_median_remove_ratio: float,
        pass2_legacy_ratio: float,
        pass1_under_remove_rate: float = 0.0,
        pass2_under_remove_rate: float = 0.0,
        pass1_seam_p90: float = 0.0,
        pass2_seam_p90: float = 0.0,
    ) -> bool:
        """判定段级重跑结果是否值得替换首轮输出。"""
        improved_remove = (
            float(pass2_median_remove_ratio) >= float(pass1_median_remove_ratio) + 0.08
        )
        improved_legacy = float(pass2_legacy_ratio) <= float(pass1_legacy_ratio) - 0.15
        improved_under_remove = float(pass2_under_remove_rate) <= float(pass1_under_remove_rate) - 0.12
        seam_guard_ok = float(pass2_seam_p90) <= float(pass1_seam_p90) + 0.005
        return bool((improved_remove or improved_legacy or improved_under_remove) and seam_guard_ok)

    @staticmethod
    def _split_sequence_by_cut_indices(
        total_length: int,
        cut_indices: Set[int],
    ) -> List[Tuple[int, int]]:
        """按硬切点把序列拆成多个子区间，区间采用 [start, end) 表示。"""
        if total_length <= 0:
            return []
        boundaries = [0]
        for idx in sorted(cut_indices):
            if 0 < idx < total_length:
                boundaries.append(int(idx))
        boundaries.append(total_length)

        ranges: List[Tuple[int, int]] = []
        for i in range(len(boundaries) - 1):
            start = int(boundaries[i])
            end = int(boundaries[i + 1])
            if end > start:
                ranges.append((start, end))
        return ranges

    @staticmethod
    def _stabilize_propainter_sequence(
        roi_sequence: List[np.ndarray],
        roi_mask: np.ndarray,
        temporal_strength: float,
        forced_reset_indices: Optional[Set[int]] = None,
        quarantine_indices: Optional[Set[int]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """
        ProPainter 专用轻量时序稳定器。

        目标是降低局部闪烁，同时避免高运动场景拖影。
        """
        if not roi_sequence:
            return [], {
                "hard_cut_resets": 0,
                "cold_start_frames": 0,
                "stabilize_applied_frames": 0,
                "cut_quarantine_frames": 0,
            }

        mask_bool = np.asarray(roi_mask) > 0
        if mask_bool.ndim != 2 or int(np.count_nonzero(mask_bool)) < 16:
            return roi_sequence, {
                "hard_cut_resets": 0,
                "cold_start_frames": 0,
                "stabilize_applied_frames": 0,
                "cut_quarantine_frames": 0,
            }

        stable_sequence: List[np.ndarray] = []
        prev_stable: Optional[np.ndarray] = None
        prev_gray: Optional[np.ndarray] = None
        forced_set = set(forced_reset_indices or set())
        quarantine_set = set(quarantine_indices or set())
        cold_start_remaining = 0
        cold_start_window = 2
        diagnostics = {
            "hard_cut_resets": 0,
            "cold_start_frames": 0,
            "stabilize_applied_frames": 0,
            "cut_quarantine_frames": 0,
        }

        for idx, current in enumerate(roi_sequence):
            current_u8 = VideoProcessor._as_uint8_bgr(current)
            current_gray = cv2.cvtColor(current_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_stable is None or prev_gray is None:
                stable_sequence.append(current_u8.copy())
                prev_stable = stable_sequence[-1]
                prev_gray = current_gray
                continue

            if idx in forced_set:
                diagnostics["hard_cut_resets"] += 1
                cold_start_remaining = max(cold_start_remaining, cold_start_window)
                stable_sequence.append(current_u8.copy())
                prev_stable = stable_sequence[-1]
                prev_gray = current_gray
                continue

            shift_xy, response = cv2.phaseCorrelate(prev_gray, current_gray)
            shift_x = float(shift_xy[0]) if np.isfinite(shift_xy[0]) else 0.0
            shift_y = float(shift_xy[1]) if np.isfinite(shift_xy[1]) else 0.0
            shift_x = float(np.clip(shift_x, -5.0, 5.0))
            shift_y = float(np.clip(shift_y, -5.0, 5.0))
            response_score = float(response) if np.isfinite(response) else 0.0

            warp = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
            aligned_prev = cv2.warpAffine(
                prev_stable,
                warp,
                (current_u8.shape[1], current_u8.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            aligned_prev_gray = cv2.cvtColor(aligned_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
            motion_level = float(np.mean(np.abs(current_gray[mask_bool] - aligned_prev_gray[mask_bool])) / 255.0)

            # ROI 内部兜底硬切检测，防止异常情况下历史颜色延续。
            if response_score < 0.05 and motion_level > 0.18:
                diagnostics["hard_cut_resets"] += 1
                cold_start_remaining = max(cold_start_remaining, cold_start_window)
                stable_sequence.append(current_u8.copy())
                prev_stable = stable_sequence[-1]
                prev_gray = current_gray
                continue

            prev_weight = (0.55 - 0.85 * motion_level) * float(temporal_strength)
            prev_weight = float(np.clip(prev_weight, 0.00, 0.45))
            if idx in quarantine_set:
                prev_weight = 0.0
                diagnostics["cut_quarantine_frames"] += 1
            elif cold_start_remaining > 0:
                cold_start_remaining -= 1
                diagnostics["cold_start_frames"] += 1
                prev_weight = 0.0

            if prev_weight <= 1e-6:
                stable_frame = current_u8.copy()
            else:
                stable_f = current_u8.astype(np.float32)
                prev_f = aligned_prev.astype(np.float32)
                stable_f[mask_bool] = (
                    stable_f[mask_bool] * (1.0 - prev_weight)
                    + prev_f[mask_bool] * prev_weight
                )
                stable_frame = np.clip(stable_f, 0.0, 255.0).astype(np.uint8)
                diagnostics["stabilize_applied_frames"] += 1

            stable_sequence.append(stable_frame)
            prev_stable = stable_frame
            prev_gray = cv2.cvtColor(prev_stable, cv2.COLOR_BGR2GRAY).astype(np.float32)

        return stable_sequence, diagnostics

    @staticmethod
    def _evaluate_propainter_frame_quality(
        roi_original: np.ndarray,
        roi_candidate: np.ndarray,
        core_mask: np.ndarray,
        transition_mask: np.ndarray,
        previous_selected_roi: Optional[np.ndarray],
        is_hard_cut_frame: bool = False,
    ) -> Dict[str, Any]:
        """ProPainter 单帧质量评估（含时域项）。"""
        base_eval = VideoProcessor._evaluate_lama_frame_quality(
            roi_original=roi_original,
            roi_candidate=roi_candidate,
            core_mask=core_mask,
            transition_mask=transition_mask,
        )
        temporal_warp_error = 0.0
        temporal_jump_core = 0.0
        original_similarity_core = 0.0
        remove_energy_core = 0.0
        residual_hf_corr = 0.0
        core_bool = (np.asarray(core_mask) > 0)
        if core_bool.ndim == 2 and int(np.count_nonzero(core_bool)) >= 8:
            original_u8 = VideoProcessor._as_uint8_bgr(roi_original).astype(np.float32)
            candidate_u8 = VideoProcessor._as_uint8_bgr(roi_candidate).astype(np.float32)
            core_abs_delta = np.abs(candidate_u8[core_bool] - original_u8[core_bool]).mean() / 255.0
            remove_energy_core = float(core_abs_delta)
            original_similarity_core = float(np.clip(1.0 - remove_energy_core, 0.0, 1.0))
            original_gray = cv2.cvtColor(original_u8.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            candidate_gray_for_hf = cv2.cvtColor(candidate_u8.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
                np.float32
            )
            original_hf = np.abs(cv2.Laplacian(original_gray, cv2.CV_32F, ksize=3))[core_bool].reshape(-1)
            candidate_hf = np.abs(cv2.Laplacian(candidate_gray_for_hf, cv2.CV_32F, ksize=3))[
                core_bool
            ].reshape(-1)
            if original_hf.size >= 8 and candidate_hf.size == original_hf.size:
                original_hf = original_hf - float(original_hf.mean())
                candidate_hf = candidate_hf - float(candidate_hf.mean())
                denom = float(np.linalg.norm(original_hf) * np.linalg.norm(candidate_hf))
                if denom > 1e-6:
                    corr = float(np.dot(original_hf, candidate_hf) / denom)
                    residual_hf_corr = float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))
        eval_mask = (np.asarray(transition_mask) > 0)
        if int(np.count_nonzero(eval_mask)) < 8:
            eval_mask = core_bool
        if (
            previous_selected_roi is not None
            and eval_mask.ndim == 2
            and int(np.count_nonzero(eval_mask)) >= 8
        ):
            prev_u8 = VideoProcessor._as_uint8_bgr(previous_selected_roi)
            curr_u8 = VideoProcessor._as_uint8_bgr(roi_candidate)
            if prev_u8.shape[:2] != curr_u8.shape[:2]:
                prev_u8 = cv2.resize(
                    prev_u8,
                    (curr_u8.shape[1], curr_u8.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            prev_gray = cv2.cvtColor(prev_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
            curr_gray = cv2.cvtColor(curr_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
            shift_xy, _ = cv2.phaseCorrelate(prev_gray, curr_gray)
            shift_x = float(shift_xy[0]) if np.isfinite(shift_xy[0]) else 0.0
            shift_y = float(shift_xy[1]) if np.isfinite(shift_xy[1]) else 0.0
            shift_x = float(np.clip(shift_x, -4.0, 4.0))
            shift_y = float(np.clip(shift_y, -4.0, 4.0))
            warp = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
            aligned_prev = cv2.warpAffine(
                prev_gray,
                warp,
                (curr_gray.shape[1], curr_gray.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            temporal_warp_error = float(
                np.mean(np.abs(curr_gray[eval_mask] - aligned_prev[eval_mask])) / 255.0
            )
            core_eval_mask = core_bool if int(np.count_nonzero(core_bool)) >= 8 else eval_mask
            temporal_jump_core = float(
                np.mean(np.abs(curr_gray[core_eval_mask] - aligned_prev[core_eval_mask])) / 255.0
            )

        reappear_flag = (
            (not is_hard_cut_frame)
            and (
                (original_similarity_core > 0.94 and temporal_jump_core > 0.08)
                or (residual_hf_corr > 0.80 and temporal_jump_core > 0.06)
            )
        )
        score = float(base_eval["score"]) + max(0.0, temporal_warp_error - 0.03) * 0.65
        score += max(0.0, original_similarity_core - 0.90) * 0.90
        score += max(0.0, temporal_jump_core - 0.05) * 0.70
        score += max(0.0, residual_hf_corr - 0.65) * 0.95
        out = dict(base_eval)
        out["temporal_warp_error"] = float(temporal_warp_error)
        out["temporal_jump_core"] = float(temporal_jump_core)
        out["original_similarity_core"] = float(original_similarity_core)
        out["remove_energy_core"] = float(remove_energy_core)
        out["residual_hf_corr"] = float(residual_hf_corr)
        out["reappear_score"] = float(
            max(0.0, original_similarity_core - 0.90)
            + max(0.0, temporal_jump_core - 0.05)
            + max(0.0, residual_hf_corr - 0.72)
        )
        out["temporal_bad_flag"] = bool(temporal_warp_error > 0.10)
        out["reappear_flag"] = bool(reappear_flag)
        out["score"] = float(score)
        return out

    @staticmethod
    def _apply_propainter_remove_sufficiency(
        quality: Dict[str, Any],
        remove_energy_reference: float,
        previous_selected_remove_ratio: Optional[float],
        is_hard_cut_frame: bool,
    ) -> Dict[str, Any]:
        """给单候选追加“去除充分性”判据并重算分数。"""
        output = dict(quality)
        reference = max(1e-4, float(remove_energy_reference))
        remove_energy = float(output.get("remove_energy_core", 0.0))
        remove_ratio = float(np.clip(remove_energy / reference, 0.0, 4.0))
        residual_hf_corr = float(output.get("residual_hf_corr", 0.0))
        under_remove_flag = (not is_hard_cut_frame) and (
            remove_ratio < 0.60 or residual_hf_corr > 0.72
        )

        score = float(output.get("score", 0.0))
        score += max(0.0, 0.70 - remove_ratio) * 1.10

        reappear_flag = bool(output.get("reappear_flag", False))
        if (
            not is_hard_cut_frame
            and previous_selected_remove_ratio is not None
            and float(previous_selected_remove_ratio) - remove_ratio > 0.22
            and remove_ratio < 0.60
        ):
            reappear_flag = True

        output["remove_ratio"] = float(remove_ratio)
        output["under_remove_flag"] = bool(under_remove_flag)
        output["remove_energy_reference"] = float(reference)
        output["residual_hf_corr"] = float(residual_hf_corr)
        output["reappear_flag"] = bool(reappear_flag)
        output["score"] = float(score)
        return output

    @staticmethod
    def _augment_propainter_evaluations_for_remove_sufficiency(
        evaluations: Dict[str, Dict[str, Any]],
        previous_selected_remove_ratio: Optional[float],
        is_hard_cut_frame: bool,
    ) -> Tuple[Dict[str, Dict[str, Any]], float]:
        """给 raw/stable/legacy 三候选统一补充 remove_ratio 相关指标。"""
        remove_reference = max(
            1e-4,
            float(evaluations.get("raw_v2", {}).get("remove_energy_core", 0.0)),
            float(evaluations.get("stable_v2", {}).get("remove_energy_core", 0.0)),
        )
        updated: Dict[str, Dict[str, Any]] = {}
        for name, quality in evaluations.items():
            updated[name] = VideoProcessor._apply_propainter_remove_sufficiency(
                quality=quality,
                remove_energy_reference=remove_reference,
                previous_selected_remove_ratio=previous_selected_remove_ratio,
                is_hard_cut_frame=is_hard_cut_frame,
            )
        return updated, float(remove_reference)

    @staticmethod
    def _is_propainter_severe_dark(eval_item: Dict[str, Any]) -> bool:
        """判定是否属于 ProPainter 的严重暗块质量问题。"""
        if not bool(eval_item.get("dark_block_flag", False)):
            return False
        luma_shift = eval_item.get("core_luma_shift")
        texture_ratio = eval_item.get("core_texture_ratio")
        # 兼容测试桩：若缺少指标，按保守策略视为严重暗块。
        if luma_shift is None or texture_ratio is None:
            return True
        return float(luma_shift) < -0.12 and float(texture_ratio) < 0.60

    @staticmethod
    def _is_propainter_catastrophic(eval_item: Dict[str, Any]) -> bool:
        """判定候选是否达到“仅允许 legacy 接管”的灾难级风险。"""
        return (
            bool(eval_item.get("seam_extreme_flag", False))
            or VideoProcessor._is_propainter_severe_dark(eval_item)
            or float(eval_item.get("temporal_warp_error", 0.0)) > 0.18
        )

    @staticmethod
    def _is_propainter_hard_reject(eval_item: Dict[str, Any]) -> bool:
        """判定候选是否不应被用于 ProPainter 主路径。"""
        return (
            bool(eval_item.get("seam_extreme_flag", False))
            or bool(eval_item.get("temporal_bad_flag", False))
            or bool(eval_item.get("reappear_flag", False))
            or VideoProcessor._is_propainter_severe_dark(eval_item)
        )

    @staticmethod
    def _is_propainter_force_mode_block(eval_item: Dict[str, Any]) -> bool:
        """
        去除优先模式下的“极端拒绝”条件。

        与 hard_reject 的区别：允许 temporal_bad，避免长段被 legacy 锁死。
        """
        return (
            bool(eval_item.get("seam_extreme_flag", False))
            or bool(eval_item.get("reappear_flag", False))
            or VideoProcessor._is_propainter_severe_dark(eval_item)
        )

    @staticmethod
    def _should_enable_propainter_remove_priority(
        frames_observed: int,
        legacy_frames: int,
        original_similarity_avg: float,
        probe_window: int = 20,
    ) -> bool:
        """
        判定是否进入“去除优先模式”，防止段级 legacy 锁死。

        规则：前 probe_window 帧里 legacy 比例和与原图相似度都异常偏高。
        """
        probe_window = max(1, int(probe_window))
        if int(frames_observed) < probe_window:
            return False
        legacy_ratio = float(legacy_frames) / max(1.0, float(frames_observed))
        similarity = float(original_similarity_avg)
        if legacy_ratio >= 0.90 and similarity >= 0.88:
            return True
        if legacy_ratio >= 0.65 and similarity >= 0.82:
            return True
        return False

    @staticmethod
    def _compute_propainter_segment_probe_window(segment_total_frames: int) -> int:
        """
        计算 ProPainter 段级去除优先探测窗口。

        设计目标：
        - 短段尽快触发（避免 5 帧段“最后一帧才触发”）；
        - 长段保持稳健（避免过早误触发）。
        """
        total = max(1, int(segment_total_frames))
        if total <= 6:
            return 3
        if total <= 12:
            return 4
        return int(np.clip(round(total * 0.25), 5, 20))

    @staticmethod
    def _select_propainter_frame_candidate(
        evaluations: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, Dict[str, int]]:
        """在 raw_v2 / stable_v2 / legacy 之间选择最佳候选。"""
        v2_candidates = ("stable_v2", "raw_v2")
        v2_eligible = [
            name
            for name in v2_candidates
            if not VideoProcessor._is_propainter_hard_reject(evaluations[name])
        ]
        if v2_eligible:
            selected = min(v2_eligible, key=lambda name: float(evaluations[name]["score"]))
        else:
            selected = min(v2_candidates, key=lambda name: float(evaluations[name]["score"]))

        both_v2_catastrophic = all(
            VideoProcessor._is_propainter_catastrophic(evaluations[name]) for name in v2_candidates
        )
        if both_v2_catastrophic and not VideoProcessor._is_propainter_hard_reject(evaluations["legacy"]):
            selected = "legacy"

        dark_rejects = 0
        seam_rejects = 0
        reappear_rejects = 0
        for candidate_name in v2_candidates:
            if candidate_name == selected:
                continue
            if evaluations[candidate_name].get("dark_block_flag", False):
                dark_rejects += 1
            if evaluations[candidate_name].get("seam_bad_flag", False):
                seam_rejects += 1
            if evaluations[candidate_name].get("reappear_flag", False):
                reappear_rejects += 1
        return selected, {
            "dark_rejects": dark_rejects,
            "seam_rejects": seam_rejects,
            "reappear_rejects": reappear_rejects,
            "legacy_catastrophic_only": 1 if selected == "legacy" and both_v2_catastrophic else 0,
        }

    @staticmethod
    def _apply_propainter_frame_hysteresis(
        selected_name: str,
        evaluations: Dict[str, Dict[str, Any]],
        previous_name: Optional[str],
        legacy_advantage_streak: int,
    ) -> Tuple[str, int, Dict[str, int]]:
        """候选切换滞回与 legacy 进入门控。"""
        stats = {"hold_count": 0, "switch_count": 0, "legacy_blocked": 0}
        if (
            previous_name not in ("stable_v2", "raw_v2", "legacy")
            or previous_name not in evaluations
        ):
            next_streak = 1 if selected_name == "legacy" else 0
            return selected_name, next_streak, stats

        prev_eval = evaluations[previous_name]
        selected_eval = evaluations.get(selected_name, prev_eval)
        prev_risky = (
            bool(prev_eval.get("dark_block_flag", False))
            or bool(prev_eval.get("seam_extreme_flag", False))
            or bool(prev_eval.get("reappear_flag", False))
            or bool(prev_eval.get("under_remove_flag", False))
        )

        if selected_name != previous_name and not prev_risky:
            if float(selected_eval.get("score", 0.0)) >= float(prev_eval.get("score", 0.0)) * 0.82:
                selected_name = previous_name
                selected_eval = prev_eval
                stats["hold_count"] = 1

        if selected_name == "legacy":
            both_v2_catastrophic = all(
                VideoProcessor._is_propainter_catastrophic(evaluations[name])
                for name in ("stable_v2", "raw_v2")
            )
            fallback_name = min(
                ("stable_v2", "raw_v2"),
                key=lambda name: float(evaluations.get(name, selected_eval).get("score", 1e9)),
            )
            if not both_v2_catastrophic:
                selected_name = fallback_name
                stats["legacy_blocked"] = 1
                legacy_advantage_streak = 0
            else:
                legacy_score = float(evaluations.get("legacy", selected_eval).get("score", 1e9))
                best_v2_score = min(
                    float(evaluations.get("stable_v2", selected_eval).get("score", 1e9)),
                    float(evaluations.get("raw_v2", selected_eval).get("score", 1e9)),
                )
                catastrophic_advantage = legacy_score <= best_v2_score * 0.92
                if previous_name == "legacy":
                    legacy_advantage_streak = (
                        legacy_advantage_streak + 1 if catastrophic_advantage else 0
                    )
                else:
                    legacy_advantage_streak = 1 if catastrophic_advantage else 0
                if previous_name != "legacy" and legacy_advantage_streak < 2:
                    selected_name = fallback_name
                    stats["legacy_blocked"] = 1
        else:
            legacy_advantage_streak = 0

        if selected_name != previous_name:
            stats["switch_count"] = 1
        return selected_name, legacy_advantage_streak, stats

    @staticmethod
    def _select_propainter_sequence_viterbi(
        frame_evaluations: List[Dict[str, Dict[str, Any]]],
        cut_quarantine_indices: Optional[Set[int]] = None,
        switch_penalty: float = 0.045,
        legacy_penalty: float = 0.08,
        reappear_penalty: float = 1.0,
    ) -> Tuple[List[str], int]:
        """
        对单段候选执行序列级路径优化，减少单帧抖动切换。

        状态为 `stable_v2/raw_v2/legacy`，legacy 只在灾难条件下可用。
        """
        if not frame_evaluations:
            return [], 0
        states = ("stable_v2", "raw_v2", "legacy")
        state_to_idx = {name: idx for idx, name in enumerate(states)}
        inf = 1e12
        quarantine = set(cut_quarantine_indices or set())
        n = len(frame_evaluations)
        dp = np.full((n, len(states)), inf, dtype=np.float64)
        back = np.full((n, len(states)), -1, dtype=np.int32)

        for i in range(n):
            evals = frame_evaluations[i]
            both_v2_catastrophic = all(
                VideoProcessor._is_propainter_catastrophic(evals[name]) for name in ("stable_v2", "raw_v2")
            )
            legacy_allowed = both_v2_catastrophic and not VideoProcessor._is_propainter_hard_reject(
                evals["legacy"]
            )
            if i in quarantine:
                legacy_allowed = False
            for state in states:
                eval_item = evals[state]
                state_cost = float(eval_item.get("score", 0.0))
                state_cost += reappear_penalty * float(eval_item.get("reappear_score", 0.0))
                if state == "legacy":
                    state_cost += legacy_penalty
                    if not legacy_allowed:
                        state_cost = inf
                if i == 0:
                    dp[i, state_to_idx[state]] = state_cost
                    continue
                best_prev_cost = inf
                best_prev_idx = -1
                for prev_state in states:
                    prev_idx = state_to_idx[prev_state]
                    transition_cost = 0.0
                    if prev_state != state:
                        if i in quarantine or (i - 1) in quarantine:
                            transition_cost = 0.0
                        else:
                            transition_cost = switch_penalty
                    candidate_cost = float(dp[i - 1, prev_idx]) + float(transition_cost)
                    if candidate_cost < best_prev_cost:
                        best_prev_cost = candidate_cost
                        best_prev_idx = prev_idx
                dp[i, state_to_idx[state]] = state_cost + best_prev_cost
                back[i, state_to_idx[state]] = best_prev_idx

            if np.all(dp[i] >= inf * 0.5):
                fallback_state = min(
                    ("stable_v2", "raw_v2"),
                    key=lambda name: float(evals[name].get("score", 1e9)),
                )
                dp[i, state_to_idx[fallback_state]] = float(evals[fallback_state].get("score", 0.0))
                if i > 0:
                    prev_best = int(np.argmin(dp[i - 1]))
                    back[i, state_to_idx[fallback_state]] = prev_best

        end_state_idx = int(np.argmin(dp[n - 1]))
        indices: List[int] = [end_state_idx]
        for i in range(n - 1, 0, -1):
            prev_idx = int(back[i, indices[-1]])
            if prev_idx < 0:
                prev_idx = int(np.argmin(dp[i - 1]))
            indices.append(prev_idx)
        indices.reverse()
        selected_names = [states[idx] for idx in indices]
        switches = 0
        for i in range(1, len(selected_names)):
            if selected_names[i] != selected_names[i - 1]:
                switches += 1
        return selected_names, int(switches)

    @staticmethod
    def _suppress_propainter_selection_islands(
        selected_names: List[str],
        frame_evaluations: List[Dict[str, Dict[str, Any]]],
        cut_quarantine_indices: Optional[Set[int]] = None,
        max_score_ratio: float = 1.05,
    ) -> Tuple[List[str], int]:
        """
        抑制 Viterbi 路径中的 1~2 帧状态孤岛（仅 raw/stable 间）。

        支持两种模式：
        - A-B-A（1 帧孤岛）
        - A-B-B-A（2 帧孤岛）
        """
        total = len(selected_names)
        if total == 0 or len(frame_evaluations) != total:
            return list(selected_names), 0

        names = list(selected_names)
        quarantine = set(cut_quarantine_indices or set())
        rewrites = 0
        v2_states = {"raw_v2", "stable_v2"}

        def _can_rewrite(index: int, target_state: str, current_state: str) -> bool:
            if index < 0 or index >= total:
                return False
            if index in quarantine or target_state not in v2_states or current_state not in v2_states:
                return False
            evals = frame_evaluations[index]
            target_eval = evals.get(target_state, {})
            current_eval = evals.get(current_state, {})
            if VideoProcessor._is_propainter_hard_reject(target_eval):
                return False
            target_score = float(target_eval.get("score", 1e9))
            current_score = float(current_eval.get("score", 1e9))
            return target_score <= current_score * float(max_score_ratio)

        # 先处理 2 帧孤岛，避免被 1 帧改写拆分。
        for idx in range(1, total - 2):
            left = names[idx - 1]
            mid1 = names[idx]
            mid2 = names[idx + 1]
            right = names[idx + 2]
            if left != right or mid1 != mid2 or left == mid1:
                continue
            if left not in v2_states or mid1 not in v2_states:
                continue
            if (idx in quarantine) or ((idx + 1) in quarantine):
                continue
            if _can_rewrite(idx, left, mid1) and _can_rewrite(idx + 1, left, mid2):
                names[idx] = left
                names[idx + 1] = left
                rewrites += 2

        # 再处理 1 帧孤岛。
        for idx in range(1, total - 1):
            left = names[idx - 1]
            mid = names[idx]
            right = names[idx + 1]
            if left != right or left == mid:
                continue
            if left not in v2_states or mid not in v2_states:
                continue
            if _can_rewrite(idx, left, mid):
                names[idx] = left
                rewrites += 1

        return names, int(rewrites)

    @staticmethod
    def _detect_propainter_micro_flicker_flags(
        selected_qualities: List[Dict[str, Any]],
        cut_quarantine_indices: Optional[Set[int]] = None,
        window_radius: int = 2,
    ) -> List[bool]:
        """
        检测“近失型”微闪烁帧（可能未触发 reappear/under_remove）。
        """
        total = len(selected_qualities)
        if total == 0:
            return []

        quarantine = set(cut_quarantine_indices or set())
        flags: List[bool] = [False for _ in range(total)]
        radius = max(1, int(window_radius))

        for idx in range(total):
            if idx in quarantine:
                continue
            win_start = max(0, idx - radius)
            win_end = min(total, idx + radius + 1)
            window_items = [selected_qualities[j] for j in range(win_start, win_end) if j not in quarantine]
            if len(window_items) < 3:
                continue

            local_jump_median = float(
                np.median([float(item.get("temporal_jump_core", 0.0)) for item in window_items])
            )
            local_remove_median = float(
                np.median([float(item.get("remove_ratio", 0.0)) for item in window_items])
            )
            local_hf_median = float(
                np.median([float(item.get("residual_hf_corr", 0.0)) for item in window_items])
            )

            item = selected_qualities[idx]
            jump = float(item.get("temporal_jump_core", 0.0))
            remove_ratio = float(item.get("remove_ratio", 0.0))
            residual_hf = float(item.get("residual_hf_corr", 0.0))
            hit = jump > max(0.060, local_jump_median + 0.025) and (
                remove_ratio < local_remove_median - 0.12
                or residual_hf > local_hf_median + 0.08
            )
            flags[idx] = bool(hit)
        return flags

    @staticmethod
    def _reevaluate_propainter_sequence_qualities(
        rois: List[np.ndarray],
        qualities: List[Dict[str, Any]],
        roi_originals: List[np.ndarray],
        core_mask: np.ndarray,
        transition_mask: np.ndarray,
        excluded_indices: Set[int],
    ) -> List[Dict[str, Any]]:
        """
        按最终输出序列重算 ProPainter 质量指标，保证时域项与结果一致。
        """
        updated: List[Dict[str, Any]] = [dict(item) for item in qualities]
        total = len(rois)
        for idx in range(total):
            reevaluated = VideoProcessor._evaluate_propainter_frame_quality(
                roi_original=roi_originals[idx],
                roi_candidate=rois[idx],
                core_mask=core_mask,
                transition_mask=transition_mask,
                previous_selected_roi=(rois[idx - 1] if idx > 0 else None),
                is_hard_cut_frame=(idx in excluded_indices),
            )
            remove_reference = max(
                1e-4,
                float(updated[idx].get("remove_energy_reference", 0.0)),
                float(reevaluated.get("remove_energy_core", 0.0)),
            )
            updated[idx] = VideoProcessor._apply_propainter_remove_sufficiency(
                quality=reevaluated,
                remove_energy_reference=remove_reference,
                previous_selected_remove_ratio=float(updated[idx - 1].get("remove_ratio", 1.0))
                if idx > 0
                else None,
                is_hard_cut_frame=(idx in excluded_indices),
            )
        return updated

    @staticmethod
    def _warp_propainter_roi_to_anchor(
        source_roi: np.ndarray,
        anchor_roi: np.ndarray,
    ) -> np.ndarray:
        """把 source ROI 通过相位相关平移对齐到 anchor ROI。"""
        source_u8 = VideoProcessor._as_uint8_bgr(source_roi)
        anchor_u8 = VideoProcessor._as_uint8_bgr(anchor_roi)
        source_gray = cv2.cvtColor(source_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        anchor_gray = cv2.cvtColor(anchor_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shift_xy, _ = cv2.phaseCorrelate(source_gray, anchor_gray)
        shift_x = float(shift_xy[0]) if np.isfinite(shift_xy[0]) else 0.0
        shift_y = float(shift_xy[1]) if np.isfinite(shift_xy[1]) else 0.0
        shift_x = float(np.clip(shift_x, -6.0, 6.0))
        shift_y = float(np.clip(shift_y, -6.0, 6.0))
        warp = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
        return cv2.warpAffine(
            source_u8,
            warp,
            (anchor_u8.shape[1], anchor_u8.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    @staticmethod
    def _build_propainter_neighbor_interpolation(
        previous_roi: np.ndarray,
        next_roi: np.ndarray,
        anchor_roi: np.ndarray,
        blend_mask: np.ndarray,
    ) -> np.ndarray:
        """利用前后帧对齐结果生成中间帧候选（仅作用于掩码区）。"""
        anchor_u8 = VideoProcessor._as_uint8_bgr(anchor_roi)
        mask_bool = np.asarray(blend_mask) > 0
        if mask_bool.ndim != 2 or int(np.count_nonzero(mask_bool)) < 8:
            return anchor_u8

        aligned_prev = VideoProcessor._warp_propainter_roi_to_anchor(previous_roi, anchor_u8)
        aligned_next = VideoProcessor._warp_propainter_roi_to_anchor(next_roi, anchor_u8)
        blended = anchor_u8.astype(np.float32)
        blended_neighbors = (
            aligned_prev.astype(np.float32) * 0.5 + aligned_next.astype(np.float32) * 0.5
        )
        blended[mask_bool] = blended_neighbors[mask_bool]
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _repair_propainter_short_reappear_bursts(
        selected_rois: List[np.ndarray],
        selected_qualities: List[Dict[str, Any]],
        stable_candidates: List[np.ndarray],
        raw_candidates: Optional[List[np.ndarray]],
        roi_originals: List[np.ndarray],
        core_mask: np.ndarray,
        transition_mask: np.ndarray,
        forced_reset_indices: Set[int],
        cold_start_window: int = 2,
        max_burst_length: int = 3,
        cut_quarantine_indices: Optional[Set[int]] = None,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, int]]:
        """
        修复 1~3 帧短突发回闪。

        仅处理非硬切且非冷启动窗口内的突发帧。
        """
        total = len(selected_rois)
        if total == 0:
            return selected_rois, selected_qualities, {
                "burst_fix_attempts": 0,
                "burst_fix_accepted_frames": 0,
                "burst_fix_rejected_frames": 0,
            }

        repaired_rois = [VideoProcessor._as_uint8_bgr(roi) for roi in selected_rois]
        repaired_qualities = [dict(q) for q in selected_qualities]
        excluded_indices: Set[int] = set(cut_quarantine_indices or set())
        for cut_idx in sorted(forced_reset_indices):
            if cut_idx < 0 or cut_idx >= total:
                continue
            for offset in range(0, cold_start_window + 1):
                idx = cut_idx + offset
                if idx < total:
                    excluded_indices.add(idx)

        flagged_indices = [
            idx
            for idx, q in enumerate(repaired_qualities)
            if (
                bool(q.get("reappear_flag", False))
                or bool(q.get("under_remove_flag", False))
            )
            and idx not in excluded_indices
        ]
        if not flagged_indices:
            return repaired_rois, repaired_qualities, {
                "burst_fix_attempts": 0,
                "burst_fix_accepted_frames": 0,
                "burst_fix_rejected_frames": 0,
            }

        stats = {
            "burst_fix_attempts": 0,
            "burst_fix_accepted_frames": 0,
            "burst_fix_rejected_frames": 0,
        }
        bursts: List[Tuple[int, int]] = []
        burst_start = flagged_indices[0]
        burst_end = burst_start
        for idx in flagged_indices[1:]:
            if idx == burst_end + 1:
                burst_end = idx
                continue
            bursts.append((burst_start, burst_end))
            burst_start = idx
            burst_end = idx
        bursts.append((burst_start, burst_end))

        core_bool = (np.asarray(core_mask) > 0)
        blend_mask = core_mask if int(np.count_nonzero(core_bool)) >= 8 else transition_mask

        for burst_start, burst_end in bursts:
            burst_len = burst_end - burst_start + 1
            if burst_len > max_burst_length:
                continue
            if burst_start <= 0 or burst_end >= total - 1:
                continue
            if bool(repaired_qualities[burst_start - 1].get("reappear_flag", False)):
                continue
            if bool(repaired_qualities[burst_end + 1].get("reappear_flag", False)):
                continue

            stats["burst_fix_attempts"] += 1
            for idx in range(burst_start, burst_end + 1):
                baseline_quality = repaired_qualities[idx]
                best_roi = repaired_rois[idx]
                best_quality = baseline_quality
                accepted = False

                stable_candidate = VideoProcessor._as_uint8_bgr(stable_candidates[idx])
                stable_quality = VideoProcessor._evaluate_propainter_frame_quality(
                    roi_original=roi_originals[idx],
                    roi_candidate=stable_candidate,
                    core_mask=core_mask,
                    transition_mask=transition_mask,
                    previous_selected_roi=(repaired_rois[idx - 1] if idx > 0 else None),
                    is_hard_cut_frame=False,
                )
                stable_quality = VideoProcessor._apply_propainter_remove_sufficiency(
                    quality=stable_quality,
                    remove_energy_reference=float(baseline_quality.get("remove_energy_reference", 1e-4)),
                    previous_selected_remove_ratio=float(
                        repaired_qualities[idx - 1].get("remove_ratio", 1.0)
                    )
                    if idx > 0
                    else None,
                    is_hard_cut_frame=False,
                )
                if (
                    not stable_quality.get("reappear_flag", False)
                    and not stable_quality.get("seam_extreme_flag", False)
                    and float(stable_quality.get("remove_ratio", 0.0)) >= 0.65
                    and float(stable_quality.get("score", 1e9))
                    <= float(baseline_quality.get("score", 1e9)) * 1.02
                ):
                    best_roi = stable_candidate
                    best_quality = stable_quality
                    accepted = True

                if (
                    (not accepted or best_quality.get("under_remove_flag", False))
                    and raw_candidates is not None
                    and idx < len(raw_candidates)
                ):
                    raw_candidate = VideoProcessor._as_uint8_bgr(raw_candidates[idx])
                    raw_quality = VideoProcessor._evaluate_propainter_frame_quality(
                        roi_original=roi_originals[idx],
                        roi_candidate=raw_candidate,
                        core_mask=core_mask,
                        transition_mask=transition_mask,
                        previous_selected_roi=(repaired_rois[idx - 1] if idx > 0 else None),
                        is_hard_cut_frame=False,
                    )
                    raw_quality = VideoProcessor._apply_propainter_remove_sufficiency(
                        quality=raw_quality,
                        remove_energy_reference=float(baseline_quality.get("remove_energy_reference", 1e-4)),
                        previous_selected_remove_ratio=float(
                            repaired_qualities[idx - 1].get("remove_ratio", 1.0)
                        )
                        if idx > 0
                        else None,
                        is_hard_cut_frame=False,
                    )
                    if (
                        not raw_quality.get("reappear_flag", False)
                        and not raw_quality.get("seam_extreme_flag", False)
                        and float(raw_quality.get("remove_ratio", 0.0)) >= 0.65
                        and float(raw_quality.get("score", 1e9))
                        <= float(baseline_quality.get("score", 1e9)) * 1.02
                    ):
                        best_roi = raw_candidate
                        best_quality = raw_quality
                        accepted = True

                if (not accepted or best_quality.get("reappear_flag", False)) and 0 < idx < total - 1:
                    interp_candidate = VideoProcessor._build_propainter_neighbor_interpolation(
                        previous_roi=repaired_rois[idx - 1],
                        next_roi=repaired_rois[idx + 1],
                        anchor_roi=repaired_rois[idx],
                        blend_mask=blend_mask,
                    )
                    interp_quality = VideoProcessor._evaluate_propainter_frame_quality(
                        roi_original=roi_originals[idx],
                        roi_candidate=interp_candidate,
                        core_mask=core_mask,
                        transition_mask=transition_mask,
                        previous_selected_roi=(repaired_rois[idx - 1] if idx > 0 else None),
                        is_hard_cut_frame=False,
                    )
                    interp_quality = VideoProcessor._apply_propainter_remove_sufficiency(
                        quality=interp_quality,
                        remove_energy_reference=float(baseline_quality.get("remove_energy_reference", 1e-4)),
                        previous_selected_remove_ratio=float(
                            repaired_qualities[idx - 1].get("remove_ratio", 1.0)
                        )
                        if idx > 0
                        else None,
                        is_hard_cut_frame=False,
                    )
                    if (
                        not interp_quality.get("reappear_flag", False)
                        and not interp_quality.get("seam_extreme_flag", False)
                        and float(interp_quality.get("remove_ratio", 0.0)) >= 0.65
                        and float(interp_quality.get("score", 1e9))
                        <= float(baseline_quality.get("score", 1e9)) * 1.02
                    ):
                        best_roi = interp_candidate
                        best_quality = interp_quality
                        accepted = True

                if accepted:
                    repaired_rois[idx] = best_roi
                    repaired_qualities[idx] = best_quality
                    stats["burst_fix_accepted_frames"] += 1
                else:
                    stats["burst_fix_rejected_frames"] += 1

        repaired_qualities = VideoProcessor._reevaluate_propainter_sequence_qualities(
            rois=repaired_rois,
            qualities=repaired_qualities,
            roi_originals=roi_originals,
            core_mask=core_mask,
            transition_mask=transition_mask,
            excluded_indices=excluded_indices,
        )
        return repaired_rois, repaired_qualities, stats

    @staticmethod
    def _repair_propainter_micro_flicker_bursts(
        selected_rois: List[np.ndarray],
        selected_qualities: List[Dict[str, Any]],
        stable_candidates: List[np.ndarray],
        raw_candidates: Optional[List[np.ndarray]],
        roi_originals: List[np.ndarray],
        core_mask: np.ndarray,
        transition_mask: np.ndarray,
        forced_reset_indices: Set[int],
        micro_flicker_flags: List[bool],
        cold_start_window: int = 2,
        max_burst_length: int = 2,
        cut_quarantine_indices: Optional[Set[int]] = None,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, int]]:
        """
        修复 1~2 帧“近失型”微闪烁。

        与 reappear 修复不同，此处只处理低概率短突发，避免扩大时域干预。
        """
        total = len(selected_rois)
        if total == 0:
            return selected_rois, selected_qualities, {
                "micro_burst_fix_attempts": 0,
                "micro_burst_fix_accepted_frames": 0,
                "micro_burst_fix_rejected_frames": 0,
            }

        repaired_rois = [VideoProcessor._as_uint8_bgr(roi) for roi in selected_rois]
        repaired_qualities = [dict(q) for q in selected_qualities]
        excluded_indices: Set[int] = set(cut_quarantine_indices or set())
        for cut_idx in sorted(forced_reset_indices):
            if cut_idx < 0 or cut_idx >= total:
                continue
            for offset in range(0, cold_start_window + 1):
                idx = cut_idx + offset
                if idx < total:
                    excluded_indices.add(idx)

        flags = list(micro_flicker_flags[:total])
        if len(flags) < total:
            flags.extend([False] * (total - len(flags)))
        flagged_indices = [idx for idx in range(total) if bool(flags[idx]) and idx not in excluded_indices]
        if not flagged_indices:
            return repaired_rois, repaired_qualities, {
                "micro_burst_fix_attempts": 0,
                "micro_burst_fix_accepted_frames": 0,
                "micro_burst_fix_rejected_frames": 0,
            }

        stats = {
            "micro_burst_fix_attempts": 0,
            "micro_burst_fix_accepted_frames": 0,
            "micro_burst_fix_rejected_frames": 0,
        }
        bursts: List[Tuple[int, int]] = []
        burst_start = flagged_indices[0]
        burst_end = burst_start
        for idx in flagged_indices[1:]:
            if idx == burst_end + 1:
                burst_end = idx
            else:
                bursts.append((burst_start, burst_end))
                burst_start = idx
                burst_end = idx
        bursts.append((burst_start, burst_end))

        core_bool = (np.asarray(core_mask) > 0)
        blend_mask = core_mask if int(np.count_nonzero(core_bool)) >= 8 else transition_mask

        for burst_start, burst_end in bursts:
            burst_len = burst_end - burst_start + 1
            if burst_len > max_burst_length:
                continue
            stats["micro_burst_fix_attempts"] += 1

            for idx in range(burst_start, burst_end + 1):
                baseline_quality = repaired_qualities[idx]
                baseline_score = max(1e-6, float(baseline_quality.get("score", 1e9)))
                baseline_remove_ratio = float(baseline_quality.get("remove_ratio", 0.0))
                baseline_jump = float(baseline_quality.get("temporal_jump_core", 0.0))
                baseline_hf = float(baseline_quality.get("residual_hf_corr", 1.0))

                candidates: List[np.ndarray] = []
                stable_candidate = VideoProcessor._as_uint8_bgr(stable_candidates[idx])
                candidates.append(stable_candidate)
                if raw_candidates is not None and idx < len(raw_candidates):
                    candidates.append(VideoProcessor._as_uint8_bgr(raw_candidates[idx]))
                if 0 < idx < total - 1:
                    interp_candidate = VideoProcessor._build_propainter_neighbor_interpolation(
                        previous_roi=repaired_rois[idx - 1],
                        next_roi=repaired_rois[idx + 1],
                        anchor_roi=repaired_rois[idx],
                        blend_mask=blend_mask,
                    )
                    candidates.append(interp_candidate)

                best_roi = repaired_rois[idx]
                best_quality = baseline_quality
                accepted = False
                for candidate_roi in candidates:
                    candidate_quality = VideoProcessor._evaluate_propainter_frame_quality(
                        roi_original=roi_originals[idx],
                        roi_candidate=candidate_roi,
                        core_mask=core_mask,
                        transition_mask=transition_mask,
                        previous_selected_roi=(repaired_rois[idx - 1] if idx > 0 else None),
                        is_hard_cut_frame=False,
                    )
                    candidate_quality = VideoProcessor._apply_propainter_remove_sufficiency(
                        quality=candidate_quality,
                        remove_energy_reference=float(
                            baseline_quality.get("remove_energy_reference", 1e-4)
                        ),
                        previous_selected_remove_ratio=float(
                            repaired_qualities[idx - 1].get("remove_ratio", 1.0)
                        )
                        if idx > 0
                        else None,
                        is_hard_cut_frame=False,
                    )

                    candidate_score = float(candidate_quality.get("score", 1e9))
                    candidate_remove_ratio = float(candidate_quality.get("remove_ratio", 0.0))
                    candidate_jump = float(candidate_quality.get("temporal_jump_core", 0.0))
                    candidate_hf = float(candidate_quality.get("residual_hf_corr", 1.0))
                    improves_temporal = (
                        candidate_jump <= baseline_jump * 0.85
                        or candidate_hf <= baseline_hf - 0.05
                    )
                    if (
                        candidate_score <= baseline_score * 1.01
                        and candidate_remove_ratio >= max(0.62, baseline_remove_ratio - 0.03)
                        and improves_temporal
                        and not bool(candidate_quality.get("seam_extreme_flag", False))
                        and not bool(candidate_quality.get("reappear_flag", False))
                        and not bool(candidate_quality.get("under_remove_flag", False))
                    ):
                        if (
                            (not accepted)
                            or candidate_score < float(best_quality.get("score", 1e9))
                        ):
                            best_roi = candidate_roi
                            best_quality = candidate_quality
                            accepted = True

                if accepted:
                    repaired_rois[idx] = best_roi
                    repaired_qualities[idx] = best_quality
                    stats["micro_burst_fix_accepted_frames"] += 1
                else:
                    stats["micro_burst_fix_rejected_frames"] += 1

        repaired_qualities = VideoProcessor._reevaluate_propainter_sequence_qualities(
            rois=repaired_rois,
            qualities=repaired_qualities,
            roi_originals=roi_originals,
            core_mask=core_mask,
            transition_mask=transition_mask,
            excluded_indices=excluded_indices,
        )
        return repaired_rois, repaired_qualities, stats

    @staticmethod
    def _should_accept_propainter_rescue(
        selected_quality: Dict[str, Any],
        rescue_quality: Dict[str, Any],
    ) -> bool:
        """判断 ProPainter rescue 结果是否可用。"""
        return (
            float(rescue_quality.get("score", 0.0))
            < float(selected_quality.get("score", 0.0)) * 0.95
            and not bool(rescue_quality.get("dark_block_flag", False))
        )

    @staticmethod
    def _apply_propainter_ring_clone(
        roi_original: np.ndarray,
        roi_candidate: np.ndarray,
        transition_outer_mask: np.ndarray,
    ) -> Tuple[np.ndarray, bool, bool]:
        """
        对过渡外环做常态梯度域缝合，降低方形边界感。

        返回 `(output_roi, used_clone, fallback)`。
        """
        src = VideoProcessor._as_uint8_bgr(roi_candidate)
        dst = VideoProcessor._as_uint8_bgr(roi_original)
        outer_mask = ((np.asarray(transition_outer_mask) > 0).astype(np.uint8) * 255)
        if int(np.count_nonzero(outer_mask)) < 24:
            return src, False, False

        x, y, w, h = cv2.boundingRect(outer_mask)
        center = (int(x + w / 2), int(y + h / 2))
        flag_names = ("MIXED_CLONE_WIDE", "MIXED_CLONE", "NORMAL_CLONE_WIDE", "NORMAL_CLONE")
        ring_bool = outer_mask > 0
        last_error: Optional[Exception] = None
        for flag_name in flag_names:
            if not hasattr(cv2, flag_name):
                continue
            try:
                ring_cloned = cv2.seamlessClone(src, dst, outer_mask, center, int(getattr(cv2, flag_name)))
                output = src.copy()
                output[ring_bool] = ring_cloned[ring_bool]
                return output, True, False
            except Exception as exc:  # pragma: no cover - 运行时兜底
                last_error = exc
                continue
        if last_error is not None:
            logger.debug("ProPainter ring clone fallback: %s", last_error)
        return src, False, True

    @staticmethod
    def _rescue_propainter_with_seamless_clone(
        roi_original: np.ndarray,
        roi_candidate: np.ndarray,
        rounded_mask: np.ndarray,
    ) -> np.ndarray:
        """
        ProPainter 应急梯度域融合。

        优先使用 MIXED_CLONE_WIDE（如果 OpenCV 版本支持），否则降级到 MIXED_CLONE。
        """
        src = VideoProcessor._as_uint8_bgr(roi_candidate)
        dst = VideoProcessor._as_uint8_bgr(roi_original)
        mask = ((np.asarray(rounded_mask) > 0).astype(np.uint8) * 255)
        if int(np.count_nonzero(mask)) < 16:
            return src
        x, y, w, h = cv2.boundingRect(mask)
        center = (int(x + w / 2), int(y + h / 2))

        flag_names = ("MIXED_CLONE_WIDE", "MIXED_CLONE", "NORMAL_CLONE_WIDE", "NORMAL_CLONE")
        last_error: Optional[Exception] = None
        for flag_name in flag_names:
            if not hasattr(cv2, flag_name):
                continue
            try:
                return cv2.seamlessClone(src, dst, mask, center, int(getattr(cv2, flag_name)))
            except Exception as exc:  # pragma: no cover - 运行时兜底
                last_error = exc
                continue
        if last_error:
            raise last_error
        return src

    @staticmethod
    def _stabilize_lama_sequence(
        roi_sequence: List[np.ndarray],
        roi_mask: np.ndarray,
        temporal_strength: float,
        forced_reset_indices: Optional[Set[int]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """
        对 ROI 序列做时序稳定：
        - 检测硬切并重置；
        - 相邻帧对齐后做自适应混合；
        - 输出稳定序列与诊断统计。
        """
        if not roi_sequence:
            return [], {
                "hard_cut_resets_total": 0,
                "hard_cut_resets_forced": 0,
                "hard_cut_resets_roi": 0,
                "cold_start_frames": 0,
            }

        mask_bool = np.asarray(roi_mask) > 0
        if mask_bool.ndim != 2 or int(np.count_nonzero(mask_bool)) == 0:
            return roi_sequence, {
                "hard_cut_resets_total": 0,
                "hard_cut_resets_forced": 0,
                "hard_cut_resets_roi": 0,
                "cold_start_frames": 0,
            }

        stable_sequence: List[np.ndarray] = []
        prev_stable: Optional[np.ndarray] = None
        prev_gray: Optional[np.ndarray] = None
        diagnostics = {
            "hard_cut_resets_total": 0,
            "hard_cut_resets_forced": 0,
            "hard_cut_resets_roi": 0,
            "cold_start_frames": 0,
        }
        forced_set: Set[int] = set(forced_reset_indices or set())
        cold_start_remaining = 0
        cold_start_window = 3

        for idx, current in enumerate(roi_sequence):
            current_u8 = VideoProcessor._as_uint8_bgr(current)
            if prev_stable is None or prev_gray is None:
                stable_frame = current_u8.copy()
                stable_sequence.append(stable_frame)
                prev_stable = stable_frame
                prev_gray = cv2.cvtColor(prev_stable, cv2.COLOR_BGR2GRAY).astype(np.float32)
                continue

            current_gray = cv2.cvtColor(current_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if idx in forced_set:
                diagnostics["hard_cut_resets_total"] += 1
                diagnostics["hard_cut_resets_forced"] += 1
                cold_start_remaining = max(cold_start_remaining, cold_start_window)
                stable_frame = current_u8.copy()
                stable_sequence.append(stable_frame)
                prev_stable = stable_frame
                prev_gray = current_gray
                continue

            shift_xy, response = cv2.phaseCorrelate(prev_gray, current_gray)
            shift_x = float(shift_xy[0]) if np.isfinite(shift_xy[0]) else 0.0
            shift_y = float(shift_xy[1]) if np.isfinite(shift_xy[1]) else 0.0
            shift_x = float(np.clip(shift_x, -6.0, 6.0))
            shift_y = float(np.clip(shift_y, -6.0, 6.0))

            prev_pixels = prev_gray[mask_bool]
            current_pixels = current_gray[mask_bool]
            abs_diff_level = float(np.mean(np.abs(current_pixels - prev_pixels)) / 255.0)
            gray_mean_delta = float(np.abs(current_pixels.mean() - prev_pixels.mean()) / 255.0)
            prev_lap = cv2.Laplacian(prev_gray, cv2.CV_32F, ksize=3)
            current_lap = cv2.Laplacian(current_gray, cv2.CV_32F, ksize=3)
            texture_delta = float(
                np.mean(np.abs(current_lap[mask_bool] - prev_lap[mask_bool])) / 64.0
            )
            texture_delta = float(np.clip(texture_delta, 0.0, 1.0))
            response_score = float(response) if np.isfinite(response) else 0.0

            hard_cut_votes = 0
            hard_cut_votes += 1 if response_score < 0.07 else 0
            hard_cut_votes += 1 if abs_diff_level > 0.20 else 0
            hard_cut_votes += 1 if gray_mean_delta > 0.14 else 0
            hard_cut_votes += 1 if texture_delta > 0.28 else 0
            hard_cut = hard_cut_votes >= 2 and (abs_diff_level > 0.16 or texture_delta > 0.22)
            if hard_cut:
                diagnostics["hard_cut_resets_total"] += 1
                diagnostics["hard_cut_resets_roi"] += 1
                cold_start_remaining = max(cold_start_remaining, cold_start_window)
                stable_frame = current_u8.copy()
                stable_sequence.append(stable_frame)
                prev_stable = stable_frame
                prev_gray = current_gray
                continue

            warp = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
            aligned_prev = cv2.warpAffine(
                prev_stable,
                warp,
                (current_u8.shape[1], current_u8.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            aligned_prev_gray = cv2.cvtColor(aligned_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
            motion_map = np.abs(current_gray - aligned_prev_gray)
            motion_level = float(motion_map[mask_bool].mean() / 255.0)
            prev_weight = (0.68 - 1.00 * motion_level) * float(temporal_strength)
            prev_weight = float(np.clip(prev_weight, 0.02, 0.62))
            if cold_start_remaining > 0:
                cold_start_remaining -= 1
                diagnostics["cold_start_frames"] += 1
                prev_weight = 0.0

            current_f = current_u8.astype(np.float32)
            aligned_prev_f = aligned_prev.astype(np.float32)
            smoothed_f = current_f * (1.0 - prev_weight) + aligned_prev_f * prev_weight
            stable_frame = current_u8.copy()
            stable_frame[mask_bool] = np.clip(smoothed_f[mask_bool], 0.0, 255.0).astype(np.uint8)

            stable_sequence.append(stable_frame)
            prev_stable = stable_frame
            prev_gray = cv2.cvtColor(prev_stable, cv2.COLOR_BGR2GRAY).astype(np.float32)

        return stable_sequence, diagnostics

    def get_video_info(self, video_path: str) -> VideoInfo:
        """读取视频基础信息（分辨率/FPS/帧数/音频存在性等）。"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video_path}')

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = ''.join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

        cap.release()

        duration = frame_count / fps if fps > 0 else 0
        has_audio = self._check_audio(video_path)

        return VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            has_audio=has_audio,
            codec=codec_str,
        )

    def _check_audio(self, video_path: str) -> bool:
        """
        检测视频是否包含音频流。

        优先顺序：
        1) ffprobe 精确检查；
        2) ffmpeg 输出文本兜底判断。
        """
        ffprobe_bin = resolve_ffprobe_path()
        if ffprobe_bin:
            try:
                result = subprocess.run(
                    [
                        ffprobe_bin,
                        '-v',
                        'error',
                        '-select_streams',
                        'a',
                        '-show_entries',
                        'stream=codec_type',
                        '-of',
                        'default=noprint_wrappers=1:nokey=1',
                        video_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return bool(result.stdout.strip())
            except Exception as exc:
                logger.warning(f'ffprobe audio check failed ({video_path}): {exc}')

        ffmpeg_bin = resolve_ffmpeg_path()
        if ffmpeg_bin:
            try:
                result = subprocess.run(
                    [ffmpeg_bin, '-hide_banner', '-i', video_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                stream_output = (result.stdout or '') + '\n' + (result.stderr or '')
                return 'audio:' in stream_output.lower()
            except Exception as exc:
                logger.warning(f'ffmpeg fallback audio check failed ({video_path}): {exc}')

        logger.warning('FFprobe/FFmpeg not available, unable to detect audio stream')
        return False

    def extract_frames(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> Generator[np.ndarray, None, None]:
        """按顺序迭代视频帧，可选同时落盘到目录。"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video_path}')

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if output_dir:
                frame_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.png')
                cv2.imwrite(frame_path, frame)

            yield frame
            frame_idx += 1

        cap.release()

    def process_video(
        self,
        video_path: str,
        output_path: str,
        annotation_segments: Optional[List[Dict[str, Any]]],
        model_id: str = 'lama_roi',
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        手工标注视频处理主入口。

        输入：
        - annotation_segments: 标记段列表（必填且至少有 1 条启用段）
        - model_id: 期望模型（可能被回退）
        输出：
        - 输出文件路径、模型回退信息、命中/跳过帧统计。
        """
        if self._is_processing:
            raise RuntimeError('Already processing a video')

        if not annotation_segments:
            raise ValueError('No enabled annotation segments provided')

        self._is_processing = True
        self._should_stop = False

        try:
            video_info = self.get_video_info(video_path)

            if status_callback:
                status_callback('Loading models...')

            if self.remover and not self.remover.is_loaded():
                self.remover.load_model()

            model_registry = self._get_model_registry()
            active_engine, resolved_model = model_registry.resolve(model_id)
            requested_model_id = resolved_model.requested_model_id
            effective_model_id = resolved_model.effective_model_id
            model_warning = resolved_model.warning

            temp_dir = tempfile.mkdtemp()

            if status_callback:
                status_callback('Extracting frames...')

            cap = cv2.VideoCapture(video_path)
            frame_count = video_info.frame_count
            fps = video_info.fps
            width = video_info.width
            height = video_info.height

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_output = os.path.join(temp_dir, 'output_no_audio.mp4')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            frame_idx = 0
            start_ts = time.time()
            safe_total = max(1, frame_count)
            hit_frames = 0
            skipped_frames = 0
            processed_frames = 0
            lama_enhancement_fallbacks = 0
            lama_pass2_fallbacks = 0
            lama_hard_cut_resets_total = 0
            lama_hard_cut_resets_forced = 0
            lama_hard_cut_resets_roi = 0
            lama_cold_start_frames = 0
            lama_blend_v2_fallbacks = 0
            lama_blend_v2_used_frames = 0
            lama_transition_band_width_sum = 0.0
            lama_transition_band_width_count = 0
            lama_seam_delta_before_sum = 0.0
            lama_seam_delta_after_sum = 0.0
            lama_seam_delta_count = 0
            lama_frame_guard_total = 0
            lama_frame_guard_choose_stage2_v2 = 0
            lama_frame_guard_choose_stage1_v2 = 0
            lama_frame_guard_choose_legacy = 0
            lama_frame_guard_dark_block_rejects = 0
            lama_frame_guard_seam_rejects = 0
            lama_rescue_attempts = 0
            lama_rescue_accepted = 0
            lama_rescue_rejected = 0
            lama_frame_guard_switch_total = 0
            lama_frame_guard_switch_suppressed = 0
            lama_final_micro_smooth_applied = 0
            propainter_v2_used_frames = 0
            propainter_v2_fallbacks = 0
            propainter_hard_cut_splits = 0
            propainter_stabilize_applied_frames = 0
            propainter_frame_guard_total = 0
            propainter_choose_raw_v2 = 0
            propainter_choose_stable_v2 = 0
            propainter_choose_legacy = 0
            propainter_frame_guard_dark_block_rejects = 0
            propainter_frame_guard_seam_rejects = 0
            propainter_rescue_attempts = 0
            propainter_rescue_accepted = 0
            propainter_rescue_rejected = 0
            propainter_seam_delta_before_sum = 0.0
            propainter_seam_delta_after_sum = 0.0
            propainter_seam_delta_count = 0
            propainter_reappear_flags_total = 0
            propainter_hysteresis_hold_count = 0
            propainter_hysteresis_switch_count = 0
            propainter_burst_fix_attempts = 0
            propainter_burst_fix_accepted_frames = 0
            propainter_burst_fix_rejected_frames = 0
            propainter_legacy_blocked_by_guard = 0
            propainter_reappear_score_before_sum = 0.0
            propainter_reappear_score_after_sum = 0.0
            propainter_reappear_score_count = 0
            propainter_under_remove_flags_total = 0
            propainter_remove_ratio_before_sum = 0.0
            propainter_remove_ratio_after_sum = 0.0
            propainter_segment_rerun_attempts = 0
            propainter_segment_rerun_accepted = 0
            propainter_segment_rerun_rejected = 0
            propainter_legacy_catastrophic_only_count = 0
            propainter_cut_quarantine_frames = 0
            propainter_viterbi_switches = 0
            propainter_ring_clone_used_frames = 0
            propainter_ring_clone_fallbacks = 0
            propainter_residual_hf_corr_before_sum = 0.0
            propainter_residual_hf_corr_after_sum = 0.0
            propainter_seam_p90_before_sum = 0.0
            propainter_seam_p90_after_sum = 0.0
            propainter_seam_p90_count = 0
            propainter_micro_flicker_flags_total = 0
            propainter_selection_island_rewrites = 0
            propainter_micro_burst_fix_attempts = 0
            propainter_micro_burst_fix_accepted_frames = 0
            propainter_micro_burst_fix_rejected_frames = 0
            propainter_temporal_jump_before_microfix_sum = 0.0
            propainter_temporal_jump_after_microfix_sum = 0.0
            propainter_remove_ratio_before_microfix_sum = 0.0
            propainter_remove_ratio_after_microfix_sum = 0.0
            propainter_microfix_metric_count = 0

            normalized_segments: List[Dict[str, Any]] = []
            max_frame = max(0, frame_count - 1)
            # 先把输入标记段做一轮过滤和规范化，避免运行期重复判断。
            for seg in annotation_segments:
                if not isinstance(seg, dict):
                    continue
                if not bool(seg.get('enabled', True)):
                    continue

                rect = seg.get('rect', {})
                if not isinstance(rect, dict):
                    continue

                start_frame = int(seg.get('start_frame', 0))
                end_frame = int(seg.get('end_frame', start_frame))
                start_frame = max(0, min(start_frame, max_frame))
                end_frame = max(0, min(end_frame, max_frame))
                if end_frame < start_frame:
                    start_frame, end_frame = end_frame, start_frame

                expand_px = max(0, int(seg.get('expand_px', 5)))
                feather_px = max(0, int(seg.get('feather_px', 3)))
                if self._resolve_segment_geometry(
                    rect=rect,
                    expand_px=expand_px,
                    feather_px=feather_px,
                    frame_width=width,
                    frame_height=height,
                    use_lama_adaptive=False,
                    use_propainter_adaptive=(effective_model_id == 'propainter_roi'),
                ) is None:
                    continue

                seg_order = len(normalized_segments)
                seg_id = str(seg.get('id', ''))
                normalized_segments.append(
                    {
                        'id': seg_id,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'rect': {
                            'x': int(rect.get('x', 0)),
                            'y': int(rect.get('y', 0)),
                            'width': int(rect.get('width', 0)),
                            'height': int(rect.get('height', 0)),
                        },
                        'expand_px': expand_px,
                        'feather_px': feather_px,
                        '_order': seg_order,
                        '_area': int(
                            max(1, int(rect.get('width', 0))) * max(1, int(rect.get('height', 0)))
                        ),
                        '_state_key': f'{seg_id}@{seg_order}:{start_frame}-{end_frame}',
                    }
                )

            if not normalized_segments:
                raise ValueError('No enabled annotation segments provided')

            if status_callback:
                status_callback('Processing frames...')

            propainter_segment_runtime: Dict[str, Dict[str, Any]] = {}

            def _emit_runtime_progress() -> None:
                # 运行时进度：按已写入帧估算 ETA。
                elapsed = max(0.0, time.time() - start_ts)
                fps_process = processed_frames / max(1e-6, elapsed)
                eta_seconds = (frame_count - processed_frames) / max(1e-6, fps_process)
                progress_message = (
                    f'Manual annotation processing {processed_frames}/{frame_count} '
                    f'(hit={hit_frames}, skip={skipped_frames})'
                )
                self._emit_progress(
                    progress_callback=progress_callback,
                    progress=processed_frames / safe_total,
                    message=progress_message,
                    processed_frames=processed_frames,
                    total_frames=frame_count,
                    estimated_time=self._format_eta(eta_seconds),
                )

            def _write_result_frame(frame_to_write: np.ndarray) -> None:
                # 统一写帧并按节流规则推送进度。
                nonlocal processed_frames
                out.write(frame_to_write)
                processed_frames += 1
                if progress_callback and (
                    processed_frames % 10 == 0 or processed_frames >= frame_count
                ):
                    _emit_runtime_progress()

            def _apply_single_segment_on_frames(
                segment: Dict[str, Any],
                segment_frames: List[np.ndarray],
            ) -> List[np.ndarray]:
                """
                对一个连续帧块应用单个标记段，返回变更后的帧序列。

                这是单段处理核心：
                - 计算 ROI 与掩码；
                - 跑模型推理；
                - 做融合与质量守卫；
                - 写回帧并返回（不直接输出）。
                """
                nonlocal active_engine, effective_model_id, model_warning, hit_frames
                nonlocal lama_enhancement_fallbacks, lama_pass2_fallbacks
                nonlocal lama_hard_cut_resets_total, lama_hard_cut_resets_forced
                nonlocal lama_hard_cut_resets_roi, lama_cold_start_frames
                nonlocal lama_blend_v2_fallbacks, lama_blend_v2_used_frames
                nonlocal lama_transition_band_width_sum, lama_transition_band_width_count
                nonlocal lama_seam_delta_before_sum, lama_seam_delta_after_sum
                nonlocal lama_seam_delta_count
                nonlocal lama_frame_guard_total, lama_frame_guard_choose_stage2_v2
                nonlocal lama_frame_guard_choose_stage1_v2, lama_frame_guard_choose_legacy
                nonlocal lama_frame_guard_dark_block_rejects, lama_frame_guard_seam_rejects
                nonlocal lama_rescue_attempts, lama_rescue_accepted, lama_rescue_rejected
                nonlocal lama_frame_guard_switch_total, lama_frame_guard_switch_suppressed
                nonlocal lama_final_micro_smooth_applied
                nonlocal propainter_v2_used_frames, propainter_v2_fallbacks
                nonlocal propainter_hard_cut_splits, propainter_stabilize_applied_frames
                nonlocal propainter_frame_guard_total, propainter_choose_raw_v2
                nonlocal propainter_choose_stable_v2, propainter_choose_legacy
                nonlocal propainter_frame_guard_dark_block_rejects
                nonlocal propainter_frame_guard_seam_rejects
                nonlocal propainter_rescue_attempts, propainter_rescue_accepted
                nonlocal propainter_rescue_rejected
                nonlocal propainter_seam_delta_before_sum, propainter_seam_delta_after_sum
                nonlocal propainter_seam_delta_count
                nonlocal propainter_reappear_flags_total
                nonlocal propainter_hysteresis_hold_count, propainter_hysteresis_switch_count
                nonlocal propainter_burst_fix_attempts, propainter_burst_fix_accepted_frames
                nonlocal propainter_burst_fix_rejected_frames, propainter_legacy_blocked_by_guard
                nonlocal propainter_reappear_score_before_sum, propainter_reappear_score_after_sum
                nonlocal propainter_reappear_score_count
                nonlocal propainter_under_remove_flags_total
                nonlocal propainter_remove_ratio_before_sum, propainter_remove_ratio_after_sum
                nonlocal propainter_segment_rerun_attempts, propainter_segment_rerun_accepted
                nonlocal propainter_segment_rerun_rejected
                nonlocal propainter_legacy_catastrophic_only_count
                nonlocal propainter_cut_quarantine_frames, propainter_viterbi_switches
                nonlocal propainter_selection_island_rewrites
                nonlocal propainter_ring_clone_used_frames, propainter_ring_clone_fallbacks
                nonlocal propainter_residual_hf_corr_before_sum, propainter_residual_hf_corr_after_sum
                nonlocal propainter_seam_p90_before_sum, propainter_seam_p90_after_sum
                nonlocal propainter_seam_p90_count
                nonlocal propainter_micro_flicker_flags_total
                nonlocal propainter_micro_burst_fix_attempts
                nonlocal propainter_micro_burst_fix_accepted_frames
                nonlocal propainter_micro_burst_fix_rejected_frames
                nonlocal propainter_temporal_jump_before_microfix_sum
                nonlocal propainter_temporal_jump_after_microfix_sum
                nonlocal propainter_remove_ratio_before_microfix_sum
                nonlocal propainter_remove_ratio_after_microfix_sum
                nonlocal propainter_microfix_metric_count
                if not segment_frames:
                    return []
                if segment is None:
                    return [frame_item.copy() for frame_item in segment_frames]

                if self.remover is None:
                    return [frame_item.copy() for frame_item in segment_frames]

                geometry = self._resolve_segment_geometry(
                    rect=segment.get('rect', {}),
                    expand_px=int(segment.get('expand_px', 5)),
                    feather_px=int(segment.get('feather_px', 3)),
                    frame_width=width,
                    frame_height=height,
                    use_lama_adaptive=(effective_model_id == 'lama_roi'),
                    use_propainter_adaptive=(effective_model_id == 'propainter_roi'),
                )
                if geometry is None:
                    return [frame_item.copy() for frame_item in segment_frames]

                roi_bounds = geometry['roi_bounds']
                mask_box = geometry['mask_box']
                x1, y1, x2, y2 = [int(v) for v in roi_bounds]
                mx1, my1, mx2, my2 = [int(v) for v in mask_box]
                if x2 <= x1 or y2 <= y1:
                    return [frame_item.copy() for frame_item in segment_frames]

                roi_mask_template = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                roi_mask_template[my1:my2, mx1:mx2] = 255
                roi_frames = [frame_item[y1:y2, x1:x2] for frame_item in segment_frames]
                rect = segment.get('rect', {}) if isinstance(segment.get('rect'), dict) else {}
                rect_w = int(rect.get('width', max(1, mx2 - mx1)))
                rect_h = int(rect.get('height', max(1, my2 - my1)))
                segment_base_processed = processed_frames
                segment_total_frames = max(1, len(segment_frames))
                feather_px = int(geometry.get('feather_px', segment.get('feather_px', 3)))
                lama_inpaint_mask_stage1_template = self._build_lama_inpaint_mask(
                    core_mask=roi_mask_template,
                    feather_px=feather_px,
                    rect_w=rect_w,
                    rect_h=rect_h,
                )
                lama_inpaint_mask_stage2_template = self._build_lama_inpaint_mask_stage2(
                    stage1_mask=lama_inpaint_mask_stage1_template,
                    feather_px=feather_px,
                    rect_w=rect_w,
                    rect_h=rect_h,
                )
                hard_mask = (roi_mask_template > 0)[:, :, np.newaxis]
                blend_mask = None
                lama_final_rois: Optional[List[np.ndarray]] = None
                propainter_final_rois: Optional[List[np.ndarray]] = None
                propainter_forced_reset_indices: Set[int] = set()

                def _engine_progress_callback(progress_meta: Any, *legacy_args: Any) -> None:
                    # 把模型内部阶段进度映射成全局进度事件，回传给 UI。
                    if not progress_callback:
                        return

                    payload: Dict[str, Any]
                    if isinstance(progress_meta, dict):
                        payload = dict(progress_meta)
                    elif legacy_args:
                        payload = {'step': progress_meta, 'total': legacy_args[0]}
                    elif isinstance(progress_meta, (tuple, list)) and len(progress_meta) >= 2:
                        payload = {'step': progress_meta[0], 'total': progress_meta[1]}
                    elif isinstance(progress_meta, (int, float)):
                        payload = {'progress': progress_meta}
                    else:
                        payload = {'message': str(progress_meta or '').strip()}

                    phase = str(payload.get('phase') or 'infer').strip().lower() or 'infer'
                    step_raw = payload.get('step')
                    total_raw = payload.get('total')
                    ratio = None
                    step_value = None
                    total_value = None

                    try:
                        if total_raw is not None:
                            total_value = max(1, int(total_raw))
                        if step_raw is not None and total_value is not None:
                            step_value = min(max(0, int(step_raw)), total_value)
                            ratio = float(step_value) / float(total_value)
                    except (TypeError, ValueError):
                        ratio = None
                        step_value = None
                        total_value = None

                    if ratio is None:
                        try:
                            ratio = float(payload.get('progress'))
                        except (TypeError, ValueError):
                            ratio = None

                    estimated_processed = segment_base_processed
                    if ratio is not None:
                        ratio = min(max(ratio, 0.0), 1.0)
                        estimated_processed = min(
                            frame_count,
                            int(round(segment_base_processed + ratio * segment_total_frames)),
                        )
                    stage_message = str(
                        payload.get('message')
                        or f'{effective_model_id} {phase} {step_value if step_value is not None else ""}/{total_value if total_value is not None else ""}'
                    ).strip()
                    opaque_infer = bool(payload.get('opaque_infer', phase == 'infer'))

                    extra_payload: Dict[str, Any] = {
                        'phase': phase,
                        'opaque_infer': opaque_infer,
                    }
                    if step_value is not None:
                        extra_payload['step'] = step_value
                    if total_value is not None:
                        extra_payload['total'] = total_value
                    if ratio is not None:
                        extra_payload['progress'] = ratio

                    self._emit_progress(
                        progress_callback=progress_callback,
                        progress=estimated_processed / safe_total,
                        message=stage_message,
                        processed_frames=estimated_processed,
                        total_frames=frame_count,
                        extra=extra_payload,
                    )

                def _build_gaussian_blend_mask() -> Optional[np.ndarray]:
                    # 非 LaMa v2 路径使用的简单高斯融合掩码。
                    if feather_px <= 0:
                        return None
                    kernel_size = max(3, feather_px * 2 + 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    gaussian_mask = cv2.GaussianBlur(
                        roi_mask_template,
                        (kernel_size, kernel_size),
                        0,
                    ).astype(np.float32) / 255.0
                    return gaussian_mask[:, :, np.newaxis]

                def _prepare_rois(inpainted_list: List[np.ndarray]) -> List[np.ndarray]:
                    # 校验输出帧数与尺寸，并统一到 ROI 尺寸。
                    if len(inpainted_list) != len(segment_frames):
                        raise RuntimeError(
                            f'Engine {effective_model_id} returned {len(inpainted_list)} frames, '
                            f'expected {len(segment_frames)}'
                        )
                    prepared: List[np.ndarray] = []
                    for idx_inner in range(len(segment_frames)):
                        roi_original = roi_frames[idx_inner]
                        prepared_roi = self._as_uint8_bgr(inpainted_list[idx_inner])
                        if prepared_roi.shape[:2] != roi_original.shape[:2]:
                            prepared_roi = cv2.resize(
                                prepared_roi,
                                (roi_original.shape[1], roi_original.shape[0]),
                                interpolation=cv2.INTER_CUBIC,
                            )
                        prepared.append(prepared_roi)
                    return prepared

                def _run_lama_two_stage(
                    source_rois: List[np.ndarray],
                ) -> Tuple[List[np.ndarray], List[np.ndarray], bool]:
                    # LaMa 双阶段策略：先 pass1，再 pass2，并加质量门控。
                    nonlocal lama_pass2_fallbacks
                    stage1_masks = [lama_inpaint_mask_stage1_template.copy() for _ in source_rois]
                    stage1_output = active_engine.inpaint_roi_sequence(
                        source_rois,
                        stage1_masks,
                        progress_callback=_engine_progress_callback,
                    )
                    stage1_prepared = _prepare_rois(stage1_output)
                    stage2_masks = [lama_inpaint_mask_stage2_template.copy() for _ in source_rois]
                    try:
                        stage2_output = active_engine.inpaint_roi_sequence(
                            stage1_prepared,
                            stage2_masks,
                            progress_callback=_engine_progress_callback,
                        )
                        stage2_prepared = _prepare_rois(stage2_output)
                        pass2_ok, rejected_frames = self._is_lama_pass2_acceptable(
                            stage1_rois=stage1_prepared,
                            stage2_rois=stage2_prepared,
                            core_mask=roi_mask_template,
                        )
                        if not pass2_ok:
                            lama_pass2_fallbacks += 1
                            logger.warning(
                                (
                                    'LaMa pass2 rejected by quality gate '
                                    '(rejected_frames=%d/%d), fallback to pass1 output'
                                ),
                                rejected_frames,
                                len(stage1_prepared),
                            )
                            return stage1_prepared, stage2_prepared, False
                        return stage1_prepared, stage2_prepared, True
                    except Exception as stage2_exc:
                        lama_pass2_fallbacks += 1
                        logger.warning(
                            'LaMa pass2 failed, fallback to pass1 output: %s',
                            stage2_exc,
                        )
                        return stage1_prepared, stage1_prepared, False

                lama_stage1_rois: Optional[List[np.ndarray]] = None
                lama_stage2_rois: Optional[List[np.ndarray]] = None
                lama_pass2_segment_ok = False

                try:
                    if effective_model_id == 'lama_roi':
                        (
                            lama_stage1_rois,
                            lama_stage2_rois,
                            lama_pass2_segment_ok,
                        ) = _run_lama_two_stage(roi_frames)
                        prepared_rois = (
                            lama_stage2_rois if lama_pass2_segment_ok else lama_stage1_rois
                        )
                    elif effective_model_id == 'propainter_roi':
                        propainter_options = self._compute_propainter_infer_options(
                            rect_w=rect_w,
                            rect_h=rect_h,
                            feather_px=feather_px,
                            segment_total_frames=segment_total_frames,
                            fps=fps,
                        )
                        propainter_forced_reset_indices = self._detect_hard_cuts_in_run(
                            segment_frames,
                            roi_bounds=(x1, y1, x2, y2),
                        )
                        split_ranges = self._split_sequence_by_cut_indices(
                            total_length=len(segment_frames),
                            cut_indices=propainter_forced_reset_indices,
                        )
                        if not split_ranges:
                            split_ranges = [(0, len(segment_frames))]
                        propainter_hard_cut_splits += max(0, len(split_ranges) - 1)

                        inpainted_rois: List[np.ndarray] = []
                        for split_start, split_end in split_ranges:
                            sub_frames = roi_frames[split_start:split_end]
                            sub_masks = [roi_mask_template.copy() for _ in sub_frames]
                            sub_rois = active_engine.inpaint_roi_sequence(
                                sub_frames,
                                sub_masks,
                                progress_callback=_engine_progress_callback,
                                propainter_options=propainter_options,
                            )
                            inpainted_rois.extend(sub_rois)
                        prepared_rois = _prepare_rois(inpainted_rois)
                    else:
                        roi_masks = [roi_mask_template.copy() for _ in segment_frames]
                        inpainted_rois = active_engine.inpaint_roi_sequence(
                            roi_frames,
                            roi_masks,
                            progress_callback=_engine_progress_callback,
                        )
                        prepared_rois = _prepare_rois(inpainted_rois)
                except Exception as engine_exc:
                    # 时序模型失败时自动回退 LaMa，避免整任务失败。
                    if effective_model_id != 'lama_roi':
                        logger.warning(
                            'Model %s failed during sequence inference (%s), fallback to lama_roi',
                            effective_model_id,
                            engine_exc,
                        )
                        fallback_engine, fallback_resolved = model_registry.resolve('lama_roi')
                        active_engine = fallback_engine
                        effective_model_id = fallback_resolved.effective_model_id
                        fallback_warning = (
                            f'Model {requested_model_id} failed during inference and fallback to '
                            f'{effective_model_id}: {engine_exc}'
                        )
                        model_warning = (
                            f'{model_warning} {fallback_warning}'.strip()
                            if model_warning
                            else fallback_warning
                        )
                        (
                            lama_stage1_rois,
                            lama_stage2_rois,
                            lama_pass2_segment_ok,
                        ) = _run_lama_two_stage(roi_frames)
                        prepared_rois = (
                            lama_stage2_rois if lama_pass2_segment_ok else lama_stage1_rois
                        )
                    else:
                        raise

                if effective_model_id == 'lama_roi':
                    try:
                        # LaMa 增强路径：多候选融合 + 质量评估 + 时序平滑 + rescue。
                        if lama_stage1_rois is None or lama_stage2_rois is None:
                            raise RuntimeError('LaMa stage candidates are unavailable')
                        transition_masks = self._build_propainter_transition_masks_v3(
                            core_mask=roi_mask_template,
                            inpaint_mask=lama_inpaint_mask_stage1_template,
                            feather_px=feather_px,
                            rect_w=rect_w,
                            rect_h=rect_h,
                        )
                        core_replace_mask = transition_masks['core_replace_mask']
                        transition_mask = transition_masks['transition_mask']
                        context_mask = transition_masks['context_mask']
                        rounded_mask = transition_masks['rounded_mask']
                        rounded_area = int(np.count_nonzero(rounded_mask))
                        transition_band_width = float(transition_masks.get('transition_band_width', 0.0))
                        if transition_band_width > 0:
                            lama_transition_band_width_sum += transition_band_width
                            lama_transition_band_width_count += 1

                        stage2_corrected_rois = [
                            self._apply_boundary_color_correction_weighted(
                                roi_original=roi_frames[idx],
                                inpainted_roi=lama_stage2_rois[idx],
                                core_replace_mask=core_replace_mask,
                                transition_mask=transition_mask,
                                context_mask=context_mask,
                                feather_px=feather_px,
                            )
                            for idx in range(len(lama_stage2_rois))
                        ]
                        stage1_corrected_rois = [
                            self._apply_boundary_color_correction_weighted(
                                roi_original=roi_frames[idx],
                                inpainted_roi=lama_stage1_rois[idx],
                                core_replace_mask=core_replace_mask,
                                transition_mask=transition_mask,
                                context_mask=context_mask,
                                feather_px=feather_px,
                            )
                            for idx in range(len(lama_stage1_rois))
                        ]
                        temporal_strength = self._compute_lama_temporal_strength(
                            segment_total_frames=segment_total_frames,
                            frame_width=width,
                            frame_height=height,
                        )
                        forced_reset_indices = self._detect_hard_cuts_in_run(segment_frames)
                        stage2_stable_rois, stage2_stabilization_diag = self._stabilize_lama_sequence(
                            roi_sequence=stage2_corrected_rois,
                            roi_mask=lama_inpaint_mask_stage1_template,
                            temporal_strength=temporal_strength,
                            forced_reset_indices=forced_reset_indices,
                        )
                        stage1_stable_rois, stage1_stabilization_diag = self._stabilize_lama_sequence(
                            roi_sequence=stage1_corrected_rois,
                            roi_mask=lama_inpaint_mask_stage1_template,
                            temporal_strength=temporal_strength,
                            forced_reset_indices=forced_reset_indices,
                        )
                        stabilization_diag = (
                            stage2_stabilization_diag if lama_pass2_segment_ok else stage1_stabilization_diag
                        )
                        lama_hard_cut_resets_total += int(
                            stabilization_diag.get('hard_cut_resets_total', 0)
                        )
                        lama_hard_cut_resets_forced += int(
                            stabilization_diag.get('hard_cut_resets_forced', 0)
                        )
                        lama_hard_cut_resets_roi += int(
                            stabilization_diag.get('hard_cut_resets_roi', 0)
                        )
                        lama_cold_start_frames += int(
                            stabilization_diag.get('cold_start_frames', 0)
                        )

                        legacy_alpha = self._build_lama_blend_alpha(
                            core_mask=roi_mask_template,
                            inpaint_mask=lama_inpaint_mask_stage1_template,
                            feather_px=feather_px,
                            rect_w=rect_w,
                            rect_h=rect_h,
                        )
                        transition_nonzero = int(np.count_nonzero(transition_mask)) > 8
                        lap_levels = 4 if min(y2 - y1, x2 - x1) >= 72 else 3
                        lama_final_rois = []
                        previous_selected_name: Optional[str] = None
                        previous_output_roi: Optional[np.ndarray] = None
                        for idx in range(len(stage1_stable_rois)):
                            roi_original = roi_frames[idx]
                            alpha = self._build_edge_aware_alpha(
                                core_replace_mask=core_replace_mask,
                                transition_mask=transition_mask,
                                reference_roi=roi_original,
                                feather_px=feather_px,
                            )
                            stage2_v2_roi = self._laplacian_blend_roi(
                                roi_original=roi_original,
                                roi_inpainted=stage2_stable_rois[idx],
                                alpha=alpha,
                                levels=lap_levels,
                            )
                            stage2_v2_roi = self._harmonize_transition_seam(
                                roi_original=roi_original,
                                blended_roi=stage2_v2_roi,
                                core_replace_mask=core_replace_mask,
                                transition_mask=transition_mask,
                            )
                            stage1_v2_roi = self._laplacian_blend_roi(
                                roi_original=roi_original,
                                roi_inpainted=stage1_stable_rois[idx],
                                alpha=alpha,
                                levels=lap_levels,
                            )
                            stage1_v2_roi = self._harmonize_transition_seam(
                                roi_original=roi_original,
                                blended_roi=stage1_v2_roi,
                                core_replace_mask=core_replace_mask,
                                transition_mask=transition_mask,
                            )
                            legacy_roi = np.clip(
                                roi_original.astype(np.float32) * (1.0 - legacy_alpha)
                                + stage1_stable_rois[idx].astype(np.float32) * legacy_alpha,
                                0.0,
                                255.0,
                            ).astype(np.uint8)

                            evaluations = {
                                'stage2_v2': self._evaluate_lama_frame_quality(
                                    roi_original=roi_original,
                                    roi_candidate=stage2_v2_roi,
                                    core_mask=roi_mask_template,
                                    transition_mask=transition_mask,
                                ),
                                'stage1_v2': self._evaluate_lama_frame_quality(
                                    roi_original=roi_original,
                                    roi_candidate=stage1_v2_roi,
                                    core_mask=roi_mask_template,
                                    transition_mask=transition_mask,
                                ),
                                'legacy': self._evaluate_lama_frame_quality(
                                    roi_original=roi_original,
                                    roi_candidate=legacy_roi,
                                    core_mask=roi_mask_template,
                                    transition_mask=transition_mask,
                                ),
                            }
                            lama_frame_guard_total += 1
                            selected_name, reject_stats = self._select_lama_frame_candidate(evaluations)
                            lama_frame_guard_dark_block_rejects += int(reject_stats['dark_rejects'])
                            lama_frame_guard_seam_rejects += int(reject_stats['seam_rejects'])
                            if previous_selected_name is not None and selected_name != previous_selected_name:
                                lama_frame_guard_switch_total += 1
                            selected_name, switch_suppressed = self._apply_frame_guard_hysteresis(
                                selected_name=selected_name,
                                evaluations=evaluations,
                                previous_name=previous_selected_name,
                            )
                            if switch_suppressed:
                                lama_frame_guard_switch_suppressed += 1
                            legacy_score = float(evaluations['legacy']['score'])

                            candidate_rois = {
                                'stage2_v2': stage2_v2_roi,
                                'stage1_v2': stage1_v2_roi,
                                'legacy': legacy_roi,
                            }
                            selected_roi = candidate_rois[selected_name]
                            selected_quality = evaluations[selected_name]

                            if (
                                selected_name in ('stage2_v2', 'stage1_v2')
                                and rounded_area >= 256
                                and (
                                    selected_quality['dark_block_flag']
                                    or selected_quality['seam_bad_flag']
                                )
                            ):
                                lama_rescue_attempts += 1
                                rescue_ok = False
                                try:
                                    rescue_roi = self._rescue_blend_with_seamless_clone(
                                        roi_original=roi_original,
                                        roi_candidate=selected_roi,
                                        rounded_mask=rounded_mask,
                                    )
                                    rescue_quality = self._evaluate_lama_frame_quality(
                                        roi_original=roi_original,
                                        roi_candidate=rescue_roi,
                                        core_mask=roi_mask_template,
                                        transition_mask=transition_mask,
                                    )
                                    if self._should_accept_lama_rescue(
                                        selected_quality=selected_quality,
                                        rescue_quality=rescue_quality,
                                    ):
                                        selected_roi = rescue_roi
                                        selected_quality = rescue_quality
                                        rescue_ok = True
                                        lama_rescue_accepted += 1
                                except Exception as rescue_exc:
                                    logger.warning('LaMa rescue blend failed: %s', rescue_exc)

                                if not rescue_ok:
                                    lama_rescue_rejected += 1
                                    if (
                                        selected_name != 'legacy'
                                        and legacy_score <= float(selected_quality['score']) * 1.15
                                    ):
                                        selected_name = 'legacy'
                                        selected_roi = legacy_roi
                                        selected_quality = evaluations['legacy']

                            selected_roi, micro_smooth_applied = self._apply_final_micro_smoothing(
                                current_roi=selected_roi,
                                previous_roi=previous_output_roi,
                                smoothing_mask=rounded_mask,
                                force_reset=(idx in forced_reset_indices),
                            )
                            if micro_smooth_applied:
                                lama_final_micro_smooth_applied += 1

                            if selected_name == 'stage2_v2':
                                lama_frame_guard_choose_stage2_v2 += 1
                                lama_blend_v2_used_frames += 1
                            elif selected_name == 'stage1_v2':
                                lama_frame_guard_choose_stage1_v2 += 1
                                lama_blend_v2_used_frames += 1
                            else:
                                lama_frame_guard_choose_legacy += 1

                            lama_final_rois.append(selected_roi)
                            previous_selected_name = selected_name
                            previous_output_roi = selected_roi
                            if transition_nonzero:
                                before_roi = (
                                    stage2_stable_rois[idx]
                                    if selected_name == 'stage2_v2'
                                    else stage1_stable_rois[idx]
                                )
                                lama_seam_delta_before_sum += self._compute_seam_delta(
                                    roi_original=roi_original,
                                    roi_candidate=before_roi,
                                    seam_mask=transition_mask,
                                )
                                lama_seam_delta_after_sum += self._compute_seam_delta(
                                    roi_original=roi_original,
                                    roi_candidate=selected_roi,
                                    seam_mask=transition_mask,
                                )
                                lama_seam_delta_count += 1
                    except Exception as lama_exc:
                        # 增强链路异常时回退到高斯融合，优先保证任务完成。
                        logger.warning(
                            'LaMa enhancement pipeline failed, fallback to gaussian blend: %s',
                            lama_exc,
                        )
                        lama_enhancement_fallbacks += 1
                        lama_blend_v2_fallbacks += 1
                        lama_final_rois = None
                        blend_mask = _build_gaussian_blend_mask()
                elif effective_model_id == 'propainter_roi':
                    try:
                        transition_masks = self._build_lama_transition_masks(
                            core_mask=roi_mask_template,
                            inpaint_mask=lama_inpaint_mask_stage1_template,
                            feather_px=feather_px,
                            rect_w=rect_w,
                            rect_h=rect_h,
                        )
                        core_replace_mask = transition_masks['core_replace_mask']
                        transition_mask = transition_masks['transition_mask']
                        context_mask = transition_masks['context_mask']
                        rounded_mask = transition_masks['rounded_mask']
                        transition_inner_mask = transition_masks.get('transition_inner_mask', transition_mask)
                        transition_outer_mask = transition_masks.get('transition_outer_mask', transition_mask)
                        rounded_area = int(np.count_nonzero(rounded_mask))
                        transition_nonzero = int(np.count_nonzero(transition_mask)) > 8

                        legacy_alpha = self._build_lama_blend_alpha(
                            core_mask=roi_mask_template,
                            inpaint_mask=lama_inpaint_mask_stage1_template,
                            feather_px=feather_px,
                            rect_w=rect_w,
                            rect_h=rect_h,
                        )
                        lap_levels = 4 if min(y2 - y1, x2 - x1) >= 180 else 3
                        segment_start_frame = int(segment.get('start_frame', 0))
                        segment_end_frame = int(segment.get('end_frame', segment_start_frame))
                        segment_state_key = str(
                            segment.get(
                                '_state_key',
                                f"{str(segment.get('id', ''))}@{int(segment.get('_order', 0))}:"
                                f"{int(segment.get('start_frame', 0))}-{int(segment.get('end_frame', 0))}",
                            )
                        )
                        segment_declared_total = max(
                            1,
                            int(segment.get('end_frame', 0)) - int(segment.get('start_frame', 0)) + 1,
                        )
                        segment_runtime = propainter_segment_runtime.setdefault(
                            segment_state_key,
                            {
                                'observed_frames': 0,
                                'legacy_frames': 0,
                                'original_similarity_sum': 0.0,
                                'reappear_sum': 0.0,
                                'force_remove_mode': False,
                                'force_remove_trigger_frame': -1,
                                'force_remove_overrides_total': 0,
                                'declared_total_frames': segment_declared_total,
                            },
                        )
                        segment_runtime['declared_total_frames'] = max(
                            int(segment_runtime.get('declared_total_frames', 0)),
                            segment_declared_total,
                        )
                        segment_observed_frames = int(segment_runtime.get('observed_frames', 0))
                        segment_legacy_frames = int(segment_runtime.get('legacy_frames', 0))
                        segment_similarity_sum_total = float(
                            segment_runtime.get('original_similarity_sum', 0.0)
                        )
                        segment_reappear_sum_total = float(segment_runtime.get('reappear_sum', 0.0))

                        propainter_cut_quarantine_indices = self._compute_cut_quarantine_indices(
                            total_length=len(prepared_rois),
                            cut_indices=propainter_forced_reset_indices,
                            before=1,
                            after=3,
                        )

                        def _run_propainter_frame_selection(
                            prepared_source_rois: List[np.ndarray],
                            legacy_source_rois: Optional[List[np.ndarray]] = None,
                        ) -> Dict[str, Any]:
                            corrected_rois_local = [
                                self._apply_boundary_color_correction_weighted(
                                    roi_original=roi_frames[idx],
                                    inpainted_roi=prepared_source_rois[idx],
                                    core_replace_mask=core_replace_mask,
                                    transition_mask=transition_mask,
                                    context_mask=context_mask,
                                    feather_px=feather_px,
                                )
                                for idx in range(len(prepared_source_rois))
                            ]
                            temporal_strength_local = self._compute_propainter_temporal_strength(
                                segment_total_frames=segment_total_frames,
                                frame_width=width,
                                frame_height=height,
                            )
                            stable_rois_local, stable_diag_local = self._stabilize_propainter_sequence(
                                roi_sequence=corrected_rois_local,
                                roi_mask=lama_inpaint_mask_stage1_template,
                                temporal_strength=temporal_strength_local,
                                forced_reset_indices=propainter_forced_reset_indices,
                                quarantine_indices=propainter_cut_quarantine_indices,
                            )
                            frame_candidates: List[Dict[str, Any]] = []
                            pass_ring_clone_used = 0
                            pass_ring_clone_fallbacks = 0
                            previous_remove_ratio_anchor: Optional[float] = None

                            for idx in range(len(prepared_source_rois)):
                                roi_original = roi_frames[idx]
                                is_hard_cut_frame = idx in propainter_cut_quarantine_indices
                                eval_prev_roi = (
                                    None
                                    if is_hard_cut_frame or idx <= 0
                                    else stable_rois_local[idx - 1]
                                )
                                alpha_local = self._build_edge_aware_alpha_v3(
                                    core_replace_mask=core_replace_mask,
                                    transition_inner_mask=transition_inner_mask,
                                    transition_outer_mask=transition_outer_mask,
                                    reference_roi=roi_original,
                                )
                                raw_v2_base = self._laplacian_blend_roi(
                                    roi_original=roi_original,
                                    roi_inpainted=corrected_rois_local[idx],
                                    alpha=alpha_local,
                                    levels=lap_levels,
                                )
                                raw_v2_base = self._harmonize_transition_seam(
                                    roi_original=roi_original,
                                    blended_roi=raw_v2_base,
                                    core_replace_mask=core_replace_mask,
                                    transition_mask=transition_mask,
                                )
                                raw_v2_roi, raw_ring_used, raw_ring_fallback = self._apply_propainter_ring_clone(
                                    roi_original=roi_original,
                                    roi_candidate=raw_v2_base,
                                    transition_outer_mask=transition_outer_mask,
                                )
                                pass_ring_clone_used += 1 if raw_ring_used else 0
                                pass_ring_clone_fallbacks += 1 if raw_ring_fallback else 0
                                raw_seam_before = self._compute_seam_delta(
                                    roi_original=roi_original,
                                    roi_candidate=raw_v2_base,
                                    seam_mask=transition_mask,
                                )
                                raw_seam_after = self._compute_seam_delta(
                                    roi_original=roi_original,
                                    roi_candidate=raw_v2_roi,
                                    seam_mask=transition_mask,
                                )
                                if raw_seam_after > raw_seam_before * 1.03:
                                    raw_v2_roi = raw_v2_base

                                stable_v2_base = self._laplacian_blend_roi(
                                    roi_original=roi_original,
                                    roi_inpainted=stable_rois_local[idx],
                                    alpha=alpha_local,
                                    levels=lap_levels,
                                )
                                stable_v2_base = self._harmonize_transition_seam(
                                    roi_original=roi_original,
                                    blended_roi=stable_v2_base,
                                    core_replace_mask=core_replace_mask,
                                    transition_mask=transition_mask,
                                )
                                stable_v2_roi, stable_ring_used, stable_ring_fallback = (
                                    self._apply_propainter_ring_clone(
                                        roi_original=roi_original,
                                        roi_candidate=stable_v2_base,
                                        transition_outer_mask=transition_outer_mask,
                                    )
                                )
                                pass_ring_clone_used += 1 if stable_ring_used else 0
                                pass_ring_clone_fallbacks += 1 if stable_ring_fallback else 0
                                stable_seam_before = self._compute_seam_delta(
                                    roi_original=roi_original,
                                    roi_candidate=stable_v2_base,
                                    seam_mask=transition_mask,
                                )
                                stable_seam_after = self._compute_seam_delta(
                                    roi_original=roi_original,
                                    roi_candidate=stable_v2_roi,
                                    seam_mask=transition_mask,
                                )
                                if stable_seam_after > stable_seam_before * 1.03:
                                    stable_v2_roi = stable_v2_base

                                legacy_input_rois = (
                                    legacy_source_rois
                                    if legacy_source_rois is not None
                                    else prepared_source_rois
                                )
                                legacy_roi = np.clip(
                                    roi_original.astype(np.float32) * (1.0 - legacy_alpha)
                                    + legacy_input_rois[idx].astype(np.float32) * legacy_alpha,
                                    0.0,
                                    255.0,
                                ).astype(np.uint8)

                                evaluations = {
                                    'raw_v2': self._evaluate_propainter_frame_quality(
                                        roi_original=roi_original,
                                        roi_candidate=raw_v2_roi,
                                        core_mask=roi_mask_template,
                                        transition_mask=transition_mask,
                                        previous_selected_roi=eval_prev_roi,
                                        is_hard_cut_frame=is_hard_cut_frame,
                                    ),
                                    'stable_v2': self._evaluate_propainter_frame_quality(
                                        roi_original=roi_original,
                                        roi_candidate=stable_v2_roi,
                                        core_mask=roi_mask_template,
                                        transition_mask=transition_mask,
                                        previous_selected_roi=eval_prev_roi,
                                        is_hard_cut_frame=is_hard_cut_frame,
                                    ),
                                    'legacy': self._evaluate_propainter_frame_quality(
                                        roi_original=roi_original,
                                        roi_candidate=legacy_roi,
                                        core_mask=roi_mask_template,
                                        transition_mask=transition_mask,
                                        previous_selected_roi=eval_prev_roi,
                                        is_hard_cut_frame=is_hard_cut_frame,
                                    ),
                                }
                                evaluations, remove_reference = (
                                    self._augment_propainter_evaluations_for_remove_sufficiency(
                                        evaluations=evaluations,
                                        previous_selected_remove_ratio=previous_remove_ratio_anchor,
                                        is_hard_cut_frame=is_hard_cut_frame,
                                    )
                                )
                                previous_remove_ratio_anchor = max(
                                    float(evaluations['raw_v2'].get('remove_ratio', 0.0)),
                                    float(evaluations['stable_v2'].get('remove_ratio', 0.0)),
                                )
                                frame_candidates.append(
                                    {
                                        'evaluations': evaluations,
                                        'candidate_rois': {
                                            'raw_v2': raw_v2_roi,
                                            'stable_v2': stable_v2_roi,
                                            'legacy': legacy_roi,
                                        },
                                        'remove_reference': float(remove_reference),
                                        'is_hard_cut_frame': bool(is_hard_cut_frame),
                                        'raw_seam_before': float(raw_seam_before),
                                        'stable_seam_before': float(stable_seam_before),
                                    }
                                )

                            viterbi_names, viterbi_switches = self._select_propainter_sequence_viterbi(
                                frame_evaluations=[item['evaluations'] for item in frame_candidates],
                                cut_quarantine_indices=propainter_cut_quarantine_indices,
                                switch_penalty=0.045,
                                legacy_penalty=0.08,
                                reappear_penalty=1.0,
                            )
                            viterbi_names, selection_island_rewrites = (
                                self._suppress_propainter_selection_islands(
                                    selected_names=viterbi_names,
                                    frame_evaluations=[item['evaluations'] for item in frame_candidates],
                                    cut_quarantine_indices=propainter_cut_quarantine_indices,
                                    max_score_ratio=1.05,
                                )
                            )
                            viterbi_switches = sum(
                                1
                                for i in range(1, len(viterbi_names))
                                if viterbi_names[i] != viterbi_names[i - 1]
                            )
                            pass_output_rois: List[np.ndarray] = []
                            pass_qualities: List[Dict[str, Any]] = []
                            stable_candidates_local: List[np.ndarray] = []
                            raw_candidates_local: List[np.ndarray] = []
                            previous_selected_name_local: Optional[str] = None
                            previous_output_roi_local: Optional[np.ndarray] = None
                            previous_selected_remove_ratio_local: Optional[float] = None
                            legacy_advantage_streak_local = 0
                            pass_choose_raw = 0
                            pass_choose_stable = 0
                            pass_choose_legacy = 0
                            pass_similarity_sum = 0.0
                            pass_reappear_sum = 0.0
                            pass_frame_guard_total = 0
                            pass_dark_rejects = 0
                            pass_seam_rejects = 0
                            pass_legacy_catastrophic_only = 0
                            pass_hysteresis_hold = 0
                            pass_hysteresis_switch = 0
                            pass_legacy_blocked = 0
                            pass_rescue_attempts = 0
                            pass_rescue_accepted = 0
                            pass_rescue_rejected = 0
                            pass_under_remove_count = 0
                            pass_reappear_count = 0
                            pass_remove_ratios: List[float] = []
                            pass_residual_hf_corr_values: List[float] = []
                            pass_seam_before_values: List[float] = []
                            pass_seam_after_values: List[float] = []

                            for idx in range(len(frame_candidates)):
                                roi_original = roi_frames[idx]
                                frame_item = frame_candidates[idx]
                                evaluations = frame_item['evaluations']
                                candidate_rois = frame_item['candidate_rois']
                                is_hard_cut_frame = bool(frame_item['is_hard_cut_frame'])
                                pass_frame_guard_total += 1
                                selected_name = viterbi_names[idx] if idx < len(viterbi_names) else 'stable_v2'
                                if is_hard_cut_frame and selected_name == 'legacy':
                                    selected_name = min(
                                        ('stable_v2', 'raw_v2'),
                                        key=lambda name: float(evaluations[name].get('score', 1e9)),
                                    )
                                selected_name, legacy_advantage_streak_local, hysteresis_stats = (
                                    self._apply_propainter_frame_hysteresis(
                                        selected_name=selected_name,
                                        evaluations=evaluations,
                                        previous_name=previous_selected_name_local,
                                        legacy_advantage_streak=legacy_advantage_streak_local,
                                    )
                                )
                                pass_hysteresis_hold += int(hysteresis_stats['hold_count'])
                                pass_hysteresis_switch += int(hysteresis_stats['switch_count'])
                                pass_legacy_blocked += int(hysteresis_stats['legacy_blocked'])
                                if is_hard_cut_frame and selected_name == 'legacy':
                                    selected_name = min(
                                        ('stable_v2', 'raw_v2'),
                                        key=lambda name: float(evaluations[name].get('score', 1e9)),
                                    )
                                    pass_legacy_blocked += 1

                                selected_roi = candidate_rois[selected_name]
                                selected_quality = evaluations[selected_name]
                                both_v2_catastrophic = all(
                                    self._is_propainter_catastrophic(evaluations[name])
                                    for name in ('stable_v2', 'raw_v2')
                                )
                                if selected_name == 'legacy' and both_v2_catastrophic:
                                    pass_legacy_catastrophic_only += 1

                                for candidate_name in ('stable_v2', 'raw_v2'):
                                    if candidate_name == selected_name:
                                        continue
                                    if evaluations[candidate_name].get('dark_block_flag', False):
                                        pass_dark_rejects += 1
                                    if evaluations[candidate_name].get('seam_bad_flag', False):
                                        pass_seam_rejects += 1

                                if (
                                    (not is_hard_cut_frame)
                                    and selected_name in ('raw_v2', 'stable_v2')
                                    and rounded_area >= 256
                                    and (
                                        selected_quality.get('dark_block_flag', False)
                                        or selected_quality.get('seam_bad_flag', False)
                                        or selected_quality.get('under_remove_flag', False)
                                    )
                                ):
                                    pass_rescue_attempts += 1
                                    rescue_ok = False
                                    try:
                                        rescue_roi = self._rescue_propainter_with_seamless_clone(
                                            roi_original=roi_original,
                                            roi_candidate=selected_roi,
                                            rounded_mask=rounded_mask,
                                        )
                                        rescue_quality = self._evaluate_propainter_frame_quality(
                                            roi_original=roi_original,
                                            roi_candidate=rescue_roi,
                                            core_mask=roi_mask_template,
                                            transition_mask=transition_mask,
                                            previous_selected_roi=(
                                                None if is_hard_cut_frame else previous_output_roi_local
                                            ),
                                            is_hard_cut_frame=is_hard_cut_frame,
                                        )
                                        rescue_quality = self._apply_propainter_remove_sufficiency(
                                            quality=rescue_quality,
                                            remove_energy_reference=float(frame_item['remove_reference']),
                                            previous_selected_remove_ratio=previous_selected_remove_ratio_local,
                                            is_hard_cut_frame=is_hard_cut_frame,
                                        )
                                        if self._should_accept_propainter_rescue(
                                            selected_quality=selected_quality,
                                            rescue_quality=rescue_quality,
                                        ):
                                            selected_roi = rescue_roi
                                            selected_quality = rescue_quality
                                            rescue_ok = True
                                            pass_rescue_accepted += 1
                                    except Exception as rescue_exc:
                                        logger.warning('ProPainter rescue blend failed: %s', rescue_exc)

                                    if not rescue_ok:
                                        pass_rescue_rejected += 1
                                        if (
                                            selected_name != 'legacy'
                                            and self._is_propainter_catastrophic(selected_quality)
                                            and not is_hard_cut_frame
                                        ):
                                            selected_name = 'legacy'
                                            selected_roi = candidate_rois['legacy']
                                            selected_quality = evaluations['legacy']

                                selected_roi, _ = self._apply_final_micro_smoothing(
                                    current_roi=selected_roi,
                                    previous_roi=previous_output_roi_local,
                                    smoothing_mask=rounded_mask,
                                    force_reset=(idx in propainter_cut_quarantine_indices),
                                )
                                if selected_name == 'stable_v2':
                                    pass_choose_stable += 1
                                elif selected_name == 'raw_v2':
                                    pass_choose_raw += 1
                                else:
                                    pass_choose_legacy += 1

                                pass_similarity_sum += float(
                                    selected_quality.get('original_similarity_core', 0.0)
                                )
                                if bool(selected_quality.get('reappear_flag', False)):
                                    pass_reappear_sum += 1.0
                                    pass_reappear_count += 1
                                if bool(selected_quality.get('under_remove_flag', False)):
                                    pass_under_remove_count += 1
                                pass_remove_ratios.append(float(selected_quality.get('remove_ratio', 0.0)))
                                pass_residual_hf_corr_values.append(
                                    float(selected_quality.get('residual_hf_corr', 0.0))
                                )
                                pass_seam_before_values.append(
                                    self._compute_seam_delta(
                                        roi_original=roi_original,
                                        roi_candidate=prepared_source_rois[idx],
                                        seam_mask=transition_mask,
                                    )
                                )
                                pass_seam_after_values.append(
                                    self._compute_seam_delta(
                                        roi_original=roi_original,
                                        roi_candidate=selected_roi,
                                        seam_mask=transition_mask,
                                    )
                                )

                                pass_output_rois.append(selected_roi)
                                pass_qualities.append(dict(selected_quality))
                                stable_candidates_local.append(candidate_rois['stable_v2'])
                                raw_candidates_local.append(candidate_rois['raw_v2'])
                                previous_selected_name_local = selected_name
                                previous_output_roi_local = selected_roi
                                previous_selected_remove_ratio_local = float(
                                    selected_quality.get('remove_ratio', 1.0)
                                )

                            flagged_indices = [
                                idx
                                for idx, item in enumerate(pass_qualities)
                                if bool(item.get('reappear_flag', False))
                                or bool(item.get('under_remove_flag', False))
                            ]
                            burst_count = 0
                            if flagged_indices:
                                burst_count = 1
                                for idx in range(1, len(flagged_indices)):
                                    if flagged_indices[idx] != flagged_indices[idx - 1] + 1:
                                        burst_count += 1
                            pass_frames = max(1, len(prepared_source_rois))
                            return {
                                'prepared_rois': prepared_source_rois,
                                'final_rois': pass_output_rois,
                                'selected_qualities': pass_qualities,
                                'stable_candidates': stable_candidates_local,
                                'raw_candidates': raw_candidates_local,
                                'choose_raw': pass_choose_raw,
                                'choose_stable': pass_choose_stable,
                                'choose_legacy': pass_choose_legacy,
                                'v2_used': pass_choose_raw + pass_choose_stable,
                                'legacy_ratio': float(pass_choose_legacy) / float(pass_frames),
                                'original_similarity_sum': pass_similarity_sum,
                                'reappear_sum': pass_reappear_sum,
                                'under_remove_count': pass_under_remove_count,
                                'under_remove_rate': float(pass_under_remove_count) / float(pass_frames),
                                'reappear_count': pass_reappear_count,
                                'burst_count': int(burst_count),
                                'median_remove_ratio': float(
                                    np.median(np.asarray(pass_remove_ratios, dtype=np.float32))
                                )
                                if pass_remove_ratios
                                else 0.0,
                                'median_residual_hf_corr': float(
                                    np.median(np.asarray(pass_residual_hf_corr_values, dtype=np.float32))
                                )
                                if pass_residual_hf_corr_values
                                else 0.0,
                                'seam_p90_before': float(
                                    np.percentile(np.asarray(pass_seam_before_values, dtype=np.float32), 90)
                                )
                                if pass_seam_before_values
                                else 0.0,
                                'seam_p90_after': float(
                                    np.percentile(np.asarray(pass_seam_after_values, dtype=np.float32), 90)
                                )
                                if pass_seam_after_values
                                else 0.0,
                                'remove_ratio_values': pass_remove_ratios,
                                'residual_hf_corr_values': pass_residual_hf_corr_values,
                                'frame_guard_total': pass_frame_guard_total,
                                'frame_guard_dark_rejects': pass_dark_rejects,
                                'frame_guard_seam_rejects': pass_seam_rejects,
                                'legacy_allowed_catastrophic_only': pass_legacy_catastrophic_only,
                                'hysteresis_hold': pass_hysteresis_hold,
                                'hysteresis_switch': pass_hysteresis_switch,
                                'legacy_blocked': pass_legacy_blocked,
                                'rescue_attempts': pass_rescue_attempts,
                                'rescue_accepted': pass_rescue_accepted,
                                'rescue_rejected': pass_rescue_rejected,
                                'stabilize_applied_frames': int(
                                    stable_diag_local.get('stabilize_applied_frames', 0)
                                ),
                                'cut_quarantine_frames': int(
                                    stable_diag_local.get('cut_quarantine_frames', 0)
                                ),
                                'viterbi_switches': int(viterbi_switches),
                                'selection_island_rewrites': int(selection_island_rewrites),
                                'ring_clone_used': int(pass_ring_clone_used),
                                'ring_clone_fallbacks': int(pass_ring_clone_fallbacks),
                            }

                        pass1_result = _run_propainter_frame_selection(
                            prepared_source_rois=prepared_rois,
                            legacy_source_rois=None,
                        )
                        selected_pass = pass1_result
                        pass1_frames = max(1, len(pass1_result['prepared_rois']))
                        rerun_triggered = self._should_rerun_propainter_segment(
                            legacy_ratio=float(pass1_result['legacy_ratio']),
                            median_remove_ratio=float(pass1_result['median_remove_ratio']),
                            reappear_count=int(pass1_result['reappear_count']),
                            frame_count=pass1_frames,
                            median_residual_hf_corr=float(
                                pass1_result.get('median_residual_hf_corr', 0.0)
                            ),
                            under_remove_rate=float(pass1_result.get('under_remove_rate', 0.0)),
                            burst_count=int(pass1_result.get('burst_count', 0)),
                        )
                        if rerun_triggered:
                            propainter_segment_rerun_attempts += 1
                            rerun_options = self._compute_propainter_rerun_options(
                                base_options=propainter_options,
                                rect_w=rect_w,
                                rect_h=rect_h,
                            )
                            try:
                                rerun_rois: List[np.ndarray] = []
                                for split_start, split_end in split_ranges:
                                    sub_frames = roi_frames[split_start:split_end]
                                    sub_masks = [roi_mask_template.copy() for _ in sub_frames]
                                    sub_rois = active_engine.inpaint_roi_sequence(
                                        sub_frames,
                                        sub_masks,
                                        progress_callback=_engine_progress_callback,
                                        propainter_options=rerun_options,
                                    )
                                    rerun_rois.extend(sub_rois)
                                rerun_prepared_rois = _prepare_rois(rerun_rois)
                                pass2_result = _run_propainter_frame_selection(
                                    prepared_source_rois=rerun_prepared_rois,
                                    legacy_source_rois=pass1_result['final_rois'],
                                )
                                if self._should_accept_propainter_rerun(
                                    pass1_median_remove_ratio=float(
                                        pass1_result['median_remove_ratio']
                                    ),
                                    pass1_legacy_ratio=float(pass1_result['legacy_ratio']),
                                    pass2_median_remove_ratio=float(
                                        pass2_result['median_remove_ratio']
                                    ),
                                    pass2_legacy_ratio=float(pass2_result['legacy_ratio']),
                                    pass1_under_remove_rate=float(
                                        pass1_result.get('under_remove_rate', 0.0)
                                    ),
                                    pass2_under_remove_rate=float(
                                        pass2_result.get('under_remove_rate', 0.0)
                                    ),
                                    pass1_seam_p90=float(pass1_result.get('seam_p90_after', 0.0)),
                                    pass2_seam_p90=float(pass2_result.get('seam_p90_after', 0.0)),
                                ):
                                    selected_pass = pass2_result
                                    propainter_segment_rerun_accepted += 1
                                else:
                                    propainter_segment_rerun_rejected += 1
                            except Exception as rerun_exc:
                                logger.warning(
                                    'ProPainter segment rerun failed, keep pass1: %s',
                                    rerun_exc,
                                )
                                propainter_segment_rerun_rejected += 1

                        propainter_final_rois = list(selected_pass['final_rois'])
                        propainter_selected_qualities: List[Dict[str, Any]] = [
                            dict(item) for item in selected_pass['selected_qualities']
                        ]
                        propainter_reappear_score_before_sum += sum(
                            float(item.get('original_similarity_core', 0.0))
                            for item in propainter_selected_qualities
                        )
                        propainter_remove_ratio_before_sum += sum(
                            float(item.get('remove_ratio', 0.0)) for item in propainter_selected_qualities
                        )
                        propainter_residual_hf_corr_before_sum += sum(
                            float(item.get('residual_hf_corr', 0.0))
                            for item in propainter_selected_qualities
                        )
                        propainter_reappear_score_count += len(propainter_selected_qualities)

                        (
                            propainter_final_rois,
                            propainter_selected_qualities,
                            burst_fix_stats,
                        ) = self._repair_propainter_short_reappear_bursts(
                            selected_rois=propainter_final_rois,
                            selected_qualities=propainter_selected_qualities,
                            stable_candidates=list(selected_pass['stable_candidates']),
                            raw_candidates=list(selected_pass['raw_candidates']),
                            roi_originals=roi_frames,
                            core_mask=roi_mask_template,
                            transition_mask=transition_mask,
                            forced_reset_indices=propainter_forced_reset_indices,
                            cold_start_window=2,
                            max_burst_length=3,
                            cut_quarantine_indices=propainter_cut_quarantine_indices,
                        )
                        propainter_burst_fix_attempts += int(burst_fix_stats['burst_fix_attempts'])
                        propainter_burst_fix_accepted_frames += int(
                            burst_fix_stats['burst_fix_accepted_frames']
                        )
                        propainter_burst_fix_rejected_frames += int(
                            burst_fix_stats['burst_fix_rejected_frames']
                        )
                        propainter_microfix_metric_count += len(propainter_selected_qualities)
                        propainter_temporal_jump_before_microfix_sum += sum(
                            float(item.get('temporal_jump_core', 0.0))
                            for item in propainter_selected_qualities
                        )
                        propainter_remove_ratio_before_microfix_sum += sum(
                            float(item.get('remove_ratio', 0.0))
                            for item in propainter_selected_qualities
                        )
                        micro_flicker_flags = self._detect_propainter_micro_flicker_flags(
                            selected_qualities=propainter_selected_qualities,
                            cut_quarantine_indices=propainter_cut_quarantine_indices,
                            window_radius=2,
                        )
                        propainter_micro_flicker_flags_total += int(
                            sum(1 for flag in micro_flicker_flags if flag)
                        )
                        (
                            propainter_final_rois,
                            propainter_selected_qualities,
                            micro_fix_stats,
                        ) = self._repair_propainter_micro_flicker_bursts(
                            selected_rois=propainter_final_rois,
                            selected_qualities=propainter_selected_qualities,
                            stable_candidates=list(selected_pass['stable_candidates']),
                            raw_candidates=list(selected_pass['raw_candidates']),
                            roi_originals=roi_frames,
                            core_mask=roi_mask_template,
                            transition_mask=transition_mask,
                            forced_reset_indices=propainter_forced_reset_indices,
                            micro_flicker_flags=micro_flicker_flags,
                            cold_start_window=2,
                            max_burst_length=2,
                            cut_quarantine_indices=propainter_cut_quarantine_indices,
                        )
                        propainter_micro_burst_fix_attempts += int(
                            micro_fix_stats['micro_burst_fix_attempts']
                        )
                        propainter_micro_burst_fix_accepted_frames += int(
                            micro_fix_stats['micro_burst_fix_accepted_frames']
                        )
                        propainter_micro_burst_fix_rejected_frames += int(
                            micro_fix_stats['micro_burst_fix_rejected_frames']
                        )
                        propainter_temporal_jump_after_microfix_sum += sum(
                            float(item.get('temporal_jump_core', 0.0))
                            for item in propainter_selected_qualities
                        )
                        propainter_remove_ratio_after_microfix_sum += sum(
                            float(item.get('remove_ratio', 0.0))
                            for item in propainter_selected_qualities
                        )

                        propainter_stabilize_applied_frames += int(
                            selected_pass['stabilize_applied_frames']
                        )
                        propainter_frame_guard_total += int(selected_pass['frame_guard_total'])
                        propainter_choose_raw_v2 += int(selected_pass['choose_raw'])
                        propainter_choose_stable_v2 += int(selected_pass['choose_stable'])
                        propainter_choose_legacy += int(selected_pass['choose_legacy'])
                        propainter_v2_used_frames += int(selected_pass['v2_used'])
                        propainter_frame_guard_dark_block_rejects += int(
                            selected_pass['frame_guard_dark_rejects']
                        )
                        propainter_frame_guard_seam_rejects += int(
                            selected_pass['frame_guard_seam_rejects']
                        )
                        propainter_hysteresis_hold_count += int(selected_pass['hysteresis_hold'])
                        propainter_hysteresis_switch_count += int(selected_pass['hysteresis_switch'])
                        propainter_viterbi_switches += int(selected_pass.get('viterbi_switches', 0))
                        propainter_selection_island_rewrites += int(
                            selected_pass.get('selection_island_rewrites', 0)
                        )
                        propainter_legacy_blocked_by_guard += int(selected_pass['legacy_blocked'])
                        propainter_rescue_attempts += int(selected_pass['rescue_attempts'])
                        propainter_rescue_accepted += int(selected_pass['rescue_accepted'])
                        propainter_rescue_rejected += int(selected_pass['rescue_rejected'])
                        propainter_cut_quarantine_frames += int(
                            selected_pass.get('cut_quarantine_frames', 0)
                        )
                        propainter_ring_clone_used_frames += int(
                            selected_pass.get('ring_clone_used', 0)
                        )
                        propainter_ring_clone_fallbacks += int(
                            selected_pass.get('ring_clone_fallbacks', 0)
                        )
                        propainter_legacy_catastrophic_only_count += int(
                            selected_pass['legacy_allowed_catastrophic_only']
                        )

                        segment_choose_raw = int(selected_pass['choose_raw'])
                        segment_choose_stable = int(selected_pass['choose_stable'])
                        segment_choose_legacy = int(selected_pass['choose_legacy'])
                        segment_frames_count = max(1, len(propainter_final_rois))
                        segment_v2_used = segment_choose_raw + segment_choose_stable
                        segment_original_similarity_sum = 0.0
                        segment_reappear_sum = 0.0
                        segment_under_remove_count = 0
                        segment_remove_ratios: List[float] = []
                        for quality_item in propainter_selected_qualities:
                            similarity = float(quality_item.get('original_similarity_core', 0.0))
                            segment_original_similarity_sum += similarity
                            if bool(quality_item.get('reappear_flag', False)):
                                segment_reappear_sum += 1.0
                                propainter_reappear_flags_total += 1
                            if bool(quality_item.get('under_remove_flag', False)):
                                segment_under_remove_count += 1
                            segment_remove_ratios.append(float(quality_item.get('remove_ratio', 0.0)))
                            propainter_reappear_score_after_sum += similarity
                            propainter_remove_ratio_after_sum += float(
                                quality_item.get('remove_ratio', 0.0)
                            )
                            propainter_residual_hf_corr_after_sum += float(
                                quality_item.get('residual_hf_corr', 0.0)
                            )
                        propainter_under_remove_flags_total += int(segment_under_remove_count)

                        if transition_nonzero:
                            before_rois_for_seam = selected_pass['prepared_rois']
                            for idx, selected_roi in enumerate(propainter_final_rois):
                                roi_original = roi_frames[idx]
                                propainter_seam_delta_before_sum += self._compute_seam_delta(
                                    roi_original=roi_original,
                                    roi_candidate=before_rois_for_seam[idx],
                                    seam_mask=transition_mask,
                                )
                                propainter_seam_delta_after_sum += self._compute_seam_delta(
                                    roi_original=roi_original,
                                    roi_candidate=selected_roi,
                                    seam_mask=transition_mask,
                                )
                                propainter_seam_delta_count += 1
                        propainter_seam_p90_before_sum += float(
                            selected_pass.get('seam_p90_before', 0.0)
                        )
                        propainter_seam_p90_after_sum += float(
                            selected_pass.get('seam_p90_after', 0.0)
                        )
                        propainter_seam_p90_count += 1

                        segment_remove_ratio_median = (
                            float(np.median(np.asarray(segment_remove_ratios, dtype=np.float32)))
                            if segment_remove_ratios
                            else 0.0
                        )
                        segment_residual_hf_corr_median = float(
                            selected_pass.get('median_residual_hf_corr', 0.0)
                        )
                        segment_under_remove_rate = float(selected_pass.get('under_remove_rate', 0.0))
                        segment_seam_p90_after = float(selected_pass.get('seam_p90_after', 0.0))
                        segment_observed_frames += segment_frames_count
                        segment_legacy_frames += segment_choose_legacy
                        segment_similarity_sum_total += segment_original_similarity_sum
                        segment_reappear_sum_total += segment_reappear_sum

                        logger.info(
                            (
                                'ProPainter segment stats: segment_id=%s, frame_range=%d-%d, '
                                'frames=%d, choose_raw=%d, choose_stable=%d, choose_legacy=%d, '
                                'v2_used_ratio=%.3f, reappear_avg=%.3f, '
                                'under_remove_avg=%.3f, median_remove_ratio=%.3f, '
                                'median_residual_hf_corr=%.3f, seam_p90_after=%.4f, '
                                'original_similarity_avg=%.3f, forced_mode_triggered=%s, '
                                'forced_mode_trigger_frame=%d, forced_mode_overrides=%d, '
                                'global_observed=%d, global_legacy=%d'
                            ),
                            str(segment.get('id', '')),
                            segment_start_frame,
                            segment_end_frame,
                            segment_frames_count,
                            segment_choose_raw,
                            segment_choose_stable,
                            segment_choose_legacy,
                            float(segment_v2_used) / float(segment_frames_count),
                            float(segment_reappear_sum) / float(segment_frames_count),
                            segment_under_remove_rate,
                            segment_remove_ratio_median,
                            segment_residual_hf_corr_median,
                            segment_seam_p90_after,
                            float(segment_original_similarity_sum) / float(segment_frames_count),
                            False,
                            -1,
                            0,
                            segment_observed_frames,
                            segment_legacy_frames,
                        )
                        segment_runtime['observed_frames'] = segment_observed_frames
                        segment_runtime['legacy_frames'] = segment_legacy_frames
                        segment_runtime['original_similarity_sum'] = segment_similarity_sum_total
                        segment_runtime['reappear_sum'] = segment_reappear_sum_total
                        segment_runtime['force_remove_mode'] = False
                        segment_runtime['force_remove_trigger_frame'] = -1
                        segment_runtime['force_remove_overrides_total'] = int(
                            segment_runtime.get('force_remove_overrides_total', 0)
                        )
                    except Exception as propainter_exc:
                        logger.warning(
                            'ProPainter enhancement pipeline failed, fallback to gaussian blend: %s',
                            propainter_exc,
                        )
                        propainter_v2_fallbacks += 1
                        propainter_final_rois = None
                        blend_mask = _build_gaussian_blend_mask()
                else:
                    blend_mask = _build_gaussian_blend_mask()

                result_frames: List[np.ndarray] = []
                for idx, frame_item in enumerate(segment_frames):
                    roi_original = roi_frames[idx]
                    inpainted_roi = prepared_rois[idx]
                    if lama_final_rois is not None:
                        blended_roi = lama_final_rois[idx]
                    elif propainter_final_rois is not None:
                        blended_roi = propainter_final_rois[idx]
                    elif blend_mask is not None:
                        blended_roi = (
                            roi_original.astype(np.float32) * (1.0 - blend_mask)
                            + inpainted_roi.astype(np.float32) * blend_mask
                        ).astype(np.uint8)
                    else:
                        blended_roi = np.where(hard_mask, inpainted_roi, roi_original)

                    result = frame_item.copy()
                    result[y1:y2, x1:x2] = blended_roi
                    result_frames.append(result)

                return result_frames

            def _flush_segment_run(
                segments_in_run: List[Dict[str, Any]],
                segment_frames: List[np.ndarray],
            ) -> None:
                """把同一签名帧块按多段顺序依次处理后统一写出。"""
                nonlocal hit_frames, skipped_frames
                if not segment_frames:
                    return
                if not segments_in_run:
                    for frame_item in segment_frames:
                        _write_result_frame(frame_item)
                        skipped_frames += 1
                    return

                working_frames = [frame_item.copy() for frame_item in segment_frames]
                for segment_item in segments_in_run:
                    working_frames = _apply_single_segment_on_frames(segment_item, working_frames)
                for output_frame in working_frames:
                    _write_result_frame(output_frame)
                    hit_frames += 1

            buffered_signature: Optional[Tuple[str, ...]] = None
            buffered_segments: List[Dict[str, Any]] = []
            buffered_frames: List[np.ndarray] = []

            while True:
                # 主读帧循环：按“命中段连续块”进行分段处理，非命中帧直接透传。
                if self._should_stop:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                active_segments = self._resolve_active_annotation_segments(
                    frame_idx=frame_idx,
                    segments=normalized_segments,
                )

                if not active_segments:
                    if buffered_frames:
                        _flush_segment_run(buffered_segments, buffered_frames)
                        buffered_signature = None
                        buffered_segments = []
                        buffered_frames = []
                    _write_result_frame(frame)
                    skipped_frames += 1
                else:
                    active_signature = tuple(
                        f"{str(seg.get('id', ''))}@{int(seg.get('_order', 0))}"
                        for seg in active_segments
                    )
                    if buffered_signature is None:
                        buffered_signature = active_signature
                        buffered_segments = list(active_segments)
                    elif active_signature != buffered_signature:
                        _flush_segment_run(buffered_segments, buffered_frames)
                        buffered_frames = []
                        buffered_signature = active_signature
                        buffered_segments = list(active_segments)
                    buffered_frames.append(frame)

                frame_idx += 1

            if not self._should_stop and buffered_frames:
                _flush_segment_run(buffered_segments, buffered_frames)

            cap.release()
            out.release()

            if self._should_stop:
                shutil.rmtree(temp_dir)
                return {
                    'output_path': '',
                    'requested_model_id': requested_model_id,
                    'effective_model_id': effective_model_id,
                    'model_warning': model_warning,
                    'stopped': True,
                }

            if status_callback:
                status_callback('Finalizing video...')

            self._emit_progress(
                progress_callback=progress_callback,
                progress=0.99,
                message='Finalizing output video...',
                processed_frames=frame_count,
                total_frames=frame_count,
                estimated_time='--:--',
            )

            normalized_output_path = str(Path(output_path).with_suffix('.mp4'))
            if video_info.has_audio:
                final_output = self._merge_audio(temp_output, video_path, normalized_output_path)
            else:
                final_output = self._transcode_video_h264(temp_output, normalized_output_path)

            shutil.rmtree(temp_dir)

            self._emit_progress(
                progress_callback=progress_callback,
                progress=1.0,
                message='Complete!',
                processed_frames=frame_count,
                total_frames=frame_count,
                estimated_time='00:00',
            )

            if status_callback:
                status_callback('Complete!')

            total_observed = max(1, hit_frames + skipped_frames)
            logger.info(
                'Manual annotation stats: segments=%d, hit_frames=%d, skip_frames=%d, hit_ratio=%.2f%%',
                len(normalized_segments),
                hit_frames,
                skipped_frames,
                100.0 * hit_frames / total_observed,
            )
            if requested_model_id == 'lama_roi' or effective_model_id == 'lama_roi':
                seam_delta_before_avg = (
                    lama_seam_delta_before_sum / lama_seam_delta_count
                    if lama_seam_delta_count > 0
                    else 0.0
                )
                seam_delta_after_avg = (
                    lama_seam_delta_after_sum / lama_seam_delta_count
                    if lama_seam_delta_count > 0
                    else 0.0
                )
                transition_band_width_avg = (
                    lama_transition_band_width_sum / lama_transition_band_width_count
                    if lama_transition_band_width_count > 0
                    else 0.0
                )
                logger.info(
                    (
                        'LaMa enhancement stats: hard_cut_resets_total=%d, '
                        'hard_cut_resets_forced=%d, hard_cut_resets_roi=%d, '
                        'cold_start_frames=%d, enhancement_fallbacks=%d, pass2_fallbacks=%d, '
                        'blend_v2_fallbacks=%d, blend_v2_used_frames=%d, '
                        'frame_guard_total=%d, frame_guard_choose_stage2_v2=%d, '
                        'frame_guard_choose_stage1_v2=%d, frame_guard_choose_legacy=%d, '
                        'frame_guard_dark_block_rejects=%d, frame_guard_seam_rejects=%d, '
                        'frame_guard_switch_total=%d, frame_guard_switch_suppressed=%d, '
                        'rescue_attempts=%d, rescue_accepted=%d, rescue_rejected=%d, '
                        'final_micro_smooth_applied=%d, '
                        'seam_delta_before=%.6f, seam_delta_after=%.6f, '
                        'transition_band_width_avg=%.3f'
                    ),
                    lama_hard_cut_resets_total,
                    lama_hard_cut_resets_forced,
                    lama_hard_cut_resets_roi,
                    lama_cold_start_frames,
                    lama_enhancement_fallbacks,
                    lama_pass2_fallbacks,
                    lama_blend_v2_fallbacks,
                    lama_blend_v2_used_frames,
                    lama_frame_guard_total,
                    lama_frame_guard_choose_stage2_v2,
                    lama_frame_guard_choose_stage1_v2,
                    lama_frame_guard_choose_legacy,
                    lama_frame_guard_dark_block_rejects,
                    lama_frame_guard_seam_rejects,
                    lama_frame_guard_switch_total,
                    lama_frame_guard_switch_suppressed,
                    lama_rescue_attempts,
                    lama_rescue_accepted,
                    lama_rescue_rejected,
                    lama_final_micro_smooth_applied,
                    seam_delta_before_avg,
                    seam_delta_after_avg,
                    transition_band_width_avg,
                )
            if requested_model_id == 'propainter_roi' or effective_model_id == 'propainter_roi':
                propainter_seam_delta_before_avg = (
                    propainter_seam_delta_before_sum / propainter_seam_delta_count
                    if propainter_seam_delta_count > 0
                    else 0.0
                )
                propainter_seam_delta_after_avg = (
                    propainter_seam_delta_after_sum / propainter_seam_delta_count
                    if propainter_seam_delta_count > 0
                    else 0.0
                )
                propainter_reappear_score_avg_before = (
                    propainter_reappear_score_before_sum / propainter_reappear_score_count
                    if propainter_reappear_score_count > 0
                    else 0.0
                )
                propainter_reappear_score_avg_after = (
                    propainter_reappear_score_after_sum / propainter_reappear_score_count
                    if propainter_reappear_score_count > 0
                    else 0.0
                )
                propainter_remove_ratio_avg_before = (
                    propainter_remove_ratio_before_sum / propainter_reappear_score_count
                    if propainter_reappear_score_count > 0
                    else 0.0
                )
                propainter_remove_ratio_avg_after = (
                    propainter_remove_ratio_after_sum / propainter_reappear_score_count
                    if propainter_reappear_score_count > 0
                    else 0.0
                )
                propainter_residual_hf_corr_avg_before = (
                    propainter_residual_hf_corr_before_sum / propainter_reappear_score_count
                    if propainter_reappear_score_count > 0
                    else 0.0
                )
                propainter_residual_hf_corr_avg_after = (
                    propainter_residual_hf_corr_after_sum / propainter_reappear_score_count
                    if propainter_reappear_score_count > 0
                    else 0.0
                )
                propainter_under_remove_rate = (
                    float(propainter_under_remove_flags_total) / float(propainter_reappear_score_count)
                    if propainter_reappear_score_count > 0
                    else 0.0
                )
                propainter_seam_p90_before_avg = (
                    propainter_seam_p90_before_sum / propainter_seam_p90_count
                    if propainter_seam_p90_count > 0
                    else 0.0
                )
                propainter_seam_p90_after_avg = (
                    propainter_seam_p90_after_sum / propainter_seam_p90_count
                    if propainter_seam_p90_count > 0
                    else 0.0
                )
                propainter_temporal_jump_avg_before_microfix = (
                    propainter_temporal_jump_before_microfix_sum / propainter_microfix_metric_count
                    if propainter_microfix_metric_count > 0
                    else 0.0
                )
                propainter_temporal_jump_avg_after_microfix = (
                    propainter_temporal_jump_after_microfix_sum / propainter_microfix_metric_count
                    if propainter_microfix_metric_count > 0
                    else 0.0
                )
                propainter_remove_ratio_avg_before_microfix = (
                    propainter_remove_ratio_before_microfix_sum / propainter_microfix_metric_count
                    if propainter_microfix_metric_count > 0
                    else 0.0
                )
                propainter_remove_ratio_avg_after_microfix = (
                    propainter_remove_ratio_after_microfix_sum / propainter_microfix_metric_count
                    if propainter_microfix_metric_count > 0
                    else 0.0
                )
                logger.info(
                    (
                        'ProPainter enhancement stats: '
                        'hard_cut_splits=%d, stabilize_applied_frames=%d, '
                        'v2_used_frames=%d, v2_fallbacks=%d, '
                        'frame_guard_total=%d, choose_raw_v2=%d, choose_stable_v2=%d, '
                        'choose_legacy=%d, frame_guard_dark_block_rejects=%d, '
                        'frame_guard_seam_rejects=%d, reappear_flags_total=%d, '
                        'hysteresis_hold_count=%d, hysteresis_switch_count=%d, '
                        'cut_quarantine_frames=%d, viterbi_switches=%d, '
                        'selection_island_rewrites=%d, '
                        'legacy_blocked_by_guard=%d, burst_fix_attempts=%d, '
                        'burst_fix_accepted_frames=%d, burst_fix_rejected_frames=%d, '
                        'micro_flicker_flags_total=%d, '
                        'micro_burst_fix_attempts=%d, micro_burst_fix_accepted_frames=%d, '
                        'micro_burst_fix_rejected_frames=%d, '
                        'under_remove_flags_total=%d, legacy_allowed_catastrophic_only_count=%d, '
                        'segment_rerun_attempts=%d, segment_rerun_accepted=%d, '
                        'segment_rerun_rejected=%d, '
                        'ring_clone_used_frames=%d, ring_clone_fallbacks=%d, '
                        'rescue_attempts=%d, rescue_accepted=%d, rescue_rejected=%d, '
                        'reappear_score_avg_before=%.6f, reappear_score_avg_after=%.6f, '
                        'remove_ratio_avg_before=%.6f, remove_ratio_avg_after=%.6f, '
                        'remove_ratio_avg_before_microfix=%.6f, '
                        'remove_ratio_avg_after_microfix=%.6f, '
                        'residual_hf_corr_avg_before=%.6f, residual_hf_corr_avg_after=%.6f, '
                        'temporal_jump_avg_before_microfix=%.6f, '
                        'temporal_jump_avg_after_microfix=%.6f, '
                        'under_remove_rate=%.6f, '
                        'seam_delta_before=%.6f, seam_delta_after=%.6f, '
                        'seam_p90_before=%.6f, seam_p90_after=%.6f'
                    ),
                    propainter_hard_cut_splits,
                    propainter_stabilize_applied_frames,
                    propainter_v2_used_frames,
                    propainter_v2_fallbacks,
                    propainter_frame_guard_total,
                    propainter_choose_raw_v2,
                    propainter_choose_stable_v2,
                    propainter_choose_legacy,
                    propainter_frame_guard_dark_block_rejects,
                    propainter_frame_guard_seam_rejects,
                    propainter_reappear_flags_total,
                    propainter_hysteresis_hold_count,
                    propainter_hysteresis_switch_count,
                    propainter_cut_quarantine_frames,
                    propainter_viterbi_switches,
                    propainter_selection_island_rewrites,
                    propainter_legacy_blocked_by_guard,
                    propainter_burst_fix_attempts,
                    propainter_burst_fix_accepted_frames,
                    propainter_burst_fix_rejected_frames,
                    propainter_micro_flicker_flags_total,
                    propainter_micro_burst_fix_attempts,
                    propainter_micro_burst_fix_accepted_frames,
                    propainter_micro_burst_fix_rejected_frames,
                    propainter_under_remove_flags_total,
                    propainter_legacy_catastrophic_only_count,
                    propainter_segment_rerun_attempts,
                    propainter_segment_rerun_accepted,
                    propainter_segment_rerun_rejected,
                    propainter_ring_clone_used_frames,
                    propainter_ring_clone_fallbacks,
                    propainter_rescue_attempts,
                    propainter_rescue_accepted,
                    propainter_rescue_rejected,
                    propainter_reappear_score_avg_before,
                    propainter_reappear_score_avg_after,
                    propainter_remove_ratio_avg_before,
                    propainter_remove_ratio_avg_after,
                    propainter_remove_ratio_avg_before_microfix,
                    propainter_remove_ratio_avg_after_microfix,
                    propainter_residual_hf_corr_avg_before,
                    propainter_residual_hf_corr_avg_after,
                    propainter_temporal_jump_avg_before_microfix,
                    propainter_temporal_jump_avg_after_microfix,
                    propainter_under_remove_rate,
                    propainter_seam_delta_before_avg,
                    propainter_seam_delta_after_avg,
                    propainter_seam_p90_before_avg,
                    propainter_seam_p90_after_avg,
                )

            return {
                'output_path': final_output,
                'requested_model_id': requested_model_id,
                'effective_model_id': effective_model_id,
                'model_warning': model_warning,
                'stopped': False,
                'hit_frames': hit_frames,
                'skip_frames': skipped_frames,
            }

        except Exception as e:
            logger.error(f'Video processing failed: {e}')
            raise
        finally:
            self._is_processing = False
            self._should_stop = False

    def _merge_audio(
        self,
        video_path: str,
        audio_source: str,
        output_path: str,
    ) -> str:
        """把处理后视频与原音频合并并转成标准 H.264/AAC MP4。"""
        ffmpeg_bin = resolve_ffmpeg_path()
        if not ffmpeg_bin:
            raise RuntimeError(
                'FFmpeg runtime not found. Unable to generate required H.264 MP4 output.'
            )

        try:
            cmd = [
                ffmpeg_bin,
                '-y',
                '-i',
                video_path,
                '-i',
                audio_source,
                '-map',
                '0:v:0',
                '-map',
                '1:a:0?',
                '-c:v',
                'libx264',
                '-preset',
                'medium',
                '-pix_fmt',
                'yuv420p',
                '-vf',
                'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                '-movflags',
                '+faststart',
                '-c:a',
                'aac',
                '-b:a',
                '128k',
                '-shortest',
                output_path,
            ]

            subprocess.run(cmd, capture_output=True, check=True, timeout=1200)
            return output_path

        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b'')
            if isinstance(stderr, bytes):
                stderr_text = stderr.decode('utf-8', errors='replace')
            else:
                stderr_text = str(stderr)
            logger.error(f'Failed to render H.264 MP4 with audio: {stderr_text}')
            raise RuntimeError('Failed to encode H.264 MP4 output with audio') from e

    def _transcode_video_h264(self, video_path: str, output_path: str) -> str:
        """无音频场景下，仅转码视频到 H.264 MP4。"""
        ffmpeg_bin = resolve_ffmpeg_path()
        if not ffmpeg_bin:
            raise RuntimeError(
                'FFmpeg runtime not found. Unable to generate required H.264 MP4 output.'
            )

        cmd = [
            ffmpeg_bin,
            '-y',
            '-i',
            video_path,
            '-c:v',
            'libx264',
            '-preset',
            'medium',
            '-pix_fmt',
            'yuv420p',
            '-vf',
            'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-movflags',
            '+faststart',
            '-an',
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=1200)
            return output_path
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b'')
            if isinstance(stderr, bytes):
                stderr_text = stderr.decode('utf-8', errors='replace')
            else:
                stderr_text = str(stderr)
            logger.error(f'Failed to render H.264 MP4 without audio: {stderr_text}')
            raise RuntimeError('Failed to encode H.264 MP4 output') from e

    def stop_processing(self) -> None:
        """请求停止处理（在主循环内生效）。"""
        self._should_stop = True

    def is_processing(self) -> bool:
        """返回当前是否处于处理中。"""
        return self._is_processing
