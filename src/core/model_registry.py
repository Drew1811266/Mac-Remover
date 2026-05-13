"""
模型路由注册器。

作用可以理解为“模型总开关”：
- 根据 `model_id` 返回对应引擎；
- 同时把“请求模型 / 实际生效模型 / 警告信息”打包返回给上层。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..utils.logger import logger
from .remover import WatermarkRemover


SUPPORTED_MODEL_IDS = ("lama_roi",)
LEGACY_MODEL_ALIASES = {
    "sttn_roi": "lama_roi",
}


@dataclass
class ResolvedModel:
    """模型解析结果：前端可据此提示是否发生了回退。"""
    requested_model_id: str
    effective_model_id: str
    warning: str = ""


class BaseInpaintEngine:
    """统一的引擎抽象基类。"""
    model_id = "base"
    display_name = "Base"

    def __init__(self, remover: WatermarkRemover):
        self.remover = remover

    def load(self) -> None:
        raise NotImplementedError

    def inpaint_roi(self, roi: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inpaint_roi_sequence(self, roi_frames, roi_masks, progress_callback=None, **kwargs):
        # 默认实现：逐帧调用单帧接口。时序模型可覆盖这个方法做批量推理。
        return [self.inpaint_roi(frame, mask) for frame, mask in zip(roi_frames, roi_masks)]

    def is_available(self) -> bool:
        return True

    def get_warning(self) -> str:
        return ""


class LamaRoiEngine(BaseInpaintEngine):
    """内置 LaMa 引擎。"""
    model_id = "lama_roi"
    display_name = "LaMa-ROI"

    def load(self) -> None:
        if not self.remover.is_loaded():
            self.remover.load_model()

    def inpaint_roi(self, roi: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        return self.remover.inpaint(roi, roi_mask)

    def inpaint_roi_sequence(self, roi_frames, roi_masks, progress_callback=None, **kwargs):
        """
        LaMa 序列接口：逐帧推理并输出结构化进度。

        进度节流规则：
        - 每 2 帧至少一次；
        - 或每 0.25 秒至少一次；
        - 最后一帧强制发一次 100%。
        """
        frames = list(roi_frames or [])
        masks = list(roi_masks or [])
        total = min(len(frames), len(masks))
        if total <= 0:
            return []

        outputs = []
        last_emit_ts = 0.0
        for idx in range(total):
            outputs.append(self.inpaint_roi(frames[idx], masks[idx]))
            if not progress_callback:
                continue
            now = time.monotonic()
            should_emit = (
                idx == total - 1
                or ((idx + 1) % 2 == 0)
                or (now - last_emit_ts >= 0.25)
            )
            if not should_emit:
                continue
            step = idx + 1
            progress_callback(
                {
                    "phase": "infer",
                    "step": step,
                    "total": total,
                    "progress": float(step) / float(total),
                    "opaque_infer": True,
                    "message": f"LaMa infer {step}/{total}",
                }
            )
            last_emit_ts = now
        return outputs


class ModelRegistry:
    """模型注册与解析入口。"""
    def __init__(self, remover: WatermarkRemover):
        self._remover = remover
        self._engines: Dict[str, BaseInpaintEngine] = {}

    @staticmethod
    def normalize_model_id(model_id: Optional[str]) -> str:
        """标准化模型 ID；未知值回落到默认 `lama_roi`。"""
        raw = str(model_id or "").strip().lower()
        raw = LEGACY_MODEL_ALIASES.get(raw, raw)
        if raw in SUPPORTED_MODEL_IDS:
            return raw
        return "lama_roi"

    def _create_engine(self, model_id: str) -> BaseInpaintEngine:
        """按模型 ID 创建对应引擎实例。"""
        if model_id == "lama_roi":
            return LamaRoiEngine(self._remover)
        return LamaRoiEngine(self._remover)

    def get_engine(self, model_id: str) -> BaseInpaintEngine:
        """获取（或懒创建）引擎实例。"""
        model_id = self.normalize_model_id(model_id)
        engine = self._engines.get(model_id)
        if engine is None:
            engine = self._create_engine(model_id)
            self._engines[model_id] = engine
        return engine

    def resolve(self, requested_model_id: Optional[str]) -> Tuple[BaseInpaintEngine, ResolvedModel]:
        """
        解析最终可用引擎。

        返回：
        - 可直接调用的引擎对象；
        - 解析信息（用于 UI 提示是否发生回退）。
        """
        raw_requested = str(requested_model_id or "").strip().lower()
        requested = self.normalize_model_id(raw_requested)
        engine = self.get_engine(requested)
        warning = ""

        if raw_requested in LEGACY_MODEL_ALIASES:
            warning = (
                f"Model {raw_requested} has been removed. "
                f"Automatically switched to {requested}."
            )

        try:
            engine.load()
        except Exception as exc:
            warning = f"{engine.display_name} load failed: {exc}. Falling back to LaMa-ROI."
            logger.warning(warning)
            fallback = self.get_engine("lama_roi")
            fallback.load()
            return fallback, ResolvedModel(
                requested_model_id=requested,
                effective_model_id="lama_roi",
                warning=warning,
            )

        if not engine.is_available():
            warning = engine.get_warning() or (
                f"{engine.display_name} backend unavailable. Falling back to LaMa-ROI."
            )
            fallback = self.get_engine("lama_roi")
            fallback.load()
            return fallback, ResolvedModel(
                requested_model_id=requested,
                effective_model_id="lama_roi",
                warning=warning,
            )

        return engine, ResolvedModel(
            requested_model_id=requested,
            effective_model_id=requested,
            warning=warning,
        )
