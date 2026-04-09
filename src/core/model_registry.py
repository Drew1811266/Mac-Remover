"""
模型路由注册器。

作用可以理解为“模型总开关”：
- 根据 `model_id` 返回对应引擎；
- ProPainter 不可用时自动回退到 LaMa；
- 同时把“请求模型 / 实际生效模型 / 警告信息”打包返回给上层。
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.logger import logger
from .remover import WatermarkRemover


SUPPORTED_MODEL_IDS = ("lama_roi", "propainter_roi")
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


class OptionalAdapterEngine(BaseInpaintEngine):
    """
    Adapter engine for optional third-party models.

    If the adapter module cannot be imported or fails at runtime,
    this engine reports unavailable and callers can fall back to LaMa.
    """

    adapter_module = ""

    def __init__(self, remover: WatermarkRemover):
        super().__init__(remover)
        self._adapter = None
        self._available = False
        self._warning = ""

    def load(self) -> None:
        # 仅首次加载；加载失败后写 warning，交给上层做回退。
        if self._available:
            return

        if self._adapter is not None:
            return

        try:
            module = importlib.import_module(self.adapter_module)
            adapter_cls = getattr(module, "Adapter", None)
            if adapter_cls is None:
                self._warning = (
                    f"{self.display_name} adapter missing Adapter class. "
                    "Falling back to LaMa-ROI."
                )
                logger.warning(self._warning)
                self._adapter = False
                return
            self._adapter = adapter_cls()
            if hasattr(self._adapter, "load"):
                self._adapter.load()
            self._available = True
        except Exception as exc:
            self._warning = (
                f"{self.display_name} backend unavailable: {exc}. "
                "Falling back to LaMa-ROI."
            )
            logger.warning(self._warning)
            self._adapter = False
            self._available = False

    def inpaint_roi(self, roi: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        # 运行期失败也会降级：标记不可用并抛出异常让上层切换 LaMa。
        if not self._available or not self._adapter:
            raise RuntimeError(f"{self.display_name} backend not available")
        try:
            return self._adapter.inpaint_roi(roi, roi_mask)
        except Exception as exc:
            self._available = False
            self._warning = (
                f"{self.display_name} inference failed: {exc}. "
                "Falling back to LaMa-ROI."
            )
            logger.warning(self._warning)
            raise

    def inpaint_roi_sequence(self, roi_frames, roi_masks, progress_callback=None, **kwargs):
        if not self._available or not self._adapter:
            raise RuntimeError(f"{self.display_name} backend not available")

        def _normalize_progress_payload(*cb_args: Any, **cb_kwargs: Any) -> Optional[Dict[str, Any]]:
            raw = cb_args[0] if cb_args else cb_kwargs.get("progress")
            if isinstance(raw, dict):
                payload: Dict[str, Any] = dict(raw)
                payload["phase"] = str(payload.get("phase") or "infer").strip().lower() or "infer"
                payload["opaque_infer"] = bool(payload.get("opaque_infer", payload["phase"] == "infer"))
                return payload

            step_raw: Any = None
            total_raw: Any = None
            if len(cb_args) >= 2:
                step_raw, total_raw = cb_args[0], cb_args[1]
            elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
                step_raw, total_raw = raw[0], raw[1]
            elif cb_kwargs.get("step") is not None and cb_kwargs.get("total") is not None:
                step_raw, total_raw = cb_kwargs.get("step"), cb_kwargs.get("total")

            if step_raw is not None and total_raw is not None:
                try:
                    total = max(1, int(total_raw))
                    step = min(max(0, int(step_raw)), total)
                    return {
                        "phase": "infer",
                        "step": step,
                        "total": total,
                        "progress": float(step) / float(total),
                        "opaque_infer": True,
                        "message": f"{self.display_name} infer {step}/{total}",
                    }
                except (TypeError, ValueError):
                    return None

            candidate = cb_kwargs.get("progress", raw)
            try:
                ratio = float(candidate)
            except (TypeError, ValueError):
                ratio = None
            if ratio is None:
                return None

            ratio = min(max(ratio, 0.0), 1.0)
            return {
                "phase": "infer",
                "progress": ratio,
                "opaque_infer": True,
                "message": f"{self.display_name} infer {int(round(ratio * 100))}%",
            }

        def _bridge_progress_callback(*cb_args: Any, **cb_kwargs: Any) -> None:
            if not progress_callback:
                return
            payload = _normalize_progress_payload(*cb_args, **cb_kwargs)
            if payload is None:
                return
            try:
                progress_callback(payload)
            except TypeError:
                # 兼容极少数 adapter 仍使用(step, total) 形式。
                step = payload.get("step")
                total = payload.get("total")
                if step is not None and total is not None:
                    progress_callback(step, total)

        try:
            if hasattr(self._adapter, "inpaint_roi_sequence"):
                try:
                    return self._adapter.inpaint_roi_sequence(
                        roi_frames,
                        roi_masks,
                        progress_callback=_bridge_progress_callback if progress_callback else None,
                        **kwargs,
                    )
                except TypeError:
                    return self._adapter.inpaint_roi_sequence(roi_frames, roi_masks)
            return super().inpaint_roi_sequence(
                roi_frames,
                roi_masks,
                progress_callback=progress_callback,
                **kwargs,
            )
        except Exception as exc:
            self._available = False
            self._warning = (
                f"{self.display_name} inference failed: {exc}. "
                "Falling back to LaMa-ROI."
            )
            logger.warning(self._warning)
            raise

    def is_available(self) -> bool:
        return self._available

    def get_warning(self) -> str:
        return self._warning


class ProPainterRoiEngine(OptionalAdapterEngine):
    """ProPainter 适配引擎。"""
    model_id = "propainter_roi"
    display_name = "ProPainter-ROI"
    adapter_module = "src.core.optional_adapters.propainter_adapter"


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
        if model_id == "propainter_roi":
            return ProPainterRoiEngine(self._remover)
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
