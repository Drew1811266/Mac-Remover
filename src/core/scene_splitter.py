"""
智能镜头分切模块（稳定优先）。

策略：
1) Layer A: FFmpeg scene score 快速候选；
2) Layer B: PySceneDetect 自适应复核；
3) Layer C: 仅在低置信度时使用 TransNetV2（默认 CPU）补充复核。

最终输出稳定的 segments 列表，并内置短段合并、长段拆分与固定时长兜底。
"""

from __future__ import annotations

import math
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


SCENE_SCORE_THRESHOLD = 0.26
ADAPTIVE_THRESHOLD = 3.2
MIN_SCENE_SEC = 1.2
SECONDARY_SHORT_MERGE_SEC = 2.0
MAX_SCENE_SEC = 4.0
FALLBACK_SEGMENT_SEC = 3.0
MIN_SCENE_FRAMES = 12
MERGE_TOLERANCE_SEC = 0.20


class SceneSplitCancelled(RuntimeError):
    """镜头分切被取消。"""


@dataclass(frozen=True)
class SceneSegment:
    idx: int
    start: float
    end: float
    duration: float


@dataclass(frozen=True)
class SceneSplitResult:
    segments: Tuple[SceneSegment, ...]
    split_mode: str
    warnings: Tuple[str, ...]
    cuts: Tuple[float, ...]
    stats: Dict[str, Any]


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


def _clamp_time(value: float, *, duration_sec: float) -> float:
    return max(0.0, min(float(duration_sec), float(value)))


def _merge_cuts(cuts: Iterable[float], *, tolerance: float = MERGE_TOLERANCE_SEC) -> List[float]:
    ordered = sorted([max(0.0, float(item)) for item in cuts if float(item) >= 0.0])
    if not ordered:
        return []
    merged: List[float] = [ordered[0]]
    for value in ordered[1:]:
        if abs(value - merged[-1]) <= tolerance:
            merged[-1] = (merged[-1] + value) / 2.0
        else:
            merged.append(value)
    return merged


def _timecode_to_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    getter = getattr(value, "get_seconds", None)
    if callable(getter):
        try:
            return max(0.0, float(getter()))
        except Exception:
            return 0.0
    try:
        return max(0.0, float(value))
    except Exception:
        return 0.0


class SceneSplitter:
    """基于规则 + 轻量模型复核的镜头分切器。"""

    def __init__(self, *, ffmpeg_bin: str) -> None:
        self.ffmpeg_bin = str(ffmpeg_bin)

    def split(
        self,
        *,
        input_path: str,
        duration_sec: float,
        fps: float,
        cancel_event: Optional[Any] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> SceneSplitResult:
        duration_sec = max(0.1, float(duration_sec))
        fps = max(1.0, float(fps))
        warnings: List[str] = []
        split_mode = "rule"

        _safe_emit(
            progress_callback,
            {
                "phase": "prepare",
                "progress": 0.02,
                "message": "Scene split: collecting FFmpeg candidates...",
            },
        )
        ffmpeg_cuts = self._detect_ffmpeg_cuts(
            input_path=input_path,
            duration_sec=duration_sec,
            cancel_event=cancel_event,
        )

        _safe_emit(
            progress_callback,
            {
                "phase": "prepare",
                "progress": 0.07,
                "message": "Scene split: running adaptive rule review...",
            },
        )
        pyscene_cuts, pyscene_warning = self._detect_pyscenedetect_cuts(
            input_path=input_path,
            fps=fps,
            cancel_event=cancel_event,
        )
        if pyscene_warning:
            warnings.append(pyscene_warning)

        rule_cuts = self._merge_rule_cuts(
            ffmpeg_cuts=ffmpeg_cuts,
            pyscene_cuts=pyscene_cuts,
            duration_sec=duration_sec,
        )
        rough_segments = self._to_segments(
            cuts=rule_cuts,
            duration_sec=duration_sec,
            normalize=False,
        )

        need_transnet, trigger_reason = self._should_trigger_transnet(
            ffmpeg_cuts=ffmpeg_cuts,
            pyscene_cuts=pyscene_cuts,
            rough_segments=rough_segments,
            duration_sec=duration_sec,
        )
        transnet_cuts: List[float] = []

        if need_transnet:
            _safe_emit(
                progress_callback,
                {
                    "phase": "prepare",
                    "progress": 0.11,
                    "message": "Scene split: low confidence detected, running TransNetV2 review...",
                },
            )
            transnet_cuts, transnet_warning = self._detect_transnet_cuts(
                input_path=input_path,
                fps=fps,
                cancel_event=cancel_event,
            )
            if transnet_warning:
                warnings.append(transnet_warning)
            if transnet_cuts:
                split_mode = "hybrid"
                rule_cuts = _merge_cuts(rule_cuts + transnet_cuts, tolerance=MERGE_TOLERANCE_SEC)
            else:
                split_mode = "rule"
                warnings.append(f"Scene split review fallback: {trigger_reason}.")

        normalized = self._to_segments(
            cuts=rule_cuts,
            duration_sec=duration_sec,
            normalize=True,
        )
        short_merge_count = max(
            0,
            len([seg for seg in rough_segments if seg.duration < SECONDARY_SHORT_MERGE_SEC])
            - len([seg for seg in normalized if seg.duration < SECONDARY_SHORT_MERGE_SEC]),
        )

        if not rule_cuts:
            split_mode = "fallback"
            warnings.append("Scene split fallback: fixed 3.0s segmentation applied.")
            normalized = self._fallback_segments(duration_sec=duration_sec)
        elif not normalized:
            split_mode = "fallback"
            warnings.append("Scene split fallback: fixed 3.0s segmentation applied.")
            normalized = self._fallback_segments(duration_sec=duration_sec)

        _safe_emit(
            progress_callback,
            {
                "phase": "prepare",
                "progress": 0.15,
                "message": f"Scene split ready: {len(normalized)} segment(s).",
            },
        )

        return SceneSplitResult(
            segments=tuple(normalized),
            split_mode=split_mode,
            warnings=tuple(warnings),
            cuts=tuple(rule_cuts),
            stats={
                "ffmpeg_cut_count": len(ffmpeg_cuts),
                "pyscene_cut_count": len(pyscene_cuts),
                "transnet_cut_count": len(transnet_cuts),
                "duration_sec": float(duration_sec),
                "short_merge_count": int(short_merge_count),
            },
        )

    def _run_capture(
        self,
        cmd: Sequence[str],
        *,
        timeout_sec: float,
        cancel_event: Optional[Any],
    ) -> Tuple[str, str, int]:
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        started = time.time()
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()
                    raise SceneSplitCancelled("Scene split cancelled")
                if (time.time() - started) > timeout_sec:
                    proc.kill()
                    raise RuntimeError(f"Scene split command timeout: {' '.join(cmd)}")
                if proc.poll() is not None:
                    break
                time.sleep(0.08)
            stdout, stderr = proc.communicate(timeout=5)
            return str(stdout or ""), str(stderr or ""), int(proc.returncode or 0)
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

    def _detect_ffmpeg_cuts(
        self,
        *,
        input_path: str,
        duration_sec: float,
        cancel_event: Optional[Any],
    ) -> List[float]:
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-i",
            input_path,
            "-filter:v",
            f"select='gt(scene,{SCENE_SCORE_THRESHOLD})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
        timeout = max(20.0, float(duration_sec) * 1.4 + 20.0)
        stdout, stderr, _ = self._run_capture(cmd, timeout_sec=timeout, cancel_event=cancel_event)
        content = f"{stdout}\n{stderr}"
        pattern = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
        cuts = [float(match.group(1)) for match in pattern.finditer(content)]
        return _merge_cuts(cuts, tolerance=MERGE_TOLERANCE_SEC)

    def _detect_pyscenedetect_cuts(
        self,
        *,
        input_path: str,
        fps: float,
        cancel_event: Optional[Any],
    ) -> Tuple[List[float], str]:
        if cancel_event is not None and cancel_event.is_set():
            raise SceneSplitCancelled("Scene split cancelled")
        try:
            from scenedetect import detect
            from scenedetect.detectors import AdaptiveDetector, ContentDetector
        except Exception:
            return [], "PySceneDetect unavailable; using FFmpeg rule-only scene split."

        min_scene_len = max(int(MIN_SCENE_FRAMES), int(round(float(fps) * 0.5)))
        cuts: List[float] = []
        warning = ""
        try:
            adaptive_scenes = detect(
                input_path,
                AdaptiveDetector(
                    adaptive_threshold=ADAPTIVE_THRESHOLD,
                    min_scene_len=min_scene_len,
                ),
            )
            content_scenes = detect(
                input_path,
                ContentDetector(min_scene_len=min_scene_len),
            )
            cuts.extend(self._cuts_from_scene_list(adaptive_scenes))
            cuts.extend(self._cuts_from_scene_list(content_scenes))
        except Exception as exc:
            warning = f"PySceneDetect failed ({exc}); falling back to FFmpeg rule cuts."
            cuts = []
        return _merge_cuts(cuts, tolerance=MERGE_TOLERANCE_SEC), warning

    @staticmethod
    def _cuts_from_scene_list(scene_list: Any) -> List[float]:
        cuts: List[float] = []
        if not scene_list:
            return cuts
        for idx, pair in enumerate(scene_list):
            if idx == 0:
                continue
            try:
                start = pair[0]
            except Exception:
                continue
            sec = _timecode_to_seconds(start)
            if sec > 0:
                cuts.append(sec)
        return cuts

    def _should_trigger_transnet(
        self,
        *,
        ffmpeg_cuts: Sequence[float],
        pyscene_cuts: Sequence[float],
        rough_segments: Sequence[SceneSegment],
        duration_sec: float,
    ) -> Tuple[bool, str]:
        if abs(len(ffmpeg_cuts) - len(pyscene_cuts)) >= 3:
            return True, "ffmpeg/pyscene cut count diverged"

        short_ratio = 0.0
        if rough_segments:
            short_cnt = len([seg for seg in rough_segments if seg.duration < MIN_SCENE_SEC])
            short_ratio = float(short_cnt) / float(len(rough_segments))
        if short_ratio > 0.30:
            return True, "too many short fragments"

        if float(duration_sec) > 8.0 and any(seg.duration > MAX_SCENE_SEC for seg in rough_segments):
            return True, "long scene remained in >8s video"

        merged = _merge_cuts(list(ffmpeg_cuts) + list(pyscene_cuts), tolerance=MERGE_TOLERANCE_SEC)
        dense_ratio = 0.0
        if len(merged) >= 2:
            dense = 0
            for idx in range(len(merged) - 1):
                if (merged[idx + 1] - merged[idx]) < 0.2:
                    dense += 1
            dense_ratio = float(dense) / float(max(1, len(merged) - 1))
        if dense_ratio > 0.30:
            return True, "adjacent scene cuts are overly dense"

        return False, ""

    def _detect_transnet_cuts(
        self,
        *,
        input_path: str,
        fps: float,
        cancel_event: Optional[Any],
    ) -> Tuple[List[float], str]:
        if cancel_event is not None and cancel_event.is_set():
            raise SceneSplitCancelled("Scene split cancelled")
        try:
            import numpy as np
            from transnetv2_pytorch import TransNetV2
        except Exception:
            return [], "TransNetV2 unavailable; skipped low-confidence review."

        try:
            model = TransNetV2(device="cpu")
        except Exception:
            try:
                model = TransNetV2()
            except Exception as exc:
                return [], f"TransNetV2 init failed ({exc}); skipped low-confidence review."

        try:
            prediction = model.predict_video(input_path)
        except Exception as exc:
            return [], f"TransNetV2 predict failed ({exc}); skipped low-confidence review."

        scenes: Any = None
        prediction_arr: Any = prediction
        if isinstance(prediction, tuple) and prediction:
            prediction_arr = prediction[0]

        if hasattr(model, "predictions_to_scenes"):
            try:
                scenes = model.predictions_to_scenes(prediction_arr)
            except Exception:
                scenes = None

        if scenes is None:
            # 兜底：把概率峰值阈值化为切点，尽量避免完全放弃复核。
            try:
                prob = np.asarray(prediction_arr).astype("float32").reshape(-1)
                frame_ids = np.where(prob >= 0.55)[0].tolist()
                if not frame_ids:
                    return [], "TransNetV2 returned no stable cuts."
                grouped: List[List[int]] = []
                for frame_id in frame_ids:
                    if not grouped or frame_id - grouped[-1][-1] > 3:
                        grouped.append([int(frame_id)])
                    else:
                        grouped[-1].append(int(frame_id))
                cuts = [float(int(sum(group) / len(group))) / float(max(1.0, fps)) for group in grouped]
                return _merge_cuts(cuts, tolerance=MERGE_TOLERANCE_SEC), ""
            except Exception:
                return [], "TransNetV2 output parse failed; skipped low-confidence review."

        cuts: List[float] = []
        try:
            for idx in range(1, len(scenes)):
                start_frame = int(scenes[idx][0])
                cuts.append(float(start_frame) / float(max(1.0, fps)))
        except Exception:
            return [], "TransNetV2 scene conversion failed; skipped low-confidence review."
        return _merge_cuts(cuts, tolerance=MERGE_TOLERANCE_SEC), ""

    def _merge_rule_cuts(
        self,
        *,
        ffmpeg_cuts: Sequence[float],
        pyscene_cuts: Sequence[float],
        duration_sec: float,
    ) -> List[float]:
        ffmpeg_cuts = _merge_cuts(ffmpeg_cuts)
        pyscene_cuts = _merge_cuts(pyscene_cuts)
        if not ffmpeg_cuts and not pyscene_cuts:
            return []
        if not pyscene_cuts:
            return [c for c in ffmpeg_cuts if 0.0 < c < duration_sec]
        if not ffmpeg_cuts:
            return [c for c in pyscene_cuts if 0.0 < c < duration_sec]

        # 规则融合：优先保留 PySceneDetect 结果，并吸收附近 FFmpeg 候选。
        merged: List[float] = list(pyscene_cuts)
        for cut in ffmpeg_cuts:
            near = any(abs(cut - ref) <= 0.35 for ref in pyscene_cuts)
            if near:
                merged.append(cut)
        merged = _merge_cuts(merged, tolerance=MERGE_TOLERANCE_SEC)
        return [c for c in merged if 0.0 < c < duration_sec]

    def _to_segments(
        self,
        *,
        cuts: Sequence[float],
        duration_sec: float,
        normalize: bool,
    ) -> List[SceneSegment]:
        duration_sec = max(0.1, float(duration_sec))
        boundaries = [0.0]
        boundaries.extend([_clamp_time(cut, duration_sec=duration_sec) for cut in cuts if 0.0 < cut < duration_sec])
        boundaries.append(duration_sec)
        boundaries = _merge_cuts(boundaries, tolerance=0.02)
        if not boundaries:
            boundaries = [0.0, duration_sec]
        if boundaries[0] > 0.0:
            boundaries.insert(0, 0.0)
        if boundaries[-1] < duration_sec:
            boundaries.append(duration_sec)

        segments: List[List[float]] = []
        for idx in range(len(boundaries) - 1):
            start = float(boundaries[idx])
            end = float(boundaries[idx + 1])
            if end - start <= 0.03:
                continue
            segments.append([start, end])

        if not segments:
            return []

        if normalize:
            segments = self._merge_short_segments(segments, min_duration=MIN_SCENE_SEC)
            segments = self._merge_short_segments(segments, min_duration=SECONDARY_SHORT_MERGE_SEC)
            segments = self._split_long_segments(segments)

        result: List[SceneSegment] = []
        for idx, (start, end) in enumerate(segments, start=1):
            s = _clamp_time(start, duration_sec=duration_sec)
            e = _clamp_time(end, duration_sec=duration_sec)
            if e - s <= 0.03:
                continue
            result.append(
                SceneSegment(
                    idx=idx,
                    start=round(s, 3),
                    end=round(e, 3),
                    duration=round(max(0.0, e - s), 3),
                )
            )
        return result

    def _merge_short_segments(self, segments: List[List[float]], *, min_duration: float) -> List[List[float]]:
        threshold = max(0.1, float(min_duration))
        merged = [list(item) for item in segments]
        i = 0
        while i < len(merged):
            start, end = merged[i]
            duration = end - start
            if duration >= threshold or len(merged) == 1:
                i += 1
                continue

            if i == 0:
                merged[1][0] = merged[0][0]
                del merged[0]
                continue
            if i == len(merged) - 1:
                merged[i - 1][1] = merged[i][1]
                del merged[i]
                i = max(0, i - 1)
                continue

            left_dur = merged[i - 1][1] - merged[i - 1][0]
            right_dur = merged[i + 1][1] - merged[i + 1][0]
            if left_dur <= right_dur:
                merged[i - 1][1] = merged[i][1]
            else:
                merged[i + 1][0] = merged[i][0]
            del merged[i]
            i = max(0, i - 1)
        return merged

    def _split_long_segments(self, segments: List[List[float]]) -> List[List[float]]:
        split: List[List[float]] = []
        for start, end in segments:
            duration = float(end - start)
            if duration <= MAX_SCENE_SEC:
                split.append([start, end])
                continue
            piece_count = int(max(2, math.ceil(duration / MAX_SCENE_SEC)))
            piece = duration / float(piece_count)
            for idx in range(piece_count):
                piece_start = start + float(idx) * piece
                piece_end = end if idx == piece_count - 1 else start + float(idx + 1) * piece
                if piece_end - piece_start <= 0.03:
                    continue
                split.append([piece_start, piece_end])
        return split

    def _fallback_segments(self, *, duration_sec: float) -> List[SceneSegment]:
        duration_sec = max(0.1, float(duration_sec))
        segments: List[SceneSegment] = []
        cursor = 0.0
        idx = 1
        while cursor < duration_sec - 0.02:
            end = min(duration_sec, cursor + FALLBACK_SEGMENT_SEC)
            segments.append(
                SceneSegment(
                    idx=idx,
                    start=round(cursor, 3),
                    end=round(end, 3),
                    duration=round(max(0.0, end - cursor), 3),
                )
            )
            idx += 1
            cursor = end
        return segments


__all__ = [
    "FALLBACK_SEGMENT_SEC",
    "MAX_SCENE_SEC",
    "MIN_SCENE_SEC",
    "SECONDARY_SHORT_MERGE_SEC",
    "SceneSegment",
    "SceneSplitCancelled",
    "SceneSplitResult",
    "SceneSplitter",
]
