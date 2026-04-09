"""
ProPainter 适配器（真实后端桥接）。

作用：
- 把 ROI 帧与掩码写入临时目录；
- 调用官方 ProPainter 推理脚本；
- 读取结果视频并还原为帧序列。
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from ...utils.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROPAINTER_ROOT = PROJECT_ROOT / "models" / "third_party" / "ProPainter"
PROPAINTER_SCRIPT = PROPAINTER_ROOT / "inference_propainter.py"
PROPAINTER_WEIGHTS = PROPAINTER_ROOT / "weights"
MIN_PROCESS_SIDE = 128
REQUIRED_WEIGHT_FILES = (
    "raft-things.pth",
    "recurrent_flow_completion.pth",
    "ProPainter.pth",
)
DEFAULT_INFER_OPTIONS = {
    "mask_dilation": 4,
    "neighbor_length": 10,
    "ref_stride": 10,
    "subvideo_length": 80,
    "raft_iter": 20,
    "save_fps": 24,
}
PROPAINTER_RATIO_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
PROPAINTER_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)%")


class Adapter:
    """ProPainter 模型适配器实现。"""
    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        """加载前检查：脚本存在、权重目录可用。"""
        if self._loaded:
            return
        if not PROPAINTER_ROOT.exists():
            raise RuntimeError(
                f"ProPainter source not found at {PROPAINTER_ROOT}. "
                "Run scripts/download_models.py --propainter to deploy ProPainter."
            )
        if not PROPAINTER_SCRIPT.exists():
            raise RuntimeError(f"ProPainter inference script missing: {PROPAINTER_SCRIPT}")

        # We don't hard-fail if weights are missing because ProPainter can auto-download,
        # but we keep this flag for diagnostics and deployment checks.
        missing = [name for name in REQUIRED_WEIGHT_FILES if not (PROPAINTER_WEIGHTS / name).exists()]
        if missing and not PROPAINTER_WEIGHTS.exists():
            raise RuntimeError(
                f"ProPainter weights directory missing: {PROPAINTER_WEIGHTS}. "
                "Run scripts/download_models.py --propainter to deploy local weights."
            )
        self._loaded = True

    @staticmethod
    def _normalize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """统一掩码到目标尺寸的 0/255 单通道格式。"""
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    @staticmethod
    def _read_video_frames(video_path: Path, expected_count: int) -> List[np.ndarray]:
        """读取输出视频并修正帧数（补齐或裁剪）。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open ProPainter output video: {video_path}")

        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"ProPainter produced empty output video: {video_path}")

        if len(frames) < expected_count:
            frames.extend([frames[-1].copy() for _ in range(expected_count - len(frames))])
        elif len(frames) > expected_count:
            frames = frames[:expected_count]
        return frames

    @staticmethod
    def _normalize_infer_options(options: dict | None) -> dict:
        """清洗 ProPainter 内部推理参数，避免传入非法值。"""
        merged = dict(DEFAULT_INFER_OPTIONS)
        if isinstance(options, dict):
            merged.update(options)

        def _int_opt(name: str, minimum: int, maximum: int) -> int:
            try:
                value = int(merged.get(name, DEFAULT_INFER_OPTIONS[name]))
            except (TypeError, ValueError):
                value = int(DEFAULT_INFER_OPTIONS[name])
            return max(minimum, min(maximum, value))

        return {
            "mask_dilation": _int_opt("mask_dilation", 0, 48),
            "neighbor_length": _int_opt("neighbor_length", 2, 24),
            "ref_stride": _int_opt("ref_stride", 2, 24),
            "subvideo_length": _int_opt("subvideo_length", 20, 240),
            "raft_iter": _int_opt("raft_iter", 8, 48),
            "save_fps": _int_opt("save_fps", 1, 120),
        }

    @staticmethod
    def _parse_progress_line(line: str) -> Optional[Dict[str, float]]:
        """
        解析 ProPainter 日志中的进度片段。

        兼容 tqdm 常见格式：
        - `12/40`
        - `30.5%`
        """
        text = str(line or "").strip()
        if not text:
            return None

        ratio_match = PROPAINTER_RATIO_RE.search(text)
        if ratio_match:
            try:
                step = int(ratio_match.group(1))
                total = max(1, int(ratio_match.group(2)))
                step = min(max(0, step), total)
                return {
                    "step": float(step),
                    "total": float(total),
                    "progress": float(step) / float(total),
                }
            except (TypeError, ValueError):
                pass

        percent_match = PROPAINTER_PERCENT_RE.search(text)
        if percent_match:
            try:
                ratio = max(0.0, min(100.0, float(percent_match.group(1)))) / 100.0
                return {"progress": ratio}
            except (TypeError, ValueError):
                return None
        return None

    def inpaint_roi_sequence(
        self,
        roi_frames: Iterable[np.ndarray],
        roi_masks: Iterable[np.ndarray],
        progress_callback=None,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        ProPainter 时序修复主入口。

        注意：
        - 为兼容模型输入，ROI 会最小放大到 `MIN_PROCESS_SIDE`；
        - 推理后再缩回原 ROI 尺寸。
        """
        self.load()

        frames = [np.asarray(frame, dtype=np.uint8) for frame in roi_frames]
        masks = [np.asarray(mask) for mask in roi_masks]
        if not frames:
            return []
        if len(frames) != len(masks):
            raise RuntimeError("ProPainter adapter expects roi_frames and roi_masks with equal length")

        h, w = frames[0].shape[:2]
        for frame in frames:
            if frame.shape[:2] != (h, w):
                raise RuntimeError(
                    "ProPainter adapter requires consistent ROI shape within one segment"
                )
        infer_options = self._normalize_infer_options(kwargs.get("propainter_options"))

        temp_dir = Path(tempfile.mkdtemp(prefix="wmr_propainter_"))
        try:
            frame_dir = temp_dir / "frames"
            mask_dir = temp_dir / "masks"
            out_root = temp_dir / "out"
            frame_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            out_root.mkdir(parents=True, exist_ok=True)

            process_w = max(w, MIN_PROCESS_SIDE)
            process_h = max(h, MIN_PROCESS_SIDE)
            if process_w % 2:
                process_w += 1
            if process_h % 2:
                process_h += 1

            for idx, mask in enumerate(masks):
                frame = frames[idx]
                if (process_h, process_w) != (h, w):
                    frame = cv2.resize(frame, (process_w, process_h), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(frame_dir / f"{idx:06d}.png"), frame)

                normalized_mask = self._normalize_mask(mask, (h, w))
                if (process_h, process_w) != (h, w):
                    normalized_mask = cv2.resize(
                        normalized_mask,
                        (process_w, process_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                cv2.imwrite(str(mask_dir / f"{idx:06d}.png"), normalized_mask)

            cmd = [
                sys.executable,
                str(PROPAINTER_SCRIPT),
                "--video",
                str(frame_dir),
                "--mask",
                str(mask_dir),
                "--output",
                str(out_root),
                "--width",
                str(process_w),
                "--height",
                str(process_h),
                "--mask_dilation",
                str(infer_options["mask_dilation"]),
                "--neighbor_length",
                str(infer_options["neighbor_length"]),
                "--ref_stride",
                str(infer_options["ref_stride"]),
                "--subvideo_length",
                str(infer_options["subvideo_length"]),
                "--raft_iter",
                str(infer_options["raft_iter"]),
                "--save_fps",
                str(infer_options["save_fps"]),
            ]
            env = os.environ.copy()
            timeout_sec = max(600, len(frames) * 12)
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROPAINTER_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            stdout_lines: List[str] = []
            stderr_lines: List[str] = []
            stream_events: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()

            def _reader(stream, stream_name: str, sink: List[str]) -> None:
                try:
                    while True:
                        line = stream.readline()
                        if not line:
                            break
                        sink.append(line.rstrip("\n"))
                        stream_events.put((stream_name, line))
                finally:
                    stream_events.put((stream_name, None))

            stdout_thread = threading.Thread(
                target=_reader,
                args=(proc.stdout, "stdout", stdout_lines),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_reader,
                args=(proc.stderr, "stderr", stderr_lines),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            streams_closed = 0
            infer_progress = 0.0
            saw_structured_progress = False
            start_ts = time.monotonic()
            last_heartbeat_ts = start_ts
            while True:
                if time.monotonic() - start_ts > timeout_sec:
                    proc.kill()
                    raise RuntimeError(f"ProPainter inference timeout after {timeout_sec}s")

                try:
                    stream_name, line = stream_events.get(timeout=0.2)
                except queue.Empty:
                    stream_name, line = "", ""

                now = time.monotonic()
                if line is None and stream_name:
                    streams_closed += 1
                elif line:
                    text = str(line).strip()
                    if text:
                        parsed = self._parse_progress_line(text)
                        if parsed is not None and progress_callback:
                            saw_structured_progress = True
                            if parsed.get("step") is not None and parsed.get("total") is not None:
                                step_value = int(parsed["step"])
                                total_value = max(1, int(parsed["total"]))
                                infer_progress = max(
                                    infer_progress,
                                    min(1.0, max(0.0, float(parsed.get("progress", 0.0)))),
                                )
                                progress_callback(
                                    {
                                        "phase": "infer",
                                        "step": step_value,
                                        "total": total_value,
                                        "progress": infer_progress,
                                        "opaque_infer": True,
                                        "message": f"ProPainter infer {step_value}/{total_value}",
                                    }
                                )
                            else:
                                infer_progress = max(
                                    infer_progress,
                                    min(1.0, max(0.0, float(parsed.get("progress", 0.0)))),
                                )
                                progress_callback(
                                    {
                                        "phase": "infer",
                                        "progress": infer_progress,
                                        "opaque_infer": True,
                                        "message": f"ProPainter infer {int(round(infer_progress * 100))}%",
                                    }
                                )
                should_emit_heartbeat = progress_callback and (now - last_heartbeat_ts >= 1.0)
                if should_emit_heartbeat:
                    if not saw_structured_progress:
                        infer_progress = min(0.96, infer_progress + 0.005)
                    progress_callback(
                        {
                            "phase": "infer",
                            "progress": infer_progress,
                            "opaque_infer": True,
                            "message": "ProPainter infer heartbeat (estimated)",
                        }
                    )
                    last_heartbeat_ts = now

                if proc.poll() is not None and streams_closed >= 2:
                    break

            stdout_thread.join(timeout=0.5)
            stderr_thread.join(timeout=0.5)

            if proc.returncode != 0:
                stderr_tail = "\n".join(stderr_lines[-20:]).strip()
                stdout_tail = "\n".join(stdout_lines[-20:]).strip()
                raise RuntimeError(
                    f"ProPainter inference failed (code {proc.returncode}): "
                    f"{stderr_tail or stdout_tail}"
                )

            output_video = out_root / frame_dir.name / "inpaint_out.mp4"
            if not output_video.exists():
                stdout_tail = "\n".join(stdout_lines[-20:]).strip()
                raise RuntimeError(
                    f"ProPainter output file missing: {output_video}. Logs: {stdout_tail}"
                )

            out_frames = self._read_video_frames(output_video, len(frames))
            if progress_callback:
                progress_callback(
                    {
                        "phase": "infer",
                        "step": len(frames),
                        "total": len(frames),
                        "progress": 1.0,
                        "opaque_infer": True,
                        "message": f"ProPainter infer {len(frames)}/{len(frames)}",
                    }
                )
            resized: List[np.ndarray] = []
            for frame in out_frames:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
                resized.append(frame)
            if stdout_lines:
                logger.debug("ProPainter stdout tail: %s", " | ".join(stdout_lines[-5:]))
            return resized
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def inpaint_roi(self, roi: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """单帧接口：复用序列接口。"""
        result = self.inpaint_roi_sequence([roi], [roi_mask])
        if not result:
            raise RuntimeError("ProPainter adapter returned empty frame")
        return result[0]
