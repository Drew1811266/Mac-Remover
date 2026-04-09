"""
人工打标 sidecar（旁路标注文件）工具。

这个文件负责：
- 根据视频路径计算 sidecar 路径；
- 读取视频基础信息并生成“文件指纹”；
- 规范化标记段数据（坐标、帧区间、默认值）；
- 读写 `.wmr.json` 标注文件。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


SIDECAR_SUFFIX = ".wmr.json"


def _now_iso() -> str:
    """返回当前 UTC 时间（ISO 格式）。"""
    return datetime.now(timezone.utc).isoformat()


def build_sidecar_path(video_path: str) -> Path:
    """根据视频路径生成 sidecar 文件路径。"""
    p = Path(video_path)
    return p.with_name(f"{p.name}{SIDECAR_SUFFIX}")


def file_sha1(path: str, chunk_size: int = 1024 * 1024) -> str:
    """按块计算文件 SHA1，避免一次性读入大文件。"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_video_meta(path: str) -> Dict[str, Any]:
    """
    读取视频基础元数据并附带文件指纹。

    失败场景：
    - 视频无法被 OpenCV 打开时抛出异常。
    """
    stat = os.stat(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    return {
        "path": str(Path(path).resolve()),
        "basename": Path(path).name,
        "sha1": file_sha1(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "width": max(0, width),
        "height": max(0, height),
        "fps": max(0.0, fps),
        "frame_count": max(0, frame_count),
    }


def _coerce_int(v: Any, default: int = 0) -> int:
    """把输入安全转成 int，失败时给默认值。"""
    try:
        return int(v)
    except Exception:
        return int(default)


def _normalize_rect(
    rect: Dict[str, Any],
    width: int,
    height: int
) -> Dict[str, int]:
    """
    规范化矩形坐标：
    - 坐标限制在画面内；
    - 宽高至少为 1；
    - 超出边界时自动裁切。
    """
    x = _coerce_int(rect.get("x", 0), 0)
    y = _coerce_int(rect.get("y", 0), 0)
    w = _coerce_int(rect.get("width", 0), 0)
    h = _coerce_int(rect.get("height", 0), 0)

    x = max(0, min(x, max(0, width - 1)))
    y = max(0, min(y, max(0, height - 1)))
    w = max(1, w)
    h = max(1, h)

    if x + w > width:
        w = max(1, width - x)
    if y + h > height:
        h = max(1, height - y)

    return {"x": x, "y": y, "width": w, "height": h}


def normalize_segments(
    segments: List[Dict[str, Any]],
    video_meta: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    规范化“标记段”列表，确保每段都可安全用于后续处理。

    包括：
    - 过滤非法项；
    - 修正 start/end 帧范围；
    - 补齐 id / expand / feather / enabled 等默认字段。
    """
    width = int(video_meta.get("width", 0))
    height = int(video_meta.get("height", 0))
    frame_count = int(video_meta.get("frame_count", 0))
    max_frame = max(0, frame_count - 1)
    now = _now_iso()

    normalized: List[Dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        rect = item.get("rect", {})
        if not isinstance(rect, dict):
            continue

        start_frame = _coerce_int(item.get("start_frame", 0), 0)
        end_frame = _coerce_int(item.get("end_frame", start_frame), start_frame)
        start_frame = max(0, min(start_frame, max_frame))
        end_frame = max(0, min(end_frame, max_frame))
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        normalized.append({
            "id": str(item.get("id") or uuid.uuid4().hex),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "rect": _normalize_rect(rect, width, height),
            "expand_px": max(0, _coerce_int(item.get("expand_px", 5), 5)),
            "feather_px": max(0, _coerce_int(item.get("feather_px", 3), 3)),
            "enabled": bool(item.get("enabled", True)),
            "created_at": str(item.get("created_at") or now),
            "updated_at": now,
        })

    return normalized


def save_sidecar(
    video_path: str,
    segments: List[Dict[str, Any]],
    video_meta: Optional[Dict[str, Any]] = None
) -> Tuple[Path, Dict[str, Any]]:
    """
    保存标注 sidecar 文件。

    注意：
    - 总是使用当前视频的最新指纹写入，忽略传入的旧 meta。
    """
    current_meta = get_video_meta(video_path)
    normalized_segments = normalize_segments(segments or [], current_meta)

    sidecar = build_sidecar_path(video_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    # Persist current file signature; ignore stale incoming meta.
    payload: Dict[str, Any] = {
        "version": "1.0",
        "video_meta": current_meta,
        "segments": normalized_segments,
        "updated_at": _now_iso(),
    }

    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return sidecar, payload


def load_sidecar(video_path: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]], Optional[str]]:
    """
    读取并校验 sidecar。

    返回值中的 warning 用于提示“视频已变化，标注不自动套用”。
    """
    sidecar = build_sidecar_path(video_path)
    if not sidecar.exists():
        return None, None, "Annotation file not found"

    with open(sidecar, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid annotation file format")

    current_meta = get_video_meta(video_path)
    meta = data.get("video_meta") or {}
    mismatch = (
        str(meta.get("sha1", "")) != str(current_meta.get("sha1", "")) or
        int(meta.get("size", -1)) != int(current_meta.get("size", -2)) or
        int(meta.get("mtime_ns", -1)) != int(current_meta.get("mtime_ns", -2))
    )

    warning: Optional[str] = None
    if mismatch:
        warning = "Video fingerprint mismatch. Annotation not auto-applied."

    raw_segments = data.get("segments") if isinstance(data.get("segments"), list) else []
    normalized_segments = normalize_segments(raw_segments, current_meta)

    payload = {
        "version": str(data.get("version", "1.0")),
        "video_meta": current_meta,
        "segments": normalized_segments,
        "updated_at": str(data.get("updated_at") or _now_iso()),
    }
    return sidecar, payload, warning


def delete_sidecar(video_path: str) -> bool:
    """删除 sidecar；文件不存在也视为成功。"""
    sidecar = build_sidecar_path(video_path)
    if not sidecar.exists():
        return True
    sidecar.unlink()
    return True
