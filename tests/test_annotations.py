"""标注 sidecar 模块测试。

重点验证：
1. 标记段归一化（边界裁剪、默认值填充）。
2. sidecar 保存/加载是否能往返一致。
3. 视频指纹变化时是否给出告警。
4. sidecar 删除逻辑是否生效。
"""

import os
import tempfile
import unittest

import cv2
import numpy as np

from src.core.annotations import (
    build_sidecar_path,
    delete_sidecar,
    load_sidecar,
    normalize_segments,
    save_sidecar,
)


def _create_test_video(path: str, width: int = 64, height: int = 48, fps: int = 10, frames: int = 12) -> None:
    """生成一个小视频样本，供测试读取元信息和 sidecar 绑定。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot create test video")
    for idx in range(frames):
        frame = np.full((height, width, 3), idx * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class AnnotationSidecarTests(unittest.TestCase):
    """`src.core.annotations` 的核心行为回归测试。"""

    def test_normalize_segments_bounds_and_defaults(self):
        """验证越界输入会被裁剪，并补齐默认字段。"""
        video_meta = {
            "width": 1920,
            "height": 1080,
            "frame_count": 100,
        }
        segments = [
            {
                "start_frame": 120,
                "end_frame": -2,
                "rect": {"x": -30, "y": 1000, "width": 5000, "height": 5000},
            }
        ]

        normalized = normalize_segments(segments, video_meta)
        self.assertEqual(len(normalized), 1)
        seg = normalized[0]
        self.assertEqual(seg["start_frame"], 0)
        self.assertEqual(seg["end_frame"], 99)
        self.assertGreaterEqual(seg["rect"]["x"], 0)
        self.assertGreaterEqual(seg["rect"]["y"], 0)
        self.assertLessEqual(seg["rect"]["x"] + seg["rect"]["width"], 1920)
        self.assertLessEqual(seg["rect"]["y"] + seg["rect"]["height"], 1080)
        self.assertEqual(seg["expand_px"], 5)
        self.assertEqual(seg["feather_px"], 3)
        self.assertTrue(seg["enabled"])

    def test_save_and_load_sidecar_roundtrip(self):
        """验证 sidecar 可成功保存并读回同一份结构。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "sample.mp4")
            _create_test_video(video_path)

            segments = [
                {
                    "id": "seg_1",
                    "start_frame": 2,
                    "end_frame": 8,
                    "rect": {"x": 10, "y": 6, "width": 12, "height": 10},
                    "enabled": True,
                }
            ]

            sidecar_path, payload = save_sidecar(video_path, segments)
            self.assertTrue(sidecar_path.exists())
            self.assertEqual(payload["video_meta"]["basename"], "sample.mp4")
            self.assertEqual(len(payload["segments"]), 1)

            loaded_path, loaded_payload, warning = load_sidecar(video_path)
            self.assertEqual(str(loaded_path), str(sidecar_path))
            self.assertIsNone(warning)
            self.assertEqual(len(loaded_payload["segments"]), 1)
            self.assertEqual(loaded_payload["segments"][0]["id"], "seg_1")

    def test_load_sidecar_warns_on_video_fingerprint_mismatch(self):
        """验证视频文件被改写后，加载 sidecar 会返回 warning。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "sample.mp4")
            _create_test_video(video_path, frames=8)

            save_sidecar(
                video_path,
                [
                    {
                        "id": "seg_a",
                        "start_frame": 1,
                        "end_frame": 4,
                        "rect": {"x": 3, "y": 2, "width": 9, "height": 9},
                    }
                ],
            )

            # 重写视频，触发 sha1/size/mtime 指纹变化。
            _create_test_video(video_path, frames=10)

            _, payload, warning = load_sidecar(video_path)
            self.assertIsNotNone(payload)
            self.assertTrue(warning)

    def test_delete_sidecar(self):
        """验证删除 sidecar 后文件确实不存在。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "sample.mp4")
            _create_test_video(video_path)
            save_sidecar(
                video_path,
                [
                    {
                        "id": "seg_x",
                        "start_frame": 0,
                        "end_frame": 0,
                        "rect": {"x": 1, "y": 1, "width": 5, "height": 5},
                    }
                ],
            )

            sidecar = build_sidecar_path(video_path)
            self.assertTrue(sidecar.exists())
            self.assertTrue(delete_sidecar(video_path))
            self.assertFalse(sidecar.exists())


if __name__ == "__main__":
    unittest.main()
