"""手动标记段调度测试。"""

import unittest

from src.core.video_processor import VideoProcessor


class ManualSegmentSchedulerTests(unittest.TestCase):
    def test_last_added_segment_wins_on_overlap(self):
        """兼容旧接口：重叠区间仍保留 last-added wins 语义。"""
        segments = [
            {"id": "a", "start_frame": 0, "end_frame": 10},
            {"id": "b", "start_frame": 5, "end_frame": 8},
            {"id": "c", "start_frame": 8, "end_frame": 15},
        ]

        self.assertEqual(
            VideoProcessor._resolve_active_annotation_segment(6, segments)["id"],
            "b",
        )
        self.assertEqual(
            VideoProcessor._resolve_active_annotation_segment(8, segments)["id"],
            "c",
        )
        self.assertEqual(
            VideoProcessor._resolve_active_annotation_segment(12, segments)["id"],
            "c",
        )
        self.assertIsNone(VideoProcessor._resolve_active_annotation_segment(16, segments))

    def test_resolve_active_segments_returns_all_in_area_priority_order(self):
        """新接口：同帧返回全部命中段，并按面积降序排序。"""
        segments = [
            {
                "id": "small-late",
                "start_frame": 0,
                "end_frame": 12,
                "_order": 2,
                "_area": 25,
                "rect": {"x": 0, "y": 0, "width": 5, "height": 5},
            },
            {
                "id": "large-early",
                "start_frame": 0,
                "end_frame": 12,
                "_order": 0,
                "_area": 400,
                "rect": {"x": 0, "y": 0, "width": 20, "height": 20},
            },
            {
                "id": "middle",
                "start_frame": 0,
                "end_frame": 12,
                "_order": 1,
                "_area": 100,
                "rect": {"x": 0, "y": 0, "width": 10, "height": 10},
            },
        ]
        active = VideoProcessor._resolve_active_annotation_segments(6, segments)
        self.assertEqual([seg["id"] for seg in active], ["large-early", "middle", "small-late"])

    def test_resolve_active_segments_uses_creation_order_for_ties(self):
        """面积相同按创建顺序稳定排序。"""
        segments = [
            {
                "id": "first",
                "start_frame": 0,
                "end_frame": 10,
                "_order": 0,
                "_area": 100,
                "rect": {"x": 0, "y": 0, "width": 10, "height": 10},
            },
            {
                "id": "second",
                "start_frame": 0,
                "end_frame": 10,
                "_order": 1,
                "_area": 100,
                "rect": {"x": 0, "y": 0, "width": 10, "height": 10},
            },
        ]
        active = VideoProcessor._resolve_active_annotation_segments(4, segments)
        self.assertEqual([seg["id"] for seg in active], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
