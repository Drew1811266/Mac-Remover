"""智能镜头分切策略测试。"""

import unittest
from unittest.mock import patch

from src.core.scene_splitter import SceneSplitter


class SceneSplitterTests(unittest.TestCase):
    def test_split_rule_mode_with_stable_detectors(self):
        splitter = SceneSplitter(ffmpeg_bin="/embedded/ffmpeg")

        with patch.object(splitter, "_detect_ffmpeg_cuts", return_value=[2.0, 4.0, 6.0, 8.0]):
            with patch.object(splitter, "_detect_pyscenedetect_cuts", return_value=([2.1, 4.0, 6.1, 7.9], "")):
                with patch.object(splitter, "_detect_transnet_cuts", side_effect=AssertionError("should not run")):
                    result = splitter.split(
                        input_path="/tmp/input.mp4",
                        duration_sec=10.0,
                        fps=24.0,
                    )
        self.assertEqual(result.split_mode, "rule")
        self.assertGreaterEqual(len(result.segments), 3)
        self.assertTrue(all(seg.duration >= 1.2 for seg in result.segments))

    def test_split_hybrid_mode_when_transnet_triggered(self):
        splitter = SceneSplitter(ffmpeg_bin="/embedded/ffmpeg")

        with patch.object(splitter, "_detect_ffmpeg_cuts", return_value=[1.0, 2.0, 3.0, 4.0, 5.0]):
            with patch.object(splitter, "_detect_pyscenedetect_cuts", return_value=([2.0], "")):
                with patch.object(splitter, "_detect_transnet_cuts", return_value=([1.4, 2.8, 4.2], "")):
                    result = splitter.split(
                        input_path="/tmp/input.mp4",
                        duration_sec=12.0,
                        fps=24.0,
                    )
        self.assertEqual(result.split_mode, "hybrid")
        self.assertGreaterEqual(result.stats.get("transnet_cut_count", 0), 1)
        self.assertGreater(len(result.segments), 0)

    def test_split_fallback_when_no_detector_cuts(self):
        splitter = SceneSplitter(ffmpeg_bin="/embedded/ffmpeg")

        with patch.object(splitter, "_detect_ffmpeg_cuts", return_value=[]):
            with patch.object(splitter, "_detect_pyscenedetect_cuts", return_value=([], "")):
                with patch.object(splitter, "_detect_transnet_cuts", return_value=([], "")):
                    result = splitter.split(
                        input_path="/tmp/input.mp4",
                        duration_sec=10.0,
                        fps=24.0,
                    )
        self.assertEqual(result.split_mode, "fallback")
        self.assertTrue(any("fallback" in item.lower() for item in result.warnings))
        self.assertGreaterEqual(len(result.segments), 3)

    def test_normalize_merges_short_and_splits_long(self):
        splitter = SceneSplitter(ffmpeg_bin="/embedded/ffmpeg")
        segments = splitter._to_segments(
            cuts=[0.3, 0.8, 6.5],
            duration_sec=10.0,
            normalize=True,
        )
        self.assertTrue(segments)
        self.assertTrue(all(seg.duration >= 1.2 for seg in segments))
        self.assertTrue(all(seg.duration <= 4.0 for seg in segments))

    def test_secondary_short_merge_prefers_no_tiny_segments(self):
        splitter = SceneSplitter(ffmpeg_bin="/embedded/ffmpeg")
        segments = splitter._to_segments(
            cuts=[1.1, 2.1, 3.1, 5.0],
            duration_sec=8.0,
            normalize=True,
        )
        self.assertTrue(segments)
        self.assertLessEqual(len([seg for seg in segments if seg.duration < 2.0]), 1)


if __name__ == "__main__":
    unittest.main()
