"""进度估算器测试。

验证点：进度单调、ETA 随处理推进下降、完成态 ETA 归零。
"""

import time
import unittest

from src.core.progress_estimator import ProgressEstimator


class ProgressEstimatorTests(unittest.TestCase):
    def test_progress_is_monotonic(self):
        """任何阶段更新后，整体进度都不应倒退。"""
        estimator = ProgressEstimator(total_frames=120)
        values = []

        estimator.transition_to("prepare")
        estimator.update_phase_progress("prepare", 0.3)
        values.append(float(estimator.snapshot(force_recompute=True)["progress"]))

        estimator.transition_to("load_models")
        estimator.update_phase_progress("load_models", 0.5)
        values.append(float(estimator.snapshot(force_recompute=True)["progress"]))

        # 传入更小值应被忽略，确保进度单调。
        estimator.update_phase_progress("prepare", 0.1)
        values.append(float(estimator.snapshot(force_recompute=True)["progress"]))

        estimator.transition_to("infer")
        estimator.update_processed_frames(20, 120)
        values.append(float(estimator.snapshot(force_recompute=True)["progress"]))
        estimator.update_processed_frames(60, 120)
        values.append(float(estimator.snapshot(force_recompute=True)["progress"]))

        self.assertEqual(values, sorted(values))

    def test_eta_updates_and_converges_with_frame_updates(self):
        """模拟推理推进，验证 ETA 可用且逐步收敛。"""
        estimator = ProgressEstimator(total_frames=100)
        estimator.transition_to("infer")

        snapshots = []
        for processed in (10, 25, 45, 70):
            time.sleep(0.28)
            estimator.update_processed_frames(processed, 100)
            snapshots.append(estimator.snapshot(force_recompute=True))

        eta_values = [item.get("eta_seconds") for item in snapshots if item.get("eta_seconds") is not None]
        self.assertTrue(eta_values, "Expected non-empty ETA samples")
        self.assertGreater(float(snapshots[-1].get("throughput_fps") or 0.0), 0.0)
        self.assertLess(float(eta_values[-1]), float(eta_values[0]))

    def test_complete_all_reports_zero_eta(self):
        """完成态应固定输出 progress=1 和 ETA=00:00。"""
        estimator = ProgressEstimator(total_frames=10)
        estimator.complete_all()
        snap = estimator.snapshot(force_recompute=True)
        self.assertAlmostEqual(float(snap["progress"]), 1.0, places=6)
        self.assertEqual(float(snap["eta_seconds"]), 0.0)
        self.assertEqual(str(snap["estimated_time"]), "00:00")

    def test_infer_phase_progress_and_frame_ratio_never_regress(self):
        """混合使用 infer 阶段进度与帧计数更新时，整体进度保持单调。"""
        estimator = ProgressEstimator(total_frames=200)
        estimator.update_phase_progress("prepare", 1.0)
        estimator.update_phase_progress("load_models", 1.0)
        estimator.update_phase_progress("extract", 1.0)
        estimator.update_phase_progress("infer", 0.1)
        p1 = float(estimator.snapshot(force_recompute=True)["progress"])

        estimator.update_processed_frames(20, 200)
        p2 = float(estimator.snapshot(force_recompute=True)["progress"])

        # Infer phase传入更小值后应被单调保护忽略。
        estimator.update_phase_progress("infer", 0.05)
        p3 = float(estimator.snapshot(force_recompute=True)["progress"])

        estimator.update_processed_frames(80, 200)
        p4 = float(estimator.snapshot(force_recompute=True)["progress"])

        self.assertEqual([p1, p2, p3, p4], sorted([p1, p2, p3, p4]))


if __name__ == "__main__":
    unittest.main()
