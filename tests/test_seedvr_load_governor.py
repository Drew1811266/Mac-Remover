"""SeedVR 负载治理策略测试。"""

import unittest
from unittest.mock import patch

from src.core.seedvr_runtime import SeedVRRuntime


class SeedVRLoadGovernorTests(unittest.TestCase):
    def test_cpu_normalization_and_threshold(self):
        normalized = SeedVRRuntime._normalize_cpu_percent(process_cpu_percent=640.0, logical_cpu_count=8)
        self.assertEqual(normalized, 80.0)

        streak = 0
        streak = SeedVRRuntime._next_cpu_overload_streak(streak, 79.9)
        self.assertEqual(streak, 0)
        streak = SeedVRRuntime._next_cpu_overload_streak(streak, 85.0)
        self.assertEqual(streak, 0)
        streak = SeedVRRuntime._next_cpu_overload_streak(streak, 92.0)
        streak = SeedVRRuntime._next_cpu_overload_streak(streak, 89.0)
        self.assertEqual(streak, 2)

    def test_governor_skips_without_psutil(self):
        with patch("src.core.seedvr_runtime.psutil", None):
            self.assertFalse(SeedVRRuntime._load_governor_supported())


if __name__ == "__main__":
    unittest.main()
