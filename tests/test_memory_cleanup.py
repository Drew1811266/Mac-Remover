"""统一内存回收工具测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.utils.memory_cleanup import release_unified_memory


class MemoryCleanupTests(unittest.TestCase):
    def test_release_unified_memory_without_torch(self):
        with patch("src.utils.memory_cleanup.torch", None):
            result = release_unified_memory("unit-test-no-torch")

        self.assertTrue(result["success"])
        step_names = [item.get("step") for item in result.get("steps", [])]
        self.assertIn("gc.collect", step_names)
        self.assertIn("torch", step_names)
        self.assertIn("torch_unavailable", str(result))

    def test_release_unified_memory_with_mps(self):
        mps_calls = {"count": 0}

        def _mps_empty_cache():
            mps_calls["count"] += 1

        fake_torch = SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=_mps_empty_cache,
            ),
            cuda=SimpleNamespace(
                is_available=lambda: False,
            ),
        )

        with patch("src.utils.memory_cleanup.torch", fake_torch):
            result = release_unified_memory("unit-test-mps")

        self.assertTrue(result["success"])
        self.assertEqual(mps_calls["count"], 1)
        step_names = [item.get("step") for item in result.get("steps", [])]
        self.assertIn("torch.mps.empty_cache", step_names)

    def test_release_unified_memory_with_cuda(self):
        cuda_calls = {"empty": 0, "ipc": 0}

        def _cuda_empty_cache():
            cuda_calls["empty"] += 1

        def _cuda_ipc_collect():
            cuda_calls["ipc"] += 1

        fake_torch = SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: False,
                empty_cache=lambda: None,
            ),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=_cuda_empty_cache,
                ipc_collect=_cuda_ipc_collect,
            ),
        )

        with patch("src.utils.memory_cleanup.torch", fake_torch):
            result = release_unified_memory("unit-test-cuda")

        self.assertTrue(result["success"])
        self.assertEqual(cuda_calls["empty"], 1)
        self.assertEqual(cuda_calls["ipc"], 1)
        step_names = [item.get("step") for item in result.get("steps", [])]
        self.assertIn("torch.cuda.empty_cache", step_names)
        self.assertIn("torch.cuda.ipc_collect", step_names)


if __name__ == "__main__":
    unittest.main()
