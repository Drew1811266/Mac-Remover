"""WatermarkRemover 行为测试。"""

import unittest
from unittest.mock import patch

import torch

from src.core.remover import WatermarkRemover


class WatermarkRemoverTests(unittest.TestCase):
    def test_unload_model_is_idempotent(self):
        remover = WatermarkRemover(device=torch.device("cpu"), use_fp16=False)
        remover.model = object()
        remover._is_loaded = True

        with patch("src.core.remover.release_unified_memory", return_value={"success": True}) as cleanup_mock:
            remover.unload_model()
            remover.unload_model()

        self.assertIsNone(remover.model)
        self.assertFalse(remover.is_loaded())
        self.assertEqual(cleanup_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
