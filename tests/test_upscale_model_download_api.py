"""AI 放大模型下载 API 状态机测试。"""

import time
import unittest
from unittest.mock import MagicMock, patch

from src.core.model_downloader import DownloadCancelled
from src.gui.api import API


class UpscaleModelDownloadApiTests(unittest.TestCase):
    def setUp(self):
        self.native_patch = patch.object(API, "_init_native_player", lambda _self: None)
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)

    @staticmethod
    def _wait_for_state(api: API, expected_states, timeout=5.0):
        started = time.time()
        while time.time() - started < timeout:
            task = api.get_upscale_model_download_status().get("task", {})
            state = task.get("state")
            if state in expected_states:
                return task
            time.sleep(0.03)
        raise AssertionError(f"Timeout waiting for states {expected_states}")

    def test_get_upscale_model_download_status_shape(self):
        with patch(
            "src.gui.api.list_upscale_model_download_entries",
            return_value=[
                {
                    "model_id": "realesrgan_general_x4v3",
                    "display_name": "Real-ESRGAN General x4v3",
                    "installed": True,
                    "can_redownload": True,
                    "install_hint": "hint0",
                },
                {
                    "model_id": "realesrgan_x2plus",
                    "display_name": "Real-ESRGAN x2plus",
                    "installed": False,
                    "can_redownload": True,
                    "install_hint": "hint0b",
                },
                {
                    "model_id": "seedvr2_3b_q4_k_m_gguf",
                    "display_name": "SeedVR2 3B Q4_K_M (GGUF)",
                    "installed": True,
                    "can_redownload": True,
                    "install_hint": "hint",
                },
                {
                    "model_id": "seedvr2_3b_q8_0_gguf",
                    "display_name": "SeedVR2 3B Q8_0 (GGUF)",
                    "installed": False,
                    "can_redownload": True,
                    "install_hint": "hint2",
                },
            ],
        ):
            api = API()
            payload = api.get_upscale_model_download_status()

        self.assertTrue(payload["success"])
        self.assertIn("models", payload)
        self.assertIn("task", payload)
        self.assertEqual(
            {entry["model_id"] for entry in payload["models"]},
            {
                "realesrgan_general_x4v3",
                "realesrgan_x2plus",
                "seedvr2_3b_q4_k_m_gguf",
                "seedvr2_3b_q8_0_gguf",
            },
        )
        self.assertEqual(payload["task"]["state"], "idle")

    def test_start_upscale_model_download_rejects_removed_7b(self):
        api = API()
        result = api.start_upscale_model_download({"model_id": "seedvr2_7b_fp8"})
        self.assertFalse(result["success"])
        self.assertIn("Model removed", result["error"])

    def test_start_upscale_model_download_rejects_removed_fp8(self):
        api = API()
        result = api.start_upscale_model_download({"model_id": "seedvr2_3b_fp8"})
        self.assertFalse(result["success"])
        self.assertIn("Model removed", result["error"])

    def test_start_upscale_model_download_rejects_when_running(self):
        unblock = {"done": False}

        def fake_download(model_id, force=False, progress_callback=None, cancel_event=None):
            if progress_callback:
                progress_callback(
                    {
                        "progress": 0.12,
                        "downloaded_bytes": 12,
                        "total_bytes": 100,
                        "speed_bps": 1024,
                        "current_file": "repo.tar.gz",
                        "message": "running",
                        "error": "",
                    }
                )
            while not unblock["done"]:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled("cancelled")
                time.sleep(0.02)
            return {"model_id": model_id, "installed": True, "skipped": False}

        with patch("src.gui.api.download_upscale_model", side_effect=fake_download):
            api = API()
            first = api.start_upscale_model_download({"model_id": "seedvr2_3b_q4_k_m_gguf"})
            self.assertTrue(first["success"])
            self._wait_for_state(api, {"running"})

            second = api.start_upscale_model_download({"model_id": "seedvr2_3b_q4_k_m_gguf"})
            self.assertFalse(second["success"])
            self.assertIn("already running", second["error"])

            unblock["done"] = True
            self._wait_for_state(api, {"success"})

    def test_cancel_upscale_model_download_sets_cancelled_state(self):
        def fake_download(model_id, force=False, progress_callback=None, cancel_event=None):
            if progress_callback:
                progress_callback(
                    {
                        "progress": 0.2,
                        "downloaded_bytes": 20,
                        "total_bytes": 100,
                        "speed_bps": 2048,
                        "current_file": "weights.pth",
                        "message": "running",
                        "error": "",
                    }
                )
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled("cancelled")
                time.sleep(0.02)

        with patch("src.gui.api.download_upscale_model", side_effect=fake_download):
            api = API()
            started = api.start_upscale_model_download({"model_id": "seedvr2_3b_q4_k_m_gguf"})
            self.assertTrue(started["success"])
            self._wait_for_state(api, {"running"})
            cancelled = api.cancel_upscale_model_download()
            self.assertTrue(cancelled["success"])
            task = self._wait_for_state(api, {"cancelled"})
            self.assertEqual(task["state"], "cancelled")

    def test_successful_download_invalidates_upscale_capability_cache(self):
        with patch(
            "src.gui.api.download_upscale_model",
            return_value={"model_id": "seedvr2_3b_q4_k_m_gguf", "installed": True, "skipped": False},
        ):
            api = API()
            api._upscale_processor.invalidate_capabilities_cache = MagicMock()
            started = api.start_upscale_model_download({"model_id": "seedvr2_3b_q4_k_m_gguf"})
            self.assertTrue(started["success"])
            self._wait_for_state(api, {"success"})
            api._upscale_processor.invalidate_capabilities_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
