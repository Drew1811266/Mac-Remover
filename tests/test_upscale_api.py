"""SeedVR2 放大任务 API 测试。"""

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.core.upscale_processor import UpscaleCancelled
from src.gui.api import API


class UpscaleApiTests(unittest.TestCase):
    def setUp(self):
        self.native_patch = patch.object(API, "_init_native_player", lambda _self: None)
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)

    @staticmethod
    def _wait_for_task_state(api: API, expected_states, timeout=5.0):
        started = time.time()
        while time.time() - started < timeout:
            task = api.get_upscale_task_status().get("task", {})
            state = task.get("state")
            if state in expected_states:
                return task
            time.sleep(0.03)
        raise AssertionError(f"Timeout waiting for states {expected_states}")

    def test_get_upscale_capabilities_passthrough(self):
        api = API()
        with patch.object(
            api._upscale_processor,
            "get_capabilities",
            return_value={"success": True, "engines": [], "models": []},
        ) as mocked:
            payload = api.get_upscale_capabilities({"force_refresh": True})
        self.assertTrue(payload["success"])
        mocked.assert_called_once_with(force_refresh=True)

    def test_api_init_attempts_legacy_model_cleanup(self):
        with patch(
            "src.gui.api.remove_legacy_upscale_model_files",
            return_value={"removed": [], "failed": []},
        ) as mocked:
            API()
        mocked.assert_called_once()

    def test_api_init_legacy_model_cleanup_failure_does_not_block(self):
        with patch(
            "src.gui.api.remove_legacy_upscale_model_files",
            side_effect=RuntimeError("cleanup failed"),
        ):
            api = API()
        self.assertIsNotNone(api)

    def test_start_upscale_rejects_invalid_engine(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"video")
            api = API()
            payload = {
                "input_path": input_path,
                "output_dir": td,
                "mode": "upscale_resolution",
                "engine": "invalid_engine",
                "model_id": "seedvr2_3b_q4_k_m_gguf",
                "target_preset": "1080p",
            }
            result = api.start_upscale(payload)
        self.assertFalse(result["success"])
        self.assertIn("Invalid engine", result["error"])

    def test_start_upscale_rejects_invalid_model_id(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"video")
            api = API()
            payload = {
                "input_path": input_path,
                "output_dir": td,
                "mode": "upscale_resolution",
                "engine": "seedvr2",
                "model_id": "unknown_model",
                "target_preset": "1080p",
            }
            result = api.start_upscale(payload)
        self.assertFalse(result["success"])
        self.assertIn("Invalid model_id", result["error"])

    def test_start_upscale_rejects_removed_7b(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"video")
            api = API()
            result = api.start_upscale(
                {
                    "input_path": input_path,
                    "output_dir": td,
                    "mode": "upscale_resolution",
                    "engine": "seedvr2",
                    "model_id": "seedvr2_7b_fp8",
                    "target_preset": "1080p",
                }
            )
        self.assertFalse(result["success"])
        self.assertIn("Model removed", result["error"])

    def test_start_upscale_rejects_removed_fp8(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"video")
            api = API()
            result = api.start_upscale(
                {
                    "input_path": input_path,
                    "output_dir": td,
                    "mode": "upscale_resolution",
                    "engine": "seedvr2",
                    "model_id": "seedvr2_3b_fp8",
                    "target_preset": "1080p",
                }
            )
        self.assertFalse(result["success"])
        self.assertIn("Model removed", result["error"])

    def test_start_upscale_accepts_legacy_x4_and_normalizes(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "upscaled.mp4")
            Path(input_path).write_bytes(b"video")
            Path(output_path).write_bytes(b"result")
            api = API()

            def fake_upscale_video(**kwargs):
                self.assertEqual(kwargs["same_res_strength"], "x2_then_downscale")
                return {
                    "output_path": output_path,
                    "effective_engine": "seedvr2",
                    "warning": "normalized",
                }

            with patch("src.gui.api.get_device_info", return_value=SimpleNamespace(memory_gb=64.0)):
                with patch("src.gui.api.is_upscale_model_installed", return_value=True):
                    with patch.object(
                        api._upscale_processor,
                        "get_capabilities",
                        return_value={
                            "success": True,
                            "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                        },
                    ):
                        with patch.object(api._upscale_processor, "upscale_video", side_effect=fake_upscale_video):
                            with patch.object(
                                api,
                                "prepare_video_preview",
                                return_value={"success": True, "path": output_path},
                            ):
                                started = api.start_upscale(
                                    {
                                        "input_path": input_path,
                                        "output_dir": td,
                                        "mode": "enhance_same_resolution",
                                        "engine": "seedvr2",
                                        "model_id": "seedvr2_3b_q4_k_m_gguf",
                                        "same_res_strength": "x4_then_downscale",
                                    }
                                )
                                self.assertTrue(started["success"])
                                task = self._wait_for_task_state(api, {"success"})
        self.assertEqual(task["state"], "success")

    def test_start_upscale_success_updates_task_to_success(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "upscaled.mp4")
            Path(input_path).write_bytes(b"video")
            Path(output_path).write_bytes(b"result")

            api = API()
            with patch("src.gui.api.get_device_info", return_value=SimpleNamespace(memory_gb=64.0)):
                with patch("src.gui.api.is_upscale_model_installed", return_value=True):
                    with patch.object(
                        api._upscale_processor,
                        "get_capabilities",
                        return_value={
                            "success": True,
                            "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                        },
                    ):
                        with patch.object(
                            api._upscale_processor,
                            "upscale_video",
                            return_value={
                                "output_path": output_path,
                                "effective_engine": "seedvr2",
                                "warning": "",
                            },
                        ):
                            with patch.object(
                                api,
                                "prepare_video_preview",
                                return_value={"success": True, "path": output_path},
                            ):
                                started = api.start_upscale(
                                    {
                                        "input_path": input_path,
                                        "output_dir": td,
                                        "mode": "upscale_resolution",
                                        "engine": "seedvr2",
                                        "model_id": "seedvr2_3b_q4_k_m_gguf",
                                        "target_preset": "1080p",
                                        "denoise_strength": 0.35,
                                        "keep_audio": True,
                                    }
                                )
                                self.assertTrue(started["success"])
                                task = self._wait_for_task_state(api, {"success"})

        self.assertEqual(task["state"], "success")
        self.assertEqual(task["effective_engine"], "seedvr2")
        self.assertEqual(task["output_path"], output_path)

    def test_start_upscale_progress_passthrough_contains_segment_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "upscaled.mp4")
            Path(input_path).write_bytes(b"video")
            Path(output_path).write_bytes(b"result")
            api = API()

            def fake_upscale_video(**kwargs):
                progress_callback = kwargs.get("progress_callback")
                if progress_callback:
                    progress_callback(
                        {
                            "progress": 0.35,
                            "phase": "infer",
                            "message": "Scene split: segment 1/3 (0.0s-3.3s) - infer",
                            "segment_index": 1,
                            "segment_total": 3,
                            "scene_split_mode": "hybrid",
                        }
                    )
                return {
                    "output_path": output_path,
                    "effective_engine": "seedvr2",
                    "warning": "",
                    "segment_total": 3,
                    "scene_split_mode": "hybrid",
                }

            with patch("src.gui.api.get_device_info", return_value=SimpleNamespace(memory_gb=64.0)):
                with patch("src.gui.api.is_upscale_model_installed", return_value=True):
                    with patch.object(
                        api._upscale_processor,
                        "get_capabilities",
                        return_value={
                            "success": True,
                            "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                        },
                    ):
                        with patch.object(api._upscale_processor, "upscale_video", side_effect=fake_upscale_video):
                            with patch.object(
                                api,
                                "prepare_video_preview",
                                return_value={"success": True, "path": output_path},
                            ):
                                started = api.start_upscale(
                                    {
                                        "input_path": input_path,
                                        "output_dir": td,
                                        "mode": "upscale_resolution",
                                        "engine": "seedvr2",
                                        "model_id": "seedvr2_3b_q4_k_m_gguf",
                                        "target_preset": "1080p",
                                    }
                                )
                                self.assertTrue(started["success"])
                                task = self._wait_for_task_state(api, {"success"})

        self.assertEqual(task.get("segment_total"), 3)
        self.assertEqual(task.get("segment_index"), 3)
        self.assertEqual(task.get("scene_split_mode"), "hybrid")

    def test_cancel_upscale_task_moves_state_to_cancelled(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"video")
            api = API()

            def fake_upscale_video(**kwargs):
                progress_callback = kwargs.get("progress_callback")
                cancel_event = kwargs.get("cancel_event")
                if progress_callback:
                    progress_callback(
                        {
                            "progress": 0.15,
                            "phase": "infer",
                            "message": "running",
                            "eta_seconds": 10,
                        }
                    )
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise UpscaleCancelled("cancelled")
                    time.sleep(0.02)

            with patch("src.gui.api.get_device_info", return_value=SimpleNamespace(memory_gb=64.0)):
                with patch("src.gui.api.is_upscale_model_installed", return_value=True):
                    with patch.object(
                        api._upscale_processor,
                        "get_capabilities",
                        return_value={
                            "success": True,
                            "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                        },
                    ):
                        with patch.object(api._upscale_processor, "upscale_video", side_effect=fake_upscale_video):
                            started = api.start_upscale(
                                {
                                    "input_path": input_path,
                                    "output_dir": td,
                                    "mode": "upscale_resolution",
                                    "engine": "seedvr2",
                                    "model_id": "seedvr2_3b_q4_k_m_gguf",
                                    "target_preset": "1080p",
                                }
                            )
                            self.assertTrue(started["success"])
                            self._wait_for_task_state(api, {"running"})
                            cancelled = api.cancel_upscale_task()
                            self.assertTrue(cancelled["success"])
                            task = self._wait_for_task_state(api, {"cancelled"})

        self.assertEqual(task["state"], "cancelled")

    def test_start_upscale_defaults_to_q4_model(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "upscaled.mp4")
            Path(input_path).write_bytes(b"video")
            Path(output_path).write_bytes(b"result")
            api = API()

            with patch("src.gui.api.get_device_info", return_value=SimpleNamespace(memory_gb=64.0)):
                with patch("src.gui.api.is_upscale_model_installed", return_value=True):
                    with patch.object(
                        api._upscale_processor,
                        "get_capabilities",
                        return_value={
                            "success": True,
                            "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                        },
                    ):
                        with patch.object(
                            api._upscale_processor,
                            "upscale_video",
                            return_value={
                                "output_path": output_path,
                                "effective_engine": "seedvr2",
                                "warning": "",
                            },
                        ) as mocked_upscale:
                            with patch.object(
                                api,
                                "prepare_video_preview",
                                return_value={"success": True, "path": output_path},
                            ):
                                started = api.start_upscale(
                                    {
                                        "input_path": input_path,
                                        "output_dir": td,
                                        "mode": "upscale_resolution",
                                        "engine": "seedvr2",
                                        "target_preset": "1080p",
                                    }
                                )
                                self.assertTrue(started["success"])
                                self._wait_for_task_state(api, {"success"})

            self.assertEqual(mocked_upscale.call_args.kwargs["model_id"], "seedvr2_3b_q4_k_m_gguf")

    def test_start_upscale_defaults_to_realesrgan_when_engine_missing(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "upscaled.mp4")
            Path(input_path).write_bytes(b"video")
            Path(output_path).write_bytes(b"result")
            api = API()

            with patch("src.gui.api.get_device_info", return_value=SimpleNamespace(memory_gb=64.0)):
                with patch("src.gui.api.is_upscale_model_installed", return_value=True):
                    with patch.object(
                        api._upscale_processor,
                        "get_capabilities",
                        return_value={
                            "success": True,
                            "engines": [
                                {"engine": "realesrgan", "available": True, "reason": ""},
                                {"engine": "seedvr2", "available": True, "reason": ""},
                            ],
                        },
                    ):
                        with patch.object(
                            api._upscale_processor,
                            "upscale_video",
                            return_value={
                                "output_path": output_path,
                                "effective_engine": "realesrgan",
                                "warning": "",
                            },
                        ) as mocked_upscale:
                            with patch.object(
                                api,
                                "prepare_video_preview",
                                return_value={"success": True, "path": output_path},
                            ):
                                started = api.start_upscale(
                                    {
                                        "input_path": input_path,
                                        "output_dir": td,
                                        "mode": "upscale_resolution",
                                        "target_preset": "1080p",
                                    }
                                )
                                self.assertTrue(started["success"])
                                self._wait_for_task_state(api, {"success"})

            self.assertEqual(mocked_upscale.call_args.kwargs["engine"], "realesrgan")
            self.assertEqual(mocked_upscale.call_args.kwargs["model_id"], "realesrgan_general_x4v3")

    def test_upscale_worker_success_triggers_runtime_memory_release(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "upscaled.mp4")
            Path(input_path).write_bytes(b"video")
            Path(output_path).write_bytes(b"result")
            api = API()

            with patch.object(api, "_release_upscale_runtime_memory") as release_mock:
                with patch("src.gui.api.get_device_info", return_value=SimpleNamespace(memory_gb=64.0)):
                    with patch("src.gui.api.is_upscale_model_installed", return_value=True):
                        with patch.object(
                            api._upscale_processor,
                            "get_capabilities",
                            return_value={
                                "success": True,
                                "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                            },
                        ):
                            with patch.object(
                                api._upscale_processor,
                                "upscale_video",
                                return_value={
                                    "output_path": output_path,
                                    "effective_engine": "seedvr2",
                                    "warning": "",
                                },
                            ):
                                with patch.object(
                                    api,
                                    "prepare_video_preview",
                                    return_value={"success": True, "path": output_path},
                                ):
                                    started = api.start_upscale(
                                        {
                                            "input_path": input_path,
                                            "output_dir": td,
                                            "mode": "upscale_resolution",
                                            "engine": "seedvr2",
                                            "model_id": "seedvr2_3b_q4_k_m_gguf",
                                            "target_preset": "1080p",
                                        }
                                    )
                                    self.assertTrue(started["success"])
                                    self._wait_for_task_state(api, {"success"})

            release_mock.assert_called_once_with("success")

    def test_upscale_worker_failure_triggers_runtime_memory_release(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"video")
            api = API()

            with patch.object(api, "_release_upscale_runtime_memory") as release_mock:
                with patch("src.gui.api.get_device_info", return_value=SimpleNamespace(memory_gb=64.0)):
                    with patch("src.gui.api.is_upscale_model_installed", return_value=True):
                        with patch.object(
                            api._upscale_processor,
                            "get_capabilities",
                            return_value={
                                "success": True,
                                "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                            },
                        ):
                            with patch.object(
                                api._upscale_processor,
                                "upscale_video",
                                side_effect=RuntimeError("boom"),
                            ):
                                started = api.start_upscale(
                                    {
                                        "input_path": input_path,
                                        "output_dir": td,
                                        "mode": "upscale_resolution",
                                        "engine": "seedvr2",
                                        "model_id": "seedvr2_3b_q4_k_m_gguf",
                                        "target_preset": "1080p",
                                    }
                                )
                                self.assertTrue(started["success"])
                                self._wait_for_task_state(api, {"failed"})

            release_mock.assert_called_once_with("failed")

    def test_upscale_worker_cancelled_triggers_runtime_memory_release(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"video")
            api = API()

            def fake_upscale_video(**kwargs):
                cancel_event = kwargs.get("cancel_event")
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise UpscaleCancelled("cancelled")
                    time.sleep(0.02)

            with patch.object(api, "_release_upscale_runtime_memory") as release_mock:
                with patch("src.gui.api.get_device_info", return_value=SimpleNamespace(memory_gb=64.0)):
                    with patch("src.gui.api.is_upscale_model_installed", return_value=True):
                        with patch.object(
                            api._upscale_processor,
                            "get_capabilities",
                            return_value={
                                "success": True,
                                "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                            },
                        ):
                            with patch.object(api._upscale_processor, "upscale_video", side_effect=fake_upscale_video):
                                started = api.start_upscale(
                                    {
                                        "input_path": input_path,
                                        "output_dir": td,
                                        "mode": "upscale_resolution",
                                        "engine": "seedvr2",
                                        "model_id": "seedvr2_3b_q4_k_m_gguf",
                                        "target_preset": "1080p",
                                    }
                                )
                                self.assertTrue(started["success"])
                                self._wait_for_task_state(api, {"running"})
                                cancelled = api.cancel_upscale_task()
                                self.assertTrue(cancelled["success"])
                                self._wait_for_task_state(api, {"cancelled"})

            release_mock.assert_called_once_with("cancelled")


if __name__ == "__main__":
    unittest.main()
