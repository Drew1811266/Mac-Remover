"""模型下载 API 与原子替换逻辑测试。

覆盖内容：
1. GUI API 下载状态结构与状态迁移。
2. 并发下载拦截和取消行为。
3. 下载目录原子替换失败时的回滚保护。
"""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core import model_downloader
from src.core.model_downloader import DownloadCancelled
from src.gui.api import API


class ModelDownloadApiTests(unittest.TestCase):
    def setUp(self):
        """禁用原生播放器初始化，避免测试环境依赖 macOS GUI。"""
        self.native_patch = patch.object(API, "_init_native_player", lambda _self: None)
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)

    @staticmethod
    def _wait_for_state(api: API, expected_states, timeout=5.0):
        """轮询下载任务状态，直到命中目标状态或超时。"""
        started = time.time()
        while time.time() - started < timeout:
            task = api.get_model_download_status().get("task", {})
            state = task.get("state")
            if state in expected_states:
                return task
            time.sleep(0.03)
        raise AssertionError(f"Timeout waiting for states {expected_states}")

    def test_get_model_download_status_shape(self):
        """验证下载状态接口返回字段完整且类型合理。"""
        with patch(
            "src.gui.api.list_model_download_entries",
            return_value=[
                {
                    "model_id": "lama_roi",
                    "display_name": "LaMa-ROI",
                    "installed": True,
                    "can_redownload": True,
                    "install_hint": "hint",
                }
            ],
        ):
            api = API()
            payload = api.get_model_download_status()

        self.assertTrue(payload["success"])
        self.assertIn("models", payload)
        self.assertIn("task", payload)
        self.assertEqual(payload["models"][0]["model_id"], "lama_roi")

        task = payload["task"]
        self.assertEqual(task["state"], "idle")
        self.assertEqual(task["progress"], 0.0)
        self.assertEqual(task["downloaded_bytes"], 0)
        self.assertIn("message", task)

    def test_start_download_rejects_when_running(self):
        """验证已有下载任务运行时，第二次启动会被拒绝。"""
        unblock = {"done": False}

        def fake_download(model_id, force=False, progress_callback=None, cancel_event=None):
            if progress_callback:
                progress_callback(
                    {
                        "progress": 0.1,
                        "downloaded_bytes": 10,
                        "total_bytes": 100,
                        "speed_bps": 1024,
                        "current_file": "file-a",
                        "message": "running",
                        "error": "",
                    }
                )
            while not unblock["done"]:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled("cancelled")
                time.sleep(0.02)
            if progress_callback:
                progress_callback(
                    {
                        "progress": 1.0,
                        "downloaded_bytes": 100,
                        "total_bytes": 100,
                        "speed_bps": 0,
                        "current_file": "file-a",
                        "message": "done",
                        "error": "",
                    }
                )
            return {"model_id": model_id, "installed": True, "skipped": False}

        with patch("src.gui.api.download_model", side_effect=fake_download):
            api = API()
            start_1 = api.start_model_download({"model_id": "lama_roi", "force": False})
            self.assertTrue(start_1["success"])

            self._wait_for_state(api, {"running"})

            start_2 = api.start_model_download({"model_id": "lama_roi", "force": False})
            self.assertFalse(start_2["success"])
            self.assertIn("already running", start_2["error"])

            unblock["done"] = True
            self._wait_for_state(api, {"success"})

    def test_cancel_download_sets_cancelled_state(self):
        """验证取消请求最终会把任务状态推进到 cancelled。"""
        def fake_download(model_id, force=False, progress_callback=None, cancel_event=None):
            if progress_callback:
                progress_callback(
                    {
                        "progress": 0.2,
                        "downloaded_bytes": 20,
                        "total_bytes": 100,
                        "speed_bps": 2048,
                        "current_file": "file-b",
                        "message": "running",
                        "error": "",
                    }
                )
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled("cancelled")
                time.sleep(0.02)

        with patch("src.gui.api.download_model", side_effect=fake_download):
            api = API()
            started = api.start_model_download({"model_id": "lama_roi", "force": True})
            self.assertTrue(started["success"])

            self._wait_for_state(api, {"running"})
            cancelled = api.cancel_model_download()
            self.assertTrue(cancelled["success"])

            task = self._wait_for_state(api, {"cancelled"})
            self.assertEqual(task["state"], "cancelled")


class ModelDownloaderAtomicReplaceTests(unittest.TestCase):
    def test_replace_directory_atomically_restores_previous_on_failure(self):
        """验证目录替换失败时会恢复旧目录，不丢历史内容。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target"
            staged = root / "staged"

            target.mkdir(parents=True, exist_ok=True)
            (target / "old.txt").write_text("old", encoding="utf-8")
            staged.mkdir(parents=True, exist_ok=True)
            (staged / "new.txt").write_text("new", encoding="utf-8")

            real_replace = os.replace

            def flaky_replace(src, dst):
                if src == str(staged) and dst == str(target):
                    raise RuntimeError("inject failure")
                return real_replace(src, dst)

            with patch("src.core.model_downloader.os.replace", side_effect=flaky_replace):
                with self.assertRaisesRegex(RuntimeError, "inject failure"):
                    model_downloader._replace_directory_atomically(staged, target)

            self.assertTrue((target / "old.txt").exists())
            self.assertEqual((target / "old.txt").read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main()
