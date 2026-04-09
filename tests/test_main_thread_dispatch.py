"""主线程调度工具测试。"""

import threading
import unittest
from unittest.mock import patch

from src.gui import main_thread_dispatch as dispatch


class MainThreadDispatchTests(unittest.TestCase):
    def test_run_on_main_sync_direct_on_non_darwin(self):
        with patch.object(dispatch, "_IS_DARWIN", False):
            value = dispatch.run_on_main_sync(lambda x, y: x + y, 2, 3)
        self.assertEqual(value, 5)

    def test_run_on_main_sync_uses_call_after_when_not_on_main(self):
        class _FakeAppHelper:
            @staticmethod
            def callAfter(fn):
                threading.Thread(target=fn, daemon=True).start()

        class _FakeNSThread:
            @staticmethod
            def isMainThread():
                return False

        with patch.object(dispatch, "_IS_DARWIN", True):
            with patch.object(dispatch, "_AppHelper", _FakeAppHelper):
                with patch.object(dispatch, "_NSThread", _FakeNSThread):
                    value = dispatch.run_on_main_sync(lambda x: x * 2, 7, timeout_sec=1.0)
        self.assertEqual(value, 14)

    def test_run_on_main_sync_timeout_raises(self):
        class _FakeAppHelper:
            @staticmethod
            def callAfter(_fn):
                # 模拟主线程消息循环丢失，不执行回调。
                return None

        class _FakeNSThread:
            @staticmethod
            def isMainThread():
                return False

        with patch.object(dispatch, "_IS_DARWIN", True):
            with patch.object(dispatch, "_AppHelper", _FakeAppHelper):
                with patch.object(dispatch, "_NSThread", _FakeNSThread):
                    with self.assertRaises(dispatch.MainThreadDispatchTimeoutError):
                        dispatch.run_on_main_sync(lambda: "never", timeout_sec=0.1)


if __name__ == "__main__":
    unittest.main()
