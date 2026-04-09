"""FFmpeg 运行时解析测试。

目标：验证内置 vendor 路径优先级、系统回退和缺失场景表现。
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils import ffmpeg_runtime


def _write_executable(path: Path, script: str) -> None:
    """在临时目录写入可执行脚本，模拟 ffmpeg/ffprobe 二进制。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding='utf-8')
    path.chmod(0o755)


class FFmpegRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        """每个用例前设置独立 vendor 目录并清空缓存。"""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._previous_vendor_env = os.environ.get('WMR_FFMPEG_VENDOR_DIR')
        os.environ['WMR_FFMPEG_VENDOR_DIR'] = str(Path(self._tmp.name) / 'vendor')
        ffmpeg_runtime.clear_ffmpeg_runtime_cache()

    def tearDown(self) -> None:
        """恢复环境变量并清理缓存，避免测试互相污染。"""
        if self._previous_vendor_env is None:
            os.environ.pop('WMR_FFMPEG_VENDOR_DIR', None)
        else:
            os.environ['WMR_FFMPEG_VENDOR_DIR'] = self._previous_vendor_env
        ffmpeg_runtime.clear_ffmpeg_runtime_cache()

    def test_resolve_ffmpeg_prefers_embedded(self):
        """验证 ffmpeg 优先命中内置 vendor 目录。"""
        vendor = Path(os.environ['WMR_FFMPEG_VENDOR_DIR'])
        embedded = vendor / 'darwin-arm64' / 'ffmpeg'
        _write_executable(embedded, '#!/bin/sh\nexit 0\n')

        with patch('src.utils.ffmpeg_runtime.platform.system', return_value='Darwin'):
            with patch('src.utils.ffmpeg_runtime.platform.machine', return_value='arm64'):
                with patch('src.utils.ffmpeg_runtime.shutil.which', return_value='/usr/bin/ffmpeg'):
                    resolved = ffmpeg_runtime.resolve_ffmpeg_path()

        self.assertEqual(resolved, str(embedded))

    def test_resolve_ffprobe_prefers_embedded(self):
        """验证 ffprobe 优先命中内置 vendor 目录。"""
        vendor = Path(os.environ['WMR_FFMPEG_VENDOR_DIR'])
        embedded = vendor / 'darwin-arm64' / 'ffprobe'
        _write_executable(embedded, '#!/bin/sh\necho audio\n')

        with patch('src.utils.ffmpeg_runtime.platform.system', return_value='Darwin'):
            with patch('src.utils.ffmpeg_runtime.platform.machine', return_value='arm64'):
                with patch('src.utils.ffmpeg_runtime.shutil.which', return_value='/usr/bin/ffprobe'):
                    resolved = ffmpeg_runtime.resolve_ffprobe_path()

        self.assertEqual(resolved, str(embedded))

    def test_fallback_to_system_path_when_embedded_missing(self):
        """验证内置缺失时会回退系统 PATH。"""
        with patch('src.utils.ffmpeg_runtime.platform.system', return_value='Darwin'):
            with patch('src.utils.ffmpeg_runtime.platform.machine', return_value='arm64'):
                with patch(
                    'src.utils.ffmpeg_runtime.shutil.which',
                    side_effect=lambda tool: '/usr/local/bin/' + tool,
                ):
                    ffmpeg_path = ffmpeg_runtime.resolve_ffmpeg_path()
                    ffprobe_path = ffmpeg_runtime.resolve_ffprobe_path()

        self.assertEqual(ffmpeg_path, '/usr/local/bin/ffmpeg')
        self.assertEqual(ffprobe_path, '/usr/local/bin/ffprobe')

    def test_missing_both_returns_none_and_logs(self):
        """验证内置和系统都缺失时返回 None 且 source=missing。"""
        with patch('src.utils.ffmpeg_runtime.platform.system', return_value='Darwin'):
            with patch('src.utils.ffmpeg_runtime.platform.machine', return_value='arm64'):
                with patch('src.utils.ffmpeg_runtime.shutil.which', return_value=None):
                    ffmpeg_path = ffmpeg_runtime.resolve_ffmpeg_path()
                    ffprobe_path = ffmpeg_runtime.resolve_ffprobe_path()
                    info = ffmpeg_runtime.runtime_ffmpeg_info()

        self.assertIsNone(ffmpeg_path)
        self.assertIsNone(ffprobe_path)
        self.assertEqual(info['ffmpeg']['source'], 'missing')
        self.assertEqual(info['ffprobe']['source'], 'missing')


if __name__ == '__main__':
    unittest.main()
