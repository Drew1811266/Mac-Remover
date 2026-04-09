"""ProPainter 进度桥接测试。"""

import io
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.core.optional_adapters.propainter_adapter import Adapter


class _FakePopen:
    """用于模拟可解析 tqdm 输出的 Popen。"""

    def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None, text=None, bufsize=None):
        del cwd, env, stdout, stderr, text, bufsize
        output_root = Path(cmd[cmd.index("--output") + 1])
        video_dir = Path(cmd[cmd.index("--video") + 1])
        output_video = output_root / video_dir.name / "inpaint_out.mp4"
        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_video.write_bytes(b"ok")

        self.stdout = io.StringIO("infer 1/4 [00:00<00:03]\ninfer 4/4 [00:01<00:00]\n")
        self.stderr = io.StringIO("")
        self._start = time.monotonic()
        self._duration = 0.2
        self.returncode = 0

    def poll(self):
        if time.monotonic() - self._start >= self._duration:
            return self.returncode
        return None

    def kill(self):
        self.returncode = 1
        self._duration = 0.0


class _HeartbeatOnlyPopen:
    """用于模拟不可解析日志，触发心跳 floor。"""

    def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None, text=None, bufsize=None):
        del cwd, env, stdout, stderr, text, bufsize
        output_root = Path(cmd[cmd.index("--output") + 1])
        video_dir = Path(cmd[cmd.index("--video") + 1])
        output_video = output_root / video_dir.name / "inpaint_out.mp4"
        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_video.write_bytes(b"ok")

        self.stdout = io.StringIO("starting propainter backend...\n")
        self.stderr = io.StringIO("")
        self._start = time.monotonic()
        self._duration = 1.3
        self.returncode = 0

    def poll(self):
        if time.monotonic() - self._start >= self._duration:
            return self.returncode
        return None

    def kill(self):
        self.returncode = 1
        self._duration = 0.0


class ProPainterProgressBridgeTests(unittest.TestCase):
    def _run_adapter(self, popen_cls):
        adapter = Adapter()
        frames = [np.zeros((24, 24, 3), dtype=np.uint8) for _ in range(4)]
        masks = [np.zeros((24, 24), dtype=np.uint8) for _ in range(4)]
        progress_events = []

        with patch.object(Adapter, "load", return_value=None):
            with patch("src.core.optional_adapters.propainter_adapter.subprocess.Popen", popen_cls):
                with patch.object(
                    Adapter,
                    "_read_video_frames",
                    return_value=[frame.copy() for frame in frames],
                ):
                    result = adapter.inpaint_roi_sequence(
                        frames,
                        masks,
                        progress_callback=lambda payload: progress_events.append(dict(payload)),
                    )
        return result, progress_events

    def test_parse_progress_line_supports_ratio_and_percent(self):
        ratio = Adapter._parse_progress_line("infer 12/40 [00:00<00:02]")
        percent = Adapter._parse_progress_line("inference 35.5%")
        self.assertIsNotNone(ratio)
        self.assertIsNotNone(percent)
        self.assertAlmostEqual(float(ratio["progress"]), 12.0 / 40.0, places=6)
        self.assertAlmostEqual(float(percent["progress"]), 0.355, places=6)

    def test_streaming_progress_parses_tqdm_lines(self):
        result, events = self._run_adapter(_FakePopen)
        self.assertEqual(len(result), 4)
        self.assertTrue(events)
        self.assertTrue(any(event.get("step") == 1 and event.get("total") == 4 for event in events))
        self.assertTrue(any(float(event.get("progress", 0.0)) >= 1.0 for event in events))

    def test_heartbeat_floor_emits_when_logs_not_parseable(self):
        result, events = self._run_adapter(_HeartbeatOnlyPopen)
        self.assertEqual(len(result), 4)
        self.assertTrue(events)
        heartbeat_events = [
            event for event in events
            if "heartbeat" in str(event.get("message", "")).lower()
        ]
        self.assertTrue(heartbeat_events)
        self.assertTrue(any(float(event.get("progress", 0.0)) > 0.0 for event in heartbeat_events))


if __name__ == "__main__":
    unittest.main()
