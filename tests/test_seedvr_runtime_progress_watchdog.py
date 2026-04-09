"""SeedVR runtime stall watchdog tests."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.seedvr_runtime import SeedVRRuntime, SeedVRRuntimeError


class SeedVRRuntimeProgressWatchdogTests(unittest.TestCase):
    def test_warmup_watchdog_triggers_when_no_activity(self):
        runtime = SeedVRRuntime()
        with tempfile.TemporaryDirectory() as td:
            watch_dir = Path(td)
            cmd = [sys.executable, "-c", "import time; time.sleep(2.0)"]
            with patch("src.core.seedvr_runtime.WARMUP_STALL_TIMEOUT_SEC", 0.5):
                with self.assertRaises(SeedVRRuntimeError) as ctx:
                    runtime._run_command(
                        cmd,
                        cancel_event=None,
                        progress_callback=None,
                        message="Running SeedVR2 inference...",
                        timeout_sec=5.0,
                        parse_progress=True,
                        output_watch_dir=watch_dir,
                    )
        self.assertIn("Warmup stalled", str(ctx.exception))

    def test_watchdog_does_not_trigger_when_progress_is_emitted(self):
        runtime = SeedVRRuntime()
        with tempfile.TemporaryDirectory() as td:
            watch_dir = Path(td)
            out_file = watch_dir / "out.txt"
            code = (
                "import time, pathlib\n"
                f"p = pathlib.Path(r'{out_file}')\n"
                "print('PROGRESS chunk=1/2 frames=1/2', flush=True)\n"
                "p.write_text('a', encoding='utf-8')\n"
                "time.sleep(0.15)\n"
                "print('PROGRESS chunk=2/2 frames=2/2', flush=True)\n"
                "p.write_text('b', encoding='utf-8')\n"
            )
            cmd = [sys.executable, "-u", "-c", code]
            with patch("src.core.seedvr_runtime.WARMUP_STALL_TIMEOUT_SEC", 0.5):
                with patch("src.core.seedvr_runtime.RUN_STALL_TIMEOUT_SEC", 0.5):
                    runtime._run_command(
                        cmd,
                        cancel_event=None,
                        progress_callback=None,
                        message="Running SeedVR2 inference...",
                        timeout_sec=5.0,
                        parse_progress=True,
                        output_watch_dir=watch_dir,
                    )
            self.assertTrue(out_file.exists())
            self.assertEqual(out_file.read_text(encoding="utf-8"), "b")

    def test_warmup_heartbeat_does_not_false_positive(self):
        runtime = SeedVRRuntime()
        with tempfile.TemporaryDirectory() as td:
            watch_dir = Path(td)
            code = (
                "import time\n"
                "for _ in range(12):\n"
                "    print('PROGRESS stage=warmup step=load_model_done', flush=True)\n"
                "    time.sleep(0.04)\n"
            )
            cmd = [sys.executable, "-u", "-c", code]
            with patch("src.core.seedvr_runtime.WARMUP_STALL_TIMEOUT_SEC", 0.2):
                runtime._run_command(
                    cmd,
                    cancel_event=None,
                    progress_callback=None,
                    message="Running SeedVR2 inference...",
                    timeout_sec=5.0,
                    parse_progress=True,
                    output_watch_dir=watch_dir,
                )

    def test_run_watchdog_triggers_without_forward_progress(self):
        runtime = SeedVRRuntime()
        with tempfile.TemporaryDirectory() as td:
            watch_dir = Path(td)
            code = (
                "import time\n"
                "print('PROGRESS chunk=1/2 frames=1/2', flush=True)\n"
                "for _ in range(20):\n"
                "    print('heartbeat', flush=True)\n"
                "    time.sleep(0.05)\n"
            )
            cmd = [sys.executable, "-u", "-c", code]
            with patch("src.core.seedvr_runtime.WARMUP_STALL_TIMEOUT_SEC", 1.0):
                with patch("src.core.seedvr_runtime.RUN_STALL_TIMEOUT_SEC", 0.4):
                    with self.assertRaises(SeedVRRuntimeError) as ctx:
                        runtime._run_command(
                            cmd,
                            cancel_event=None,
                            progress_callback=None,
                            message="Running SeedVR2 inference...",
                            timeout_sec=5.0,
                            parse_progress=True,
                            output_watch_dir=watch_dir,
                        )
        self.assertIn("Inference stalled", str(ctx.exception))

    def test_cpu_time_growth_does_not_exit_warmup_early(self):
        runtime = SeedVRRuntime()
        with tempfile.TemporaryDirectory() as td:
            watch_dir = Path(td)
            code = (
                "import time\n"
                "x = 0\n"
                "end = time.time() + 0.3\n"
                "while time.time() < end:\n"
                "    x += 1\n"
                "print('done', x)\n"
            )
            cmd = [sys.executable, "-u", "-c", code]
            with patch("src.core.seedvr_runtime.WARMUP_STALL_TIMEOUT_SEC", 0.6):
                with patch("src.core.seedvr_runtime.RUN_STALL_TIMEOUT_SEC", 0.1):
                    runtime._run_command(
                        cmd,
                        cancel_event=None,
                        progress_callback=None,
                        message="Running SeedVR2 inference...",
                        timeout_sec=5.0,
                        parse_progress=True,
                        output_watch_dir=watch_dir,
                    )

    def test_cpu_activity_without_progress_still_triggers_run_stall(self):
        runtime = SeedVRRuntime()
        with tempfile.TemporaryDirectory() as td:
            watch_dir = Path(td)
            code = (
                "import time\n"
                "print('PROGRESS chunk=1/2 frames=1/2', flush=True)\n"
                "x = 0\n"
                "end = time.time() + 0.6\n"
                "while time.time() < end:\n"
                "    x += 1\n"
                "print('PROGRESS chunk=2/2 frames=2/2', flush=True)\n"
            )
            cmd = [sys.executable, "-u", "-c", code]
            with patch("src.core.seedvr_runtime.WARMUP_STALL_TIMEOUT_SEC", 0.5):
                with patch("src.core.seedvr_runtime.RUN_STALL_TIMEOUT_SEC", 0.2):
                    with self.assertRaises(SeedVRRuntimeError) as ctx:
                        runtime._run_command(
                            cmd,
                            cancel_event=None,
                            progress_callback=None,
                            message="Running SeedVR2 inference...",
                            timeout_sec=5.0,
                            parse_progress=True,
                            output_watch_dir=watch_dir,
                        )
        self.assertIn("Inference stalled", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
