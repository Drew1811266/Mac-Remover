"""SeedVR runtime command assembly tests."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core import seedvr_runtime as runtime_module
from src.core.seedvr_runtime import SeedVRRuntime


class SeedVRRuntimeTests(unittest.TestCase):
    def _run_with_capture(
        self,
        *,
        cache_dit: bool = False,
        cache_vae: bool = False,
        ffmpeg_bin: str = "",
    ) -> tuple[list[str], str, dict[str, object]]:
        runtime = SeedVRRuntime()
        captured: dict[str, object] = {}

        def fake_run_command(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["message"] = kwargs.get("message", "")
            captured["env_overrides"] = kwargs.get("env_overrides", {})

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            fake_result = out_dir / "generated.mp4"
            with patch.object(runtime, "get_status", return_value={"ready": True}):
                with patch.object(runtime, "_run_command", side_effect=fake_run_command):
                    with patch.object(runtime, "_find_latest_video", return_value=fake_result):
                        runtime.run_inference(
                            input_path=str(Path(td) / "input.mp4"),
                            output_dir=str(out_dir),
                            dit_model_name="seedvr2_ema_3b-Q4_K_M.gguf",
                            target_short_resolution=1080,
                            denoise_strength=0.35,
                            same_res_strength="x2_then_downscale",
                            cache_dit=cache_dit,
                            cache_vae=cache_vae,
                            ffmpeg_bin=ffmpeg_bin,
                        )

        return list(captured["cmd"]), str(captured["message"]), captured

    def test_run_inference_defaults_to_mps_first_without_cache(self):
        cmd, message, _ = self._run_with_capture()
        self.assertGreaterEqual(len(cmd), 2)
        self.assertEqual(cmd[1], "-u")
        self.assertIn("--video_backend", cmd)
        self.assertEqual(cmd[cmd.index("--video_backend") + 1], "ffmpeg")
        self.assertIn("--dit_offload_device", cmd)
        self.assertIn("--vae_offload_device", cmd)
        self.assertIn("--tensor_offload_device", cmd)
        self.assertEqual(cmd[cmd.index("--dit_offload_device") + 1], "none")
        self.assertEqual(cmd[cmd.index("--vae_offload_device") + 1], "none")
        self.assertEqual(cmd[cmd.index("--tensor_offload_device") + 1], "cpu")
        self.assertNotIn("--cache_dit", cmd)
        self.assertNotIn("--cache_vae", cmd)
        self.assertIn("device=mps", message)
        self.assertIn("backend=ffmpeg", message)
        self.assertIn("offload=none", message)

    def test_run_inference_adds_cache_flags_only_when_enabled(self):
        cmd, _, _ = self._run_with_capture(cache_dit=True, cache_vae=True)
        self.assertIn("--cache_dit", cmd)
        self.assertIn("--cache_vae", cmd)

    def test_run_inference_injects_ffmpeg_bin_dir_into_path(self):
        cmd, _, captured = self._run_with_capture(ffmpeg_bin="/opt/homebrew/bin/ffmpeg")
        self.assertEqual(cmd[cmd.index("--video_backend") + 1], "ffmpeg")
        env_overrides = dict(captured.get("env_overrides") or {})
        self.assertIn("PATH", env_overrides)
        self.assertTrue(str(env_overrides.get("PATH") or "").startswith("/opt/homebrew/bin:"))

    def test_build_env_injects_thread_caps(self):
        with patch("src.core.seedvr_runtime.os.cpu_count", return_value=10):
            env = SeedVRRuntime._build_env()
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual(env.get(key), "8")
        self.assertEqual(env.get("PYTHONUNBUFFERED"), "1")

    def test_watchdog_timeout_constants_are_split(self):
        self.assertGreater(runtime_module.WARMUP_STALL_TIMEOUT_SEC, runtime_module.RUN_STALL_TIMEOUT_SEC)
        self.assertEqual(runtime_module.WARMUP_STALL_TIMEOUT_SEC, 240.0)
        self.assertEqual(runtime_module.RUN_STALL_TIMEOUT_SEC, 90.0)

    def test_load_governor_skips_throttle_during_warmup_grace(self):
        runtime = SeedVRRuntime()
        with tempfile.TemporaryDirectory() as td:
            watch_dir = Path(td)
            cmd = [
                sys.executable,
                "-u",
                "-c",
                (
                    "import time\n"
                    "for _ in range(8):\n"
                    "    print('PROGRESS stage=warmup step=load_model_done', flush=True)\n"
                    "    time.sleep(0.03)\n"
                ),
            ]
            with patch("src.core.seedvr_runtime.LOAD_CAP_MIN_STREAK", 1):
                with patch("src.core.seedvr_runtime.LOAD_CAP_WARMUP_GRACE_SEC", 999.0):
                    with patch.object(SeedVRRuntime, "_normalize_cpu_percent", return_value=100.0):
                        with patch.object(SeedVRRuntime, "_throttle_process", return_value=0.0) as throttle_mock:
                            runtime._run_command(
                                cmd,
                                cancel_event=None,
                                progress_callback=None,
                                message="Running SeedVR2 inference...",
                                timeout_sec=5.0,
                                parse_progress=True,
                                output_watch_dir=watch_dir,
                            )
                            throttle_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
