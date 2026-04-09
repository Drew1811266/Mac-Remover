"""SeedVR runtime backend fallback tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.seedvr_runtime import SeedVRRuntime, SeedVRRuntimeError


class SeedVRRuntimeBackendFallbackTests(unittest.TestCase):
    def test_run_inference_falls_back_to_opencv_after_ffmpeg_failure(self):
        runtime = SeedVRRuntime()
        seen_cmds: list[list[str]] = []
        messages: list[str] = []

        def fake_run_command(cmd, **kwargs):
            seen_cmds.append(list(cmd))
            backend = cmd[cmd.index("--video_backend") + 1]
            if backend == "ffmpeg":
                raise SeedVRRuntimeError(
                    "SeedVR command failed (code 1): --video_backend ffmpeg unknown encoder"
                )

        def progress(payload):
            messages.append(str(payload.get("message") or ""))

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            fake_result = out_dir / "generated.mp4"
            with patch.object(runtime, "get_status", return_value={"ready": True}):
                with patch.object(runtime, "_run_command", side_effect=fake_run_command):
                    with patch.object(runtime, "_find_latest_video", return_value=fake_result):
                        result = runtime.run_inference(
                            input_path=str(Path(td) / "input.mp4"),
                            output_dir=str(out_dir),
                            dit_model_name="seedvr2_ema_3b-Q4_K_M.gguf",
                            target_short_resolution=1080,
                            denoise_strength=0.2,
                            same_res_strength="x2_then_downscale",
                            ffmpeg_bin="/opt/homebrew/bin/ffmpeg",
                            progress_callback=progress,
                        )

        self.assertEqual(result, str(fake_result))
        self.assertEqual(len(seen_cmds), 2)
        self.assertEqual(seen_cmds[0][seen_cmds[0].index("--video_backend") + 1], "ffmpeg")
        self.assertEqual(seen_cmds[1][seen_cmds[1].index("--video_backend") + 1], "opencv")
        self.assertTrue(any("retrying with OpenCV backend" in msg for msg in messages))


if __name__ == "__main__":
    unittest.main()

