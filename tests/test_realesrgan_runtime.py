"""Real-ESRGAN runtime 基础行为测试。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.realesrgan_runtime import RealESRGANRuntime


class RealESRGANRuntimeTests(unittest.TestCase):
    def test_get_status_reports_python_missing(self):
        runtime = RealESRGANRuntime()
        with patch("src.core.realesrgan_runtime.resolve_python312", return_value=""):
            status = runtime.get_status()
        self.assertFalse(status["ready"])
        self.assertIn("Python 3.12", status["reason"])

    def test_run_inference_builds_unbuffered_worker_command(self):
        runtime = RealESRGANRuntime()
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"input")

            captured = {"cmd": []}

            def fake_run_command(cmd, **kwargs):
                captured["cmd"] = list(cmd)
                output_flag = "--output"
                out_path = cmd[cmd.index(output_flag) + 1]
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_bytes(b"result")

            with patch.object(runtime, "get_status", return_value={"ready": True}):
                with patch.object(runtime, "_run_command", side_effect=fake_run_command):
                    output = runtime.run_inference(
                        input_path=input_path,
                        output_dir=td,
                        model_id="realesrgan_general_x4v3",
                        outscale=1.5,
                        denoise_strength=0.35,
                        tile=256,
                        tile_pad=10,
                        pre_pad=0,
                    )

            self.assertTrue(output.endswith("realesrgan_output.mp4"))
            self.assertIn("-u", captured["cmd"])
            self.assertIn("--model-id", captured["cmd"])
            self.assertIn("realesrgan_general_x4v3", captured["cmd"])


if __name__ == "__main__":
    unittest.main()

