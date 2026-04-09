"""SeedVR2 放大处理器核心逻辑测试。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.scene_splitter import SceneSegment, SceneSplitResult
from src.core.seedvr_runtime import SeedVRRuntimeError
from src.core.upscale_processor import UpscaleCancelled, UpscaleProcessor


class UpscaleProcessorTests(unittest.TestCase):
    def test_stall_detection_treats_timeout_as_retryable(self):
        processor = UpscaleProcessor()
        self.assertTrue(processor._is_stall_error("SeedVR command timed out after 360s"))
        self.assertTrue(processor._is_stall_error("FFmpeg backend produced no progress output after warmup"))

    def test_compute_target_resolution_keeps_aspect_and_avoids_downscale(self):
        processor = UpscaleProcessor()
        # 720p -> 1080p
        w, h = processor._compute_target_resolution(1280, 720, "upscale_resolution", "1080p")
        self.assertEqual((w, h), (1920, 1080))
        # 已高于目标时不应降分辨率
        w2, h2 = processor._compute_target_resolution(2560, 1440, "upscale_resolution", "1080p")
        self.assertEqual((w2, h2), (2560, 1440))
        # 同分辨率增强保持原尺寸
        w3, h3 = processor._compute_target_resolution(1280, 720, "enhance_same_resolution", None)
        self.assertEqual((w3, h3), (1280, 720))

    def test_get_capabilities_shape(self):
        processor = UpscaleProcessor()
        with patch("src.core.upscale_processor.resolve_ffmpeg_path", return_value="/embedded/ffmpeg"):
            with patch.object(UpscaleProcessor, "_check_filter_available", return_value=True):
                with patch.object(processor._runtime, "get_status", return_value={"ready": True, "reason": ""}):
                    with patch(
                        "src.core.upscale_processor.is_upscale_model_installed",
                        side_effect=lambda model_id: model_id == "seedvr2_3b_q4_k_m_gguf",
                    ):
                        payload = processor.get_capabilities(force_refresh=True)
        self.assertTrue(payload["success"])
        self.assertIn("engines", payload)
        self.assertIn("models", payload)
        self.assertIn("defaults", payload)
        self.assertEqual(payload["defaults"]["engine"], "seedvr2")
        self.assertEqual(payload["same_res_strengths"], ["x2_then_downscale"])
        self.assertEqual(payload["target_presets"], ["1080p"])
        self.assertEqual(
            {entry["model_id"] for entry in payload["models"]},
            {
                "realesrgan_general_x4v3",
                "realesrgan_x2plus",
                "seedvr2_3b_q8_0_gguf",
                "seedvr2_3b_q4_k_m_gguf",
            },
        )
        self.assertEqual(payload["defaults"]["model_id"], "seedvr2_3b_q4_k_m_gguf")

    def test_prepare_seedvr_input_720p_auto_preprocesses_non_720p(self):
        processor = UpscaleProcessor()
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"stub-input")

            def fake_ffmpeg(self, **kwargs):
                out_path = Path(kwargs["cmd"][-1])
                out_path.write_bytes(b"prepared")

            with patch.object(UpscaleProcessor, "_run_ffmpeg_with_progress", fake_ffmpeg):
                prepared_path, work_dir, warnings = processor._prepare_seedvr_input_720p(
                    ffmpeg_bin="/embedded/ffmpeg",
                    input_path=input_path,
                    source_w=1920,
                    source_h=1080,
                    duration_sec=20.0,
                    cancel_event=None,
                    progress_callback=None,
                )

            self.assertNotEqual(prepared_path, input_path)
            self.assertIsNotNone(work_dir)
            self.assertTrue(os.path.exists(prepared_path))
            self.assertTrue(any("preprocessed input to 720p" in item.lower() for item in warnings))

    def test_ffmpeg_commands_include_thread_cap(self):
        processor = UpscaleProcessor()
        with tempfile.TemporaryDirectory() as td:
            source_path = os.path.join(td, "source.mp4")
            video_path = os.path.join(td, "video.mp4")
            output_path = os.path.join(td, "out.mp4")
            seg_a = os.path.join(td, "seg_a.mp4")
            seg_b = os.path.join(td, "seg_b.mp4")
            Path(source_path).write_bytes(b"src")
            Path(video_path).write_bytes(b"video")
            Path(seg_a).write_bytes(b"a")
            Path(seg_b).write_bytes(b"b")

            captured_cmds: list[list[str]] = []

            def fake_ffmpeg(_self, **kwargs):
                cmd = list(kwargs["cmd"])
                captured_cmds.append(cmd)
                Path(cmd[-1]).write_bytes(b"ok")

            with patch.object(UpscaleProcessor, "_resolve_ffmpeg_thread_cap", return_value=6):
                with patch.object(UpscaleProcessor, "_run_ffmpeg_with_progress", fake_ffmpeg):
                    processor._prepare_seedvr_input_720p(
                        ffmpeg_bin="/embedded/ffmpeg",
                        input_path=source_path,
                        source_w=1920,
                        source_h=1080,
                        duration_sec=8.0,
                        cancel_event=None,
                        progress_callback=None,
                    )
                    processor._extract_segment_without_audio(
                        ffmpeg_bin="/embedded/ffmpeg",
                        source_path=source_path,
                        output_path=os.path.join(td, "segment.mp4"),
                        start_sec=0.0,
                        end_sec=1.0,
                        cancel_event=None,
                        progress_callback=None,
                        progress_start=0.2,
                        progress_span=0.2,
                        message="extract",
                        segment_index=1,
                        segment_total=2,
                        scene_split_mode="rule",
                    )
                    processor._concat_segments(
                        ffmpeg_bin="/embedded/ffmpeg",
                        segment_paths=[seg_a, seg_b],
                        output_path=output_path,
                        work_dir=Path(td),
                        duration_sec=2.0,
                        cancel_event=None,
                        progress_callback=None,
                        progress_start=0.8,
                        progress_span=0.1,
                        scene_split_mode="rule",
                    )
                    processor._mux_audio(
                        ffmpeg_bin="/embedded/ffmpeg",
                        video_path=video_path,
                        source_path=source_path,
                        output_path=os.path.join(td, "mux.mp4"),
                        keep_audio=True,
                        duration_sec=2.0,
                        cancel_event=None,
                        progress_callback=None,
                        scene_split_mode="rule",
                    )
                    processor._finalize_with_ffmpeg(
                        ffmpeg_bin="/embedded/ffmpeg",
                        ai_output_path=video_path,
                        source_path=None,
                        output_path=os.path.join(td, "final.mp4"),
                        mode="upscale_resolution",
                        source_w=1280,
                        source_h=720,
                        target_w=1920,
                        target_h=1080,
                        same_res_strength="x2_then_downscale",
                        denoise_strength=0.3,
                        keep_audio=False,
                        duration_sec=2.0,
                        estimated_total_sec=2.0,
                        cancel_event=None,
                        progress_callback=None,
                        prefer_libplacebo=False,
                        scene_split_mode="rule",
                    )

            self.assertGreaterEqual(len(captured_cmds), 5)
            for cmd in captured_cmds:
                self.assertIn("-threads", cmd)
                self.assertEqual(cmd[cmd.index("-threads") + 1], "6")

    def test_upscale_video_runs_segmented_pipeline_and_builds_output(self):
        processor = UpscaleProcessor()
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"stub-input")

            with patch("src.core.upscale_processor.resolve_ffmpeg_path", return_value="/embedded/ffmpeg"):
                with patch.object(
                    UpscaleProcessor,
                    "get_capabilities",
                    return_value={
                        "success": True,
                        "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                        "ffmpeg": {"libplacebo_available": False},
                    },
                ):
                    with patch.object(
                        UpscaleProcessor,
                        "_video_meta",
                        side_effect=[
                            {"width": 1280, "height": 720, "fps": 24.0, "frame_count": 576, "duration_sec": 24.0},
                            {"width": 1280, "height": 720, "fps": 24.0, "frame_count": 240, "duration_sec": 10.0},
                            {"width": 1280, "height": 720, "fps": 24.0, "frame_count": 336, "duration_sec": 14.0},
                        ],
                    ):
                        output_holder = {"path": ""}
                        seen_segments = {"count": 0}

                        split_result = SceneSplitResult(
                            segments=(
                                SceneSegment(idx=1, start=0.0, end=4.0, duration=4.0),
                                SceneSegment(idx=2, start=4.0, end=10.0, duration=6.0),
                            ),
                            split_mode="hybrid",
                            warnings=("split warning",),
                            cuts=(4.0,),
                            stats={"ffmpeg_cut_count": 1},
                        )

                        def fake_split(_processor, **kwargs):
                            self.assertEqual(kwargs["input_path"], input_path)
                            return split_result

                        def fake_extract(self, **kwargs):
                            Path(kwargs["output_path"]).write_bytes(b"segment-input")

                        def fake_seedvr_infer(_processor, **kwargs):
                            seen_segments["count"] += 1
                            self.assertIn(kwargs["segment_index"], [1, 2])
                            self.assertEqual(kwargs["same_res_strength"], "x2_then_downscale")
                            work_dir = Path(tempfile.mkdtemp(prefix="ut-seedvr-", dir=td))
                            ai_output = work_dir / "ai_output.mp4"
                            ai_output.write_bytes(b"ai")
                            return str(ai_output), work_dir, ["auto safe profile"]

                        def fake_finalize(self, **kwargs):
                            Path(kwargs["output_path"]).write_bytes(b"segment-post")

                        def fake_concat(self, **kwargs):
                            Path(kwargs["output_path"]).write_bytes(b"merged")
                            return ["concat warning"]

                        def fake_mux(self, **kwargs):
                            output_path = kwargs["output_path"]
                            output_holder["path"] = output_path
                            Path(output_path).write_bytes(b"ok")

                        with patch("src.core.upscale_processor.release_unified_memory") as cleanup_mock:
                            with patch("src.core.upscale_processor.is_upscale_model_installed", return_value=True):
                                with patch.object(UpscaleProcessor, "_split_video_scenes", fake_split):
                                    with patch.object(UpscaleProcessor, "_extract_segment_without_audio", fake_extract):
                                        with patch.object(UpscaleProcessor, "_concat_segments", fake_concat):
                                            with patch.object(UpscaleProcessor, "_mux_audio", fake_mux):
                                                with patch.object(UpscaleProcessor, "_run_seedvr2_inference", fake_seedvr_infer):
                                                    with patch.object(UpscaleProcessor, "_finalize_with_ffmpeg", fake_finalize):
                                                        result = processor.upscale_video(
                                                            input_path=input_path,
                                                            output_dir=td,
                                                            mode="enhance_same_resolution",
                                                            engine="seedvr2",
                                                            model_id="seedvr2_3b_q4_k_m_gguf",
                                                            target_preset=None,
                                                            same_res_strength="x4_then_downscale",
                                                            denoise_strength=0.3,
                                                            keep_audio=True,
                                                        )
                        cleanup_reasons = [str(call.args[0]) for call in cleanup_mock.call_args_list if call.args]
                        self.assertIn("upscale_segment_finalize:1/2", cleanup_reasons)
                        self.assertIn("upscale_segment_finalize:2/2", cleanup_reasons)
                        self.assertIn("upscale_processor_finalize", cleanup_reasons)

            self.assertTrue(os.path.exists(output_holder["path"]))
            self.assertEqual(seen_segments["count"], 2)
            self.assertEqual(result["target_width"], 1280)
            self.assertEqual(result["target_height"], 720)
            self.assertEqual(result["effective_engine"], "seedvr2")
            self.assertEqual(result["model_id"], "seedvr2_3b_q4_k_m_gguf")
            self.assertEqual(result["scene_split_mode"], "hybrid")
            self.assertEqual(result["segment_total"], 2)
            self.assertIn("Automatically switched to x2", result["warning"])
            self.assertIn("segment-level memory cleanup", result["warning"])

    def test_upscale_video_cancelled_raises_upscale_cancelled(self):
        processor = UpscaleProcessor()
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"stub-input")
            with patch("src.core.upscale_processor.resolve_ffmpeg_path", return_value="/embedded/ffmpeg"):
                with patch.object(
                    UpscaleProcessor,
                    "get_capabilities",
                    return_value={
                        "success": True,
                        "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                        "ffmpeg": {"libplacebo_available": False},
                    },
                ):
                    with patch.object(
                        UpscaleProcessor,
                        "_video_meta",
                        return_value={
                            "width": 1280,
                            "height": 720,
                            "fps": 24.0,
                            "frame_count": 576,
                            "duration_sec": 24.0,
                        },
                    ):
                        with patch("src.core.upscale_processor.is_upscale_model_installed", return_value=True):
                            with patch.object(
                                UpscaleProcessor,
                                "_split_video_scenes",
                                side_effect=UpscaleCancelled("cancelled"),
                            ):
                                with self.assertRaises(UpscaleCancelled):
                                    processor.upscale_video(
                                        input_path=input_path,
                                        output_dir=td,
                                        mode="upscale_resolution",
                                        engine="seedvr2",
                                        model_id="seedvr2_3b_q4_k_m_gguf",
                                        target_preset="1080p",
                                        same_res_strength="x2_then_downscale",
                                        denoise_strength=0.3,
                                        keep_audio=True,
                                    )

    def test_upscale_video_short_clip_bypasses_scene_split(self):
        processor = UpscaleProcessor()
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"stub-input")

            with patch("src.core.upscale_processor.resolve_ffmpeg_path", return_value="/embedded/ffmpeg"):
                with patch.object(
                    UpscaleProcessor,
                    "get_capabilities",
                    return_value={
                        "success": True,
                        "engines": [{"engine": "seedvr2", "available": True, "reason": ""}],
                        "ffmpeg": {"libplacebo_available": False},
                    },
                ):
                    with patch.object(
                        UpscaleProcessor,
                        "_video_meta",
                        side_effect=[
                            {"width": 1280, "height": 720, "fps": 24.0, "frame_count": 240, "duration_sec": 10.0},
                            {"width": 1280, "height": 720, "fps": 24.0, "frame_count": 240, "duration_sec": 10.0},
                        ],
                    ):
                        with patch("src.core.upscale_processor.is_upscale_model_installed", return_value=True):
                            with patch.object(
                                UpscaleProcessor,
                                "_split_video_scenes",
                                side_effect=AssertionError("short clip should bypass scene splitter"),
                            ):
                                with patch.object(UpscaleProcessor, "_extract_segment_without_audio") as extract_mock:
                                    with patch.object(UpscaleProcessor, "_run_seedvr2_inference") as infer_mock:
                                        with patch.object(UpscaleProcessor, "_finalize_with_ffmpeg") as finalize_mock:
                                            with patch.object(UpscaleProcessor, "_concat_segments") as concat_mock:
                                                with patch.object(UpscaleProcessor, "_mux_audio") as mux_mock:
                                                    segment_input = Path(td) / "segment_input.mp4"
                                                    segment_post = Path(td) / "segment_post.mp4"
                                                    merged = Path(td) / "merged.mp4"
                                                    final_out = Path(td) / "final.mp4"

                                                    def fake_extract(**kwargs):
                                                        Path(kwargs["output_path"]).write_bytes(b"segment-input")

                                                    def fake_infer(**kwargs):
                                                        work_dir = Path(tempfile.mkdtemp(prefix="ut-seedvr-", dir=td))
                                                        ai_output = work_dir / "ai_output.mp4"
                                                        ai_output.write_bytes(b"ai")
                                                        return str(ai_output), work_dir, []

                                                    def fake_finalize(**kwargs):
                                                        Path(kwargs["output_path"]).write_bytes(b"segment-post")

                                                    def fake_concat(**kwargs):
                                                        Path(kwargs["output_path"]).write_bytes(b"merged")
                                                        return []

                                                    def fake_mux(**kwargs):
                                                        Path(kwargs["output_path"]).write_bytes(b"ok")

                                                    extract_mock.side_effect = fake_extract
                                                    infer_mock.side_effect = fake_infer
                                                    finalize_mock.side_effect = fake_finalize
                                                    concat_mock.side_effect = fake_concat
                                                    mux_mock.side_effect = fake_mux

                                                    result = processor.upscale_video(
                                                        input_path=input_path,
                                                        output_dir=td,
                                                        mode="upscale_resolution",
                                                        engine="seedvr2",
                                                        model_id="seedvr2_3b_q4_k_m_gguf",
                                                        target_preset="1080p",
                                                        same_res_strength="x2_then_downscale",
                                                        denoise_strength=0.3,
                                                        keep_audio=False,
                                                    )

            self.assertEqual(result["segment_total"], 1)
            self.assertEqual(result["scene_split_mode"], "bypass_short_video")
            self.assertIn("Scene split bypassed for short input", result["warning"])

    def test_run_seedvr2_inference_retries_on_memory_error(self):
        processor = UpscaleProcessor()
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"stub-input")

            calls = {"count": 0}
            captured_kwargs = []

            def fake_run_inference(**kwargs):
                calls["count"] += 1
                captured_kwargs.append(kwargs)
                if calls["count"] == 1:
                    raise SeedVRRuntimeError("MPS backend out of memory")
                out_dir = Path(kwargs["output_dir"])
                out_path = out_dir / "ok.mp4"
                out_path.write_bytes(b"ok")
                return str(out_path)

            with patch("src.core.upscale_processor.detect_system_memory_gb", return_value=(24.0, 9.0)):
                with patch.object(processor._runtime, "run_inference", side_effect=fake_run_inference):
                    generated, work_dir, warnings = processor._run_seedvr2_inference(
                        ffmpeg_bin="/embedded/ffmpeg",
                        input_path=input_path,
                        model_id="seedvr2_3b_q4_k_m_gguf",
                        mode="upscale_resolution",
                        same_res_strength="x2_then_downscale",
                        denoise_strength=0.35,
                        source_w=1280,
                        source_h=720,
                        target_w=1920,
                        target_h=1080,
                        duration_sec=50.0,
                        cancel_event=None,
                        progress_callback=None,
                        estimated_total_sec=60.0,
                    )

            self.assertEqual(calls["count"], 2)
            self.assertTrue(os.path.exists(generated))
            self.assertTrue(work_dir.exists())
            self.assertTrue(any("emergency low-memory profile" in w for w in warnings))
            self.assertTrue(any("MPS-first execution policy" in w for w in warnings))
            self.assertEqual(captured_kwargs[0]["dit_offload_device"], "none")
            self.assertEqual(captured_kwargs[0]["vae_offload_device"], "none")
            self.assertEqual(captured_kwargs[0]["tensor_offload_device"], "cpu")
            self.assertEqual(captured_kwargs[1]["dit_offload_device"], "cpu")
            self.assertEqual(captured_kwargs[1]["vae_offload_device"], "cpu")
            self.assertEqual(captured_kwargs[1]["tensor_offload_device"], "cpu")
            self.assertEqual(captured_kwargs[0]["video_backend_preference"], "ffmpeg")
            self.assertEqual(captured_kwargs[0]["ffmpeg_bin"], "/embedded/ffmpeg")
            self.assertFalse(captured_kwargs[0]["cache_dit"])
            self.assertFalse(captured_kwargs[0]["cache_vae"])
            self.assertFalse(captured_kwargs[1]["cache_dit"])
            self.assertFalse(captured_kwargs[1]["cache_vae"])
            self.assertGreaterEqual(float(captured_kwargs[0]["timeout_sec"]), 360.0)
            shutil_path = Path(work_dir)
            if shutil_path.exists():
                import shutil
                shutil.rmtree(shutil_path, ignore_errors=True)

    def test_run_seedvr2_inference_retries_on_stall_once(self):
        processor = UpscaleProcessor()
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"stub-input")

            calls = {"count": 0}
            captured_kwargs = []

            def fake_run_inference(**kwargs):
                calls["count"] += 1
                captured_kwargs.append(kwargs)
                if calls["count"] == 1:
                    raise SeedVRRuntimeError("Inference stalled (no forward progress for 90s)")
                out_path = Path(kwargs["output_dir"]) / "ok.mp4"
                out_path.write_bytes(b"ok")
                return str(out_path)

            with patch("src.core.upscale_processor.detect_system_memory_gb", return_value=(24.0, 20.0)):
                with patch.object(processor._runtime, "run_inference", side_effect=fake_run_inference):
                    generated, work_dir, warnings = processor._run_seedvr2_inference(
                        ffmpeg_bin="/embedded/ffmpeg",
                        input_path=input_path,
                        model_id="seedvr2_3b_q4_k_m_gguf",
                        mode="upscale_resolution",
                        same_res_strength="x2_then_downscale",
                        denoise_strength=0.35,
                        source_w=1280,
                        source_h=720,
                        target_w=1920,
                        target_h=1080,
                        duration_sec=10.0,
                        cancel_event=None,
                        progress_callback=None,
                        estimated_total_sec=60.0,
                    )

            self.assertEqual(calls["count"], 2)
            self.assertTrue(os.path.exists(generated))
            self.assertTrue(work_dir.exists())
            self.assertTrue(any("No forward progress detected" in w for w in warnings))
            self.assertEqual(captured_kwargs[0]["dit_offload_device"], "none")
            self.assertEqual(captured_kwargs[0]["vae_offload_device"], "none")
            self.assertEqual(captured_kwargs[1]["dit_offload_device"], "none")
            self.assertEqual(captured_kwargs[1]["vae_offload_device"], "none")
            self.assertEqual(captured_kwargs[0]["video_backend_preference"], "ffmpeg")
            self.assertEqual(captured_kwargs[0]["ffmpeg_bin"], "/embedded/ffmpeg")
            self.assertEqual(captured_kwargs[0]["tensor_offload_device"], "cpu")
            self.assertEqual(captured_kwargs[1]["tensor_offload_device"], "cpu")
            self.assertFalse(captured_kwargs[0]["cache_dit"])
            self.assertFalse(captured_kwargs[0]["cache_vae"])
            self.assertFalse(captured_kwargs[1]["cache_dit"])
            self.assertFalse(captured_kwargs[1]["cache_vae"])
            self.assertGreaterEqual(float(captured_kwargs[0]["timeout_sec"]), 360.0)
            shutil_path = Path(work_dir)
            if shutil_path.exists():
                import shutil
                shutil.rmtree(shutil_path, ignore_errors=True)

    def test_run_seedvr2_inference_warmup_stall_retry_message(self):
        processor = UpscaleProcessor()
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.mp4")
            Path(input_path).write_bytes(b"stub-input")

            calls = {"count": 0}
            progress_messages: list[str] = []

            def fake_run_inference(**kwargs):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise SeedVRRuntimeError("Warmup stalled (no activity for 240s)")
                out_path = Path(kwargs["output_dir"]) / "ok.mp4"
                out_path.write_bytes(b"ok")
                return str(out_path)

            def on_progress(payload):
                progress_messages.append(str(payload.get("message") or ""))

            with patch("src.core.upscale_processor.detect_system_memory_gb", return_value=(24.0, 20.0)):
                with patch.object(processor._runtime, "run_inference", side_effect=fake_run_inference):
                    generated, work_dir, _warnings = processor._run_seedvr2_inference(
                        ffmpeg_bin="/embedded/ffmpeg",
                        input_path=input_path,
                        model_id="seedvr2_3b_q4_k_m_gguf",
                        mode="upscale_resolution",
                        same_res_strength="x2_then_downscale",
                        denoise_strength=0.35,
                        source_w=1280,
                        source_h=720,
                        target_w=1920,
                        target_h=1080,
                        duration_sec=10.0,
                        cancel_event=None,
                        progress_callback=on_progress,
                        estimated_total_sec=60.0,
                    )

            self.assertEqual(calls["count"], 2)
            self.assertTrue(os.path.exists(generated))
            self.assertTrue(any("Warmup timed out, retrying with stall-recovery profile..." in item for item in progress_messages))
            shutil_path = Path(work_dir)
            if shutil_path.exists():
                import shutil
                shutil.rmtree(shutil_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
