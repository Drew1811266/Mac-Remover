"""手动标注流程端到端回归测试。

本文件覆盖：
1. `VideoProcessor` 的手动标注主链路与 LaMa 细节策略。
2. FFmpeg/FFprobe 路径解析在处理流程中的接线。
3. GUI API 在 manual-only 模式下的参数校验与进度回调结构。
"""

import os
import sys
import subprocess
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.core.annotations import build_sidecar_path, save_sidecar
from src.core.video_processor import VideoProcessor
from src.gui.api import API


def _create_test_video(path: str, width: int = 64, height: int = 48, fps: int = 10, frames: int = 12) -> None:
    """生成测试视频，避免测试依赖外部媒体资源。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot create test video")
    for idx in range(frames):
        frame = np.full((height, width, 3), idx * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class _StubRemover:
    """最小可用 remover 桩对象：仅返回输入帧，便于聚焦流程测试。"""

    def __init__(self):
        self.unload_calls = 0

    def is_loaded(self) -> bool:
        return True

    def load_model(self) -> None:
        return None

    def inpaint(self, image, mask):
        return image

    def unload_model(self) -> None:
        self.unload_calls += 1


class _StubProcessor:
    """最小可用 processor 桩对象：直接写出文件并返回成功结构。"""

    def process_video(
        self,
        video_path,
        output_path,
        annotation_segments,
        model_id="lama_roi",
        progress_callback=None,
        status_callback=None,
    ):
        Path(output_path).write_bytes(b"ok")
        return {
            "output_path": output_path,
            "requested_model_id": model_id,
            "effective_model_id": model_id,
            "model_warning": "",
        }

    def stop_processing(self):
        return None


class ManualOnlyVideoProcessorTests(unittest.TestCase):
    """`VideoProcessor` 手动标注与 LaMa 增强逻辑测试集。"""

    def test_process_video_rejects_when_all_segments_disabled(self):
        """验证全部标记段禁用时会直接报错。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "output.mp4")
            _create_test_video(video_path)

            processor = VideoProcessor(remover=_StubRemover())

            with self.assertRaisesRegex(ValueError, "No enabled annotation segments provided"):
                processor.process_video(
                    video_path=video_path,
                    output_path=output_path,
                    annotation_segments=[
                        {
                            "id": "seg-disabled",
                            "enabled": False,
                            "start_frame": 0,
                            "end_frame": 1,
                            "rect": {"x": 0, "y": 0, "width": 10, "height": 10},
                        }
                    ],
                )

    def test_process_video_clips_out_of_bounds_roi(self):
        """验证越界 ROI 会被裁剪，仍可完成导出。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "output.mp4")
            _create_test_video(video_path)

            processor = VideoProcessor(remover=_StubRemover())

            result = processor.process_video(
                video_path=video_path,
                output_path=output_path,
                annotation_segments=[
                    {
                        "id": "seg-1",
                        "enabled": True,
                        "start_frame": 0,
                        "end_frame": 8,
                        "rect": {"x": -1000, "y": -1000, "width": 5000, "height": 5000},
                        "expand_px": 50,
                        "feather_px": 5,
                    }
                ],
                model_id="lama_roi",
            )

            self.assertEqual(result["output_path"], output_path)
            self.assertTrue(os.path.exists(result["output_path"]))

    def test_check_audio_uses_resolved_ffprobe_path(self):
        """验证音频检测调用了运行时解析后的 ffprobe 路径。"""
        processor = VideoProcessor(remover=_StubRemover())
        with patch('src.core.video_processor.resolve_ffprobe_path', return_value='/embedded/ffprobe'):
            with patch('src.core.video_processor.subprocess.run') as run_mock:
                run_mock.return_value = subprocess.CompletedProcess(
                    args=['/embedded/ffprobe'],
                    returncode=0,
                    stdout='audio\n',
                    stderr='',
                )
                has_audio = processor._check_audio('/tmp/demo.mp4')

        self.assertTrue(has_audio)
        self.assertTrue(run_mock.called)
        called_args = run_mock.call_args[0][0]
        self.assertEqual(called_args[0], '/embedded/ffprobe')

    def test_merge_audio_uses_resolved_ffmpeg_path(self):
        """验证音视频合并调用了运行时解析后的 ffmpeg 路径。"""
        processor = VideoProcessor(remover=_StubRemover())
        with patch('src.core.video_processor.resolve_ffmpeg_path', return_value='/embedded/ffmpeg'):
            with patch('src.core.video_processor.subprocess.run') as run_mock:
                run_mock.return_value = subprocess.CompletedProcess(
                    args=['/embedded/ffmpeg'],
                    returncode=0,
                    stdout='',
                    stderr='',
                )
                output = processor._merge_audio('/tmp/video.mp4', '/tmp/source.mp4', '/tmp/out.mp4')

        self.assertEqual(output, '/tmp/out.mp4')
        self.assertTrue(run_mock.called)
        called_args = run_mock.call_args[0][0]
        self.assertEqual(called_args[0], '/embedded/ffmpeg')

    def test_lama_blend_alpha_keeps_core_opaque_and_expands_two_stage_masks(self):
        """验证 LaMa 双阶段掩码与 alpha 核心区特性符合预期。"""
        core_mask = np.zeros((40, 120), dtype=np.uint8)
        core_mask[20:24, 50:64] = 255

        inpaint_mask_stage1 = VideoProcessor._build_lama_inpaint_mask(
            core_mask=core_mask,
            feather_px=8,
            rect_w=14,
            rect_h=4,
        )
        inpaint_mask_stage2 = VideoProcessor._build_lama_inpaint_mask_stage2(
            stage1_mask=inpaint_mask_stage1,
            feather_px=8,
            rect_w=14,
            rect_h=4,
        )
        alpha = VideoProcessor._build_lama_blend_alpha(
            core_mask=core_mask,
            inpaint_mask=inpaint_mask_stage2,
            feather_px=8,
            rect_w=14,
            rect_h=4,
        )

        core_bool = core_mask > 0
        ring_bool = (inpaint_mask_stage2 > 0) & ~core_bool
        self.assertGreater(np.count_nonzero(inpaint_mask_stage1), np.count_nonzero(core_mask))
        self.assertGreater(np.count_nonzero(inpaint_mask_stage2), np.count_nonzero(inpaint_mask_stage1))
        self.assertTrue(np.all(alpha[core_bool, 0] >= 0.999))
        self.assertTrue(np.any(alpha[ring_bool, 0] < 1.0))

    def test_lama_transition_masks_keep_expected_relationships(self):
        """验证过渡掩码各区域关系（核心/过渡/上下文）正确。"""
        core_mask = np.zeros((64, 96), dtype=np.uint8)
        core_mask[24:40, 30:56] = 255
        stage1 = VideoProcessor._build_lama_inpaint_mask(
            core_mask=core_mask,
            feather_px=7,
            rect_w=26,
            rect_h=16,
        )
        masks = VideoProcessor._build_lama_transition_masks(
            core_mask=core_mask,
            inpaint_mask=stage1,
            feather_px=7,
            rect_w=26,
            rect_h=16,
        )

        rounded = masks["rounded_mask"] > 0
        core_replace = masks["core_replace_mask"] > 0
        transition = masks["transition_mask"] > 0
        context = masks["context_mask"] > 0
        self.assertGreater(np.count_nonzero(core_replace), 0)
        self.assertGreater(np.count_nonzero(transition), 0)
        self.assertGreater(np.count_nonzero(context), 0)
        self.assertTrue(np.all(core_replace <= rounded))
        self.assertTrue(np.all(transition <= rounded))
        self.assertEqual(np.count_nonzero(transition & core_replace), 0)
        self.assertEqual(np.count_nonzero(context & rounded), 0)
        self.assertGreater(masks["transition_band_width"], 0.0)

    def test_lama_edge_aware_alpha_has_core_one_and_transition_fractional(self):
        """验证边缘感知 alpha：核心接近 1，过渡带为分数。"""
        core_mask = np.zeros((56, 88), dtype=np.uint8)
        core_mask[20:34, 28:52] = 255
        stage1 = VideoProcessor._build_lama_inpaint_mask(
            core_mask=core_mask,
            feather_px=6,
            rect_w=24,
            rect_h=14,
        )
        masks = VideoProcessor._build_lama_transition_masks(
            core_mask=core_mask,
            inpaint_mask=stage1,
            feather_px=6,
            rect_w=24,
            rect_h=14,
        )
        yy, xx = np.indices((56, 88))
        reference = np.dstack(
            [
                np.clip(110 + xx * 0.9, 0, 255),
                np.clip(100 + yy * 1.1, 0, 255),
                np.clip(90 + xx * 0.5 + yy * 0.4, 0, 255),
            ]
        ).astype(np.uint8)
        alpha = VideoProcessor._build_edge_aware_alpha(
            core_replace_mask=masks["core_replace_mask"],
            transition_mask=masks["transition_mask"],
            reference_roi=reference,
            feather_px=6,
        )[:, :, 0]

        core = masks["core_replace_mask"] > 0
        transition = masks["transition_mask"] > 0
        outside = ~(core | transition)
        self.assertTrue(np.all(alpha[core] >= 0.999))
        self.assertTrue(np.any(alpha[transition] < 0.95))
        self.assertTrue(np.any(alpha[transition] > 0.05))
        self.assertTrue(np.all(alpha[outside] <= 1e-6))

    def test_lama_v2_blending_reduces_transition_seam_delta(self):
        """验证 v2 融合策略能降低缝合区域误差。"""
        h, w = 72, 112
        yy, xx = np.indices((h, w))
        roi_original = np.dstack(
            [
                np.clip(120 + xx * 0.7, 0, 255),
                np.clip(115 + yy * 0.8, 0, 255),
                np.clip(100 + xx * 0.45 + yy * 0.35, 0, 255),
            ]
        ).astype(np.uint8)

        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_mask[26:44, 38:66] = 255
        stage1 = VideoProcessor._build_lama_inpaint_mask(
            core_mask=core_mask,
            feather_px=7,
            rect_w=28,
            rect_h=18,
        )
        masks = VideoProcessor._build_lama_transition_masks(
            core_mask=core_mask,
            inpaint_mask=stage1,
            feather_px=7,
            rect_w=28,
            rect_h=18,
        )

        roi_inpainted = roi_original.copy()
        inpaint_bool = stage1 > 0
        roi_inpainted[inpaint_bool] = np.clip(
            roi_inpainted[inpaint_bool].astype(np.float32) * np.array([0.63, 0.60, 0.56], dtype=np.float32)
            + np.array([32.0, 8.0, -6.0], dtype=np.float32),
            0.0,
            255.0,
        ).astype(np.uint8)

        old_blended = np.where(stage1[:, :, np.newaxis] > 0, roi_inpainted, roi_original)

        corrected = VideoProcessor._apply_boundary_color_correction_weighted(
            roi_original=roi_original,
            inpainted_roi=roi_inpainted,
            core_replace_mask=masks["core_replace_mask"],
            transition_mask=masks["transition_mask"],
            context_mask=masks["context_mask"],
            feather_px=7,
        )
        alpha_v2 = VideoProcessor._build_edge_aware_alpha(
            core_replace_mask=masks["core_replace_mask"],
            transition_mask=masks["transition_mask"],
            reference_roi=roi_original,
            feather_px=7,
        )
        v2_blended = VideoProcessor._laplacian_blend_roi(
            roi_original=roi_original,
            roi_inpainted=corrected,
            alpha=alpha_v2,
            levels=4,
        )
        v2_blended = VideoProcessor._harmonize_transition_seam(
            roi_original=roi_original,
            blended_roi=v2_blended,
            core_replace_mask=masks["core_replace_mask"],
            transition_mask=masks["transition_mask"],
        )

        seam_eval_mask = masks["transition_mask"]

        seam_before = VideoProcessor._compute_seam_delta(
            roi_original=roi_original,
            roi_candidate=old_blended,
            seam_mask=seam_eval_mask,
        )
        seam_after = VideoProcessor._compute_seam_delta(
            roi_original=roi_original,
            roi_candidate=v2_blended,
            seam_mask=seam_eval_mask,
        )
        self.assertLess(seam_after, seam_before)

    def test_evaluate_lama_frame_quality_flags_dark_block_candidate(self):
        """验证质量评估可识别“暗块塌陷”候选帧。"""
        h, w = 64, 96
        roi_original = np.full((h, w, 3), 185, dtype=np.uint8)
        roi_candidate = roi_original.copy()
        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_mask[22:42, 30:58] = 255
        stage1 = VideoProcessor._build_lama_inpaint_mask(
            core_mask=core_mask,
            feather_px=7,
            rect_w=28,
            rect_h=20,
        )
        transition_masks = VideoProcessor._build_lama_transition_masks(
            core_mask=core_mask,
            inpaint_mask=stage1,
            feather_px=7,
            rect_w=28,
            rect_h=20,
        )
        roi_candidate[stage1 > 0] = np.array([90, 88, 82], dtype=np.uint8)
        roi_candidate = cv2.GaussianBlur(roi_candidate, (7, 7), 0)
        quality = VideoProcessor._evaluate_lama_frame_quality(
            roi_original=roi_original,
            roi_candidate=roi_candidate,
            core_mask=core_mask,
            transition_mask=transition_masks["transition_mask"],
        )
        self.assertTrue(quality["dark_block_flag"])
        self.assertLess(quality["core_luma_shift"], -0.07)

    def test_frame_guard_rejects_dark_candidate_and_selects_safe_candidate(self):
        """验证帧守卫会拒绝暗块候选并选择更安全方案。"""
        evaluations = {
            "stage2_v2": {
                "score": 0.01,
                "dark_block_flag": True,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
            },
            "stage1_v2": {
                "score": 0.04,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
            },
            "legacy": {
                "score": 0.06,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
            },
        }
        selected, reject_stats = VideoProcessor._select_lama_frame_candidate(evaluations)
        self.assertEqual(selected, "stage1_v2")
        self.assertEqual(reject_stats["dark_rejects"], 1)

    def test_frame_guard_falls_back_to_legacy_when_v2_is_worse(self):
        """验证 v2 评分更差时会回退 legacy 方案。"""
        evaluations = {
            "stage2_v2": {
                "score": 0.20,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
            },
            "stage1_v2": {
                "score": 0.18,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
            },
            "legacy": {
                "score": 0.12,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
            },
        }
        selected, _ = VideoProcessor._select_lama_frame_candidate(evaluations)
        self.assertEqual(selected, "legacy")

    def test_frame_guard_hysteresis_suppresses_minor_switch(self):
        """验证滞回机制会抑制小幅抖动切换。"""
        evaluations = {
            "stage2_v2": {
                "score": 0.100,
                "dark_block_flag": False,
                "seam_extreme_flag": False,
            },
            "stage1_v2": {
                "score": 0.112,
                "dark_block_flag": False,
                "seam_extreme_flag": False,
            },
            "legacy": {
                "score": 0.130,
                "dark_block_flag": False,
                "seam_extreme_flag": False,
            },
        }
        selected, suppressed = VideoProcessor._apply_frame_guard_hysteresis(
            selected_name="stage2_v2",
            evaluations=evaluations,
            previous_name="stage1_v2",
        )
        self.assertTrue(suppressed)
        self.assertEqual(selected, "stage1_v2")

    def test_frame_guard_hysteresis_allows_switch_when_previous_bad(self):
        """验证上一帧方案劣化时，滞回机制允许切换。"""
        evaluations = {
            "stage2_v2": {
                "score": 0.08,
                "dark_block_flag": False,
                "seam_extreme_flag": False,
            },
            "stage1_v2": {
                "score": 0.09,
                "dark_block_flag": True,
                "seam_extreme_flag": False,
            },
            "legacy": {
                "score": 0.11,
                "dark_block_flag": False,
                "seam_extreme_flag": False,
            },
        }
        selected, suppressed = VideoProcessor._apply_frame_guard_hysteresis(
            selected_name="stage2_v2",
            evaluations=evaluations,
            previous_name="stage1_v2",
        )
        self.assertFalse(suppressed)
        self.assertEqual(selected, "stage2_v2")

    def test_lama_rescue_accept_and_reject_rules(self):
        """验证救援策略的接受/拒绝判定规则。"""
        selected_quality = {"score": 0.20, "dark_block_flag": True}
        accepted_rescue = {"score": 0.17, "dark_block_flag": False}
        rejected_rescue = {"score": 0.195, "dark_block_flag": False}
        rejected_dark_rescue = {"score": 0.16, "dark_block_flag": True}

        self.assertTrue(
            VideoProcessor._should_accept_lama_rescue(
                selected_quality=selected_quality,
                rescue_quality=accepted_rescue,
            )
        )
        self.assertFalse(
            VideoProcessor._should_accept_lama_rescue(
                selected_quality=selected_quality,
                rescue_quality=rejected_rescue,
            )
        )
        self.assertFalse(
            VideoProcessor._should_accept_lama_rescue(
                selected_quality=selected_quality,
                rescue_quality=rejected_dark_rescue,
            )
        )

    def test_rescue_blend_with_seamless_clone_returns_valid_frame(self):
        """验证 seamless clone 救援融合能返回合法图像。"""
        h, w = 80, 120
        roi_original = np.full((h, w, 3), 172, dtype=np.uint8)
        roi_candidate = roi_original.copy()
        rounded_mask = np.zeros((h, w), dtype=np.uint8)
        rounded_mask[22:60, 35:88] = 255
        roi_candidate[rounded_mask > 0] = np.array([80, 72, 68], dtype=np.uint8)

        rescued = VideoProcessor._rescue_blend_with_seamless_clone(
            roi_original=roi_original,
            roi_candidate=roi_candidate,
            rounded_mask=rounded_mask,
        )
        self.assertEqual(rescued.shape, roi_original.shape)
        self.assertEqual(rescued.dtype, np.uint8)

    def test_final_micro_smoothing_applies_in_low_motion(self):
        """验证低运动场景会触发最终微平滑。"""
        h, w = 72, 104
        current = np.full((h, w, 3), 140, dtype=np.uint8)
        previous = current.copy()
        current[24:48, 32:68] = np.array([160, 156, 152], dtype=np.uint8)
        previous[24:48, 32:68] = np.array([150, 146, 142], dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:52, 28:72] = 255

        smoothed, applied = VideoProcessor._apply_final_micro_smoothing(
            current_roi=current,
            previous_roi=previous,
            smoothing_mask=mask,
            force_reset=False,
        )
        self.assertTrue(applied)
        self.assertEqual(smoothed.shape, current.shape)

    def test_final_micro_smoothing_skips_when_force_reset(self):
        """验证强制重置时不会应用微平滑。"""
        h, w = 64, 96
        current = np.full((h, w, 3), 120, dtype=np.uint8)
        previous = np.full((h, w, 3), 90, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[16:44, 24:60] = 255

        smoothed, applied = VideoProcessor._apply_final_micro_smoothing(
            current_roi=current,
            previous_roi=previous,
            smoothing_mask=mask,
            force_reset=True,
        )
        self.assertFalse(applied)
        self.assertTrue(np.array_equal(smoothed, current))

    def test_detect_hard_cuts_in_run_detects_scene_switch(self):
        """验证硬切检测能识别场景突变帧。"""
        h, w = 90, 160
        frames = []
        for i in range(8):
            frame = np.full((h, w, 3), 35, dtype=np.uint8)
            if i >= 4:
                frame = np.full((h, w, 3), 220, dtype=np.uint8)
            frames.append(frame)

        forced = VideoProcessor._detect_hard_cuts_in_run(frames)
        self.assertIn(4, forced)

    def test_lama_stabilizer_resets_on_forced_hard_cut(self):
        """验证强制硬切点会触发稳定器重置。"""
        h, w = 48, 64
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[14:34, 20:46] = 255

        sequence = []
        for idx in range(8):
            frame = np.full((h, w, 3), 30, dtype=np.uint8)
            if idx < 4:
                frame[14:34, 20:46] = 60
            else:
                frame[14:34, 20:46] = 220
            sequence.append(frame)

        stabilized, diag = VideoProcessor._stabilize_lama_sequence(
            roi_sequence=sequence,
            roi_mask=roi_mask,
            temporal_strength=1.0,
            forced_reset_indices={4},
        )

        cut_idx = 4
        cut_diff = np.mean(
            np.abs(
                stabilized[cut_idx][roi_mask > 0].astype(np.float32)
                - sequence[cut_idx][roi_mask > 0].astype(np.float32)
            )
        )
        self.assertGreaterEqual(diag["hard_cut_resets_total"], 1)
        self.assertGreaterEqual(diag["hard_cut_resets_forced"], 1)
        self.assertGreaterEqual(diag["cold_start_frames"], 1)
        self.assertLess(cut_diff, 2.0)

    def test_lama_stabilizer_keeps_temporal_smoothing_without_hard_cut(self):
        """验证无硬切时稳定器可降低时域抖动。"""
        rng = np.random.default_rng(0)
        h, w = 48, 64
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[14:34, 20:46] = 255
        roi_bool = roi_mask > 0

        sequence = []
        for idx in range(10):
            frame = np.full((h, w, 3), 90, dtype=np.uint8)
            noise = rng.normal(loc=0.0, scale=6.0, size=(h, w, 3)).astype(np.float32)
            local = np.clip(130.0 + idx * 0.5 + noise, 0, 255).astype(np.uint8)
            frame[roi_bool] = local[roi_bool]
            sequence.append(frame)

        stabilized, diag = VideoProcessor._stabilize_lama_sequence(
            roi_sequence=sequence,
            roi_mask=roi_mask,
            temporal_strength=1.0,
        )

        def _temporal_energy(frames):
            values = []
            for i in range(1, len(frames)):
                diff = np.abs(
                    frames[i][roi_bool].astype(np.float32) - frames[i - 1][roi_bool].astype(np.float32)
                )
                values.append(float(diff.mean()))
            return float(np.mean(values))

        raw_energy = _temporal_energy(sequence)
        smooth_energy = _temporal_energy(stabilized)
        self.assertEqual(diag["hard_cut_resets_total"], 0)
        self.assertEqual(diag["hard_cut_resets_forced"], 0)
        self.assertEqual(diag["hard_cut_resets_roi"], 0)
        self.assertLess(smooth_energy, raw_energy)

    def test_lama_dual_stage_calls_two_passes(self):
        """验证 LaMa 双阶段流程会调用两次 inpaint。"""
        class _Engine:
            def __init__(self):
                self.calls = 0

            def inpaint_roi_sequence(self, roi_frames, roi_masks, progress_callback=None):
                self.calls += 1
                return [np.asarray(f).copy() for f in roi_frames]

        class _Registry:
            def __init__(self, engine):
                self._engine = engine

            def resolve(self, requested_model_id):
                info = type(
                    "R",
                    (),
                    {
                        "requested_model_id": "lama_roi",
                        "effective_model_id": "lama_roi",
                        "warning": "",
                    },
                )
                return self._engine, info

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "output.mp4")
            _create_test_video(video_path, frames=10)

            engine = _Engine()
            processor = VideoProcessor(remover=_StubRemover())
            with patch.object(processor, "_get_model_registry", return_value=_Registry(engine)):
                with patch.object(processor, "_transcode_video_h264", return_value=output_path):
                    result = processor.process_video(
                        video_path=video_path,
                        output_path=output_path,
                        annotation_segments=[
                            {
                                "id": "seg-1",
                                "enabled": True,
                                "start_frame": 0,
                                "end_frame": 9,
                                "rect": {"x": 4, "y": 4, "width": 16, "height": 12},
                            }
                        ],
                        model_id="lama_roi",
                    )

        self.assertEqual(result["output_path"], output_path)
        self.assertEqual(engine.calls, 2)

    def test_lama_dual_stage_falls_back_to_pass1_when_pass2_fails(self):
        """验证第二阶段失败时仍能用第一阶段结果导出。"""
        class _Engine:
            def __init__(self):
                self.calls = 0

            def inpaint_roi_sequence(self, roi_frames, roi_masks, progress_callback=None):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("pass2 failed")
                return [np.asarray(f).copy() for f in roi_frames]

        class _Registry:
            def __init__(self, engine):
                self._engine = engine

            def resolve(self, requested_model_id):
                info = type(
                    "R",
                    (),
                    {
                        "requested_model_id": "lama_roi",
                        "effective_model_id": "lama_roi",
                        "warning": "",
                    },
                )
                return self._engine, info

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "output.mp4")
            _create_test_video(video_path, frames=10)

            engine = _Engine()
            processor = VideoProcessor(remover=_StubRemover())
            with patch.object(processor, "_get_model_registry", return_value=_Registry(engine)):
                with patch.object(processor, "_transcode_video_h264", return_value=output_path):
                    result = processor.process_video(
                        video_path=video_path,
                        output_path=output_path,
                        annotation_segments=[
                            {
                                "id": "seg-1",
                                "enabled": True,
                                "start_frame": 0,
                                "end_frame": 9,
                                "rect": {"x": 4, "y": 4, "width": 16, "height": 12},
                            }
                        ],
                        model_id="lama_roi",
                    )

        self.assertEqual(result["output_path"], output_path)
        self.assertEqual(engine.calls, 2)

    def test_lama_pass2_quality_gate_rejects_dark_collapse(self):
        """验证 pass2 质量门会拒绝明显暗塌结果。"""
        core_mask = np.zeros((48, 64), dtype=np.uint8)
        core_mask[16:30, 18:42] = 255
        stage1 = [np.full((48, 64, 3), 180, dtype=np.uint8) for _ in range(6)]
        stage2 = [np.full((48, 64, 3), 20, dtype=np.uint8) for _ in range(6)]

        ok, rejected = VideoProcessor._is_lama_pass2_acceptable(
            stage1_rois=stage1,
            stage2_rois=stage2,
            core_mask=core_mask,
        )
        self.assertFalse(ok)
        self.assertGreaterEqual(rejected, 1)

    def test_lama_blend_v2_fallback_keeps_export_successful(self):
        """验证 v2 融合失败时可回退且导出成功。"""
        class _Engine:
            def inpaint_roi_sequence(self, roi_frames, roi_masks, progress_callback=None, **kwargs):
                _ = kwargs
                return [np.asarray(f).copy() for f in roi_frames]

        class _Registry:
            def __init__(self, engine):
                self._engine = engine

            def resolve(self, requested_model_id):
                info = type(
                    "R",
                    (),
                    {
                        "requested_model_id": "lama_roi",
                        "effective_model_id": "lama_roi",
                        "warning": "",
                    },
                )
                return self._engine, info

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "output.mp4")
            _create_test_video(video_path, frames=10)

            engine = _Engine()
            processor = VideoProcessor(remover=_StubRemover())
            with patch.object(processor, "_get_model_registry", return_value=_Registry(engine)):
                with patch.object(processor, "_transcode_video_h264", return_value=output_path):
                    with patch.object(
                        VideoProcessor,
                        "_laplacian_blend_roi",
                        side_effect=RuntimeError("blend v2 failed"),
                    ):
                        result = processor.process_video(
                            video_path=video_path,
                            output_path=output_path,
                            annotation_segments=[
                                {
                                    "id": "seg-1",
                                    "enabled": True,
                                    "start_frame": 0,
                                    "end_frame": 9,
                                    "rect": {"x": 4, "y": 4, "width": 16, "height": 12},
                                }
                            ],
                            model_id="lama_roi",
                        )

        self.assertEqual(result["output_path"], output_path)

    def test_non_lama_path_does_not_use_blend_v2(self):
        """验证非 LaMa 模型路径不会调用 LaMa v2 融合。"""
        class _Engine:
            def inpaint_roi_sequence(self, roi_frames, roi_masks, progress_callback=None):
                return [np.asarray(f).copy() for f in roi_frames]

        class _Registry:
            def __init__(self, engine):
                self._engine = engine

            def resolve(self, requested_model_id):
                info = type(
                    "R",
                    (),
                    {
                        "requested_model_id": "custom_temporal_roi",
                        "effective_model_id": "custom_temporal_roi",
                        "warning": "",
                    },
                )
                return self._engine, info

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "output.mp4")
            _create_test_video(video_path, frames=8)

            processor = VideoProcessor(remover=_StubRemover())
            with patch.object(processor, "_get_model_registry", return_value=_Registry(_Engine())):
                with patch.object(processor, "_transcode_video_h264", return_value=output_path):
                    with patch.object(VideoProcessor, "_laplacian_blend_roi") as blend_mock:
                        result = processor.process_video(
                            video_path=video_path,
                            output_path=output_path,
                            annotation_segments=[
                                {
                                    "id": "seg-1",
                                    "enabled": True,
                                    "start_frame": 0,
                                    "end_frame": 7,
                                    "rect": {"x": 3, "y": 3, "width": 14, "height": 10},
                                }
                            ],
                            model_id="custom_temporal_roi",
                        )

        self.assertEqual(result["output_path"], output_path)
        blend_mock.assert_not_called()

    def test_propainter_infer_options_adapt_to_roi_and_segment(self):
        """验证 ProPainter 推理参数会按 ROI 与段长度自适应。"""
        opts_short = VideoProcessor._compute_propainter_infer_options(
            rect_w=80,
            rect_h=40,
            feather_px=6,
            segment_total_frames=60,
            fps=30.0,
        )
        opts_long = VideoProcessor._compute_propainter_infer_options(
            rect_w=240,
            rect_h=110,
            feather_px=8,
            segment_total_frames=360,
            fps=29.97,
        )
        self.assertGreaterEqual(opts_short["mask_dilation"], 4)
        self.assertEqual(opts_short["neighbor_length"], 12)
        self.assertEqual(opts_short["ref_stride"], 8)
        self.assertEqual(opts_short["subvideo_length"], 100)
        self.assertEqual(opts_short["raft_iter"], 24)
        self.assertEqual(opts_short["save_fps"], 30)

        self.assertGreaterEqual(opts_long["mask_dilation"], opts_short["mask_dilation"])
        self.assertEqual(opts_long["neighbor_length"], 14)
        self.assertEqual(opts_long["ref_stride"], 6)
        self.assertEqual(opts_long["subvideo_length"], 80)

    def test_propainter_split_sequence_by_cut_indices(self):
        """验证硬切索引会正确切分子区间。"""
        ranges = VideoProcessor._split_sequence_by_cut_indices(
            total_length=12,
            cut_indices={4, 9},
        )
        self.assertEqual(ranges, [(0, 4), (4, 9), (9, 12)])

    def test_propainter_cut_quarantine_indices_cover_cut_window(self):
        """验证切镜头隔离窗口默认为 [cut-1, cut+3]。"""
        quarantine = VideoProcessor._compute_cut_quarantine_indices(
            total_length=12,
            cut_indices={5},
            before=1,
            after=3,
        )
        self.assertEqual(quarantine, {4, 5, 6, 7, 8})

    def test_propainter_viterbi_suppresses_single_frame_legacy_switch(self):
        """验证序列级路径优化会抑制单帧 legacy 跳变。"""
        frames = [
            {
                "stable_v2": {"score": 0.08, "reappear_score": 0.01, "seam_extreme_flag": False},
                "raw_v2": {"score": 0.09, "reappear_score": 0.01, "seam_extreme_flag": False},
                "legacy": {"score": 0.05, "reappear_score": 0.0, "seam_extreme_flag": False},
            },
            {
                "stable_v2": {"score": 0.09, "reappear_score": 0.01, "seam_extreme_flag": False},
                "raw_v2": {"score": 0.10, "reappear_score": 0.01, "seam_extreme_flag": False},
                "legacy": {"score": 0.04, "reappear_score": 0.0, "seam_extreme_flag": False},
            },
            {
                "stable_v2": {"score": 0.08, "reappear_score": 0.01, "seam_extreme_flag": False},
                "raw_v2": {"score": 0.09, "reappear_score": 0.01, "seam_extreme_flag": False},
                "legacy": {"score": 0.05, "reappear_score": 0.0, "seam_extreme_flag": False},
            },
        ]
        for item in frames:
            item["stable_v2"].update(
                {"dark_block_flag": False, "core_luma_shift": -0.01, "core_texture_ratio": 0.92, "temporal_warp_error": 0.04}
            )
            item["raw_v2"].update(
                {"dark_block_flag": False, "core_luma_shift": -0.01, "core_texture_ratio": 0.90, "temporal_warp_error": 0.05}
            )
            item["legacy"].update(
                {"dark_block_flag": False, "core_luma_shift": -0.01, "core_texture_ratio": 0.92, "temporal_warp_error": 0.02}
            )
        names, switches = VideoProcessor._select_propainter_sequence_viterbi(frames, cut_quarantine_indices=set())
        self.assertEqual(names, ["stable_v2", "stable_v2", "stable_v2"])
        self.assertEqual(switches, 0)

    def test_propainter_selection_island_rewrite_aba(self):
        """验证 A-B-A 的 1 帧孤岛会被回并。"""
        names = ["stable_v2", "raw_v2", "stable_v2"]
        evals = [
            {
                "stable_v2": {"score": 0.10, "seam_extreme_flag": False, "temporal_bad_flag": False, "reappear_flag": False, "dark_block_flag": False, "core_luma_shift": -0.01, "core_texture_ratio": 0.95},
                "raw_v2": {"score": 0.11, "seam_extreme_flag": False, "temporal_bad_flag": False, "reappear_flag": False, "dark_block_flag": False, "core_luma_shift": -0.01, "core_texture_ratio": 0.95},
                "legacy": {"score": 0.09, "seam_extreme_flag": False},
            },
            {
                "stable_v2": {"score": 0.105, "seam_extreme_flag": False, "temporal_bad_flag": False, "reappear_flag": False, "dark_block_flag": False, "core_luma_shift": -0.01, "core_texture_ratio": 0.95},
                "raw_v2": {"score": 0.100, "seam_extreme_flag": False, "temporal_bad_flag": False, "reappear_flag": False, "dark_block_flag": False, "core_luma_shift": -0.01, "core_texture_ratio": 0.95},
                "legacy": {"score": 0.08, "seam_extreme_flag": False},
            },
            {
                "stable_v2": {"score": 0.10, "seam_extreme_flag": False, "temporal_bad_flag": False, "reappear_flag": False, "dark_block_flag": False, "core_luma_shift": -0.01, "core_texture_ratio": 0.95},
                "raw_v2": {"score": 0.11, "seam_extreme_flag": False, "temporal_bad_flag": False, "reappear_flag": False, "dark_block_flag": False, "core_luma_shift": -0.01, "core_texture_ratio": 0.95},
                "legacy": {"score": 0.09, "seam_extreme_flag": False},
            },
        ]
        rewritten, rewrites = VideoProcessor._suppress_propainter_selection_islands(
            selected_names=names,
            frame_evaluations=evals,
            cut_quarantine_indices=set(),
        )
        self.assertEqual(rewritten, ["stable_v2", "stable_v2", "stable_v2"])
        self.assertEqual(rewrites, 1)

    def test_propainter_selection_island_rewrite_abba(self):
        """验证 A-B-B-A 的 2 帧孤岛会被回并。"""
        names = ["stable_v2", "raw_v2", "raw_v2", "stable_v2"]
        evals = []
        for idx in range(4):
            evals.append(
                {
                    "stable_v2": {
                        "score": 0.10 + (0.003 if idx in (1, 2) else 0.0),
                        "seam_extreme_flag": False,
                        "temporal_bad_flag": False,
                        "reappear_flag": False,
                        "dark_block_flag": False,
                        "core_luma_shift": -0.01,
                        "core_texture_ratio": 0.95,
                    },
                    "raw_v2": {
                        "score": 0.10,
                        "seam_extreme_flag": False,
                        "temporal_bad_flag": False,
                        "reappear_flag": False,
                        "dark_block_flag": False,
                        "core_luma_shift": -0.01,
                        "core_texture_ratio": 0.95,
                    },
                    "legacy": {"score": 0.08, "seam_extreme_flag": False},
                }
            )
        rewritten, rewrites = VideoProcessor._suppress_propainter_selection_islands(
            selected_names=names,
            frame_evaluations=evals,
            cut_quarantine_indices=set(),
        )
        self.assertEqual(rewritten, ["stable_v2", "stable_v2", "stable_v2", "stable_v2"])
        self.assertEqual(rewrites, 2)

    def test_propainter_selection_island_respects_hard_reject(self):
        """验证替换目标若 hard reject，则孤岛不应被改写。"""
        names = ["stable_v2", "raw_v2", "stable_v2"]
        evals = [
            {
                "stable_v2": {
                    "score": 0.10,
                    "seam_extreme_flag": False,
                    "temporal_bad_flag": False,
                    "reappear_flag": False,
                    "dark_block_flag": False,
                    "core_luma_shift": -0.01,
                    "core_texture_ratio": 0.95,
                },
                "raw_v2": {"score": 0.10, "seam_extreme_flag": False},
                "legacy": {"score": 0.09, "seam_extreme_flag": False},
            },
            {
                "stable_v2": {
                    "score": 0.09,
                    "seam_extreme_flag": True,
                    "temporal_bad_flag": False,
                    "reappear_flag": False,
                    "dark_block_flag": False,
                    "core_luma_shift": -0.01,
                    "core_texture_ratio": 0.95,
                },
                "raw_v2": {"score": 0.08, "seam_extreme_flag": False},
                "legacy": {"score": 0.07, "seam_extreme_flag": False},
            },
            {
                "stable_v2": {
                    "score": 0.10,
                    "seam_extreme_flag": False,
                    "temporal_bad_flag": False,
                    "reappear_flag": False,
                    "dark_block_flag": False,
                    "core_luma_shift": -0.01,
                    "core_texture_ratio": 0.95,
                },
                "raw_v2": {"score": 0.10, "seam_extreme_flag": False},
                "legacy": {"score": 0.09, "seam_extreme_flag": False},
            },
        ]
        rewritten, rewrites = VideoProcessor._suppress_propainter_selection_islands(
            selected_names=names,
            frame_evaluations=evals,
            cut_quarantine_indices=set(),
        )
        self.assertEqual(rewritten, names)
        self.assertEqual(rewrites, 0)

    def test_propainter_micro_flicker_detector_marks_near_miss(self):
        """验证未触发旧 flag 的近失帧也会被微闪检测命中。"""
        qualities = [
            {"temporal_jump_core": 0.020, "remove_ratio": 0.90, "residual_hf_corr": 0.42},
            {"temporal_jump_core": 0.022, "remove_ratio": 0.89, "residual_hf_corr": 0.43},
            {"temporal_jump_core": 0.110, "remove_ratio": 0.74, "residual_hf_corr": 0.50},
            {"temporal_jump_core": 0.021, "remove_ratio": 0.88, "residual_hf_corr": 0.42},
            {"temporal_jump_core": 0.024, "remove_ratio": 0.90, "residual_hf_corr": 0.44},
        ]
        flags = VideoProcessor._detect_propainter_micro_flicker_flags(
            selected_qualities=qualities,
            cut_quarantine_indices=set(),
            window_radius=2,
        )
        self.assertEqual(flags, [False, False, True, False, False])

    def test_propainter_transition_masks_v3_contains_inner_and_outer(self):
        """验证 V3 掩码会生成内外环并满足包含关系。"""
        core = np.zeros((80, 120), dtype=np.uint8)
        core[28:50, 42:76] = 255
        inpaint = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
        masks = VideoProcessor._build_propainter_transition_masks_v3(
            core_mask=core,
            inpaint_mask=inpaint,
            feather_px=8,
            rect_w=34,
            rect_h=22,
        )
        inner = masks["transition_inner_mask"] > 0
        outer = masks["transition_outer_mask"] > 0
        transition = masks["transition_mask"] > 0
        self.assertGreater(int(np.count_nonzero(inner)), 0)
        self.assertGreater(int(np.count_nonzero(outer)), 0)
        self.assertTrue(np.all((inner | outer) <= transition))

    def test_propainter_edge_aware_alpha_v3_has_expected_ranges(self):
        """验证 V3 alpha 在核心/过渡/外部区域满足预期范围。"""
        h, w = 72, 108
        core = np.zeros((h, w), dtype=np.uint8)
        core[24:46, 38:70] = 255
        inner = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        inner = ((inner > 0) & ~(core > 0)).astype(np.uint8) * 255
        outer = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
        outer = ((outer > 0) & ~(cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1) > 0)).astype(np.uint8) * 255
        roi = np.full((h, w, 3), 150, dtype=np.uint8)
        alpha = VideoProcessor._build_edge_aware_alpha_v3(
            core_replace_mask=core,
            transition_inner_mask=inner,
            transition_outer_mask=outer,
            reference_roi=roi,
        )[:, :, 0]
        self.assertAlmostEqual(float(alpha[core > 0].mean()), 1.0, delta=1e-6)
        self.assertGreater(float(alpha[inner > 0].mean()), float(alpha[outer > 0].mean()))
        outside = (core == 0) & (inner == 0) & (outer == 0)
        self.assertLess(float(alpha[outside].max()), 1e-5)

    def test_propainter_remove_priority_trigger_rule(self):
        """验证段级去除优先模式触发条件。"""
        self.assertFalse(
            VideoProcessor._should_enable_propainter_remove_priority(
                frames_observed=19,
                legacy_frames=18,
                original_similarity_avg=0.95,
                probe_window=20,
            )
        )
        self.assertFalse(
            VideoProcessor._should_enable_propainter_remove_priority(
                frames_observed=20,
                legacy_frames=12,
                original_similarity_avg=0.89,
                probe_window=20,
            )
        )
        self.assertTrue(
            VideoProcessor._should_enable_propainter_remove_priority(
                frames_observed=20,
                legacy_frames=18,
                original_similarity_avg=0.92,
                probe_window=20,
            )
        )
        self.assertTrue(
            VideoProcessor._should_enable_propainter_remove_priority(
                frames_observed=30,
                legacy_frames=21,
                original_similarity_avg=0.85,
                probe_window=20,
            )
        )

    def test_propainter_probe_window_is_aggressive_for_short_segments(self):
        """验证短段会使用更小探测窗口，避免末帧才触发。"""
        self.assertEqual(VideoProcessor._compute_propainter_segment_probe_window(5), 3)
        self.assertEqual(VideoProcessor._compute_propainter_segment_probe_window(10), 4)
        self.assertEqual(VideoProcessor._compute_propainter_segment_probe_window(120), 20)

    def test_propainter_stabilizer_reduces_temporal_energy(self):
        """验证 ProPainter 稳定器在无硬切场景可降低抖动。"""
        rng = np.random.default_rng(7)
        h, w = 56, 88
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[18:40, 24:64] = 255
        roi_bool = roi_mask > 0

        sequence = []
        for i in range(10):
            frame = np.full((h, w, 3), 96, dtype=np.uint8)
            noisy = np.clip(
                138.0 + i * 0.4 + rng.normal(loc=0.0, scale=6.2, size=(h, w, 3)),
                0.0,
                255.0,
            ).astype(np.uint8)
            frame[roi_bool] = noisy[roi_bool]
            sequence.append(frame)

        stabilized, diag = VideoProcessor._stabilize_propainter_sequence(
            roi_sequence=sequence,
            roi_mask=roi_mask,
            temporal_strength=1.0,
            forced_reset_indices=set(),
        )

        def _energy(frames):
            values = []
            for i in range(1, len(frames)):
                diff = np.abs(
                    frames[i][roi_bool].astype(np.float32)
                    - frames[i - 1][roi_bool].astype(np.float32)
                )
                values.append(float(diff.mean()))
            return float(np.mean(values))

        self.assertLess(_energy(stabilized), _energy(sequence))
        self.assertGreaterEqual(diag["stabilize_applied_frames"], 1)

    def test_propainter_frame_guard_prefers_safe_candidate(self):
        """验证 ProPainter 帧守护会拒绝暗块候选。"""
        evaluations = {
            "raw_v2": {
                "score": 0.02,
                "dark_block_flag": True,
                "core_luma_shift": -0.18,
                "core_texture_ratio": 0.45,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
            },
            "stable_v2": {
                "score": 0.05,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
            },
            "legacy": {
                "score": 0.06,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
            },
        }
        selected, reject_stats = VideoProcessor._select_propainter_frame_candidate(evaluations)
        self.assertEqual(selected, "stable_v2")
        self.assertEqual(reject_stats["dark_rejects"], 1)

    def test_propainter_frame_guard_allows_legacy_only_when_v2_catastrophic(self):
        """验证 legacy 仅在两个 v2 均灾难性风险时才可接管。"""
        safe_case = {
            "raw_v2": {
                "score": 0.30,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "temporal_warp_error": 0.04,
            },
            "stable_v2": {
                "score": 0.31,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "temporal_warp_error": 0.05,
            },
            "legacy": {
                "score": 0.02,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "temporal_warp_error": 0.01,
            },
        }
        selected_safe, safe_stats = VideoProcessor._select_propainter_frame_candidate(safe_case)
        self.assertIn(selected_safe, ("raw_v2", "stable_v2"))
        self.assertEqual(safe_stats["legacy_catastrophic_only"], 0)

        catastrophic_case = {
            "raw_v2": {
                "score": 0.40,
                "dark_block_flag": False,
                "core_luma_shift": -0.20,
                "core_texture_ratio": 0.40,
                "seam_bad_flag": True,
                "seam_extreme_flag": True,
                "temporal_bad_flag": True,
                "temporal_warp_error": 0.22,
                "reappear_flag": False,
            },
            "stable_v2": {
                "score": 0.41,
                "dark_block_flag": False,
                "core_luma_shift": -0.18,
                "core_texture_ratio": 0.45,
                "seam_bad_flag": True,
                "seam_extreme_flag": True,
                "temporal_bad_flag": True,
                "temporal_warp_error": 0.24,
                "reappear_flag": False,
            },
            "legacy": {
                "score": 0.09,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "temporal_warp_error": 0.02,
                "reappear_flag": False,
            },
        }
        selected_bad, bad_stats = VideoProcessor._select_propainter_frame_candidate(catastrophic_case)
        self.assertEqual(selected_bad, "legacy")
        self.assertEqual(bad_stats["legacy_catastrophic_only"], 1)

    def test_propainter_frame_guard_avoids_legacy_when_legacy_too_close_to_original(self):
        """验证 legacy 近似原图时，优先选择可接受的 v2 候选。"""
        evaluations = {
            "raw_v2": {
                "score": 0.08,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "reappear_flag": False,
                "original_similarity_core": 0.84,
            },
            "stable_v2": {
                "score": 0.09,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "reappear_flag": False,
                "original_similarity_core": 0.86,
            },
            "legacy": {
                "score": 0.07,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "reappear_flag": False,
                "original_similarity_core": 0.95,
            },
        }
        selected, _ = VideoProcessor._select_propainter_frame_candidate(evaluations)
        self.assertEqual(selected, "raw_v2")

    def test_propainter_frame_guard_preserves_primary_when_legacy_is_near_original(self):
        """验证轻微分数劣势下不会被 legacy 反向覆盖。"""
        evaluations = {
            "raw_v2": {
                "score": 0.11,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "reappear_flag": False,
                "original_similarity_core": 0.87,
            },
            "stable_v2": {
                "score": 0.14,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "reappear_flag": False,
                "original_similarity_core": 0.89,
            },
            "legacy": {
                "score": 0.10,
                "dark_block_flag": False,
                "seam_bad_flag": False,
                "seam_extreme_flag": False,
                "temporal_bad_flag": False,
                "reappear_flag": False,
                "original_similarity_core": 0.94,
            },
        }
        selected, _ = VideoProcessor._select_propainter_frame_candidate(evaluations)
        self.assertEqual(selected, "raw_v2")

    def test_propainter_force_mode_block_ignores_temporal_bad_only(self):
        """验证去除优先模式下 temporal_bad 不会单独阻断候选。"""
        eval_item = {
            "temporal_bad_flag": True,
            "seam_extreme_flag": False,
            "reappear_flag": False,
            "dark_block_flag": False,
            "core_luma_shift": -0.01,
            "core_texture_ratio": 0.95,
        }
        self.assertTrue(VideoProcessor._is_propainter_hard_reject(eval_item))
        self.assertFalse(VideoProcessor._is_propainter_force_mode_block(eval_item))

    def test_propainter_quality_evaluator_reports_temporal_error(self):
        """验证 ProPainter 质量评估包含时域误差项。"""
        h, w = 64, 96
        roi_original = np.full((h, w, 3), 150, dtype=np.uint8)
        roi_candidate = np.full((h, w, 3), 152, dtype=np.uint8)
        previous = np.full((h, w, 3), 80, dtype=np.uint8)
        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_mask[24:42, 30:58] = 255
        transition_mask = np.zeros((h, w), dtype=np.uint8)
        transition_mask[20:46, 26:62] = 255
        quality = VideoProcessor._evaluate_propainter_frame_quality(
            roi_original=roi_original,
            roi_candidate=roi_candidate,
            core_mask=core_mask,
            transition_mask=transition_mask,
            previous_selected_roi=previous,
        )
        self.assertIn("temporal_warp_error", quality)
        self.assertGreaterEqual(quality["temporal_warp_error"], 0.0)

    def test_propainter_quality_evaluator_detects_reappear_flag(self):
        """验证回闪判定会在高原图相似+高时域跳变时触发。"""
        h, w = 56, 88
        roi_original = np.full((h, w, 3), 180, dtype=np.uint8)
        roi_candidate = roi_original.copy()
        previous = np.full((h, w, 3), 16, dtype=np.uint8)
        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_mask[16:40, 24:60] = 255
        transition_mask = np.zeros((h, w), dtype=np.uint8)
        transition_mask[12:44, 20:64] = 255

        quality = VideoProcessor._evaluate_propainter_frame_quality(
            roi_original=roi_original,
            roi_candidate=roi_candidate,
            core_mask=core_mask,
            transition_mask=transition_mask,
            previous_selected_roi=previous,
            is_hard_cut_frame=False,
        )
        self.assertTrue(quality["reappear_flag"])
        self.assertGreater(quality["original_similarity_core"], 0.94)
        self.assertGreater(quality["temporal_jump_core"], 0.08)

        hard_cut_quality = VideoProcessor._evaluate_propainter_frame_quality(
            roi_original=roi_original,
            roi_candidate=roi_candidate,
            core_mask=core_mask,
            transition_mask=transition_mask,
            previous_selected_roi=previous,
            is_hard_cut_frame=True,
        )
        self.assertFalse(hard_cut_quality["reappear_flag"])

    def test_propainter_remove_sufficiency_marks_under_remove_and_penalizes_score(self):
        """验证去除充分性会标记 under_remove 并惩罚分数。"""
        quality = {
            "score": 0.20,
            "remove_energy_core": 0.02,
            "reappear_flag": False,
        }
        updated = VideoProcessor._apply_propainter_remove_sufficiency(
            quality=quality,
            remove_energy_reference=0.10,
            previous_selected_remove_ratio=0.80,
            is_hard_cut_frame=False,
        )
        self.assertTrue(updated["under_remove_flag"])
        self.assertTrue(updated["reappear_flag"])
        self.assertLess(updated["remove_ratio"], 0.55)
        self.assertGreater(updated["score"], quality["score"])

    def test_propainter_remove_sufficiency_marks_under_remove_by_residual_hf(self):
        """验证 residual_hf_corr 过高也会触发 under_remove。"""
        quality = {
            "score": 0.10,
            "remove_energy_core": 0.12,
            "residual_hf_corr": 0.81,
            "reappear_flag": False,
        }
        updated = VideoProcessor._apply_propainter_remove_sufficiency(
            quality=quality,
            remove_energy_reference=0.11,
            previous_selected_remove_ratio=None,
            is_hard_cut_frame=False,
        )
        self.assertTrue(updated["under_remove_flag"])
        self.assertGreater(updated["residual_hf_corr"], 0.72)

    def test_propainter_rerun_options_and_trigger_rules(self):
        """验证段级重跑参数增强与触发/接受规则。"""
        base = {
            "mask_dilation": 10,
            "neighbor_length": 12,
            "ref_stride": 8,
            "subvideo_length": 100,
            "raft_iter": 24,
            "save_fps": 30,
        }
        rerun = VideoProcessor._compute_propainter_rerun_options(base, rect_w=120, rect_h=80)
        self.assertGreaterEqual(rerun["mask_dilation"], base["mask_dilation"])
        self.assertGreaterEqual(rerun["neighbor_length"], base["neighbor_length"])
        self.assertLessEqual(rerun["ref_stride"], base["ref_stride"])
        self.assertLessEqual(rerun["subvideo_length"], 80)
        self.assertGreaterEqual(rerun["raft_iter"], base["raft_iter"])

        self.assertTrue(
            VideoProcessor._should_rerun_propainter_segment(
                legacy_ratio=0.30,
                median_remove_ratio=0.75,
                reappear_count=0,
                frame_count=100,
                median_residual_hf_corr=0.40,
                under_remove_rate=0.02,
                burst_count=0,
            )
        )
        self.assertTrue(
            VideoProcessor._should_rerun_propainter_segment(
                legacy_ratio=0.05,
                median_remove_ratio=0.50,
                reappear_count=0,
                frame_count=100,
                median_residual_hf_corr=0.40,
                under_remove_rate=0.02,
                burst_count=0,
            )
        )
        self.assertTrue(
            VideoProcessor._should_rerun_propainter_segment(
                legacy_ratio=0.03,
                median_remove_ratio=0.72,
                reappear_count=0,
                frame_count=100,
                median_residual_hf_corr=0.71,
                under_remove_rate=0.25,
                burst_count=4,
            )
        )
        self.assertTrue(
            VideoProcessor._should_accept_propainter_rerun(
                pass1_median_remove_ratio=0.50,
                pass1_legacy_ratio=0.40,
                pass2_median_remove_ratio=0.60,
                pass2_legacy_ratio=0.30,
                pass1_under_remove_rate=0.30,
                pass2_under_remove_rate=0.12,
                pass1_seam_p90=0.05,
                pass2_seam_p90=0.052,
            )
        )
        self.assertFalse(
            VideoProcessor._should_accept_propainter_rerun(
                pass1_median_remove_ratio=0.50,
                pass1_legacy_ratio=0.40,
                pass2_median_remove_ratio=0.61,
                pass2_legacy_ratio=0.25,
                pass1_under_remove_rate=0.30,
                pass2_under_remove_rate=0.10,
                pass1_seam_p90=0.05,
                pass2_seam_p90=0.070,
            )
        )

    def test_propainter_hysteresis_suppresses_minor_switch(self):
        """验证 ProPainter 候选滞回会抑制单帧轻微切换。"""
        evaluations = {
            "raw_v2": {
                "score": 0.095,
                "dark_block_flag": False,
                "seam_extreme_flag": False,
                "reappear_flag": False,
            },
            "stable_v2": {
                "score": 0.100,
                "dark_block_flag": False,
                "seam_extreme_flag": False,
                "reappear_flag": False,
            },
            "legacy": {
                "score": 0.120,
                "dark_block_flag": False,
                "seam_extreme_flag": False,
                "reappear_flag": False,
            },
        }
        selected, streak, stats = VideoProcessor._apply_propainter_frame_hysteresis(
            selected_name="raw_v2",
            evaluations=evaluations,
            previous_name="stable_v2",
            legacy_advantage_streak=0,
        )
        self.assertEqual(selected, "stable_v2")
        self.assertEqual(streak, 0)
        self.assertEqual(stats["hold_count"], 1)
        self.assertEqual(stats["switch_count"], 0)

    def test_propainter_short_burst_repair_replaces_single_reappear_frame(self):
        """验证 1 帧回闪会被 second-pass 修复。"""
        h, w = 48, 72
        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_mask[14:34, 24:48] = 255
        transition_mask = np.zeros((h, w), dtype=np.uint8)
        transition_mask[10:38, 20:52] = 255

        roi_originals = [np.full((h, w, 3), 170, dtype=np.uint8) for _ in range(3)]
        selected_rois = [
            np.full((h, w, 3), 88, dtype=np.uint8),
            roi_originals[1].copy(),
            np.full((h, w, 3), 90, dtype=np.uint8),
        ]
        stable_candidates = [
            selected_rois[0].copy(),
            np.full((h, w, 3), 92, dtype=np.uint8),
            selected_rois[2].copy(),
        ]
        selected_qualities = [
            {"score": 0.12, "reappear_flag": False},
            {"score": 1.00, "reappear_flag": True},
            {"score": 0.11, "reappear_flag": False},
        ]

        repaired_rois, repaired_qualities, stats = VideoProcessor._repair_propainter_short_reappear_bursts(
            selected_rois=selected_rois,
            selected_qualities=selected_qualities,
            stable_candidates=stable_candidates,
            raw_candidates=stable_candidates,
            roi_originals=roi_originals,
            core_mask=core_mask,
            transition_mask=transition_mask,
            forced_reset_indices=set(),
            cold_start_window=2,
            max_burst_length=3,
        )

        self.assertEqual(stats["burst_fix_attempts"], 1)
        self.assertEqual(stats["burst_fix_accepted_frames"], 1)
        self.assertFalse(np.array_equal(repaired_rois[1], selected_rois[1]))
        self.assertFalse(repaired_qualities[1]["reappear_flag"])

    def test_propainter_short_burst_repair_skips_forced_cut_window(self):
        """验证硬切冷启动窗口内不会触发回闪强修复。"""
        h, w = 48, 72
        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_mask[14:34, 24:48] = 255
        transition_mask = np.zeros((h, w), dtype=np.uint8)
        transition_mask[10:38, 20:52] = 255

        roi_originals = [np.full((h, w, 3), 170, dtype=np.uint8) for _ in range(3)]
        selected_rois = [
            np.full((h, w, 3), 88, dtype=np.uint8),
            roi_originals[1].copy(),
            np.full((h, w, 3), 90, dtype=np.uint8),
        ]
        stable_candidates = [np.full((h, w, 3), 92, dtype=np.uint8) for _ in range(3)]
        selected_qualities = [
            {"score": 0.12, "reappear_flag": False},
            {"score": 1.00, "reappear_flag": True},
            {"score": 0.11, "reappear_flag": False},
        ]

        repaired_rois, _, stats = VideoProcessor._repair_propainter_short_reappear_bursts(
            selected_rois=selected_rois,
            selected_qualities=selected_qualities,
            stable_candidates=stable_candidates,
            raw_candidates=stable_candidates,
            roi_originals=roi_originals,
            core_mask=core_mask,
            transition_mask=transition_mask,
            forced_reset_indices={1},
            cold_start_window=2,
            max_burst_length=3,
        )
        self.assertEqual(stats["burst_fix_attempts"], 0)
        self.assertTrue(np.array_equal(repaired_rois[1], selected_rois[1]))

    def test_propainter_micro_burst_repair_replaces_single_micro_flicker_frame(self):
        """验证微闪 1 帧可被 second-pass 修复。"""
        h, w = 48, 72
        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_mask[14:34, 24:48] = 255
        transition_mask = np.zeros((h, w), dtype=np.uint8)
        transition_mask[10:38, 20:52] = 255
        roi_originals = [np.full((h, w, 3), 170, dtype=np.uint8) for _ in range(3)]
        selected_rois = [
            np.full((h, w, 3), 92, dtype=np.uint8),
            np.full((h, w, 3), 168, dtype=np.uint8),
            np.full((h, w, 3), 93, dtype=np.uint8),
        ]
        stable_candidates = [
            selected_rois[0].copy(),
            np.full((h, w, 3), 90, dtype=np.uint8),
            selected_rois[2].copy(),
        ]
        selected_qualities = [
            {"score": 0.11, "remove_ratio": 0.88, "temporal_jump_core": 0.02, "residual_hf_corr": 0.42, "remove_energy_reference": 0.20},
            {"score": 0.20, "remove_ratio": 0.64, "temporal_jump_core": 0.12, "residual_hf_corr": 0.82, "remove_energy_reference": 0.20},
            {"score": 0.10, "remove_ratio": 0.90, "temporal_jump_core": 0.02, "residual_hf_corr": 0.40, "remove_energy_reference": 0.20},
        ]

        def _fake_eval(*, roi_candidate, **kwargs):
            mean_val = float(np.asarray(roi_candidate).mean())
            if mean_val > 150.0:
                return {
                    "score": 0.20,
                    "remove_energy_core": 0.05,
                    "temporal_jump_core": 0.12,
                    "residual_hf_corr": 0.82,
                    "reappear_flag": False,
                    "seam_extreme_flag": False,
                    "dark_block_flag": False,
                }
            return {
                "score": 0.15,
                "remove_energy_core": 0.15,
                "temporal_jump_core": 0.03,
                "residual_hf_corr": 0.40,
                "reappear_flag": False,
                "seam_extreme_flag": False,
                "dark_block_flag": False,
            }

        with patch.object(VideoProcessor, "_evaluate_propainter_frame_quality", side_effect=_fake_eval):
            repaired_rois, _, stats = VideoProcessor._repair_propainter_micro_flicker_bursts(
                selected_rois=selected_rois,
                selected_qualities=selected_qualities,
                stable_candidates=stable_candidates,
                raw_candidates=stable_candidates,
                roi_originals=roi_originals,
                core_mask=core_mask,
                transition_mask=transition_mask,
                forced_reset_indices=set(),
                micro_flicker_flags=[False, True, False],
                cold_start_window=2,
                max_burst_length=2,
                cut_quarantine_indices=set(),
            )
        self.assertEqual(stats["micro_burst_fix_attempts"], 1)
        self.assertEqual(stats["micro_burst_fix_accepted_frames"], 1)
        self.assertFalse(np.array_equal(repaired_rois[1], selected_rois[1]))

    def test_propainter_micro_burst_repair_skips_cut_quarantine(self):
        """验证 cut 隔离窗口会跳过微闪修复。"""
        h, w = 48, 72
        core_mask = np.zeros((h, w), dtype=np.uint8)
        core_mask[14:34, 24:48] = 255
        transition_mask = np.zeros((h, w), dtype=np.uint8)
        transition_mask[10:38, 20:52] = 255
        roi_originals = [np.full((h, w, 3), 170, dtype=np.uint8) for _ in range(3)]
        selected_rois = [np.full((h, w, 3), 100, dtype=np.uint8) for _ in range(3)]
        selected_qualities = [
            {"score": 0.11, "remove_ratio": 0.88, "temporal_jump_core": 0.02, "residual_hf_corr": 0.42, "remove_energy_reference": 0.20}
            for _ in range(3)
        ]
        repaired_rois, _, stats = VideoProcessor._repair_propainter_micro_flicker_bursts(
            selected_rois=selected_rois,
            selected_qualities=selected_qualities,
            stable_candidates=selected_rois,
            raw_candidates=selected_rois,
            roi_originals=roi_originals,
            core_mask=core_mask,
            transition_mask=transition_mask,
            forced_reset_indices=set(),
            micro_flicker_flags=[False, True, False],
            cold_start_window=2,
            max_burst_length=2,
            cut_quarantine_indices={1},
        )
        self.assertEqual(stats["micro_burst_fix_attempts"], 0)
        self.assertTrue(np.array_equal(repaired_rois[1], selected_rois[1]))

    def test_propainter_repair_path_not_used_for_non_propainter_model(self):
        """验证非 ProPainter 模型不会调用回闪修复路径。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "output.mp4")
            _create_test_video(video_path, frames=6)

            processor = VideoProcessor(remover=_StubRemover())
            with patch.object(processor, "_transcode_video_h264", return_value=output_path):
                with patch.object(
                    VideoProcessor,
                    "_repair_propainter_short_reappear_bursts",
                    side_effect=AssertionError("should not be called"),
                ):
                    with patch.object(
                        VideoProcessor,
                        "_repair_propainter_micro_flicker_bursts",
                        side_effect=AssertionError("should not be called"),
                    ):
                        result = processor.process_video(
                            video_path=video_path,
                            output_path=output_path,
                            annotation_segments=[
                                {
                                    "id": "seg-1",
                                    "enabled": True,
                                    "start_frame": 0,
                                    "end_frame": 5,
                                    "rect": {"x": 4, "y": 4, "width": 16, "height": 12},
                                }
                            ],
                            model_id="lama_roi",
                        )
        self.assertEqual(result["effective_model_id"], "lama_roi")

    def test_propainter_process_splits_by_hard_cut_and_passes_internal_options(self):
        """验证 ProPainter 路径会按硬切分段并携带内部推理参数。"""
        class _Engine:
            def __init__(self):
                self.calls = []

            def inpaint_roi_sequence(self, roi_frames, roi_masks, progress_callback=None, **kwargs):
                self.calls.append(
                    {
                        "frames": len(roi_frames),
                        "kwargs": kwargs,
                    }
                )
                return [np.asarray(f).copy() for f in roi_frames]

        class _Registry:
            def __init__(self, engine):
                self._engine = engine

            def resolve(self, requested_model_id):
                info = type(
                    "R",
                    (),
                    {
                        "requested_model_id": "propainter_roi",
                        "effective_model_id": "propainter_roi",
                        "warning": "",
                    },
                )
                return self._engine, info

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "output.mp4")
            _create_test_video(video_path, frames=10)

            engine = _Engine()
            processor = VideoProcessor(remover=_StubRemover())
            with patch.object(processor, "_get_model_registry", return_value=_Registry(engine)):
                with patch.object(processor, "_transcode_video_h264", return_value=output_path):
                    with patch.object(VideoProcessor, "_detect_hard_cuts_in_run", return_value={4}):
                        result = processor.process_video(
                            video_path=video_path,
                            output_path=output_path,
                            annotation_segments=[
                                {
                                    "id": "seg-1",
                                    "enabled": True,
                                    "start_frame": 0,
                                    "end_frame": 9,
                                    "rect": {"x": 3, "y": 3, "width": 18, "height": 14},
                                }
                            ],
                            model_id="propainter_roi",
                        )

        self.assertEqual(result["output_path"], output_path)
        self.assertEqual(len(engine.calls), 2)
        first_call_kwargs = engine.calls[0]["kwargs"]
        self.assertIn("propainter_options", first_call_kwargs)
        self.assertEqual(first_call_kwargs["propainter_options"]["raft_iter"], 24)
        self.assertEqual(first_call_kwargs["propainter_options"]["neighbor_length"], 12)

    def test_propainter_process_applies_all_active_segments_per_frame_signature(self):
        """验证重叠区间会在同帧应用多个段，而不是只处理一个。"""

        class _Engine:
            def __init__(self):
                self.calls = []

            def inpaint_roi_sequence(self, roi_frames, roi_masks, progress_callback=None, **kwargs):
                self.calls.append(
                    {
                        "frames": len(roi_frames),
                        "kwargs": kwargs,
                    }
                )
                return [np.asarray(f).copy() for f in roi_frames]

        class _Registry:
            def __init__(self, engine):
                self._engine = engine

            def resolve(self, requested_model_id):
                info = type(
                    "R",
                    (),
                    {
                        "requested_model_id": "propainter_roi",
                        "effective_model_id": "propainter_roi",
                        "warning": "",
                    },
                )
                return self._engine, info

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            output_path = os.path.join(td, "output.mp4")
            _create_test_video(video_path, frames=9)

            engine = _Engine()
            processor = VideoProcessor(remover=_StubRemover())
            with patch.object(processor, "_get_model_registry", return_value=_Registry(engine)):
                with patch.object(processor, "_transcode_video_h264", return_value=output_path):
                    with patch.object(VideoProcessor, "_detect_hard_cuts_in_run", return_value=set()):
                        with patch.object(
                            VideoProcessor,
                            "_should_rerun_propainter_segment",
                            return_value=False,
                        ):
                            result = processor.process_video(
                                video_path=video_path,
                                output_path=output_path,
                                annotation_segments=[
                                    {
                                        "id": "seg-a",
                                        "enabled": True,
                                        "start_frame": 0,
                                        "end_frame": 5,
                                        "rect": {"x": 4, "y": 4, "width": 14, "height": 10},
                                    },
                                    {
                                        "id": "seg-b",
                                        "enabled": True,
                                        "start_frame": 3,
                                        "end_frame": 8,
                                        "rect": {"x": 22, "y": 14, "width": 13, "height": 9},
                                    },
                                ],
                                model_id="propainter_roi",
                            )

        self.assertEqual(result["output_path"], output_path)
        # 0-2: A; 3-5: A+B; 6-8: B => 1 + 2 + 1 = 4 次段应用调用
        self.assertEqual(len(engine.calls), 4)
        self.assertEqual([call["frames"] for call in engine.calls], [3, 3, 3, 3])


class ManualOnlyApiTests(unittest.TestCase):
    """manual-only GUI API 接口测试集。"""

    def test_clear_session_transient_data_deletes_session_sidecars_and_recent_files(self):
        """验证会话清理会删 sidecar 并清空最近文件。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            _create_test_video(video_path)

            save_sidecar(
                video_path,
                [
                    {
                        "id": "seg-1",
                        "start_frame": 0,
                        "end_frame": 2,
                        "rect": {"x": 1, "y": 1, "width": 10, "height": 10},
                    }
                ],
            )
            sidecar = build_sidecar_path(video_path)
            self.assertTrue(sidecar.exists())

            api = API()
            api._track_session_video_path(video_path)
            api.config.recent_files = [video_path]

            result = api.clear_session_transient_data()
            self.assertTrue(result["success"], result.get("failed"))
            self.assertFalse(sidecar.exists())
            self.assertEqual(api.config.recent_files, [])

    def test_begin_select_file_no_longer_persists_recent_files(self):
        """验证异步选文件流程不再自动写 recent_files。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "picked.mp4")
            _create_test_video(video_path)

            api = API()
            api.config_manager.add_recent_file = MagicMock()
            api._run_native_file_dialog = lambda select_folder=False: video_path

            begin = api.begin_select_file()
            self.assertTrue(begin["success"])
            request_id = begin["request_id"]

            poll = {"done": False}
            for _ in range(100):
                poll = api.poll_dialog_result(request_id)
                if poll.get("done"):
                    break
                time.sleep(0.01)

            self.assertTrue(poll.get("done"))
            self.assertTrue(poll.get("success"))
            self.assertEqual(poll.get("path"), video_path)
            api.config_manager.add_recent_file.assert_not_called()

            tracked = {str(Path(p)) for p in api._session_video_paths}
            self.assertIn(str(Path(video_path).resolve()), tracked)

    def test_run_native_file_dialog_uses_pywebview_native_dispatch(self):
        """验证文件对话框走 pywebview 原生调用路径。"""
        api = API()
        fake_path = "/tmp/picked.mp4"

        fake_webview = types.SimpleNamespace(
            FileDialog=types.SimpleNamespace(OPEN="open", FOLDER="folder"),
            OPEN_DIALOG="open",
            FOLDER_DIALOG="folder",
        )

        with patch.dict(sys.modules, {"webview": fake_webview}):
            with patch.object(api, "_create_file_dialog_compat", return_value=[fake_path]) as dialog_mock:
                result = api._run_native_file_dialog(select_folder=False)

        self.assertEqual(result, fake_path)
        dialog_mock.assert_called_once()

    def test_prepare_video_preview_uses_resolved_ffmpeg_path(self):
        """验证预览转码优先使用解析后的 ffmpeg 路径。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            Path(video_path).write_bytes(b'not-a-real-video')

            api = API()
            with patch('src.gui.api.resolve_ffmpeg_path', return_value='/embedded/ffmpeg'):
                with patch('src.gui.api.subprocess.run') as run_mock:
                    run_mock.return_value = subprocess.CompletedProcess(
                        args=['/embedded/ffmpeg'],
                        returncode=0,
                        stdout='',
                        stderr='',
                    )
                    with patch.object(api, '_transcode_preview_with_opencv', return_value=False):
                        _ = api.prepare_video_preview(video_path)

        self.assertTrue(run_mock.called)
        called_args = run_mock.call_args[0][0]
        self.assertEqual(called_args[0], '/embedded/ffmpeg')

    def test_process_video_rejects_deprecated_payload_fields(self):
        """验证过期字段（如 sora_mode）会被明确拒绝。"""
        api = API()
        result = api.process_video(
            {
                "input_path": "/tmp/not_used.mp4",
                "annotation_segments": [],
                "sora_mode": True,
            }
        )
        self.assertFalse(result["success"])
        self.assertIn("Unsupported payload fields", result["error"])

    def test_process_video_requires_annotation_segments(self):
        """验证不传 annotation_segments 会直接失败。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            _create_test_video(video_path)

            api = API()
            result = api.process_video({"input_path": video_path})
            self.assertFalse(result["success"])
            self.assertIn("annotation_segments is required", result["error"])

    def test_get_settings_no_detection_section(self):
        """验证 settings 结构不再包含 detection 分支。"""
        api = API()
        settings = api.get_settings()
        self.assertNotIn("detection", settings)
        self.assertIn("model_id", settings.get("output", {}))

    def test_process_video_success_with_stub_processor(self):
        """验证 legacy output_quality 参数会被拒绝并提示迁移。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            _create_test_video(video_path)

            api = API()
            api.remover = _StubRemover()
            api.processor = _StubProcessor()
            api._ensure_models = lambda: None

            result = api.process_video(
                {
                    "input_path": video_path,
                    "output_path": td,
                    "annotation_segments": [
                        {
                            "id": "seg-1",
                            "enabled": True,
                            "start_frame": 0,
                            "end_frame": 3,
                            "rect": {"x": 1, "y": 1, "width": 10, "height": 10},
                        }
                    ],
                    "settings": {"output_quality": "high"},
                }
            )

            self.assertFalse(result["success"])
            self.assertIn("settings.output_quality is removed", result["error"])

    def test_process_video_rejects_invalid_model_id(self):
        """验证非法 model_id 会被白名单校验拒绝。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            _create_test_video(video_path)

            api = API()
            result = api.process_video(
                {
                    "input_path": video_path,
                    "output_path": td,
                    "annotation_segments": [
                        {
                            "id": "seg-1",
                            "enabled": True,
                            "start_frame": 0,
                            "end_frame": 3,
                            "rect": {"x": 1, "y": 1, "width": 10, "height": 10},
                        }
                    ],
                    "settings": {"model_id": "bad_model"},
                }
            )

            self.assertFalse(result["success"])
            self.assertIn("Invalid model_id", result["error"])

    def test_process_video_success_with_stub_processor_and_model_id(self):
        """验证合法 model_id + 桩处理器时主流程可成功返回。"""
        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            _create_test_video(video_path)

            api = API()
            remover = _StubRemover()
            api.remover = remover
            api.processor = _StubProcessor()
            api._ensure_models = lambda: None

            result = api.process_video(
                {
                    "input_path": video_path,
                    "output_path": td,
                    "annotation_segments": [
                        {
                            "id": "seg-1",
                            "enabled": True,
                            "start_frame": 0,
                            "end_frame": 3,
                            "rect": {"x": 1, "y": 1, "width": 10, "height": 10},
                        }
                    ],
                    "settings": {"model_id": "propainter_roi"},
                }
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["requested_model_id"], "propainter_roi")
            self.assertEqual(result["effective_model_id"], "propainter_roi")
            self.assertTrue(result["output_path"].endswith("_no_watermark.mp4"))
            self.assertTrue(os.path.exists(result["output_path"]))
            self.assertEqual(remover.unload_calls, 1)

    def test_process_video_failure_triggers_remover_unload(self):
        """验证去水印失败后也会触发 remover 卸载。"""

        class _FailingProcessor:
            def process_video(self, **kwargs):
                raise RuntimeError("processor failed")

            def stop_processing(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            _create_test_video(video_path)

            api = API()
            remover = _StubRemover()
            api.remover = remover
            api.processor = _FailingProcessor()
            api._ensure_models = lambda: None

            result = api.process_video(
                {
                    "input_path": video_path,
                    "output_path": td,
                    "annotation_segments": [
                        {
                            "id": "seg-1",
                            "enabled": True,
                            "start_frame": 0,
                            "end_frame": 3,
                            "rect": {"x": 1, "y": 1, "width": 10, "height": 10},
                        }
                    ],
                    "settings": {"model_id": "lama_roi"},
                }
            )

            self.assertFalse(result["success"])
            self.assertEqual(remover.unload_calls, 1)

    def test_process_video_cancelled_triggers_remover_unload(self):
        """验证去水印取消（stopped）路径也会触发 remover 卸载。"""

        class _CancelledProcessor:
            def process_video(self, video_path, output_path, annotation_segments, model_id="lama_roi", **kwargs):
                return {
                    "output_path": "",
                    "requested_model_id": model_id,
                    "effective_model_id": model_id,
                    "model_warning": "",
                    "stopped": True,
                }

            def stop_processing(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            _create_test_video(video_path)

            api = API()
            remover = _StubRemover()
            api.remover = remover
            api.processor = _CancelledProcessor()
            api._ensure_models = lambda: None

            result = api.process_video(
                {
                    "input_path": video_path,
                    "output_path": td,
                    "annotation_segments": [
                        {
                            "id": "seg-1",
                            "enabled": True,
                            "start_frame": 0,
                            "end_frame": 3,
                            "rect": {"x": 1, "y": 1, "width": 10, "height": 10},
                        }
                    ],
                    "settings": {"model_id": "lama_roi"},
                }
            )

            self.assertTrue(result["success"])
            self.assertEqual(remover.unload_calls, 1)

    def test_process_video_progress_contains_eta_phase_and_monotonic_progress(self):
        """验证 API 透出的进度事件包含 ETA/phase 且 progress 单调。"""
        class _ProgressStubProcessor:
            def process_video(
                self,
                video_path,
                output_path,
                annotation_segments,
                model_id="lama_roi",
                progress_callback=None,
                status_callback=None,
            ):
                if status_callback:
                    status_callback("Loading models...")
                    status_callback("Extracting frames...")
                    status_callback("Processing frames...")
                if progress_callback:
                    progress_callback(
                        0.1,
                        "inference",
                        1,
                        10,
                        None,
                        {"phase": "infer", "step": 1, "total": 4, "opaque_infer": True},
                    )
                    time.sleep(0.25)
                    progress_callback(
                        0.6,
                        "inference",
                        6,
                        10,
                        None,
                        {"phase": "infer", "step": 3, "total": 4, "opaque_infer": True},
                    )
                    time.sleep(0.25)
                    progress_callback(
                        0.9,
                        "compose",
                        9,
                        10,
                        None,
                        {"phase": "compose", "step": 1, "total": 1, "opaque_infer": False},
                    )
                if status_callback:
                    status_callback("Finalizing video...")
                    status_callback("Complete!")

                Path(output_path).write_bytes(b"ok")
                return {
                    "output_path": output_path,
                    "requested_model_id": model_id,
                    "effective_model_id": model_id,
                    "model_warning": "",
                }

            def stop_processing(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            _create_test_video(video_path)

            api = API()
            api.remover = _StubRemover()
            api.processor = _ProgressStubProcessor()
            api._ensure_models = lambda: None

            progress_events = []
            api.set_progress_callback(lambda payload: progress_events.append(dict(payload)))

            result = api.process_video(
                {
                    "input_path": video_path,
                    "output_path": td,
                    "annotation_segments": [
                        {
                            "id": "seg-1",
                            "enabled": True,
                            "start_frame": 0,
                            "end_frame": 3,
                            "rect": {"x": 1, "y": 1, "width": 10, "height": 10},
                        }
                    ],
                    "settings": {"model_id": "lama_roi"},
                }
            )

            self.assertTrue(result["success"])
            self.assertTrue(progress_events)
            self.assertTrue(any("phase" in event for event in progress_events))
            self.assertTrue(any("eta_seconds" in event for event in progress_events))

            progress_values = [float(event.get("progress", 0.0)) for event in progress_events if "progress" in event]
            self.assertEqual(progress_values, sorted(progress_values))

    def test_process_video_progress_does_not_jump_to_18_before_infer_updates(self):
        """验证首条 infer 回调前不会直接冲到历史 18% 档位。"""
        class _ProgressBridgeStubProcessor:
            def process_video(
                self,
                video_path,
                output_path,
                annotation_segments,
                model_id="lama_roi",
                progress_callback=None,
                status_callback=None,
            ):
                if status_callback:
                    status_callback("Preparing task...")
                    status_callback("Loading models...")
                    status_callback("Extracting frames...")
                    status_callback("Processing frames...")
                if progress_callback:
                    progress_callback(
                        0.0,
                        "inference-1",
                        None,
                        None,
                        None,
                        {"phase": "infer", "step": 1, "total": 20, "opaque_infer": True},
                    )
                    progress_callback(
                        0.0,
                        "inference-2",
                        None,
                        None,
                        None,
                        {"phase": "infer", "step": 4, "total": 20, "opaque_infer": True},
                    )
                if status_callback:
                    status_callback("Finalizing video...")
                    status_callback("Complete!")

                Path(output_path).write_bytes(b"ok")
                return {
                    "output_path": output_path,
                    "requested_model_id": model_id,
                    "effective_model_id": model_id,
                    "model_warning": "",
                }

            def stop_processing(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "input.mp4")
            _create_test_video(video_path)

            api = API()
            api.remover = _StubRemover()
            api.processor = _ProgressBridgeStubProcessor()
            api._ensure_models = lambda: None

            progress_events = []
            api.set_progress_callback(lambda payload: progress_events.append(dict(payload)))

            result = api.process_video(
                {
                    "input_path": video_path,
                    "output_path": td,
                    "annotation_segments": [
                        {
                            "id": "seg-1",
                            "enabled": True,
                            "start_frame": 0,
                            "end_frame": 3,
                            "rect": {"x": 1, "y": 1, "width": 10, "height": 10},
                        }
                    ],
                    "settings": {"model_id": "lama_roi"},
                }
            )

            self.assertTrue(result["success"])
            self.assertTrue(progress_events)

            first_infer_idx = next(
                (idx for idx, event in enumerate(progress_events) if event.get("message") == "inference-1"),
                len(progress_events),
            )
            early_progress = [
                float(event.get("progress", 0.0))
                for event in progress_events[:first_infer_idx]
                if "progress" in event
            ]
            self.assertTrue(early_progress, "expected early progress samples before infer callback")
            self.assertLess(max(early_progress), 0.18)

            infer_progress = [
                float(event.get("progress", 0.0))
                for event in progress_events
                if str(event.get("message", "")).startswith("inference-")
            ]
            self.assertGreaterEqual(len(infer_progress), 2)
            self.assertLess(infer_progress[0], infer_progress[-1])


if __name__ == "__main__":
    unittest.main()
