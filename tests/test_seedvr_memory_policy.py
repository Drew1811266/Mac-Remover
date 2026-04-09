"""SeedVR 内存策略测试。"""

import unittest

from src.core.seedvr_memory_policy import (
    LEGACY_SAME_RES_X4,
    SUPPORTED_SAME_RES_STRENGTH,
    build_emergency_seedvr_profile,
    build_stall_recovery_seedvr_profile,
    build_seedvr_memory_profile,
    normalize_same_res_strength,
)


class SeedVRMemoryPolicyTests(unittest.TestCase):
    def test_normalize_same_res_strength_alias(self):
        normalized, warnings = normalize_same_res_strength(LEGACY_SAME_RES_X4)
        self.assertEqual(normalized, SUPPORTED_SAME_RES_STRENGTH)
        self.assertTrue(warnings)

    def test_build_profile_safe(self):
        profile = build_seedvr_memory_profile(
            mode="upscale_resolution",
            same_res_strength=SUPPORTED_SAME_RES_STRENGTH,
            requested_short_resolution=1080,
            duration_sec=8.0,
            total_memory_gb=24.0,
            available_memory_gb=18.0,
        )
        self.assertEqual(profile.risk_level, "safe")
        self.assertEqual(profile.batch_size, 5)
        self.assertEqual(profile.chunk_size, 24)
        self.assertEqual(profile.max_resolution, 1080)
        self.assertTrue(profile.vae_encode_tiled)
        self.assertEqual(profile.vae_tile_size, 896)
        self.assertGreater(profile.memory_guard_min_available_gb, 0.0)
        self.assertGreater(profile.memory_guard_max_process_rss_gb, 0.0)
        self.assertEqual(profile.dit_offload_device, "none")
        self.assertEqual(profile.vae_offload_device, "none")
        self.assertEqual(profile.tensor_offload_device, "cpu")
        self.assertTrue(profile.cache_dit)
        self.assertTrue(profile.cache_vae)
        self.assertTrue(any("MPS-first execution policy" in w for w in profile.warnings))
        self.assertTrue(any("Video backend preference: ffmpeg" in w for w in profile.warnings))

    def test_build_profile_guarded(self):
        profile = build_seedvr_memory_profile(
            mode="upscale_resolution",
            same_res_strength=SUPPORTED_SAME_RES_STRENGTH,
            requested_short_resolution=1080,
            duration_sec=50.0,
            total_memory_gb=24.0,
            available_memory_gb=11.5,
        )
        self.assertEqual(profile.risk_level, "guarded")
        self.assertEqual(profile.batch_size, 5)
        self.assertEqual(profile.chunk_size, 12)
        self.assertTrue(profile.vae_encode_tiled)
        self.assertEqual(profile.max_resolution, 1080)
        self.assertEqual(profile.vae_tile_size, 768)
        self.assertGreater(profile.memory_guard_min_available_gb, 2.0)
        self.assertEqual(profile.dit_offload_device, "none")
        self.assertEqual(profile.vae_offload_device, "cpu")
        self.assertEqual(profile.tensor_offload_device, "cpu")
        self.assertFalse(profile.cache_dit)
        self.assertFalse(profile.cache_vae)

    def test_build_profile_critical_caps_to_1080(self):
        profile = build_seedvr_memory_profile(
            mode="upscale_resolution",
            same_res_strength=SUPPORTED_SAME_RES_STRENGTH,
            requested_short_resolution=2160,
            duration_sec=95.0,
            total_memory_gb=24.0,
            available_memory_gb=7.9,
        )
        self.assertEqual(profile.risk_level, "critical")
        self.assertEqual(profile.batch_size, 1)
        self.assertEqual(profile.chunk_size, 8)
        self.assertEqual(profile.target_short_resolution, 1080)
        self.assertTrue(profile.warnings)
        self.assertGreater(profile.memory_guard_min_available_gb, 2.5)
        self.assertEqual(profile.dit_offload_device, "cpu")
        self.assertEqual(profile.vae_offload_device, "cpu")
        self.assertEqual(profile.tensor_offload_device, "cpu")
        self.assertFalse(profile.cache_dit)
        self.assertFalse(profile.cache_vae)

    def test_build_emergency_profile(self):
        profile = build_emergency_seedvr_profile(requested_short_resolution=2000)
        self.assertEqual(profile.risk_level, "emergency")
        self.assertEqual(profile.batch_size, 1)
        self.assertEqual(profile.chunk_size, 4)
        self.assertEqual(profile.target_short_resolution, 1080)
        self.assertTrue(profile.vae_decode_tiled)
        self.assertEqual(profile.vae_tile_size, 512)
        self.assertGreater(profile.memory_guard_min_available_gb, 3.0)
        self.assertEqual(profile.dit_offload_device, "cpu")
        self.assertEqual(profile.vae_offload_device, "cpu")
        self.assertEqual(profile.tensor_offload_device, "cpu")
        self.assertFalse(profile.cache_dit)
        self.assertFalse(profile.cache_vae)
        self.assertTrue(any("Applied streaming profile: emergency" in w for w in profile.warnings))

    def test_build_stall_recovery_profile(self):
        profile = build_stall_recovery_seedvr_profile(requested_short_resolution=1440)
        self.assertEqual(profile.risk_level, "stall_recovery")
        self.assertEqual(profile.batch_size, 1)
        self.assertEqual(profile.chunk_size, 4)
        self.assertEqual(profile.vae_tile_size, 512)
        self.assertEqual(profile.target_short_resolution, 1080)
        self.assertEqual(profile.dit_offload_device, "none")
        self.assertEqual(profile.vae_offload_device, "none")
        self.assertEqual(profile.tensor_offload_device, "cpu")
        self.assertFalse(profile.cache_dit)
        self.assertFalse(profile.cache_vae)
        self.assertTrue(any("stall_recovery" in w for w in profile.warnings))

    def test_16g_device_is_more_conservative(self):
        profile = build_seedvr_memory_profile(
            mode="upscale_resolution",
            same_res_strength=SUPPORTED_SAME_RES_STRENGTH,
            requested_short_resolution=1080,
            duration_sec=36.0,
            total_memory_gb=16.0,
            available_memory_gb=13.2,
        )
        self.assertEqual(profile.risk_level, "guarded")

    def test_q4_model_uses_conservative_safe_profile(self):
        profile = build_seedvr_memory_profile(
            mode="upscale_resolution",
            same_res_strength=SUPPORTED_SAME_RES_STRENGTH,
            requested_short_resolution=1080,
            duration_sec=8.0,
            total_memory_gb=24.0,
            available_memory_gb=18.0,
            model_id="seedvr2_3b_q4_k_m_gguf",
        )
        self.assertEqual(profile.risk_level, "safe")
        self.assertEqual(profile.batch_size, 1)
        self.assertEqual(profile.chunk_size, 12)
        self.assertEqual(profile.vae_offload_device, "none")
        self.assertFalse(profile.cache_dit)
        self.assertFalse(profile.cache_vae)
        self.assertTrue(any("Q4 conservative startup profile applied" in w for w in profile.warnings))


if __name__ == "__main__":
    unittest.main()
