import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { UpscaleService } from './upscale.js';

describe('upscale service', () => {
  it('exposes all parity-gated upscale models', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-upscale-'));
    try {
      const service = new UpscaleService({ userDataDir: dir, appRoot: dir });
      const status = await service.getModelDownloadStatus();

      expect(status.models.map((model) => model.model_id)).toEqual([
        'realesrgan_general_x4v3',
        'realesrgan_x2plus',
        'seedvr2_3b_q8_0_gguf',
        'seedvr2_3b_q4_k_m_gguf',
      ]);
      expect(status.models.every((model) => model.install_hint.includes('Blocked'))).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('blocks upscale execution when native assets are absent', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-upscale-'));
    try {
      const service = new UpscaleService({ userDataDir: dir, appRoot: dir });
      const result = await service.startUpscale({
        input_path: path.join(dir, 'input.mp4'),
        engine: 'seedvr2',
        model_id: 'seedvr2_3b_q4_k_m_gguf',
        mode: 'enhance_same_resolution',
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('SeedVR2 native engine is blocked');
      expect(service.getTaskStatus().task.state).toBe('failed');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
