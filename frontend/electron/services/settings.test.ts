import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { loadSettings, saveSettings } from './settings.js';

describe('settings service', () => {
  it('loads stable defaults and persists supported fields', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-settings-'));
    try {
      await expect(loadSettings(dir)).resolves.toEqual({
        language: 'zh',
        theme: 'light',
        output: {
          path: path.join(os.homedir(), 'Downloads', 'WatermarkRemover'),
          model_id: 'lama_roi',
        },
      });

      const saved = await saveSettings(dir, {
        language: 'en',
        theme: 'dark',
        output: { path: path.join(dir, 'out'), model_id: 'lama_roi' },
      });
      expect(saved.success).toBe(true);

      await expect(loadSettings(dir)).resolves.toMatchObject({
        language: 'en',
        theme: 'dark',
        output: { path: path.join(dir, 'out'), model_id: 'lama_roi' },
      });

      const legacyUnsupported = await saveSettings(dir, {
        output: { path: path.join(dir, 'out'), model_id: 'propainter_roi' as never },
      });
      expect(legacyUnsupported.success).toBe(true);
      await expect(loadSettings(dir)).resolves.toMatchObject({
        output: { model_id: 'lama_roi' },
      });
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
