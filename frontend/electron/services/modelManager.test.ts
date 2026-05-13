import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { ModelManager } from './modelManager.js';

describe('model manager', () => {
  it('reports only the supported LaMa inpaint model', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-models-'));
    try {
      const manager = new ModelManager({ userDataDir: dir });
      const status = await manager.getStatus();

      expect(status.models.map((model) => model.model_id)).toEqual(['lama_roi']);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('rejects unsupported model ids at the Electron boundary', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-models-'));
    try {
      const manager = new ModelManager({ userDataDir: dir });
      const result = await manager.startDownload('sttn_roi', false);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid model_id');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
