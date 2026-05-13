import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { createVideoProcessor } from './processor.js';

describe('video processor', () => {
  it('does not require the optional native core before LaMa ONNX input validation', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-processor-'));
    try {
      const modelPath = path.join(dir, 'models', 'big-lama', '1.0.0', 'big-lama.onnx');
      await mkdir(path.dirname(modelPath), { recursive: true });
      await writeFile(modelPath, 'placeholder');

      const processor = createVideoProcessor({
        userDataDir: dir,
        appRoot: dir,
        emitProgress: () => undefined,
      });
      const result = await processor.processVideo({
        input_path: path.join(dir, 'missing.mp4'),
        output_path: dir,
        annotation_segments: [{ start_frame: 0, end_frame: 0, rect: { x: 0, y: 0, width: 8, height: 8 } }],
        settings: { model_id: 'lama_roi' },
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('File not found');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('rejects removed ProPainter requests as unsupported', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-processor-'));
    try {
      const processor = createVideoProcessor({
        userDataDir: dir,
        appRoot: dir,
        emitProgress: () => undefined,
      });
      const result = await processor.processVideo({
        input_path: path.join(dir, 'missing.mp4'),
        output_path: dir,
        annotation_segments: [{ start_frame: 0, end_frame: 0, rect: { x: 0, y: 0, width: 8, height: 8 } }],
        settings: { model_id: 'propainter_roi' as never },
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid model_id');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
