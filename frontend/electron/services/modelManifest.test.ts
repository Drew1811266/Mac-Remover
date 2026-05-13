import { describe, expect, it } from 'vitest';

import { validateManifest } from './modelManifest.js';

describe('model manifest validation', () => {
  it('accepts a manifest with model and runtime assets', () => {
    const manifest = validateManifest({
      version: 1,
      assets: [
        {
          asset_id: 'big-lama-onnx',
          kind: 'model',
          model_id: 'lama_roi',
          engine: 'lama',
          runtime_kind: 'onnx',
          version: '1.0.0',
          url: 'https://example.com/big-lama.onnx',
          sha256: 'a'.repeat(64),
          size: 123,
          file_name: 'big-lama.onnx',
          license: 'self-owned',
        },
        {
          asset_id: 'ffmpeg-darwin-arm64',
          kind: 'runtime',
          engine: 'ffmpeg',
          runtime_kind: 'ffmpeg',
          platform: 'darwin',
          arch: 'arm64',
          version: '7.1',
          url: 'https://example.com/ffmpeg.zip',
          sha256: 'b'.repeat(64),
          size: 456,
          file_name: 'ffmpeg.zip',
          license: 'LGPL-2.1-or-later',
        },
      ],
    });

    expect(manifest.assets).toHaveLength(2);
    expect(manifest.assets[0].id).toBe('big-lama-onnx');
    expect(manifest.assets[0].fileName).toBe('big-lama.onnx');
  });

  it('rejects invalid checksums and missing asset fields', () => {
    expect(() =>
      validateManifest({
        version: 1,
        assets: [{ id: 'bad', kind: 'model', url: '', sha256: 'nope' }],
      }),
    ).toThrow(/Invalid manifest asset/);
  });

  it('rejects removed ProPainter model assets', () => {
    expect(() =>
      validateManifest({
        version: 1,
        assets: [
          {
            asset_id: 'propainter-inpaint-onnx',
            kind: 'model',
            model_id: 'propainter_roi',
            engine: 'propainter',
            runtime_kind: 'onnx',
            version: '1.0.0',
            url: 'https://example.com/propainter.onnx',
            sha256: 'c'.repeat(64),
            size: 123,
            file_name: 'propainter.onnx',
            license: 'self-owned',
          },
        ],
      }),
    ).toThrow(/Invalid manifest asset/);
  });
});
