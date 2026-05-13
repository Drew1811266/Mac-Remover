import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  buildSidecarPath,
  deleteAnnotations,
  loadAnnotations,
  normalizeSegments,
  saveAnnotations,
} from './annotations.js';
import type { VideoMeta } from './types.js';

describe('annotations service', () => {
  it('builds sidecar paths next to the video file', () => {
    const videoPath = path.join('/tmp', 'sample.video.mp4');
    expect(buildSidecarPath(videoPath)).toBe(path.join('/tmp', 'sample.video.mp4.wmr.json'));
  });

  it('normalizes segment frame ranges and rect bounds', () => {
    const segments = normalizeSegments(
      [
        {
          id: 'seg-1',
          start_frame: 40,
          end_frame: 5,
          rect: { x: -10, y: 30, width: 999, height: 999 },
          expand_px: -4,
          feather_px: 2,
          enabled: true,
        },
      ],
      { width: 100, height: 80, frame_count: 50 },
    );

    expect(segments).toHaveLength(1);
    expect(segments[0]).toMatchObject({
      id: 'seg-1',
      start_frame: 5,
      end_frame: 40,
      rect: { x: 0, y: 30, width: 100, height: 50 },
      expand_px: 0,
      feather_px: 2,
      enabled: true,
    });
    expect(segments[0].created_at).toEqual(expect.any(String));
    expect(segments[0].updated_at).toEqual(expect.any(String));
  });

  it('saves, loads, warns on changed fingerprints, and deletes sidecars', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-annotations-'));
    try {
      const videoPath = path.join(dir, 'clip.mp4');
      await writeFile(videoPath, 'original');
      const meta: VideoMeta = {
        path: videoPath,
        basename: 'clip.mp4',
        sha1: 'sha-before',
        size: 8,
        mtime_ns: 1,
        width: 64,
        height: 32,
        fps: 24,
        frame_count: 10,
      };

      const saved = await saveAnnotations({
        videoPath,
        videoMeta: meta,
        segments: [{ start_frame: 0, end_frame: 2, rect: { x: 1, y: 1, width: 8, height: 8 } }],
      });
      expect(saved.success).toBe(true);

      const raw = JSON.parse(await readFile(buildSidecarPath(videoPath), 'utf8'));
      expect(raw.version).toBe('1.0');
      expect(raw.video_meta.basename).toBe('clip.mp4');

      const loaded = await loadAnnotations(videoPath, {
        ...meta,
        sha1: 'sha-after',
        size: 9,
      });
      expect(loaded.success).toBe(true);
      expect(loaded.exists).toBe(true);
      expect(loaded.warning).toContain('fingerprint mismatch');
      expect(loaded.segments).toHaveLength(1);

      const deleted = await deleteAnnotations(videoPath);
      expect(deleted.success).toBe(true);
      const afterDelete = await loadAnnotations(videoPath, meta);
      expect(afterDelete.exists).toBe(false);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
