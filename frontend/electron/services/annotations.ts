import { createHash, randomUUID } from 'node:crypto';
import { access, readFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

import type {
  AnnotationSegment,
  LoadAnnotationsResult,
  NormalizedAnnotationSegment,
  SaveAnnotationsResult,
  VideoMeta,
} from './types.js';

const SIDECAR_SUFFIX = '.wmr.json';

interface SaveAnnotationsInput {
  videoPath: string;
  segments: AnnotationSegment[];
  videoMeta: VideoMeta;
}

export function buildSidecarPath(videoPath: string): string {
  return path.join(path.dirname(videoPath), `${path.basename(videoPath)}${SIDECAR_SUFFIX}`);
}

export async function fileSha1(filePath: string): Promise<string> {
  const data = await readFile(filePath);
  return createHash('sha1').update(data).digest('hex');
}

export async function buildVideoMeta(baseMeta: Omit<VideoMeta, 'sha1' | 'size' | 'mtime_ns'>): Promise<VideoMeta> {
  const fileStat = await stat(baseMeta.path);
  return {
    ...baseMeta,
    sha1: await fileSha1(baseMeta.path),
    size: Number(fileStat.size),
    mtime_ns: Math.trunc(Number(fileStat.mtimeMs) * 1_000_000),
  };
}

export function normalizeSegments(
  segments: AnnotationSegment[],
  videoMeta: Pick<VideoMeta, 'width' | 'height' | 'frame_count'>,
): NormalizedAnnotationSegment[] {
  const width = Math.max(0, toInt(videoMeta.width, 0));
  const height = Math.max(0, toInt(videoMeta.height, 0));
  const frameCount = Math.max(0, toInt(videoMeta.frame_count, 0));
  const maxFrame = Math.max(0, frameCount - 1);
  const now = new Date().toISOString();

  return (Array.isArray(segments) ? segments : []).flatMap((segment) => {
    if (!segment || typeof segment !== 'object' || !segment.rect || typeof segment.rect !== 'object') {
      return [];
    }

    let startFrame = clamp(toInt(segment.start_frame, 0), 0, maxFrame);
    let endFrame = clamp(toInt(segment.end_frame, startFrame), 0, maxFrame);
    if (endFrame < startFrame) {
      [startFrame, endFrame] = [endFrame, startFrame];
    }

    const rectX = clamp(toInt(segment.rect.x, 0), 0, Math.max(0, width - 1));
    const rectY = clamp(toInt(segment.rect.y, 0), 0, Math.max(0, height - 1));
    const rectWidth = Math.max(1, toInt(segment.rect.width, 1));
    const rectHeight = Math.max(1, toInt(segment.rect.height, 1));

    return [
      {
        id: String(segment.id || randomUUID().replaceAll('-', '')),
        start_frame: startFrame,
        end_frame: endFrame,
        rect: {
          x: rectX,
          y: rectY,
          width: Math.max(1, Math.min(rectWidth, Math.max(1, width - rectX))),
          height: Math.max(1, Math.min(rectHeight, Math.max(1, height - rectY))),
        },
        expand_px: Math.max(0, toInt(segment.expand_px, 5)),
        feather_px: Math.max(0, toInt(segment.feather_px, 3)),
        enabled: segment.enabled !== false,
        created_at: String(segment.created_at || now),
        updated_at: now,
      },
    ];
  });
}

export async function saveAnnotations(input: SaveAnnotationsInput): Promise<SaveAnnotationsResult> {
  if (!input.videoPath) {
    return { success: false, error: 'Missing video_path' };
  }

  const normalized = normalizeSegments(input.segments, input.videoMeta);
  const sidecarPath = buildSidecarPath(input.videoPath);
  const payload = {
    version: '1.0',
    video_meta: input.videoMeta,
    segments: normalized,
    updated_at: new Date().toISOString(),
  };

  await writeFile(sidecarPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');

  return {
    success: true,
    exists: true,
    sidecar_path: sidecarPath,
    video_meta: input.videoMeta,
    segments: normalized,
  };
}

export async function loadAnnotations(videoPath: string, currentMeta: VideoMeta): Promise<LoadAnnotationsResult> {
  if (!videoPath) {
    return { success: false, error: 'Missing video_path' };
  }

  const sidecarPath = buildSidecarPath(videoPath);
  try {
    await access(sidecarPath);
  } catch {
    return { success: true, exists: false, warning: 'Annotation file not found' };
  }

  try {
    const raw = JSON.parse(await readFile(sidecarPath, 'utf8')) as {
      version?: unknown;
      video_meta?: Partial<VideoMeta>;
      segments?: AnnotationSegment[];
      updated_at?: unknown;
    };
    const storedMeta = raw.video_meta || {};
    const warning =
      storedMeta.sha1 !== currentMeta.sha1 ||
      Number(storedMeta.size ?? -1) !== currentMeta.size ||
      Number(storedMeta.mtime_ns ?? -1) !== currentMeta.mtime_ns
        ? 'Video fingerprint mismatch. Annotation not auto-applied.'
        : undefined;

    return {
      success: true,
      exists: true,
      warning,
      sidecar_path: sidecarPath,
      video_meta: currentMeta,
      segments: normalizeSegments(Array.isArray(raw.segments) ? raw.segments : [], currentMeta),
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function deleteAnnotations(videoPath: string): Promise<{ success: boolean; error?: string }> {
  if (!videoPath) {
    return { success: false, error: 'Missing video_path' };
  }
  await rm(buildSidecarPath(videoPath), { force: true });
  return { success: true };
}

function toInt(value: unknown, fallback: number): number {
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
