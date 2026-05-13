import { randomUUID } from 'node:crypto';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

import { resolveRuntimeTool } from './ffmpeg.js';
import { MediaService } from './media.js';
import { runProcess } from './processRunner.js';
import type { PreviewFrameResult, PreviewSessionResult } from './types.js';

interface PreviewSession {
  path: string;
  previewFps: number;
  totalPreviewFrames: number;
  width: number;
  height: number;
  cursor: number;
  sourceFps: number;
  sourceFrameCount: number;
  step: number;
  lastAccess: number;
}

export interface PreviewServiceOptions {
  userDataDir: string;
  appRoot: string;
}

export class PreviewService {
  private readonly sessions = new Map<string, PreviewSession>();
  private readonly media: MediaService;

  constructor(private readonly options: PreviewServiceOptions) {
    this.media = new MediaService(options);
  }

  async openVideoPreviewSession(
    input: string,
    targetFps = 15,
    maxWidth = 1280,
  ): Promise<PreviewSessionResult> {
    try {
      const meta = await this.media.getVideoMeta(input);
      const safeTargetFps = Math.max(1, Math.min(30, Math.round(targetFps || 15)));
      const step = Math.max(1, Math.round((meta.fps || 24) / safeTargetFps));
      const previewFps = (meta.fps || 24) / step;
      const totalPreviewFrames = Math.max(1, Math.ceil(Math.max(1, meta.frame_count) / step));
      const scale = maxWidth > 0 && meta.width > maxWidth ? maxWidth / meta.width : 1;
      const width = even(Math.max(2, Math.round(meta.width * scale)));
      const height = even(Math.max(2, Math.round(meta.height * scale)));
      const sessionId = randomUUID().replaceAll('-', '');

      this.sessions.set(sessionId, {
        path: meta.path,
        previewFps,
        totalPreviewFrames,
        width,
        height,
        cursor: 0,
        sourceFps: meta.fps || 24,
        sourceFrameCount: Math.max(1, meta.frame_count || 1),
        step,
        lastAccess: Date.now(),
      });

      return {
        success: true,
        session_id: sessionId,
        preview_fps: previewFps,
        total_preview_frames: totalPreviewFrames,
        width,
        height,
      };
    } catch (error) {
      return { success: false, error: errorMessage(error) };
    }
  }

  async readVideoPreviewFrame(sessionId: string, frameIndex?: number): Promise<PreviewFrameResult> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return { success: false, error: 'Preview session not found' };
    }

    const started = Date.now();
    const normalizedFrame =
      typeof frameIndex === 'number'
        ? normalizeFrameIndex(frameIndex, session.totalPreviewFrames)
        : normalizeFrameIndex(session.cursor, session.totalPreviewFrames);
    session.cursor = normalizeFrameIndex(normalizedFrame + 1, session.totalPreviewFrames);
    session.lastAccess = Date.now();

    try {
      const ffmpeg = await resolveRuntimeTool({
        tool: 'ffmpeg',
        userDataDir: this.options.userDataDir,
        appRoot: this.options.appRoot,
      });
      const sourceFrameIndex = Math.min(
        Math.max(0, normalizedFrame * session.step),
        Math.max(0, session.sourceFrameCount - 1),
      );
      const seconds = sourceFrameIndex / session.sourceFps;
      const frameBuffer = await readPreviewJpegFrame(ffmpeg.path, session, sourceFrameIndex, seconds);

      return {
        success: true,
        frame_index: normalizedFrame,
        frame_url: `data:image/jpeg;base64,${frameBuffer.toString('base64')}`,
        decode_ms: Date.now() - started,
      };
    } catch (error) {
      return { success: false, error: errorMessage(error) };
    }
  }

  closeVideoPreviewSession(sessionId: string): { success: boolean; error?: string } {
    this.sessions.delete(sessionId);
    return { success: true };
  }

  async prepareVideoPreview(input: string): Promise<{ success: boolean; path?: string; cached?: boolean; transcoded?: boolean; error?: string }> {
    try {
      const previewDir = path.join(this.options.userDataDir, 'preview-cache');
      await mkdir(previewDir, { recursive: true });
      return { success: true, path: input, cached: true, transcoded: false };
    } catch (error) {
      return { success: false, error: errorMessage(error) };
    }
  }
}

async function readPreviewJpegFrame(
  ffmpegPath: string,
  session: PreviewSession,
  sourceFrameIndex: number,
  seconds: number,
): Promise<Buffer> {
  const scaleFilter = `scale=${session.width}:${session.height},format=yuvj420p`;
  try {
    return await runPreviewFrameCommand(ffmpegPath, [
      '-v',
      'error',
      '-ss',
      seconds.toFixed(3),
      '-i',
      session.path,
      '-frames:v',
      '1',
      '-vf',
      scaleFilter,
      '-f',
      'image2pipe',
      '-vcodec',
      'mjpeg',
      'pipe:1',
    ]);
  } catch {
    return runPreviewFrameCommand(ffmpegPath, [
      '-v',
      'error',
      '-i',
      session.path,
      '-vf',
      `select=eq(n\\,${sourceFrameIndex}),${scaleFilter}`,
      '-vsync',
      'vfr',
      '-frames:v',
      '1',
      '-f',
      'image2pipe',
      '-vcodec',
      'mjpeg',
      'pipe:1',
    ]);
  }
}

async function runPreviewFrameCommand(ffmpegPath: string, args: string[]): Promise<Buffer> {
  const result = await runProcess(ffmpegPath, args, { timeoutMs: 30_000 });
  if (result.stdoutBuffer.length <= 0) {
    throw new Error('FFmpeg returned an empty preview frame');
  }
  return result.stdoutBuffer;
}

function even(value: number): number {
  return value % 2 === 0 ? value : value - 1;
}

function normalizeFrameIndex(value: number, total: number): number {
  if (total <= 0) return 0;
  const rounded = Math.trunc(value);
  return ((rounded % total) + total) % total;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
