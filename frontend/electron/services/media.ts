import path from 'node:path';
import sharp from 'sharp';

import { buildVideoMeta } from './annotations.js';
import { resolveRuntimeTool } from './ffmpeg.js';
import { runProcess } from './processRunner.js';
import type { MediaInfoResult, VideoMeta } from './types.js';

export interface MediaServiceOptions {
  userDataDir: string;
  appRoot: string;
}

export class MediaService {
  constructor(private readonly options: MediaServiceOptions) {}

  async getMediaInfo(input: string): Promise<MediaInfoResult> {
    if (!input) {
      return { success: false, error: 'Missing input path' };
    }

    if (isLikelyImage(input)) {
      try {
        const metadata = await sharp(input).metadata();
        return {
          success: true,
          type: 'image',
          width: metadata.width || 0,
          height: metadata.height || 0,
        };
      } catch (error) {
        return { success: false, error: errorMessage(error) };
      }
    }

    try {
      const meta = await this.getVideoMeta(input);
      return {
        success: true,
        type: 'video',
        fps: meta.fps,
        frame_count: meta.frame_count,
        width: meta.width,
        height: meta.height,
      };
    } catch (error) {
      return { success: false, error: errorMessage(error) };
    }
  }

  async getVideoMeta(input: string): Promise<VideoMeta> {
    const ffprobe = await resolveRuntimeTool({
      tool: 'ffprobe',
      userDataDir: this.options.userDataDir,
      appRoot: this.options.appRoot,
    });
    const result = await runProcess(ffprobe.path, [
      '-v',
      'error',
      '-select_streams',
      'v:0',
      '-show_entries',
      'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration',
      '-of',
      'json',
      input,
    ]);
    const parsed = JSON.parse(result.stdout) as {
      streams?: Array<{
        width?: number;
        height?: number;
        r_frame_rate?: string;
        avg_frame_rate?: string;
        nb_frames?: string;
        duration?: string;
      }>;
    };
    const stream = parsed.streams?.[0];
    if (!stream) {
      throw new Error(`Cannot read video stream: ${input}`);
    }
    const fps = parseFps(stream.avg_frame_rate || stream.r_frame_rate);
    const duration = Number.parseFloat(String(stream.duration || '0'));
    const frameCountFromDuration = fps > 0 && duration > 0 ? Math.round(fps * duration) : 0;
    const frameCount = Number.parseInt(String(stream.nb_frames || ''), 10) || frameCountFromDuration;

    return buildVideoMeta({
      path: path.resolve(input),
      basename: path.basename(input),
      width: Number(stream.width || 0),
      height: Number(stream.height || 0),
      fps,
      frame_count: frameCount,
    });
  }
}

export function parseFps(value: string | undefined): number {
  if (!value) return 0;
  const [num, den] = value.split('/').map((part) => Number.parseFloat(part));
  if (!Number.isFinite(num)) return 0;
  if (!Number.isFinite(den) || den === 0) return num;
  return num / den;
}

function isLikelyImage(filePath: string): boolean {
  return /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(filePath);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
