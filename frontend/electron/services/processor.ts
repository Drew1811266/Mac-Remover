import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, readdir, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { normalizeSegments } from './annotations.js';
import { resolveRuntimeTool } from './ffmpeg.js';
import { inpaintImageWithOnnx } from './inpaintOnnx.js';
import { INPAINT_MODEL_SPECS, isInpaintModelId, type InpaintModelId } from './modelCatalog.js';
import { localModelPath } from './modelManager.js';
import { runProcess } from './processRunner.js';
import type { AnnotationSegment, NormalizedAnnotationSegment, ProgressEventPayload } from './types.js';

export interface ProcessVideoPayload {
  input_path: string;
  output_path?: string;
  annotation_segments: AnnotationSegment[];
  settings?: {
    model_id?: InpaintModelId;
  };
}

export interface ProcessVideoResult {
  success: boolean;
  output_path?: string;
  requested_model_id?: string;
  effective_model_id?: string;
  model_warning?: string;
  error?: string;
}

export interface VideoProcessorOptions {
  userDataDir: string;
  appRoot: string;
  emitProgress: (payload: ProgressEventPayload) => void;
}

export interface VideoProcessor {
  processVideo(payload: ProcessVideoPayload): Promise<ProcessVideoResult>;
  stopProcessing(): { success: boolean; error?: string };
}

export function createVideoProcessor(options: VideoProcessorOptions): VideoProcessor {
  let abortController: AbortController | null = null;
  let activeChild: ChildProcessWithoutNullStreams | null = null;

  return {
    async processVideo(payload: ProcessVideoPayload): Promise<ProcessVideoResult> {
      if (shouldUseProcessorChild()) {
        const result = await processVideoInChild(payload, options, (child) => {
          activeChild = child;
        });
        activeChild = null;
        return result;
      }

      const requestedModelId = payload.settings?.model_id || 'lama_roi';
      if (!isInpaintModelId(requestedModelId)) {
        return {
          success: false,
          error: `Invalid model_id: ${String(requestedModelId || '')}. Supported values: ${Object.keys(INPAINT_MODEL_SPECS).join(', ')}`,
        };
      }
      const modelSpec = INPAINT_MODEL_SPECS[requestedModelId];
      if (!modelSpec.implemented) {
        return {
          success: false,
          requested_model_id: requestedModelId,
          effective_model_id: requestedModelId,
          error: modelSpec.blockedReason || `${modelSpec.displayName} is blocked by the non-Python equivalence gate.`,
        };
      }

      const modelPath = await localModelPath(options.userDataDir, requestedModelId);
      if (!modelPath) {
        return { success: false, error: `${modelSpec.displayName} is not installed. Download the non-Python model assets first.` };
      }

      if (!payload.input_path) {
        return { success: false, error: 'Missing input path' };
      }
      if (!Array.isArray(payload.annotation_segments) || payload.annotation_segments.length === 0) {
        return { success: false, error: 'annotation_segments is required and must be a list' };
      }
      try {
        const inputStat = await stat(payload.input_path);
        if (!inputStat.isFile()) {
          return { success: false, error: `File not found: ${payload.input_path}` };
        }
      } catch {
        return { success: false, error: `File not found: ${payload.input_path}` };
      }

      abortController = new AbortController();
      const workspace = path.join(options.userDataDir, 'tmp', `process-${randomUUID()}`);
      const framesDir = path.join(workspace, 'frames');
      const outputDir = payload.output_path || path.join(process.env.HOME || options.userDataDir, 'Downloads', 'WatermarkRemover');
      const outputPath = path.join(outputDir, `${path.parse(payload.input_path).name}_no_watermark.mp4`);

      try {
        await mkdir(framesDir, { recursive: true });
        await mkdir(outputDir, { recursive: true });

        const ffmpeg = await resolveRuntimeTool({ tool: 'ffmpeg', userDataDir: options.userDataDir, appRoot: options.appRoot });
        const ffprobe = await resolveRuntimeTool({ tool: 'ffprobe', userDataDir: options.userDataDir, appRoot: options.appRoot });
        const meta = await probeVideo(ffprobe.path, payload.input_path);
        const segments = normalizeSegments(payload.annotation_segments, {
          width: meta.width,
          height: meta.height,
          frame_count: meta.frameCount,
        }).filter((segment) => segment.enabled);

        if (segments.length === 0) {
          return { success: false, error: 'No enabled annotation segments provided' };
        }

        options.emitProgress({
          progress: 0.02,
          phase: 'extract',
          status: 'Extracting frames',
          processed_frames: 0,
          total_frames: meta.frameCount,
        });

        await runProcess(
          ffmpeg.path,
          ['-y', '-i', payload.input_path, '-vsync', '0', path.join(framesDir, 'frame_%08d.png')],
          { signal: abortController.signal, timeoutMs: 60 * 60 * 1000 },
        );

        const frames = (await readdir(framesDir)).filter((item) => item.endsWith('.png')).sort();
        const segmentsByFrame = groupSegmentsByFrame(segments, frames.length);

        for (let index = 0; index < frames.length; index += 1) {
          if (abortController.signal.aborted) {
            throw new Error('Operation cancelled');
          }
          const frameSegments = segmentsByFrame.get(index);
          if (frameSegments?.length) {
            await inpaintImageWithOnnx(modelPath, path.join(framesDir, frames[index]), frameSegments);
          }
          options.emitProgress({
            progress: 0.05 + 0.82 * ((index + 1) / Math.max(1, frames.length)),
            phase: 'infer',
            status: 'LaMa ONNX inference',
            processed_frames: index + 1,
            total_frames: frames.length,
          });
        }

        options.emitProgress({
          progress: 0.9,
          phase: 'encode',
          status: 'Encoding output',
          processed_frames: frames.length,
          total_frames: frames.length,
        });

        await runProcess(
          ffmpeg.path,
          [
            '-y',
            '-framerate',
            String(meta.fps || 24),
            '-i',
            path.join(framesDir, 'frame_%08d.png'),
            '-i',
            payload.input_path,
            '-map',
            '0:v:0',
            '-map',
            '1:a?',
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-c:a',
            'copy',
            '-shortest',
            outputPath,
          ],
          { signal: abortController.signal, timeoutMs: 60 * 60 * 1000 },
        );

        options.emitProgress({
          progress: 1,
          phase: 'finalize',
          status: 'Done',
          processed_frames: frames.length,
          total_frames: frames.length,
          eta_seconds: 0,
        });

        return {
          success: true,
          output_path: outputPath,
          requested_model_id: requestedModelId,
          effective_model_id: requestedModelId,
        };
      } catch (error) {
        return { success: false, error: error instanceof Error ? error.message : String(error) };
      } finally {
        abortController = null;
        await rm(workspace, { recursive: true, force: true });
      }
    },

    stopProcessing(): { success: boolean; error?: string } {
      abortController?.abort();
      activeChild?.kill('SIGTERM');
      activeChild = null;
      return { success: true };
    },
  };
}

async function processVideoInChild(
  payload: ProcessVideoPayload,
  options: VideoProcessorOptions,
  onChild: (child: ChildProcessWithoutNullStreams) => void,
): Promise<ProcessVideoResult> {
  const child = spawn(resolveProcessorNodeBinary(), [resolveProcessorChildScript()], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      WMR_PROCESSOR_CHILD: '1',
    },
  });
  onChild(child);

  let stdout = '';
  let stderr = '';
  let settled = false;

  const resultPromise = new Promise<ProcessVideoResult>((resolve) => {
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk: string) => {
      stdout += chunk;
      const lines = stdout.split('\n');
      stdout = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        const message = parseChildMessage(line);
        if (!message) continue;
        if (message.type === 'progress') {
          options.emitProgress(message.payload);
        } else if (message.type === 'result') {
          settled = true;
          resolve(message.payload);
        }
      }
    });

    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk: string) => {
      stderr += chunk;
    });

    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      resolve({ success: false, error: `Failed to start LaMa processor child: ${error.message}` });
    });

    child.on('exit', (code, signal) => {
      if (settled) return;
      settled = true;
      const reason = signal ? `signal ${signal}` : `exit code ${code ?? 'unknown'}`;
      const detail = stderr.trim().split('\n').slice(-6).join('\n');
      resolve({ success: false, error: `LaMa processor child stopped before completion (${reason})${detail ? `: ${detail}` : ''}` });
    });
  });

  child.stdin.end(JSON.stringify({ payload, options: { userDataDir: options.userDataDir, appRoot: options.appRoot } }));
  return resultPromise;
}

function shouldUseProcessorChild(): boolean {
  if (process.env.WMR_PROCESSOR_CHILD === '1') return false;
  return Boolean(process.versions.electron) || process.env.WMR_FORCE_PROCESSOR_CHILD === '1';
}

function resolveProcessorNodeBinary(): string {
  return process.env.WMR_NODE_BINARY || 'node';
}

function resolveProcessorChildScript(): string {
  return path.join(path.dirname(fileURLToPath(import.meta.url)), 'processorChild.js');
}

type ProcessorChildMessage =
  | { type: 'progress'; payload: ProgressEventPayload }
  | { type: 'result'; payload: ProcessVideoResult };

function parseChildMessage(line: string): ProcessorChildMessage | null {
  try {
    const parsed = JSON.parse(line) as ProcessorChildMessage;
    if (parsed?.type === 'progress' || parsed?.type === 'result') return parsed;
  } catch {
    return null;
  }
  return null;
}

async function probeVideo(ffprobePath: string, inputPath: string): Promise<{ fps: number; frameCount: number; width: number; height: number }> {
  const result = await runProcess(ffprobePath, [
    '-v',
    'error',
    '-select_streams',
    'v:0',
    '-show_entries',
    'stream=width,height,avg_frame_rate,nb_frames,duration',
    '-of',
    'json',
    inputPath,
  ]);
  const parsed = JSON.parse(result.stdout) as {
    streams?: Array<{ width?: number; height?: number; avg_frame_rate?: string; nb_frames?: string; duration?: string }>;
  };
  const stream = parsed.streams?.[0];
  if (!stream) throw new Error('Cannot read video stream');
  const fps = parseFps(stream.avg_frame_rate) || 24;
  const duration = Number.parseFloat(String(stream.duration || '0'));
  const frameCount = Number.parseInt(String(stream.nb_frames || ''), 10) || Math.max(1, Math.round(duration * fps));
  return {
    fps,
    frameCount,
    width: Number(stream.width || 0),
    height: Number(stream.height || 0),
  };
}

function parseFps(value: string | undefined): number {
  if (!value) return 0;
  const [num, den] = value.split('/').map((part) => Number.parseFloat(part));
  if (!Number.isFinite(num)) return 0;
  if (!Number.isFinite(den) || den === 0) return num;
  return num / den;
}

function groupSegmentsByFrame(
  segments: NormalizedAnnotationSegment[],
  frameCount: number,
): Map<number, NormalizedAnnotationSegment[]> {
  const result = new Map<number, NormalizedAnnotationSegment[]>();
  for (const segment of segments) {
    const start = Math.max(0, Math.min(frameCount - 1, segment.start_frame));
    const end = Math.max(0, Math.min(frameCount - 1, segment.end_frame));
    for (let frame = start; frame <= end; frame += 1) {
      const list = result.get(frame) || [];
      list.push(segment);
      result.set(frame, list);
    }
  }
  return result;
}
