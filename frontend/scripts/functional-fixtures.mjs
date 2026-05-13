#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { access, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
export const FRONTEND_ROOT = path.resolve(SCRIPT_DIR, '..');
export const PROJECT_ROOT = path.resolve(FRONTEND_ROOT, '..');
export const DEFAULT_FUNCTIONAL_ROOT = path.join(FRONTEND_ROOT, '.functional-test');

const WATERMARK_FILTER_720 =
  'drawbox=x=iw-260:y=36:w=220:h=72:color=white@0.88:t=fill,drawbox=x=iw-248:y=48:w=196:h=48:color=black@0.58:t=fill';
const WATERMARK_FILTER_WEBM =
  'drawbox=x=iw-150:y=20:w=126:h=42:color=white@0.88:t=fill,drawbox=x=iw-142:y=28:w=110:h=26:color=black@0.58:t=fill';

export async function ensureFunctionalFixtures(options = {}) {
  const root = path.resolve(options.root || DEFAULT_FUNCTIONAL_ROOT);
  const fixturesDir = path.join(root, 'fixtures');
  const manifestPath = path.join(fixturesDir, 'fixtures.json');

  if (options.force) {
    await rm(fixturesDir, { recursive: true, force: true });
  }
  await mkdir(fixturesDir, { recursive: true });

  const fixtures = {
    h264Aac: {
      name: '5s_720p_h264_aac.mp4',
      path: path.join(fixturesDir, '5s_720p_h264_aac.mp4'),
      hasAudio: true,
      expected: { width: 1280, height: 720, fps: 24, durationSeconds: 5 },
    },
    h264NoAudio: {
      name: '5s_720p_no_audio.mp4',
      path: path.join(fixturesDir, '5s_720p_no_audio.mp4'),
      hasAudio: false,
      expected: { width: 1280, height: 720, fps: 24, durationSeconds: 5 },
    },
    webm: {
      name: '3s_webm.webm',
      path: path.join(fixturesDir, '3s_webm.webm'),
      hasAudio: false,
      expected: { width: 640, height: 360, fps: 15, durationSeconds: 3 },
    },
    invalid: {
      name: 'invalid.txt',
      path: path.join(fixturesDir, 'invalid.txt'),
      hasAudio: false,
    },
  };

  const ffmpeg = resolveBundledTool('ffmpeg');
  if (!(await exists(fixtures.h264Aac.path))) {
    await generateMp4(ffmpeg, fixtures.h264Aac.path, true);
  }
  if (!(await exists(fixtures.h264NoAudio.path))) {
    await generateMp4(ffmpeg, fixtures.h264NoAudio.path, false);
  }
  if (!(await exists(fixtures.webm.path))) {
    await generateWebm(ffmpeg, fixtures.webm.path);
  }
  if (!(await exists(fixtures.invalid.path))) {
    await writeFile(fixtures.invalid.path, 'This is not a video file.\n', 'utf8');
  }

  const manifest = {
    generated_at: new Date().toISOString(),
    root,
    fixtures,
    segments: {
      single: [segment('single-top-right', 0, 95, { x: 1010, y: 28, width: 250, height: 92 })],
      multi: [
        segment('multi-top-right-a', 0, 45, { x: 1010, y: 28, width: 250, height: 92 }),
        segment('multi-top-right-b', 46, 95, { x: 1000, y: 28, width: 260, height: 96 }, 8, 4),
        segment('multi-low-center', 24, 72, { x: 520, y: 560, width: 220, height: 70 }, 6, 3),
      ],
      edge: [segment('edge-bottom-right', 0, 18, { x: 1268, y: 702, width: 96, height: 64 }, 10, 5)],
    },
  };

  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  return { ...manifest, manifestPath };
}

export async function readFunctionalFixtureManifest(root = DEFAULT_FUNCTIONAL_ROOT) {
  const manifestPath = path.join(path.resolve(root), 'fixtures', 'fixtures.json');
  return JSON.parse(await readFile(manifestPath, 'utf8'));
}

export function resolveBundledTool(tool) {
  const platformKey = runtimePlatformKey();
  const executable = process.platform === 'win32' ? `${tool}.exe` : tool;
  return path.join(PROJECT_ROOT, 'vendor', 'ffmpeg', platformKey, executable);
}

function runtimePlatformKey() {
  if (process.platform === 'darwin' && process.arch === 'arm64') return 'darwin-arm64';
  if (process.platform === 'darwin' && process.arch === 'x64') return 'darwin-x86_64';
  if (process.platform === 'win32' && process.arch === 'x64') return 'win32-x86_64';
  if (process.platform === 'win32' && process.arch === 'arm64') return 'win32-arm64';
  if (process.platform === 'linux' && process.arch === 'x64') return 'linux-x86_64';
  return `${process.platform}-${process.arch}`;
}

async function generateMp4(ffmpeg, outputPath, withAudio) {
  const args = [
    '-y',
    '-hide_banner',
    '-loglevel',
    'error',
    '-f',
    'lavfi',
    '-i',
    'testsrc2=size=1280x720:rate=24:duration=5',
  ];
  if (withAudio) {
    args.push('-f', 'lavfi', '-i', 'sine=frequency=880:sample_rate=44100:duration=5');
  }
  args.push(
    '-vf',
    WATERMARK_FILTER_720,
    '-c:v',
    'libx264',
    '-pix_fmt',
    'yuv420p',
    '-movflags',
    '+faststart',
  );
  if (withAudio) {
    args.push('-c:a', 'aac', '-shortest');
  } else {
    args.push('-an');
  }
  args.push(outputPath);
  await run(ffmpeg, args);
}

async function generateWebm(ffmpeg, outputPath) {
  const baseArgs = [
    '-y',
    '-hide_banner',
    '-loglevel',
    'error',
    '-f',
    'lavfi',
    '-i',
    'testsrc2=size=640x360:rate=15:duration=3',
    '-vf',
    WATERMARK_FILTER_WEBM,
    '-an',
  ];
  try {
    await run(ffmpeg, [...baseArgs, '-c:v', 'libvpx-vp9', '-b:v', '600k', outputPath]);
  } catch {
    await run(ffmpeg, [...baseArgs, '-c:v', 'libvpx', '-b:v', '600k', outputPath]);
  }
}

function segment(id, startFrame, endFrame, rect, expandPx = 6, featherPx = 3) {
  const now = '2026-01-01T00:00:00.000Z';
  return {
    id,
    start_frame: startFrame,
    end_frame: endFrame,
    rect,
    expand_px: expandPx,
    feather_px: featherPx,
    enabled: true,
    created_at: now,
    updated_at: now,
  };
}

async function exists(filePath) {
  try {
    const fileStat = await stat(filePath);
    return fileStat.isFile() && fileStat.size > 0;
  } catch {
    return false;
  }
}

async function run(command, args) {
  try {
    await access(command);
  } catch {
    throw new Error(`Required runtime tool is missing: ${command}`);
  }

  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    const stderr = [];
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`${command} exited with ${code}: ${Buffer.concat(stderr).toString('utf8').slice(-1200)}`));
    });
  });
}
