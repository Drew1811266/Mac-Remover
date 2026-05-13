import { access } from 'node:fs/promises';
import path from 'node:path';

export type RuntimeTool = 'ffmpeg' | 'ffprobe';

export interface RuntimeToolResolution {
  path: string;
  source: 'installed' | 'bundled' | 'path';
}

export interface RuntimeToolOptions {
  tool: RuntimeTool;
  userDataDir: string;
  appRoot: string;
  platform?: NodeJS.Platform;
  arch?: NodeJS.Architecture;
}

export function getRuntimePlatformKey(
  platform: NodeJS.Platform = process.platform,
  arch: NodeJS.Architecture = process.arch,
): string {
  if (platform === 'darwin' && arch === 'arm64') return 'darwin-arm64';
  if (platform === 'darwin' && arch === 'x64') return 'darwin-x86_64';
  if (platform === 'win32' && arch === 'x64') return 'win32-x86_64';
  if (platform === 'win32' && arch === 'arm64') return 'win32-arm64';
  if (platform === 'linux' && arch === 'x64') return 'linux-x86_64';
  return `${platform}-${arch}`;
}

export async function resolveRuntimeTool(options: RuntimeToolOptions): Promise<RuntimeToolResolution> {
  const platformKey = getRuntimePlatformKey(options.platform, options.arch);
  const executable = options.platform === 'win32' ? `${options.tool}.exe` : options.tool;
  const candidates: RuntimeToolResolution[] = [
    {
      path: path.join(options.userDataDir, 'runtime', 'ffmpeg', platformKey, executable),
      source: 'installed',
    },
    {
      path: path.resolve(options.appRoot, '..', 'vendor', 'ffmpeg', platformKey, executable),
      source: 'bundled',
    },
    {
      path: path.resolve(options.appRoot, 'vendor', 'ffmpeg', platformKey, executable),
      source: 'bundled',
    },
  ];

  for (const candidate of candidates) {
    if (await exists(candidate.path)) {
      return candidate;
    }
  }

  return { path: executable, source: 'path' };
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}
