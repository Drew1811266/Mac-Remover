import { chmod, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { getRuntimePlatformKey, resolveRuntimeTool } from './ffmpeg.js';

describe('ffmpeg runtime resolver', () => {
  it('maps supported platforms to runtime asset keys', () => {
    expect(getRuntimePlatformKey('darwin', 'arm64')).toBe('darwin-arm64');
    expect(getRuntimePlatformKey('darwin', 'x64')).toBe('darwin-x86_64');
    expect(getRuntimePlatformKey('win32', 'x64')).toBe('win32-x86_64');
  });

  it('prefers installed userData runtime assets over bundled assets', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-runtime-'));
    try {
      const userData = path.join(dir, 'userData');
      const appRoot = path.join(dir, 'app');
      const installed = path.join(userData, 'runtime', 'ffmpeg', 'darwin-arm64', 'ffmpeg');
      const bundled = path.join(appRoot, '..', 'vendor', 'ffmpeg', 'darwin-arm64', 'ffmpeg');
      await mkdir(path.dirname(installed), { recursive: true });
      await mkdir(path.dirname(bundled), { recursive: true });
      await writeFile(installed, '');
      await writeFile(bundled, '');
      await chmod(installed, 0o755);
      await chmod(bundled, 0o755);

      const resolved = await resolveRuntimeTool({
        tool: 'ffmpeg',
        userDataDir: userData,
        appRoot,
        platform: 'darwin',
        arch: 'arm64',
      });

      expect(resolved.path).toBe(installed);
      expect(resolved.source).toBe('installed');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
