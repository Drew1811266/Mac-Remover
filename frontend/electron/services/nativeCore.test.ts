import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { getNativeCoreStatus, nativeCorePath } from './nativeCore.js';

describe('native core gate', () => {
  it('resolves the platform prebuild path and reports missing prebuilds', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'wmr-native-'));
    try {
      const modulePath = nativeCorePath(dir, 'darwin', 'arm64');
      expect(modulePath).toContain(path.join('native', 'prebuilds', 'darwin-arm64', 'wmr_native.node'));

      const status = await getNativeCoreStatus(dir);
      expect(status.available).toBe(false);
      expect(status.opencvAlgorithms).toBe(false);
      expect(status.reason).toContain('Prebuilt C++ Node-API core is missing');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
