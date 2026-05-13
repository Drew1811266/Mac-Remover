import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

describe('Electron package entrypoints', () => {
  it('uses the compiled Electron main entry in package metadata and dev script', async () => {
    const packageJson = JSON.parse(await readFile(path.resolve('package.json'), 'utf8')) as {
      main?: string;
      scripts?: Record<string, string>;
    };
    const runnerScript = await readFile(path.resolve('scripts/run-electron-dev.mjs'), 'utf8');

    expect(packageJson.main).toBe('dist-electron/electron/main.js');
    expect(packageJson.scripts?.['electron:dev']).toContain('node scripts/run-electron-dev.mjs');
    expect(runnerScript).toContain("'dist-electron', 'electron', 'main.js'");
  });
});
