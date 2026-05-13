import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { rendererHtmlPath } from './ipcHandlers.js';

describe('rendererHtmlPath', () => {
  it('resolves the built renderer under the project root in dev mode', () => {
    const appRoot = path.join('/tmp', 'Mac Remover');
    expect(rendererHtmlPath(appRoot, false)).toBe(
      path.join('/tmp', 'Mac Remover', 'src', 'gui', 'templates', 'dist', 'index.html'),
    );
  });

  it('resolves packaged extraResources under resourcesPath', () => {
    const resourcesPath = path.join('/tmp', 'Mac Watermark Remover.app', 'Contents', 'Resources');
    expect(rendererHtmlPath(resourcesPath, true)).toBe(path.join(resourcesPath, 'renderer', 'index.html'));
  });
});
