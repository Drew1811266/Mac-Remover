import { describe, expect, it } from 'vitest';

import { IPC_CHANNELS, isAllowedIpcChannel } from './ipcRegistry.js';

describe('ipc registry', () => {
  it('exposes only the public renderer API channels', () => {
    expect(IPC_CHANNELS).toEqual([
      'wmr:selectFile',
      'wmr:selectFolder',
      'wmr:beginSelectFile',
      'wmr:beginSelectFolder',
      'wmr:pollDialogResult',
      'wmr:clearDialogResult',
      'wmr:getMediaInfo',
      'wmr:openVideoPreviewSession',
      'wmr:readVideoPreviewFrame',
      'wmr:closeVideoPreviewSession',
      'wmr:prepareVideoPreview',
      'wmr:loadAnnotations',
      'wmr:saveAnnotations',
      'wmr:deleteAnnotations',
      'wmr:getSettings',
      'wmr:saveSettings',
      'wmr:getDeviceInfo',
      'wmr:getModelDownloadStatus',
      'wmr:startModelDownload',
      'wmr:cancelModelDownload',
      'wmr:getUpscaleModelDownloadStatus',
      'wmr:startUpscaleModelDownload',
      'wmr:cancelUpscaleModelDownload',
      'wmr:getUpscaleCapabilities',
      'wmr:startUpscale',
      'wmr:getUpscaleTaskStatus',
      'wmr:cancelUpscaleTask',
      'wmr:processVideo',
      'wmr:stopProcessing',
      'wmr:openOutputDir',
    ]);
  });

  it('rejects unknown channels', () => {
    expect(isAllowedIpcChannel('wmr:getSettings')).toBe(true);
    expect(isAllowedIpcChannel('electron:executeJavaScript')).toBe(false);
    expect(isAllowedIpcChannel('__proto__')).toBe(false);
  });
});
