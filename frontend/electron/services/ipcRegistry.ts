export const IPC_CHANNELS = [
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
] as const;

export type IpcChannel = (typeof IPC_CHANNELS)[number];

const IPC_CHANNEL_SET = new Set<string>(IPC_CHANNELS);

export function isAllowedIpcChannel(channel: string): channel is IpcChannel {
  return IPC_CHANNEL_SET.has(channel);
}
