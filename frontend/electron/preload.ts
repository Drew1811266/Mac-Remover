import { contextBridge, ipcRenderer } from 'electron';

import { isAllowedIpcChannel, type IpcChannel } from './services/ipcRegistry.js';

function invoke<T>(channel: IpcChannel, payload?: unknown): Promise<T> {
  if (!isAllowedIpcChannel(channel)) {
    return Promise.reject(new Error(`Blocked IPC channel: ${channel}`));
  }
  return ipcRenderer.invoke(channel, payload) as Promise<T>;
}

contextBridge.exposeInMainWorld('wmr', {
  selectFile: () => invoke('wmr:selectFile'),
  select_file: () => invoke('wmr:selectFile'),
  selectFolder: () => invoke('wmr:selectFolder'),
  select_folder: () => invoke('wmr:selectFolder'),
  beginSelectFile: () => invoke('wmr:beginSelectFile'),
  begin_select_file: () => invoke('wmr:beginSelectFile'),
  beginSelectFolder: () => invoke('wmr:beginSelectFolder'),
  begin_select_folder: () => invoke('wmr:beginSelectFolder'),
  pollDialogResult: (payload: unknown) => invoke('wmr:pollDialogResult', payload),
  poll_dialog_result: (payload: unknown) => invoke('wmr:pollDialogResult', payload),
  clearDialogResult: (payload: unknown) => invoke('wmr:clearDialogResult', payload),
  clear_dialog_result: (payload: unknown) => invoke('wmr:clearDialogResult', payload),
  getMediaInfo: (payload: unknown) => invoke('wmr:getMediaInfo', payload),
  get_media_info: (payload: unknown) => invoke('wmr:getMediaInfo', payload),
  openVideoPreviewSession: (payload: unknown) => invoke('wmr:openVideoPreviewSession', payload),
  open_video_preview_session: (payload: unknown) => invoke('wmr:openVideoPreviewSession', payload),
  readVideoPreviewFrame: (payload: unknown) => invoke('wmr:readVideoPreviewFrame', payload),
  read_video_preview_frame: (payload: unknown) => invoke('wmr:readVideoPreviewFrame', payload),
  closeVideoPreviewSession: (payload: unknown) => invoke('wmr:closeVideoPreviewSession', payload),
  close_video_preview_session: (payload: unknown) => invoke('wmr:closeVideoPreviewSession', payload),
  prepare_video_preview: (payload: unknown) => invoke('wmr:prepareVideoPreview', payload),
  loadAnnotations: (payload: unknown) => invoke('wmr:loadAnnotations', payload),
  load_annotations: (payload: unknown) => invoke('wmr:loadAnnotations', payload),
  saveAnnotations: (payload: unknown) => invoke('wmr:saveAnnotations', payload),
  save_annotations: (payload: unknown) => invoke('wmr:saveAnnotations', payload),
  deleteAnnotations: (payload: unknown) => invoke('wmr:deleteAnnotations', payload),
  delete_annotations: (payload: unknown) => invoke('wmr:deleteAnnotations', payload),
  getSettings: () => invoke('wmr:getSettings'),
  get_settings: () => invoke('wmr:getSettings'),
  saveSettings: (payload: unknown) => invoke('wmr:saveSettings', payload),
  save_settings: (payload: unknown) => invoke('wmr:saveSettings', payload),
  get_device_info: () => invoke('wmr:getDeviceInfo'),
  getModelDownloadStatus: () => invoke('wmr:getModelDownloadStatus'),
  get_model_download_status: () => invoke('wmr:getModelDownloadStatus'),
  startModelDownload: (payload: unknown) => invoke('wmr:startModelDownload', payload),
  start_model_download: (payload: unknown) => invoke('wmr:startModelDownload', payload),
  cancelModelDownload: () => invoke('wmr:cancelModelDownload'),
  cancel_model_download: () => invoke('wmr:cancelModelDownload'),
  get_upscale_model_download_status: () => invoke('wmr:getUpscaleModelDownloadStatus'),
  start_upscale_model_download: (payload: unknown) => invoke('wmr:startUpscaleModelDownload', payload),
  cancel_upscale_model_download: () => invoke('wmr:cancelUpscaleModelDownload'),
  get_upscale_capabilities: (payload?: unknown) => invoke('wmr:getUpscaleCapabilities', payload),
  start_upscale: (payload: unknown) => invoke('wmr:startUpscale', payload),
  get_upscale_task_status: () => invoke('wmr:getUpscaleTaskStatus'),
  cancel_upscale_task: () => invoke('wmr:cancelUpscaleTask'),
  processVideo: (payload: unknown) => invoke('wmr:processVideo', payload),
  process_video: (payload: unknown) => invoke('wmr:processVideo', payload),
  stopProcessing: () => invoke('wmr:stopProcessing'),
  stop_processing: () => invoke('wmr:stopProcessing'),
  openOutputDir: () => invoke('wmr:openOutputDir'),
  open_output_dir: () => invoke('wmr:openOutputDir'),
});

ipcRenderer.on('wmr-progress', (_event, payload) => {
  window.dispatchEvent(new CustomEvent('wmr-progress', { detail: payload }));
});
