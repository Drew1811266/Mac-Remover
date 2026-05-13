import { randomUUID } from 'node:crypto';
import { dialog, ipcMain, shell, type BrowserWindow } from 'electron';
import path from 'node:path';

import { deleteAnnotations, loadAnnotations, saveAnnotations } from './services/annotations.js';
import { IPC_CHANNELS, type IpcChannel } from './services/ipcRegistry.js';
import { MediaService } from './services/media.js';
import { ModelManager } from './services/modelManager.js';
import { PreviewService } from './services/preview.js';
import { createVideoProcessor } from './services/processor.js';
import { loadSettings, saveSettings } from './services/settings.js';
import { UpscaleService } from './services/upscale.js';
import type { AnnotationSegment, AppSettings, ProgressEventPayload } from './services/types.js';

export interface RegisterIpcOptions {
  window: BrowserWindow;
  userDataDir: string;
  appRoot: string;
  manifestUrl?: string;
}

interface DialogResult {
  success: boolean;
  request_id?: string;
  path?: string;
  done?: boolean;
  cancelled?: boolean;
  error?: string;
}

const dialogResults = new Map<string, DialogResult>();

export function registerIpcHandlers(options: RegisterIpcOptions): void {
  const media = new MediaService({ userDataDir: options.userDataDir, appRoot: options.appRoot });
  const preview = new PreviewService({ userDataDir: options.userDataDir, appRoot: options.appRoot });
  const models = new ModelManager({ userDataDir: options.userDataDir, manifestUrl: options.manifestUrl });
  const upscale = new UpscaleService({
    userDataDir: options.userDataDir,
    appRoot: options.appRoot,
    manifestUrl: options.manifestUrl,
  });
  const processor = createVideoProcessor({
    userDataDir: options.userDataDir,
    appRoot: options.appRoot,
    emitProgress: (payload) => emitProgress(options.window, payload),
  });

  clearExistingHandlers();

  ipcMain.handle('wmr:selectFile', async () => selectFile(options.window));
  ipcMain.handle('wmr:selectFolder', async () => selectFolder(options.window));
  ipcMain.handle('wmr:getMediaInfo', async (_event, payload) => media.getMediaInfo(String(payload?.path || payload || '')));
  ipcMain.handle('wmr:openVideoPreviewSession', async (_event, payload) =>
    preview.openVideoPreviewSession(String(payload?.path || ''), Number(payload?.target_fps || 15), Number(payload?.max_width || 1280)),
  );
  ipcMain.handle('wmr:readVideoPreviewFrame', async (_event, payload) =>
    preview.readVideoPreviewFrame(String(payload?.session_id || ''), optionalNumber(payload?.frame_index)),
  );
  ipcMain.handle('wmr:closeVideoPreviewSession', async (_event, payload) =>
    preview.closeVideoPreviewSession(String(payload?.session_id || payload || '')),
  );
  ipcMain.handle('wmr:prepareVideoPreview', async (_event, payload) => preview.prepareVideoPreview(String(payload?.path || payload || '')));
  ipcMain.handle('wmr:loadAnnotations', async (_event, payload) => {
    const videoPath = String(payload?.video_path || '');
    const meta = await media.getVideoMeta(videoPath);
    return loadAnnotations(videoPath, meta);
  });
  ipcMain.handle('wmr:saveAnnotations', async (_event, payload) => {
    const videoPath = String(payload?.video_path || '');
    const meta = await media.getVideoMeta(videoPath);
    return saveAnnotations({
      videoPath,
      videoMeta: meta,
      segments: (payload?.segments || []) as AnnotationSegment[],
    });
  });
  ipcMain.handle('wmr:deleteAnnotations', async (_event, payload) => deleteAnnotations(String(payload?.video_path || payload || '')));
  ipcMain.handle('wmr:getSettings', async () => loadSettings(options.userDataDir));
  ipcMain.handle('wmr:saveSettings', async (_event, payload: Partial<AppSettings>) => saveSettings(options.userDataDir, payload));
  ipcMain.handle('wmr:getDeviceInfo', async () => ({
    device: process.platform === 'darwin' ? 'Apple/CPU' : 'CPU',
    memory: '',
    supports_fp16: false,
  }));
  ipcMain.handle('wmr:getModelDownloadStatus', async () => models.getStatus());
  ipcMain.handle('wmr:startModelDownload', async (_event, payload) =>
    models.startDownload(payload?.model_id || 'lama_roi', Boolean(payload?.force)),
  );
  ipcMain.handle('wmr:cancelModelDownload', async () => models.cancelDownload());
  ipcMain.handle('wmr:getUpscaleModelDownloadStatus', async () => upscale.getModelDownloadStatus());
  ipcMain.handle('wmr:startUpscaleModelDownload', async (_event, payload) => upscale.startModelDownload(payload));
  ipcMain.handle('wmr:cancelUpscaleModelDownload', async () => upscale.cancelModelDownload());
  ipcMain.handle('wmr:getUpscaleCapabilities', async () => upscale.getCapabilities());
  ipcMain.handle('wmr:startUpscale', async (_event, payload) => upscale.startUpscale(payload));
  ipcMain.handle('wmr:getUpscaleTaskStatus', async () => upscale.getTaskStatus());
  ipcMain.handle('wmr:cancelUpscaleTask', async () => upscale.cancelTask());
  ipcMain.handle('wmr:processVideo', async (_event, payload) => processor.processVideo(payload));
  ipcMain.handle('wmr:stopProcessing', async () => processor.stopProcessing());
  ipcMain.handle('wmr:openOutputDir', async () => {
    const settings = await loadSettings(options.userDataDir);
    await shell.openPath(settings.output.path);
    return { success: true };
  });

  ipcMain.handle('wmr:beginSelectFile', async () => beginDialogRequest(options.window, false));
  ipcMain.handle('wmr:beginSelectFolder', async () => beginDialogRequest(options.window, true));
  ipcMain.handle('wmr:pollDialogResult', async (_event, payload) => pollDialogResult(String(payload?.request_id || '')));
  ipcMain.handle('wmr:clearDialogResult', async (_event, payload) => {
    dialogResults.delete(String(payload?.request_id || ''));
    return { success: true };
  });
}

function clearExistingHandlers(): void {
  for (const channel of IPC_CHANNELS) {
    ipcMain.removeHandler(channel);
  }
}

async function selectFile(window: BrowserWindow): Promise<{ path: string } | null> {
  const result = await dialog.showOpenDialog(window, {
    properties: ['openFile'],
    filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'm4v', 'avi', 'mkv', 'webm'] }],
  });
  if (result.canceled || !result.filePaths[0]) return null;
  return { path: result.filePaths[0] };
}

async function selectFolder(window: BrowserWindow): Promise<{ path: string } | null> {
  const result = await dialog.showOpenDialog(window, { properties: ['openDirectory', 'createDirectory'] });
  if (result.canceled || !result.filePaths[0]) return null;
  return { path: result.filePaths[0] };
}

async function beginDialogRequest(window: BrowserWindow, folder: boolean): Promise<DialogResult> {
  const requestId = randomUUID();
  void (folder ? selectFolder(window) : selectFile(window))
    .then((result) => {
      dialogResults.set(requestId, {
        success: true,
        done: true,
        cancelled: !result,
        path: result?.path,
      });
    })
    .catch((error) => {
      dialogResults.set(requestId, {
        success: false,
        done: true,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  dialogResults.set(requestId, { success: true, done: false, request_id: requestId });
  return { success: true, request_id: requestId };
}

function pollDialogResult(requestId: string): DialogResult {
  return dialogResults.get(requestId) || { success: true, done: false };
}

function emitProgress(window: BrowserWindow, payload: ProgressEventPayload): void {
  if (window.isDestroyed()) return;
  window.webContents.send('wmr-progress', payload);
}

function optionalNumber(value: unknown): number | undefined {
  if (typeof value === 'undefined' || value === null) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function rendererHtmlPath(appRoot: string, isPackaged = false): string {
  if (process.env.WMR_RENDERER_URL) {
    return process.env.WMR_RENDERER_URL;
  }
  if (isPackaged) {
    return path.join(appRoot, 'renderer', 'index.html');
  }
  return path.resolve(appRoot, 'src', 'gui', 'templates', 'dist', 'index.html');
}
