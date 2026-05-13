// Electron 桌面桥接客户端。
import type { AnnotationSegment, VideoMeta } from '../types/annotation';
import type {
  AppSettings,
  UpscaleCapabilities,
  UpscaleConfig,
  UpscaleEngine,
  UpscaleModelId,
  UpscaleSameResStrength,
  UpscaleTargetPreset,
  UpscaleTaskState,
} from '../types/app';

declare global {
  interface Window {
    wmr?: Record<string, (...args: unknown[]) => Promise<unknown>>;
  }
}

const DESKTOP_READY_TIMEOUT_MS = 10000;

let apiReadyPromise: Promise<Record<string, (...args: unknown[]) => Promise<unknown>>> | null = null;

function hasApiMethods(api: unknown): api is Record<string, (...args: unknown[]) => Promise<unknown>> {
  // 这里只判断“对象存在”，具体方法名在 callApi 里逐项检查。
  return !!api && typeof api === 'object';
}

async function waitForApi(): Promise<Record<string, (...args: unknown[]) => Promise<unknown>>> {
  // 已就绪时直接复用，避免重复轮询。
  if (hasApiMethods(window.wmr)) {
    return window.wmr;
  }

  if (!apiReadyPromise) {
    // 首次调用时创建单例 Promise，所有调用方共享同一等待流程。
    apiReadyPromise = new Promise((resolve, reject) => {
      const started = Date.now();
      const interval = window.setInterval(() => {
        if (hasApiMethods(window.wmr)) {
          window.clearInterval(interval);
          resolve(window.wmr);
          return;
        }
        if (Date.now() - started >= DESKTOP_READY_TIMEOUT_MS) {
          window.clearInterval(interval);
          reject(new Error('Desktop API unavailable (timeout waiting for window.wmr)'));
        }
      }, 60);
    });
  }

  return apiReadyPromise;
}

async function callApi<T>(method: string, payload?: unknown): Promise<T> {
  // 通用 RPC 调用入口：统一拿 api、校验方法、传参。
  const api = await waitForApi();
  const methodRef = api[method];
  if (typeof methodRef !== 'function') {
    throw new Error(`Desktop API method not found: ${method}`);
  }
  if (typeof payload === 'undefined') {
    return methodRef() as Promise<T>;
  }
  return methodRef(payload) as Promise<T>;
}

// 标注保存请求结构。
export interface SaveAnnotationsPayload {
  video_path: string;
  segments: AnnotationSegment[];
  video_meta?: Partial<VideoMeta>;
}

// 媒体基础信息响应结构。
export interface MediaInfoResult {
  success: boolean;
  type?: 'video' | 'image';
  fps?: number;
  frame_count?: number;
  duration?: number;
  width?: number;
  height?: number;
  error?: string;
}

// 预览会话创建结果。
export interface PreviewSessionResult {
  success: boolean;
  session_id?: string;
  preview_fps?: number;
  total_preview_frames?: number;
  width?: number;
  height?: number;
  error?: string;
}

// 预览帧读取结果（base64 图片）。
export interface PreviewFrameResult {
  success: boolean;
  frame_index?: number;
  frame_url?: string;
  decode_ms?: number;
  error?: string;
}

// 预览视频准备结果（可能转码，也可能回退原文件）。
export interface PrepareVideoPreviewResult {
  success: boolean;
  path?: string;
  transcoded?: boolean;
  cached?: boolean;
  warning?: string;
  error?: string;
}

// sidecar 标注加载结果。
export interface LoadAnnotationsResult {
  success: boolean;
  exists?: boolean;
  warning?: string;
  error?: string;
  sidecar_path?: string;
  video_meta?: VideoMeta;
  segments?: AnnotationSegment[];
}

// 后端设置数据结构（映射到前端 AppSettings 前的原始格式）。
export interface BackendSettingsResult {
  language?: string;
  theme?: string;
  output?: {
    path?: string;
    model_id?: AppSettings['modelId'];
  };
}

// 后端设备信息结构。
export interface DeviceInfoResult {
  device: string;
  memory: string;
  supports_fp16: boolean;
}

// 视频处理请求体。
export interface ProcessVideoPayload {
  input_path: string;
  output_path?: string;
  annotation_segments: AnnotationSegment[];
  settings?: {
    model_id?: AppSettings['modelId'];
  };
}

// 视频处理响应体。
export interface ProcessVideoResult {
  success: boolean;
  output_path?: string;
  requested_model_id?: string;
  effective_model_id?: string;
  model_warning?: string;
  error?: string;
}

// AI 放大任务请求体。
export interface StartUpscalePayload {
  input_path: string;
  output_dir?: string;
  mode: UpscaleConfig['mode'];
  engine: UpscaleEngine;
  model_id: UpscaleModelId;
  target_preset?: UpscaleTargetPreset;
  same_res_strength?: UpscaleSameResStrength;
  denoise_strength?: number;
  keep_audio?: boolean;
}

// AI 放大任务状态响应。
export interface UpscaleTaskStatusResult {
  success: boolean;
  task?: {
    state?: UpscaleTaskState['state'];
    progress?: number;
    phase?: UpscaleTaskState['phase'];
    message?: string;
    eta_seconds?: number;
    input_path?: string;
    output_path?: string;
    preview_path?: string;
    mode?: UpscaleTaskState['mode'];
    engine?: UpscaleTaskState['engine'];
    effective_engine?: UpscaleTaskState['effectiveEngine'];
    model_id?: UpscaleTaskState['modelId'];
    error?: string;
    warning?: string;
    segment_index?: number;
    segment_total?: number;
    scene_split_mode?: UpscaleTaskState['sceneSplitMode'];
  };
  error?: string;
}

// 异步对话框：开始请求响应。
export interface DialogBeginResult {
  success: boolean;
  request_id?: string;
  error?: string;
}

// 异步对话框：轮询结果响应。
export interface DialogPollResult {
  success: boolean;
  done: boolean;
  path?: string;
  cancelled?: boolean;
  error?: string;
}

// 模型下载任务状态枚举。
export type ModelDownloadTaskState = 'idle' | 'running' | 'success' | 'failed' | 'cancelled';

// 模型下载列表项（安装状态 + 展示文案）。
export interface ModelDownloadEntry {
  model_id: AppSettings['modelId'];
  display_name: string;
  installed: boolean;
  can_redownload: boolean;
  install_hint: string;
}

// 单个下载任务实时状态。
export interface ModelDownloadTask {
  state: ModelDownloadTaskState;
  model_id: AppSettings['modelId'] | '';
  progress: number;
  downloaded_bytes: number;
  total_bytes: number;
  speed_bps: number;
  current_file: string;
  message: string;
  error: string;
}

// 下载状态总响应：模型列表 + 当前任务。
export interface ModelDownloadStatusResult {
  success: boolean;
  models: ModelDownloadEntry[];
  task: ModelDownloadTask;
  error?: string;
}

// AI 放大模型下载列表项。
export interface UpscaleModelDownloadEntry {
  model_id: UpscaleModelId;
  display_name: string;
  installed: boolean;
  can_redownload: boolean;
  install_hint: string;
}

// AI 放大模型下载任务状态。
export interface UpscaleModelDownloadTask {
  state: ModelDownloadTaskState;
  model_id: UpscaleModelId | '';
  progress: number;
  downloaded_bytes: number;
  total_bytes: number;
  speed_bps: number;
  current_file: string;
  message: string;
  error: string;
}

// AI 放大模型下载状态总响应。
export interface UpscaleModelDownloadStatusResult {
  success: boolean;
  models: UpscaleModelDownloadEntry[];
  task: UpscaleModelDownloadTask;
  error?: string;
}

// 对外暴露的桥接客户端。每个方法对应一个后端 API 方法名。
export const desktopClient = {
  async selectFile(): Promise<{ path: string } | null> {
    return callApi<{ path: string } | null>('select_file');
  },

  async selectFolder(): Promise<{ path: string } | null> {
    return callApi<{ path: string } | null>('select_folder');
  },

  async beginSelectFile(): Promise<DialogBeginResult> {
    return callApi<DialogBeginResult>('begin_select_file');
  },

  async beginSelectFolder(): Promise<DialogBeginResult> {
    return callApi<DialogBeginResult>('begin_select_folder');
  },

  async pollDialogResult(requestId: string): Promise<DialogPollResult> {
    return callApi<DialogPollResult>('poll_dialog_result', { request_id: requestId });
  },

  async clearDialogResult(requestId: string): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('clear_dialog_result', { request_id: requestId });
  },

  async getMediaInfo(path: string): Promise<MediaInfoResult> {
    return callApi<MediaInfoResult>('get_media_info', { path });
  },

  async openVideoPreviewSession(path: string, targetFps = 15, maxWidth = 1280): Promise<PreviewSessionResult> {
    return callApi<PreviewSessionResult>('open_video_preview_session', {
      path,
      target_fps: targetFps,
      max_width: maxWidth,
    });
  },

  async readVideoPreviewFrame(sessionId: string, frameIndex?: number): Promise<PreviewFrameResult> {
    return callApi<PreviewFrameResult>('read_video_preview_frame', {
      session_id: sessionId,
      frame_index: frameIndex,
    });
  },

  async closeVideoPreviewSession(sessionId: string): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('close_video_preview_session', {
      session_id: sessionId,
    });
  },

  async prepareVideoPreview(path: string): Promise<PrepareVideoPreviewResult> {
    return callApi<PrepareVideoPreviewResult>('prepare_video_preview', { path });
  },

  async loadAnnotations(videoPath: string): Promise<LoadAnnotationsResult> {
    return callApi<LoadAnnotationsResult>('load_annotations', { video_path: videoPath });
  },

  async saveAnnotations(payload: SaveAnnotationsPayload): Promise<LoadAnnotationsResult> {
    return callApi<LoadAnnotationsResult>('save_annotations', payload);
  },

  async deleteAnnotations(videoPath: string): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('delete_annotations', { video_path: videoPath });
  },

  async getSettings(): Promise<BackendSettingsResult> {
    return callApi<BackendSettingsResult>('get_settings');
  },

  async saveSettings(settings: {
    language: AppSettings['language'];
    theme: AppSettings['theme'];
    output: { path: string; model_id: AppSettings['modelId'] };
  }): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('save_settings', settings);
  },

  async getDeviceInfo(): Promise<DeviceInfoResult> {
    return callApi<DeviceInfoResult>('get_device_info');
  },

  async processVideo(payload: ProcessVideoPayload): Promise<ProcessVideoResult> {
    return callApi<ProcessVideoResult>('process_video', payload);
  },

  async stopProcessing(): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('stop_processing');
  },

  async openOutputDir(): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('open_output_dir');
  },

  async getModelDownloadStatus(): Promise<ModelDownloadStatusResult> {
    return callApi<ModelDownloadStatusResult>('get_model_download_status');
  },

  async startModelDownload(
    modelId: AppSettings['modelId'],
    force = false,
  ): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('start_model_download', {
      model_id: modelId,
      force,
    });
  },

  async cancelModelDownload(): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('cancel_model_download');
  },

  async getUpscaleModelDownloadStatus(): Promise<UpscaleModelDownloadStatusResult> {
    return callApi<UpscaleModelDownloadStatusResult>('get_upscale_model_download_status');
  },

  async startUpscaleModelDownload(
    modelId: UpscaleModelId,
    force = false,
  ): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('start_upscale_model_download', {
      model_id: modelId,
      force,
    });
  },

  async cancelUpscaleModelDownload(): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('cancel_upscale_model_download');
  },

  async getUpscaleCapabilities(forceRefresh = false): Promise<UpscaleCapabilities> {
    if (forceRefresh) {
      return callApi<UpscaleCapabilities>('get_upscale_capabilities', { force_refresh: true });
    }
    return callApi<UpscaleCapabilities>('get_upscale_capabilities');
  },

  async startUpscale(payload: StartUpscalePayload): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('start_upscale', payload);
  },

  async getUpscaleTaskStatus(): Promise<UpscaleTaskStatusResult> {
    return callApi<UpscaleTaskStatusResult>('get_upscale_task_status');
  },

  async cancelUpscaleTask(): Promise<{ success: boolean; error?: string }> {
    return callApi<{ success: boolean; error?: string }>('cancel_upscale_task');
  },
};

export function toFileUrl(path: string): string {
  // 把本地路径转换成浏览器可用的 file:// URL（兼容 Windows 盘符）。
  const normalized = String(path).replace(/\\/g, '/');
  const isWindowsPath = /^[a-zA-Z]:\//.test(normalized);
  const segments = normalized.split('/').map((segment, index) => {
    if (segment === '' && (index === 0 || (isWindowsPath && index === 1))) {
      return segment;
    }
    return encodeURIComponent(segment);
  });
  if (isWindowsPath) {
    return `file:///${segments.join('/')}`;
  }
  return `file://${segments.join('/')}`;
}
