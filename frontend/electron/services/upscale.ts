import {
  UPSCALE_MODEL_SPECS,
  isUpscaleModelId,
  type UpscaleEngineId,
  type UpscaleModelId,
} from './modelCatalog.js';
import { isInstalledModel, type ModelDownloadTask } from './modelManager.js';
import { getNativeCoreStatus } from './nativeCore.js';

export interface UpscaleServiceOptions {
  userDataDir: string;
  appRoot: string;
  manifestUrl?: string;
}

export interface StartUpscalePayload {
  input_path?: string;
  output_dir?: string;
  mode?: 'upscale_resolution' | 'enhance_same_resolution';
  engine?: UpscaleEngineId;
  model_id?: UpscaleModelId;
  target_preset?: '1080p';
  same_res_strength?: 'x2_then_downscale';
  denoise_strength?: number;
  keep_audio?: boolean;
}

interface UpscaleTask {
  state: 'idle' | 'running' | 'success' | 'failed' | 'cancelled';
  progress: number;
  phase: string;
  message: string;
  eta_seconds: number | null;
  input_path: string;
  output_path: string;
  preview_path: string;
  mode: string;
  engine: string;
  effective_engine: string;
  model_id: string;
  error: string;
  warning: string;
  segment_index: number;
  segment_total: number;
  scene_split_mode: string;
}

export class UpscaleService {
  private downloadTask: ModelDownloadTask = createDownloadIdleTask();
  private task: UpscaleTask = createIdleUpscaleTask();

  constructor(private readonly options: UpscaleServiceOptions) {}

  async getModelDownloadStatus() {
    const models = await Promise.all(
      Object.values(UPSCALE_MODEL_SPECS).map(async (spec) => ({
        model_id: spec.modelId,
        display_name: spec.displayName,
        installed: await isInstalledModel(this.options.userDataDir, spec.modelId),
        can_redownload: true,
        install_hint: spec.installHint,
      })),
    );
    return {
      success: true,
      models,
      task: { ...this.downloadTask },
    };
  }

  async startModelDownload(payload: unknown): Promise<{ success: boolean; error?: string }> {
    const modelId = typeof payload === 'object' && payload ? (payload as { model_id?: unknown }).model_id : '';
    if (!isUpscaleModelId(modelId)) {
      return {
        success: false,
        error: `Invalid model_id: ${String(modelId || '')}. Supported values: ${Object.keys(UPSCALE_MODEL_SPECS).join(', ')}`,
      };
    }
    const spec = UPSCALE_MODEL_SPECS[modelId];
    if (!this.options.manifestUrl) {
      this.downloadTask = {
        ...createDownloadIdleTask(),
        state: 'failed',
        model_id: modelId,
        error: `WMR_MODEL_MANIFEST_URL is not configured; cannot download non-Python assets for ${modelId}`,
        message: 'Model download blocked',
      };
      return { success: false, error: this.downloadTask.error };
    }
    if (!spec.implemented) {
      this.downloadTask = {
        ...createDownloadIdleTask(),
        state: 'failed',
        model_id: modelId,
        error: spec.blockedReason || `${spec.displayName} is blocked by the non-Python equivalence gate.`,
        message: 'Model download blocked',
      };
      return { success: false, error: this.downloadTask.error };
    }
    return {
      success: false,
      error: `${spec.displayName} download is not wired until the native runtime implementation is available.`,
    };
  }

  cancelModelDownload(): { success: boolean; error?: string } {
    if (this.downloadTask.state === 'running') {
      this.downloadTask = { ...this.downloadTask, state: 'cancelled', message: 'Model download cancelled' };
    }
    return { success: true };
  }

  async getCapabilities() {
    const nativeCore = await getNativeCoreStatus(this.options.appRoot);
    const models = await Promise.all(
      Object.values(UPSCALE_MODEL_SPECS).map(async (spec) => ({
        engine: spec.engine as UpscaleEngineId,
        model_id: spec.modelId,
        display_name: spec.displayName,
        installed: await isInstalledModel(this.options.userDataDir, spec.modelId),
      })),
    );
    return {
      success: true,
      engines: (['realesrgan', 'seedvr2'] as const).map((engine) => ({
        engine,
        display_name: engine === 'realesrgan' ? 'Real-ESRGAN Native' : 'SeedVR2 Native',
        available: false,
        reason: nativeCore.available && nativeCore.opencvAlgorithms
          ? `${engine} is blocked until official/self-owned same-model non-Python assets are supplied.`
          : nativeCore.reason,
        runtime_hint: nativeCore.path,
      })),
      models,
      modes: ['upscale_resolution', 'enhance_same_resolution'],
      target_presets: ['1080p'],
      same_res_strengths: ['x2_then_downscale'],
      defaults: {
        mode: 'enhance_same_resolution',
        engine: 'realesrgan',
        model_id: 'realesrgan_x2plus',
        target_preset: '1080p',
        same_res_strength: 'x2_then_downscale',
        denoise_strength: 0.35,
        keep_audio: true,
      },
      ffmpeg: {
        available: true,
        libplacebo_available: false,
      },
    };
  }

  async startUpscale(payload: StartUpscalePayload = {}): Promise<{ success: boolean; error?: string }> {
    const engine = payload.engine || 'realesrgan';
    const modelId = payload.model_id || (engine === 'seedvr2' ? 'seedvr2_3b_q4_k_m_gguf' : 'realesrgan_general_x4v3');
    const nativeCore = await getNativeCoreStatus(this.options.appRoot);
    const spec = isUpscaleModelId(modelId) ? UPSCALE_MODEL_SPECS[modelId] : null;
    const error =
      spec?.blockedReason ||
      (nativeCore.available && nativeCore.opencvAlgorithms
        ? `${engine} is blocked until official/self-owned same-model non-Python assets are supplied.`
        : nativeCore.reason);
    this.task = {
      ...createIdleUpscaleTask(),
      state: 'failed',
      progress: 0,
      phase: 'prepare',
      message: 'Upscale blocked by non-Python equivalence gate',
      input_path: payload.input_path || '',
      output_path: '',
      mode: payload.mode || 'enhance_same_resolution',
      engine,
      effective_engine: '',
      model_id: modelId,
      error,
    };
    return { success: false, error };
  }

  getTaskStatus() {
    return { success: true, task: { ...this.task } };
  }

  cancelTask(): { success: boolean; error?: string } {
    if (this.task.state === 'running') {
      this.task = { ...this.task, state: 'cancelled', message: 'Upscale cancelled' };
    }
    return { success: true };
  }
}

function createDownloadIdleTask(): ModelDownloadTask {
  return {
    state: 'idle',
    model_id: '',
    progress: 0,
    downloaded_bytes: 0,
    total_bytes: 0,
    speed_bps: 0,
    current_file: '',
    message: '',
    error: '',
  };
}

function createIdleUpscaleTask(): UpscaleTask {
  return {
    state: 'idle',
    progress: 0,
    phase: 'idle',
    message: '',
    eta_seconds: null,
    input_path: '',
    output_path: '',
    preview_path: '',
    mode: '',
    engine: '',
    effective_engine: '',
    model_id: '',
    error: '',
    warning: '',
    segment_index: 0,
    segment_total: 0,
    scene_split_mode: '',
  };
}
