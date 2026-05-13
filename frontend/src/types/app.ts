// 主页面标签：处理、打标、结果、AI 放大结果。
export type MainView = 'process' | 'annotate' | 'result' | 'upscale';

// 语言枚举（当前仅中英文）。
export type Language = 'zh' | 'en';

// 主题模式（浅色 / 深色）。
export type ThemeMode = 'light' | 'dark';

// 后端支持的模型 ID，前后端保持一致。
export type ModelId = 'lama_roi';

// AI 放大模式。
export type UpscaleMode = 'upscale_resolution' | 'enhance_same_resolution';

// AI 放大引擎。
export type UpscaleEngine = 'realesrgan' | 'seedvr2';

// AI 放大模型。
export type UpscaleModelId =
  | 'realesrgan_general_x4v3'
  | 'realesrgan_x2plus'
  | 'seedvr2_3b_q8_0_gguf'
  | 'seedvr2_3b_q4_k_m_gguf';

// 分辨率提升目标档位。
export type UpscaleTargetPreset = '1080p';

// 同分辨率增强强度（先超分后回缩）。
export type UpscaleSameResStrength = 'x2_then_downscale';

// 全局设置结构：会在设置页编辑，并保存到后端配置。
export interface AppSettings {
  language: Language;
  theme: ThemeMode;
  outputPath: string;
  modelId: ModelId;
}

// 处理页运行时状态（进度、阶段、速度等）。
export interface ProcessState {
  isProcessing: boolean;
  progress: number;
  statusMessage: string;
  processedFrames: number;
  totalFrames: number;
  estimatedTime: string;
  etaSeconds?: number;
  throughputFps?: number;
  phase?: 'prepare' | 'load_models' | 'extract' | 'infer' | 'compose' | 'finalize' | '';
  opaqueInfer?: boolean;
}

// 设备信息展示结构。
export interface DeviceInfoState {
  device: string;
  memory: string;
  supportsFp16: boolean;
}

// 结果页展示结构。
export interface ResultState {
  outputPath: string;
  outputUrl: string;
  mediaType: 'video' | 'image' | '';
  width: number;
  height: number;
  fps: number;
  frameCount: number;
  modelId: ModelId;
}

// 放大配置（会话级，不写全局设置文件）。
export interface UpscaleConfig {
  enabled: boolean;
  mode: UpscaleMode;
  engine: UpscaleEngine;
  modelId: UpscaleModelId;
  targetPreset: UpscaleTargetPreset;
  sameResStrength: UpscaleSameResStrength;
  denoiseStrength: number;
  keepAudio: boolean;
}

// 引擎能力信息。
export interface UpscaleEngineCapability {
  engine: UpscaleEngine;
  display_name: string;
  available: boolean;
  reason?: string;
  runtime_hint?: string;
}

// 模型能力信息。
export interface UpscaleModelCapability {
  engine: UpscaleEngine;
  model_id: UpscaleModelId;
  display_name: string;
  installed?: boolean;
}

// 后端能力聚合返回结构。
export interface UpscaleCapabilities {
  success: boolean;
  engines: UpscaleEngineCapability[];
  models: UpscaleModelCapability[];
  modes: UpscaleMode[];
  target_presets: UpscaleTargetPreset[];
  same_res_strengths: UpscaleSameResStrength[];
  defaults: {
    engine: UpscaleEngine;
    mode: UpscaleMode;
    model_id: UpscaleModelId;
    target_preset: UpscaleTargetPreset;
    same_res_strength: UpscaleSameResStrength;
    denoise_strength: number;
    keep_audio: boolean;
  };
  ffmpeg?: {
    available?: boolean;
    libplacebo_available?: boolean;
  };
  error?: string;
}

// 放大任务状态。
export interface UpscaleTaskState {
  state: 'idle' | 'running' | 'success' | 'failed' | 'cancelled';
  progress: number;
  phase: 'prepare' | 'extract' | 'infer' | 'resample' | 'compose' | 'finalize' | '';
  message: string;
  etaSeconds?: number;
  inputPath: string;
  outputPath: string;
  previewPath: string;
  mode: UpscaleMode | '';
  engine: UpscaleEngine | '';
  effectiveEngine: UpscaleEngine | '';
  modelId: UpscaleModelId | '';
  error: string;
  warning: string;
  segmentIndex?: number;
  segmentTotal?: number;
  sceneSplitMode?: 'rule' | 'hybrid' | 'fallback' | '';
}

// 放大结果页展示结构。
export interface UpscaleResultState {
  outputPath: string;
  outputUrl: string;
  previewUrl: string;
  width: number;
  height: number;
  fps: number;
  frameCount: number;
  mode: UpscaleMode | '';
  engine: UpscaleEngine | '';
  modelId: UpscaleModelId | '';
  warning: string;
}
