import path from 'node:path';

export type InpaintModelId = 'lama_roi';
export type UpscaleEngineId = 'realesrgan' | 'seedvr2';
export type UpscaleModelId =
  | 'realesrgan_general_x4v3'
  | 'realesrgan_x2plus'
  | 'seedvr2_3b_q8_0_gguf'
  | 'seedvr2_3b_q4_k_m_gguf';
export type DownloadableModelId = InpaintModelId | UpscaleModelId;

export interface ModelAssetSpec {
  assetId: string;
  fileName: string;
}

export interface ModelSpec {
  modelId: DownloadableModelId;
  displayName: string;
  engine: 'lama' | UpscaleEngineId;
  installHint: string;
  version: string;
  assetDir: string[];
  assets: ModelAssetSpec[];
  implemented: boolean;
  blockedReason?: string;
}

export const INPAINT_MODEL_SPECS: Record<InpaintModelId, ModelSpec> = {
  lama_roi: {
    modelId: 'lama_roi',
    displayName: 'LaMa ROI ONNX',
    engine: 'lama',
    installHint: 'Requires the official/self-owned Big LaMa ONNX asset.',
    version: '1.0.0',
    assetDir: ['models', 'big-lama', '1.0.0'],
    assets: [{ assetId: 'big-lama-onnx', fileName: 'big-lama.onnx' }],
    implemented: true,
  },
};

export const UPSCALE_MODEL_SPECS: Record<UpscaleModelId, ModelSpec> = {
  realesrgan_general_x4v3: {
    modelId: 'realesrgan_general_x4v3',
    displayName: 'Real-ESRGAN General x4v3 Native',
    engine: 'realesrgan',
    installHint: 'Blocked until official/self-owned Real-ESRGAN x4v3 ONNX or NCNN assets are provided.',
    version: '1.0.0',
    assetDir: ['models', 'realesrgan_general_x4v3', '1.0.0'],
    assets: [
      { assetId: 'realesrgan-general-x4v3-native', fileName: 'realesr-general-x4v3.onnx' },
      { assetId: 'realesrgan-general-wdn-x4v3-native', fileName: 'realesr-general-wdn-x4v3.onnx' },
    ],
    implemented: false,
    blockedReason: 'Real-ESRGAN native engine is blocked until same-model non-Python runtime assets are available.',
  },
  realesrgan_x2plus: {
    modelId: 'realesrgan_x2plus',
    displayName: 'Real-ESRGAN x2plus Native',
    engine: 'realesrgan',
    installHint: 'Blocked until official/self-owned Real-ESRGAN x2plus ONNX or NCNN assets are provided.',
    version: '1.0.0',
    assetDir: ['models', 'realesrgan_x2plus', '1.0.0'],
    assets: [{ assetId: 'realesrgan-x2plus-native', fileName: 'RealESRGAN_x2plus.onnx' }],
    implemented: false,
    blockedReason: 'Real-ESRGAN native engine is blocked until same-model non-Python runtime assets are available.',
  },
  seedvr2_3b_q8_0_gguf: {
    modelId: 'seedvr2_3b_q8_0_gguf',
    displayName: 'SeedVR2 3B Q8_0 Native',
    engine: 'seedvr2',
    installHint: 'Blocked until an official/self-owned non-Python SeedVR2 GGUF runner is provided.',
    version: '1.0.0',
    assetDir: ['models', 'seedvr2_3b_q8_0_gguf', '1.0.0'],
    assets: [
      { assetId: 'seedvr2-3b-q8-0-gguf', fileName: 'seedvr2_ema_3b-Q8_0.gguf' },
      { assetId: 'seedvr2-native-runner', fileName: platformRunnerName() },
    ],
    implemented: false,
    blockedReason: 'SeedVR2 native engine is blocked until a trusted non-Python GGUF runner is available.',
  },
  seedvr2_3b_q4_k_m_gguf: {
    modelId: 'seedvr2_3b_q4_k_m_gguf',
    displayName: 'SeedVR2 3B Q4_K_M Native',
    engine: 'seedvr2',
    installHint: 'Blocked until an official/self-owned non-Python SeedVR2 GGUF runner is provided.',
    version: '1.0.0',
    assetDir: ['models', 'seedvr2_3b_q4_k_m_gguf', '1.0.0'],
    assets: [
      { assetId: 'seedvr2-3b-q4-k-m-gguf', fileName: 'seedvr2_ema_3b-Q4_K_M.gguf' },
      { assetId: 'seedvr2-native-runner', fileName: platformRunnerName() },
    ],
    implemented: false,
    blockedReason: 'SeedVR2 native engine is blocked until a trusted non-Python GGUF runner is available.',
  },
};

export const ALL_MODEL_SPECS: Record<DownloadableModelId, ModelSpec> = {
  ...INPAINT_MODEL_SPECS,
  ...UPSCALE_MODEL_SPECS,
};

export function isInpaintModelId(value: unknown): value is InpaintModelId {
  return value === 'lama_roi';
}

export function isUpscaleModelId(value: unknown): value is UpscaleModelId {
  return (
    value === 'realesrgan_general_x4v3' ||
    value === 'realesrgan_x2plus' ||
    value === 'seedvr2_3b_q8_0_gguf' ||
    value === 'seedvr2_3b_q4_k_m_gguf'
  );
}

export function isDownloadableModelId(value: unknown): value is DownloadableModelId {
  return isInpaintModelId(value) || isUpscaleModelId(value);
}

export function modelInstallDir(userDataDir: string, spec: ModelSpec): string {
  return path.join(userDataDir, ...spec.assetDir);
}

export function modelAssetPath(userDataDir: string, spec: ModelSpec, asset: ModelAssetSpec): string {
  return path.join(modelInstallDir(userDataDir, spec), asset.fileName);
}

function platformRunnerName(): string {
  return process.platform === 'win32' ? 'seedvr2-runner.exe' : 'seedvr2-runner';
}
