// 应用级全局状态仓库（zustand）。
// 包含视图切换、设置草稿、处理进度、结果信息等跨页面数据。
import { create } from 'zustand';
import type {
  AppSettings,
  DeviceInfoState,
  MainView,
  ProcessState,
  ResultState,
  UpscaleConfig,
  UpscaleResultState,
  UpscaleTaskState,
} from '../types/app';

interface AppState {
  // 当前主视图。
  view: MainView;
  // 页面正在编辑的设置（可能尚未保存）。
  settings: AppSettings;
  // 最近一次成功保存/加载的设置，用于“回滚”。
  persistedSettings: AppSettings;
  // 设备信息、处理状态、结果状态。
  deviceInfo: DeviceInfoState;
  process: ProcessState;
  result: ResultState;
  upscaleConfig: UpscaleConfig;
  upscaleTask: UpscaleTaskState;
  upscaleResult: UpscaleResultState;
  setView: (view: MainView) => void;
  updateSettings: (patch: Partial<AppSettings>) => void;
  setSettingsFromBackend: (settings: AppSettings) => void;
  commitSettings: () => void;
  rollbackSettings: () => void;
  setDeviceInfo: (deviceInfo: DeviceInfoState) => void;
  updateProcess: (patch: Partial<ProcessState>) => void;
  resetProcess: () => void;
  setResult: (result: ResultState) => void;
  clearResult: () => void;
  updateUpscaleConfig: (patch: Partial<UpscaleConfig>) => void;
  setUpscaleTask: (task: UpscaleTaskState) => void;
  updateUpscaleTask: (patch: Partial<UpscaleTaskState>) => void;
  resetUpscaleTask: () => void;
  setUpscaleResult: (result: UpscaleResultState) => void;
  clearUpscaleResult: () => void;
}

const DEFAULT_SETTINGS: AppSettings = {
  language: 'zh',
  theme: 'light',
  outputPath: '',
  modelId: 'lama_roi',
};

const DEFAULT_DEVICE_INFO: DeviceInfoState = {
  device: '--',
  memory: '--',
  supportsFp16: false,
};

const DEFAULT_PROCESS_STATE: ProcessState = {
  isProcessing: false,
  progress: 0,
  statusMessage: 'idle',
  processedFrames: 0,
  totalFrames: 0,
  estimatedTime: '--:--',
  etaSeconds: undefined,
  throughputFps: undefined,
  phase: '',
  opaqueInfer: false,
};

const DEFAULT_RESULT_STATE: ResultState = {
  outputPath: '',
  outputUrl: '',
  mediaType: '',
  width: 0,
  height: 0,
  fps: 0,
  frameCount: 0,
  modelId: 'lama_roi',
};

const DEFAULT_UPSCALE_CONFIG: UpscaleConfig = {
  enabled: false,
  mode: 'upscale_resolution',
  engine: 'realesrgan',
  modelId: 'realesrgan_general_x4v3',
  targetPreset: '1080p',
  sameResStrength: 'x2_then_downscale',
  denoiseStrength: 0.35,
  keepAudio: true,
};

const DEFAULT_UPSCALE_TASK: UpscaleTaskState = {
  state: 'idle',
  progress: 0,
  phase: '',
  message: '',
  etaSeconds: undefined,
  inputPath: '',
  outputPath: '',
  previewPath: '',
  mode: '',
  engine: '',
  effectiveEngine: '',
  modelId: '',
  error: '',
  warning: '',
  segmentIndex: 0,
  segmentTotal: 0,
  sceneSplitMode: '',
};

const DEFAULT_UPSCALE_RESULT: UpscaleResultState = {
  outputPath: '',
  outputUrl: '',
  previewUrl: '',
  width: 0,
  height: 0,
  fps: 0,
  frameCount: 0,
  mode: '',
  engine: '',
  modelId: '',
  warning: '',
};

// 全局应用状态仓库：
// 负责顶部导航、设置、处理进度、结果信息等跨页面数据。
export const useAppStore = create<AppState>((set) => ({
  view: 'process',
  settings: DEFAULT_SETTINGS,
  persistedSettings: DEFAULT_SETTINGS,
  deviceInfo: DEFAULT_DEVICE_INFO,
  process: DEFAULT_PROCESS_STATE,
  result: DEFAULT_RESULT_STATE,
  upscaleConfig: DEFAULT_UPSCALE_CONFIG,
  upscaleTask: DEFAULT_UPSCALE_TASK,
  upscaleResult: DEFAULT_UPSCALE_RESULT,

  setView: (view) => set({ view }),
  // 合并更新设置草稿。
  updateSettings: (patch) =>
    set((state) => ({
      settings: {
        ...state.settings,
        ...patch,
      },
    })),
  // 从后端加载设置时，同时更新草稿和已持久化快照。
  setSettingsFromBackend: (settings) =>
    set({
      settings,
      persistedSettings: settings,
    }),
  // 保存成功后，把当前草稿标记为“已持久化”。
  commitSettings: () =>
    set((state) => ({
      persistedSettings: state.settings,
    })),
  // 保存失败时，把草稿回滚到已持久化版本。
  rollbackSettings: () =>
    set((state) => ({
      settings: state.persistedSettings,
    })),
  setDeviceInfo: (deviceInfo) => set({ deviceInfo }),
  // 合并更新处理状态，避免每次都重建整对象。
  updateProcess: (patch) =>
    set((state) => ({
      process: {
        ...state.process,
        ...patch,
      },
    })),
  resetProcess: () => set({ process: DEFAULT_PROCESS_STATE }),
  setResult: (result) => set({ result }),
  clearResult: () => set({ result: DEFAULT_RESULT_STATE }),
  updateUpscaleConfig: (patch) =>
    set((state) => ({
      upscaleConfig: {
        ...state.upscaleConfig,
        ...patch,
      },
    })),
  setUpscaleTask: (task) => set({ upscaleTask: task }),
  updateUpscaleTask: (patch) =>
    set((state) => ({
      upscaleTask: {
        ...state.upscaleTask,
        ...patch,
      },
    })),
  resetUpscaleTask: () => set({ upscaleTask: DEFAULT_UPSCALE_TASK }),
  setUpscaleResult: (result) => set({ upscaleResult: result }),
  clearUpscaleResult: () => set({ upscaleResult: DEFAULT_UPSCALE_RESULT }),
}));

export { DEFAULT_SETTINGS };
