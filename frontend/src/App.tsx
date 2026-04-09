// 应用主壳组件：
// 负责串联“处理页 / 打标页 / 结果页 / 设置侧栏”，
// 同时管理前端与 pywebview 后端的主要交互流程。
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button, ConfigProvider, Layout, SideSheet, Toast, Typography } from '@douyinfe/semi-ui';
import zhCN from '@douyinfe/semi-ui/lib/es/locale/source/zh_CN';
import enUS from '@douyinfe/semi-ui/lib/es/locale/source/en_US';
import { IconBulb, IconSetting } from '@douyinfe/semi-icons';

import { AnnotationWorkspace } from './workspace/AnnotationWorkspace';
import {
  pywebviewClient,
  toFileUrl,
  type BackendSettingsResult,
  type ModelDownloadEntry,
  type ModelDownloadTask,
  type ModelDownloadTaskState,
  type UpscaleModelDownloadEntry,
  type UpscaleModelDownloadTask,
  type UpscaleTaskStatusResult,
} from './services/pywebview';
import { useWorkspaceStore } from './store/workspace';
import { DEFAULT_SETTINGS, useAppStore } from './store/app';
import type { VideoMeta } from './types/annotation';
import type {
  AppSettings,
  MainView,
  UpscaleCapabilities,
  UpscaleEngine,
  UpscaleModelId,
  UpscaleTaskState,
} from './types/app';
import { useI18n } from './i18n/useI18n';
import { ProcessView } from './views/ProcessView';
import { ResultView } from './views/ResultView';
import { ManualView } from './views/ManualView';
import { SettingsView } from './views/SettingsView';
import { UpscaleView } from './views/UpscaleView';

const { Header, Content } = Layout;
const { Title } = Typography;

// 顶部导航流程顺序，决定按钮展示和箭头激活态。
const NAV_ITEMS: Array<{ key: MainView; labelKey: string }> = [
  { key: 'process', labelKey: 'nav.process' },
  { key: 'annotate', labelKey: 'nav.annotate' },
  { key: 'result', labelKey: 'nav.result' },
  { key: 'upscale', labelKey: 'nav.upscale' },
];

const EMPTY_META: VideoMeta = {
  path: '',
  basename: '',
  sha1: '',
  size: 0,
  mtime_ns: 0,
  width: 1280,
  height: 720,
  fps: 30,
  frame_count: 1,
};

const EMPTY_DOWNLOAD_TASK: ModelDownloadTask = {
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

const EMPTY_UPSCALE_DOWNLOAD_TASK: UpscaleModelDownloadTask = {
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

function normalizeSettingsFromBackend(raw: BackendSettingsResult): AppSettings {
  // 后端返回的设置做一次兜底清洗，保证前端状态稳定。
  const language = raw.language === 'en' ? 'en' : 'zh';
  const theme = raw.theme === 'dark' ? 'dark' : 'light';
  const modelRaw = String(raw.output?.model_id || '').toLowerCase();
  const modelId =
    modelRaw === 'lama_roi' || modelRaw === 'propainter_roi'
      ? modelRaw
      : 'lama_roi';

  return {
    ...DEFAULT_SETTINGS,
    language,
    theme,
    outputPath: raw.output?.path || '',
    modelId,
  };
}

function normalizeStatusText(raw: string, t: (key: string) => string): string {
  // 把后端原始状态文本尽量映射为统一 i18n 文案。
  const text = String(raw || '').trim();
  if (!text) return t('status.idle');
  const lower = text.toLowerCase();
  if (
    lower.includes('infer heartbeat')
    || lower.includes('heartbeat (estimated)')
  ) {
    return t('process.inferHeartbeatEstimated');
  }
  if (
    lower.includes('lama infer')
    || lower.includes('propainter infer')
    || lower.includes('inference')
  ) {
    return t('process.inferEstimated');
  }
  if (lower.includes('complete') || lower.includes('done') || text.includes('完成')) return t('status.done');
  if (lower.includes('error') || lower.includes('fail') || text.includes('失败')) return t('status.failed');
  if (
    lower.includes('processing') ||
    lower.includes('extracting') ||
    lower.includes('loading') ||
    lower.includes('preparing') ||
    text.includes('处理')
  ) {
    return t('status.running');
  }
  return text;
}

function basename(path: string): string {
  // 兼容 Windows/Unix 路径分隔符，抽取文件名用于展示。
  const parts = String(path || '').replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] ?? path;
}

function normalizeModelId(value: unknown): AppSettings['modelId'] {
  // 模型 ID 白名单校验，非法值一律回退到默认模型。
  const raw = String(value ?? '').toLowerCase();
  if (raw === 'lama_roi' || raw === 'propainter_roi') {
    return raw;
  }
  return 'lama_roi';
}

function normalizeDownloadTask(raw: Partial<ModelDownloadTask> | null | undefined): ModelDownloadTask {
  // 下载任务状态清洗：把不可信字段收敛到可控范围。
  const stateRaw = String(raw?.state || 'idle').toLowerCase();
  const state: ModelDownloadTaskState = (
    stateRaw === 'running'
    || stateRaw === 'success'
    || stateRaw === 'failed'
    || stateRaw === 'cancelled'
  )
    ? stateRaw
    : 'idle';

  const modelRaw = String(raw?.model_id || '').toLowerCase();
  const modelId: ModelDownloadTask['model_id'] = (
    modelRaw === 'lama_roi' || modelRaw === 'propainter_roi'
  )
    ? modelRaw
    : '';

  return {
    state,
    model_id: modelId,
    progress: Math.min(1, Math.max(0, Number(raw?.progress || 0))),
    downloaded_bytes: Math.max(0, Number(raw?.downloaded_bytes || 0)),
    total_bytes: Math.max(0, Number(raw?.total_bytes || 0)),
    speed_bps: Math.max(0, Number(raw?.speed_bps || 0)),
    current_file: String(raw?.current_file || ''),
    message: String(raw?.message || ''),
    error: String(raw?.error || ''),
  };
}

function normalizeUpscaleDownloadTask(
  raw: Partial<UpscaleModelDownloadTask> | null | undefined,
): UpscaleModelDownloadTask {
  const stateRaw = String(raw?.state || 'idle').toLowerCase();
  const state: UpscaleModelDownloadTask['state'] = (
    stateRaw === 'running'
    || stateRaw === 'success'
    || stateRaw === 'failed'
    || stateRaw === 'cancelled'
  )
    ? stateRaw
    : 'idle';

  const modelRaw = String(raw?.model_id || '').toLowerCase();
  const modelId: UpscaleModelDownloadTask['model_id'] = (
    modelRaw === 'realesrgan_general_x4v3'
    || modelRaw === 'realesrgan_x2plus'
    || modelRaw === 'seedvr2_3b_q8_0_gguf'
    || modelRaw === 'seedvr2_3b_q4_k_m_gguf'
  )
    ? modelRaw
    : '';

  return {
    state,
    model_id: modelId,
    progress: Math.min(1, Math.max(0, Number(raw?.progress || 0))),
    downloaded_bytes: Math.max(0, Number(raw?.downloaded_bytes || 0)),
    total_bytes: Math.max(0, Number(raw?.total_bytes || 0)),
    speed_bps: Math.max(0, Number(raw?.speed_bps || 0)),
    current_file: String(raw?.current_file || ''),
    message: String(raw?.message || ''),
    error: String(raw?.error || ''),
  };
}

function normalizeUpscaleEngine(value: unknown): UpscaleEngine {
  const raw = String(value || '').toLowerCase();
  if (raw === 'realesrgan') return 'realesrgan';
  if (raw === 'seedvr2') return 'seedvr2';
  return 'realesrgan';
}

function normalizeUpscaleModelId(value: unknown): UpscaleModelId {
  const raw = String(value ?? '').toLowerCase();
  if (raw === 'realesrgan_general_x4v3') return 'realesrgan_general_x4v3';
  if (raw === 'realesrgan_x2plus') return 'realesrgan_x2plus';
  if (raw === 'seedvr2_3b_q8_0_gguf') return 'seedvr2_3b_q8_0_gguf';
  if (raw === 'seedvr2_3b_q4_k_m_gguf') return 'seedvr2_3b_q4_k_m_gguf';
  return 'realesrgan_general_x4v3';
}

function normalizeUpscaleTask(raw: UpscaleTaskStatusResult['task'] | undefined): UpscaleTaskState {
  const stateRaw = String(raw?.state || 'idle').toLowerCase();
  const state: UpscaleTaskState['state'] = (
    stateRaw === 'running'
    || stateRaw === 'success'
    || stateRaw === 'failed'
    || stateRaw === 'cancelled'
  )
    ? stateRaw
    : 'idle';

  const phaseRaw = String(raw?.phase || '');
  const phase: UpscaleTaskState['phase'] = (
    phaseRaw === 'prepare'
    || phaseRaw === 'extract'
    || phaseRaw === 'infer'
    || phaseRaw === 'resample'
    || phaseRaw === 'compose'
    || phaseRaw === 'finalize'
  )
    ? phaseRaw
    : '';

  const modeRaw = String(raw?.mode || '');
  const mode: UpscaleTaskState['mode'] = (
    modeRaw === 'upscale_resolution' || modeRaw === 'enhance_same_resolution'
  )
    ? modeRaw
    : '';

  const splitModeRaw = String(raw?.scene_split_mode || '').toLowerCase();
  const sceneSplitMode: UpscaleTaskState['sceneSplitMode'] = (
    splitModeRaw === 'rule' || splitModeRaw === 'hybrid' || splitModeRaw === 'fallback'
  )
    ? splitModeRaw
    : '';

  const segmentIndexRaw = Number(raw?.segment_index || 0);
  const segmentTotalRaw = Number(raw?.segment_total || 0);
  const segmentIndex = Number.isFinite(segmentIndexRaw) ? Math.max(0, segmentIndexRaw) : 0;
  const segmentTotal = Number.isFinite(segmentTotalRaw) ? Math.max(0, segmentTotalRaw) : 0;

  return {
    state,
    progress: Math.min(1, Math.max(0, Number(raw?.progress || 0))),
    phase,
    message: String(raw?.message || ''),
    etaSeconds: raw?.eta_seconds !== undefined ? Math.max(0, Number(raw?.eta_seconds || 0)) : undefined,
    inputPath: String(raw?.input_path || ''),
    outputPath: String(raw?.output_path || ''),
    previewPath: String(raw?.preview_path || ''),
    mode,
    engine: raw?.engine ? normalizeUpscaleEngine(raw.engine) : '',
    effectiveEngine: raw?.effective_engine ? normalizeUpscaleEngine(raw.effective_engine) : '',
    modelId: raw?.model_id ? normalizeUpscaleModelId(raw.model_id) : '',
    error: String(raw?.error || ''),
    warning: String(raw?.warning || ''),
    segmentIndex,
    segmentTotal,
    sceneSplitMode,
  };
}

function localizeSeedVRError(message: string, t: (key: string) => string): string {
  const text = String(message || '');
  const lower = text.toLowerCase();
  if (lower.includes('real-esrgan runtime') || lower.includes('realesrgan runtime')) {
    return t('upscale.realesrgan.runtimeMissing');
  }
  if (lower.includes('unsupported real-esrgan model_id')) {
    return t('upscale.realesrgan.invalidModel');
  }
  if (lower.includes('running real-esrgan inference')) {
    return t('upscale.realesrgan.phase.infer');
  }
  if (lower.includes('real-esrgan retry with smaller tile')) {
    return t('upscale.realesrgan.retryTile');
  }
  if (lower.includes('real-esrgan command failed')) {
    return t('upscale.realesrgan.commandFailed');
  }
  if (lower.includes('warmup stalled') || lower.includes('warmup timed out')) {
    return t('upscale.seedvr.warmupRetry');
  }
  if (
    lower.includes('stage=warmup')
    || lower.includes('step=chunk_loop_start')
    || lower.includes('step=load_model_done')
    || lower.includes('step=video_backend_ready')
  ) {
    return t('upscale.seedvr.phase.warmupInit');
  }
  if (lower.includes('phase=model_warmup')) return t('upscale.seedvr.phase.warmup');
  if (lower.includes('phase=chunk_infer')) return t('upscale.seedvr.phase.chunkInfer');
  if (lower.includes('phase=flush_output')) return t('upscale.seedvr.phase.flushOutput');
  if (lower.includes('python 3.12') || lower.includes('seedvr runtime')) {
    return t('upscale.seedvr.runtimeMissing');
  }
  if (lower.includes('requires at least') && lower.includes('memory')) {
    return t('upscale.seedvr.lowMemory');
  }
  if (lower.includes('memory guard triggered') || lower.includes('out of memory')) {
    return t('upscale.seedvr.memoryGuard');
  }
  if (lower.includes('inference stalled') || lower.includes('no forward progress')) {
    return t('upscale.seedvr.stallRetry');
  }
  if (lower.includes('backend=ffmpeg') || lower.includes('video backend preference: ffmpeg')) {
    return t('upscale.seedvr.backend.ffmpeg');
  }
  if (lower.includes('ffmpeg backend failed') || lower.includes('retrying with opencv backend')) {
    return t('upscale.seedvr.backend.fallback_opencv');
  }
  if (lower.includes('model caching enabled') || lower.includes('cache_dit/cache_vae')) {
    return t('upscale.seedvr.cache.enabled');
  }
  if (lower.includes('applied streaming profile')) {
    return t('upscale.seedvr.profile.safe_or_guarded');
  }
  if (lower.includes('applied mps-first execution policy')) {
    return t('upscale.seedvr.policy.mpsFirst');
  }
  if (lower.includes('scene split fallback') || lower.includes('scene split review fallback')) {
    return t('upscale.seedvr.sceneFallback');
  }
  if (lower.includes('concat copy failed')) {
    return t('upscale.seedvr.concatFallback');
  }
  if (lower.includes('load governor active') || lower.includes('throttling to 80%')) {
    return t('upscale.seedvr.loadGovernor');
  }
  return text;
}

function sleep(ms: number): Promise<void> {
  // 轮询场景使用的小延时工具。
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export default function App() {
  // 页面本地 UI 状态（不需要持久化到全局 store 的部分）。
  const [frameImageUrl, setFrameImageUrl] = useState('');
  const [previewSessionId, setPreviewSessionId] = useState('');
  const [processPreviewFrameWidth, setProcessPreviewFrameWidth] = useState(0);
  const [processPreviewFrameHeight, setProcessPreviewFrameHeight] = useState(0);
  const [isSelectingVideo, setIsSelectingVideo] = useState(false);
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const [isProcessPlaying, setIsProcessPlaying] = useState(false);
  const [isManualPanelOpen, setIsManualPanelOpen] = useState(false);
  const [isSettingsPanelOpen, setIsSettingsPanelOpen] = useState(false);
  const [modelDownloads, setModelDownloads] = useState<ModelDownloadEntry[]>([]);
  const [downloadTask, setDownloadTask] = useState<ModelDownloadTask>(EMPTY_DOWNLOAD_TASK);
  const [isPollingDownload, setIsPollingDownload] = useState(false);
  const [upscaleModelDownloads, setUpscaleModelDownloads] = useState<UpscaleModelDownloadEntry[]>([]);
  const [upscaleDownloadTask, setUpscaleDownloadTask] = useState<UpscaleModelDownloadTask>(EMPTY_UPSCALE_DOWNLOAD_TASK);
  const [isPollingUpscaleDownload, setIsPollingUpscaleDownload] = useState(false);
  const [upscaleCapabilities, setUpscaleCapabilities] = useState<UpscaleCapabilities | null>(null);
  const [isPollingUpscaleTask, setIsPollingUpscaleTask] = useState(false);
  const frameFetchInFlightRef = useRef(false);
  const pendingFrameRef = useRef<number | null>(null);
  const activeSessionRef = useRef('');
  const lastRenderedFrameRef = useRef<number | null>(null);
  const previewSessionIdRef = useRef('');
  const previousDownloadStateRef = useRef<ModelDownloadTaskState>('idle');
  const previousUpscaleDownloadStateRef = useRef<ModelDownloadTaskState>('idle');
  const lastUpscaleOutputPathRef = useRef('');
  const upscaleDefaultsAppliedRef = useRef(false);

  // 全局状态（zustand）读取：应用级状态 + 打标工作区状态。
  const { t, language } = useI18n();
  const tRef = useRef(t);
  const view = useAppStore((state) => state.view);
  const settings = useAppStore((state) => state.settings);
  const process = useAppStore((state) => state.process);
  const deviceInfo = useAppStore((state) => state.deviceInfo);
  const result = useAppStore((state) => state.result);
  const upscaleConfig = useAppStore((state) => state.upscaleConfig);
  const upscaleTask = useAppStore((state) => state.upscaleTask);
  const upscaleResult = useAppStore((state) => state.upscaleResult);
  const setView = useAppStore((state) => state.setView);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const setSettingsFromBackend = useAppStore((state) => state.setSettingsFromBackend);
  const commitSettings = useAppStore((state) => state.commitSettings);
  const rollbackSettings = useAppStore((state) => state.rollbackSettings);
  const setDeviceInfo = useAppStore((state) => state.setDeviceInfo);
  const updateProcess = useAppStore((state) => state.updateProcess);
  const setResult = useAppStore((state) => state.setResult);
  const clearResult = useAppStore((state) => state.clearResult);
  const updateUpscaleConfig = useAppStore((state) => state.updateUpscaleConfig);
  const setUpscaleTask = useAppStore((state) => state.setUpscaleTask);
  const updateUpscaleTask = useAppStore((state) => state.updateUpscaleTask);
  const resetUpscaleTask = useAppStore((state) => state.resetUpscaleTask);
  const setUpscaleResult = useAppStore((state) => state.setUpscaleResult);
  const clearUpscaleResult = useAppStore((state) => state.clearUpscaleResult);

  const videoPath = useWorkspaceStore((state) => state.videoPath);
  const videoMeta = useWorkspaceStore((state) => state.videoMeta);
  const currentFrame = useWorkspaceStore((state) => state.currentFrame);
  const segments = useWorkspaceStore((state) => state.segments);
  const setVideoPath = useWorkspaceStore((state) => state.setVideoPath);
  const setVideoMeta = useWorkspaceStore((state) => state.setVideoMeta);
  const setCurrentFrame = useWorkspaceStore((state) => state.setCurrentFrame);
  const replaceSegments = useWorkspaceStore((state) => state.replaceSegments);
  const clearSegments = useWorkspaceStore((state) => state.clearSegments);

  const frameMax = useMemo(() => Math.max(0, (videoMeta?.frame_count ?? 1) - 1), [videoMeta?.frame_count]);
  const fps = Math.max(1, videoMeta?.fps ?? 24);
  const semiLocale = language === 'en' ? enUS : zhCN;
  const currentNavIndex = useMemo(() => NAV_ITEMS.findIndex((item) => item.key === view), [view]);

  // 让异步回调里也能拿到最新翻译函数，避免闭包旧值问题。
  useEffect(() => {
    tRef.current = t;
  }, [t]);

  useEffect(() => {
    previewSessionIdRef.current = previewSessionId;
  }, [previewSessionId]);

  useEffect(() => {
    setView('process');
  }, [setView]);

  useEffect(() => {
    // 通过 data 属性驱动全局主题与页面语言。
    document.body.setAttribute('theme-mode', settings.theme);
    document.documentElement.lang = settings.language === 'en' ? 'en' : 'zh-CN';
  }, [settings.theme, settings.language]);

  const closePreviewSession = useCallback(async (sid?: string) => {
    // 统一关闭处理页预览会话，并清理相关尺寸状态。
    const target = sid ?? previewSessionIdRef.current;
    if (!target) return;
    try {
      await pywebviewClient.closeVideoPreviewSession(target);
    } catch {
      // Best effort cleanup.
    }
    if (target === previewSessionIdRef.current) {
      previewSessionIdRef.current = '';
      setPreviewSessionId('');
      setProcessPreviewFrameWidth(0);
      setProcessPreviewFrameHeight(0);
    }
  }, []);

  useEffect(() => {
    return () => {
      void closePreviewSession(previewSessionId);
    };
  }, [closePreviewSession, previewSessionId]);

  useEffect(() => {
    // 在 process/annotate 视图之外暂停帧拉取队列。
    if ((view !== 'process' && view !== 'annotate') || !previewSessionId) {
      activeSessionRef.current = '';
      pendingFrameRef.current = null;
      lastRenderedFrameRef.current = null;
      return;
    }
    activeSessionRef.current = previewSessionId;
    lastRenderedFrameRef.current = null;
  }, [previewSessionId, view]);

  const pumpPreviewFrame = useCallback(async () => {
    // 预览帧“泵”：串行拉取并更新画面，避免并发请求打乱顺序。
    if (frameFetchInFlightRef.current) return;
    const sessionId = activeSessionRef.current;
    if (!sessionId || (view !== 'process' && view !== 'annotate')) return;

    frameFetchInFlightRef.current = true;
    try {
      while (true) {
        const frameToFetch = pendingFrameRef.current;
        pendingFrameRef.current = null;
        if (frameToFetch === null) break;

        const previousFrame = lastRenderedFrameRef.current;
        const useSequentialRead = previousFrame !== null && frameToFetch === previousFrame + 1;
        const resultFrame = useSequentialRead
          // 连续播放时走顺序读取，减少随机 seek 开销。
          ? await pywebviewClient.readVideoPreviewFrame(sessionId)
          : await pywebviewClient.readVideoPreviewFrame(sessionId, frameToFetch);

        if (sessionId !== activeSessionRef.current) break;
        if (resultFrame.success && resultFrame.frame_url) {
          setFrameImageUrl(resultFrame.frame_url);
          if (typeof resultFrame.frame_index === 'number') {
            lastRenderedFrameRef.current = resultFrame.frame_index;
          } else {
            lastRenderedFrameRef.current = frameToFetch;
          }
        }
      }
    } finally {
      frameFetchInFlightRef.current = false;
      if (pendingFrameRef.current !== null) {
        void pumpPreviewFrame();
      }
    }
  }, [view]);

  useEffect(() => {
    // 当前帧变化时触发一次帧拉取请求。
    if ((view !== 'process' && view !== 'annotate') || !previewSessionId) return;
    pendingFrameRef.current = currentFrame;
    void pumpPreviewFrame();
  }, [currentFrame, previewSessionId, pumpPreviewFrame, view]);

  useEffect(() => {
    // 处理页播放循环：按视频 fps 驱动 currentFrame 自增。
    if (view !== 'process' || !isProcessPlaying) return;
    const timer = window.setInterval(() => {
      const next = (currentFrame + 1) > frameMax ? frameMax : currentFrame + 1;
      setCurrentFrame(next);
    }, Math.max(30, Math.round(1000 / fps)));
    return () => window.clearInterval(timer);
  }, [currentFrame, fps, frameMax, isProcessPlaying, setCurrentFrame, view]);

  useEffect(() => {
    if (view === 'process' && currentFrame >= frameMax && isProcessPlaying) {
      setIsProcessPlaying(false);
    }
  }, [currentFrame, frameMax, isProcessPlaying, view]);

  useEffect(() => {
    if (view !== 'process' && isProcessPlaying) {
      setIsProcessPlaying(false);
    }
  }, [isProcessPlaying, view]);

  const loadSettings = useCallback(async () => {
    // 启动时读取后端持久化设置。
    try {
      const response = await pywebviewClient.getSettings();
      const normalized = normalizeSettingsFromBackend(response);
      setSettingsFromBackend(normalized);
    } catch {
      Toast.error(tRef.current('toast.loadSettingsFailed'));
    }
  }, [setSettingsFromBackend]);

  const refreshDeviceInfo = useCallback(async () => {
    // 周期刷新设备信息，处理期间也能看到内存变化。
    if (isSelectingVideo) return;
    try {
      const info = await pywebviewClient.getDeviceInfo();
      setDeviceInfo({
        device: info.device,
        memory: info.memory,
        supportsFp16: !!info.supports_fp16,
      });
    } catch {
      // Ignore polling errors.
    }
  }, [isSelectingVideo, setDeviceInfo]);

  const refreshModelDownloadStatus = useCallback(async () => {
    // 拉取模型下载列表和任务状态，并处理状态变化提示。
    try {
      const response = await pywebviewClient.getModelDownloadStatus();
      if (!response.success) {
        return;
      }

      const normalizedModels: ModelDownloadEntry[] = (Array.isArray(response.models) ? response.models : [])
        .map((entry) => {
          const rawModelId = String(entry?.model_id || '').toLowerCase();
          const model_id: AppSettings['modelId'] = (
            rawModelId === 'lama_roi' || rawModelId === 'propainter_roi'
          )
            ? rawModelId
            : 'lama_roi';

          return {
            ...entry,
            model_id,
            display_name: String(entry.display_name || model_id),
            installed: !!entry.installed,
            can_redownload: entry.can_redownload !== false,
            install_hint: String(entry.install_hint || ''),
          };
        });
      setModelDownloads(normalizedModels);

      const nextTask = normalizeDownloadTask(response.task);
      setDownloadTask(nextTask);
      setIsPollingDownload(nextTask.state === 'running');

      const previousState = previousDownloadStateRef.current;
      // 只在 running -> 终态 时弹一次结果提示，避免重复通知。
      if (previousState === 'running') {
        if (nextTask.state === 'success') {
          Toast.success(tRef.current('toast.modelDownloadSuccess'));
        } else if (nextTask.state === 'failed') {
          Toast.error(`${tRef.current('toast.modelDownloadFailed')}: ${nextTask.error || nextTask.message}`);
        } else if (nextTask.state === 'cancelled') {
          Toast.info(tRef.current('toast.modelDownloadCancelled'));
        }
      }
      previousDownloadStateRef.current = nextTask.state;
    } catch {
      // Ignore polling failures, next poll will retry.
    }
  }, []);

  const loadUpscaleCapabilities = useCallback(async (forceRefresh = false) => {
    // 获取 AI 放大能力列表（引擎/模型/默认参数）。
    try {
      const response = await pywebviewClient.getUpscaleCapabilities(forceRefresh);
      if (!response.success) {
        setUpscaleCapabilities(null);
        return;
      }
      setUpscaleCapabilities(response);
      if (response.defaults && !upscaleDefaultsAppliedRef.current) {
        updateUpscaleConfig({
          mode: response.defaults.mode,
          engine: response.defaults.engine,
          modelId: response.defaults.model_id,
          targetPreset: response.defaults.target_preset,
          sameResStrength: response.defaults.same_res_strength,
          denoiseStrength: response.defaults.denoise_strength,
          keepAudio: !!response.defaults.keep_audio,
        });
        upscaleDefaultsAppliedRef.current = true;
      }
    } catch {
      setUpscaleCapabilities(null);
    }
  }, [updateUpscaleConfig]);

  const refreshUpscaleModelDownloadStatus = useCallback(async () => {
    // 拉取 AI 放大模型下载状态，并在终态时刷新能力列表。
    try {
      const response = await pywebviewClient.getUpscaleModelDownloadStatus();
      if (!response.success) {
        return;
      }

      const normalizedModels: UpscaleModelDownloadEntry[] = (Array.isArray(response.models) ? response.models : [])
        .map((entry) => {
          const modelId = normalizeUpscaleModelId(entry?.model_id);
          return {
            ...entry,
            model_id: modelId,
            display_name: String(entry.display_name || modelId),
            installed: !!entry.installed,
            can_redownload: entry.can_redownload !== false,
            install_hint: String(entry.install_hint || ''),
          };
        });
      setUpscaleModelDownloads(normalizedModels);

      const nextTask = normalizeUpscaleDownloadTask(response.task);
      setUpscaleDownloadTask(nextTask);
      setIsPollingUpscaleDownload(nextTask.state === 'running');

      const previousState = previousUpscaleDownloadStateRef.current;
      if (previousState === 'running') {
        if (nextTask.state === 'success') {
          Toast.success(tRef.current('toast.upscaleModelDownloadSuccess'));
          void loadUpscaleCapabilities(true);
        } else if (nextTask.state === 'failed') {
          Toast.error(`${tRef.current('toast.upscaleModelDownloadFailed')}: ${nextTask.error || nextTask.message}`);
        } else if (nextTask.state === 'cancelled') {
          Toast.info(tRef.current('toast.upscaleModelDownloadCancelled'));
        }
      }
      previousUpscaleDownloadStateRef.current = nextTask.state;
    } catch {
      // Ignore polling failures, next poll will retry.
    }
  }, [loadUpscaleCapabilities]);

  const refreshUpscaleTaskStatus = useCallback(async () => {
    // 拉取放大任务状态，并在成功后刷新“AI 放大结果”页数据。
    try {
      const response = await pywebviewClient.getUpscaleTaskStatus();
      if (!response.success) return;
      const task = normalizeUpscaleTask(response.task);
      setUpscaleTask(task);
      setIsPollingUpscaleTask(task.state === 'running');

      if (task.state === 'success' && task.outputPath && lastUpscaleOutputPathRef.current !== task.outputPath) {
        lastUpscaleOutputPathRef.current = task.outputPath;
        try {
          const mediaInfo = await pywebviewClient.getMediaInfo(task.outputPath);
          const playbackPath = task.previewPath || task.outputPath;
          setUpscaleResult({
            outputPath: task.outputPath,
            outputUrl: toFileUrl(task.outputPath),
            previewUrl: toFileUrl(playbackPath),
            width: Number(mediaInfo.width || 0),
            height: Number(mediaInfo.height || 0),
            fps: Number(mediaInfo.fps || 0),
            frameCount: Number(mediaInfo.frame_count || 0),
            mode: task.mode,
            engine: task.effectiveEngine || task.engine,
            modelId: task.modelId,
            warning: task.warning,
          });
        } catch {
          setUpscaleResult({
            outputPath: task.outputPath,
            outputUrl: toFileUrl(task.outputPath),
            previewUrl: toFileUrl(task.previewPath || task.outputPath),
            width: 0,
            height: 0,
            fps: 0,
            frameCount: 0,
            mode: task.mode,
            engine: task.effectiveEngine || task.engine,
            modelId: task.modelId,
            warning: task.warning,
          });
        }
      }

      if (task.state === 'success' && task.outputPath) {
        setView('upscale');
      }
    } catch {
      // Ignore polling failures.
    }
  }, [setUpscaleResult, setUpscaleTask, setView]);

  useEffect(() => {
    // 应用首次加载：同步设置、设备信息和下载状态。
    void loadSettings();
    void refreshDeviceInfo();
    void refreshModelDownloadStatus();
    void refreshUpscaleModelDownloadStatus();
    void loadUpscaleCapabilities();
    void refreshUpscaleTaskStatus();
    const timer = window.setInterval(() => {
      void refreshDeviceInfo();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [
    loadSettings,
    loadUpscaleCapabilities,
    refreshDeviceInfo,
    refreshModelDownloadStatus,
    refreshUpscaleModelDownloadStatus,
    refreshUpscaleTaskStatus,
  ]);

  useEffect(() => {
    // 设置面板打开时主动刷新一次，确保显示最新下载信息。
    if (!isSettingsPanelOpen) return;
    void refreshModelDownloadStatus();
  }, [isSettingsPanelOpen, refreshModelDownloadStatus]);

  useEffect(() => {
    // 下载中高频轮询状态，让进度条更平滑。
    if (!isPollingDownload) return;
    const timer = window.setInterval(() => {
      void refreshModelDownloadStatus();
    }, 400);
    return () => window.clearInterval(timer);
  }, [isPollingDownload, refreshModelDownloadStatus]);

  useEffect(() => {
    // AI 放大模型下载中轮询状态。
    if (!isPollingUpscaleDownload) return;
    const timer = window.setInterval(() => {
      void refreshUpscaleModelDownloadStatus();
    }, 400);
    return () => window.clearInterval(timer);
  }, [isPollingUpscaleDownload, refreshUpscaleModelDownloadStatus]);

  useEffect(() => {
    // 放大任务运行中轮询状态。
    if (!isPollingUpscaleTask) return;
    const timer = window.setInterval(() => {
      void refreshUpscaleTaskStatus();
    }, 500);
    return () => window.clearInterval(timer);
  }, [isPollingUpscaleTask, refreshUpscaleTaskStatus]);

  useEffect(() => {
    // 监听后端推送的处理进度事件（wmr-progress）。
    const onProgress = (event: Event) => {
      const custom = event as CustomEvent<Record<string, unknown>>;
      const detail = custom.detail || {};
      const next: Record<string, unknown> = {};

      if (typeof detail.progress === 'number') {
        next.progress = Math.min(1, Math.max(0, Number(detail.progress)));
      }
      if (typeof detail.processed_frames === 'number') {
        next.processedFrames = Number(detail.processed_frames);
      }
      if (typeof detail.total_frames === 'number') {
        next.totalFrames = Number(detail.total_frames);
      }
      if (typeof detail.estimated_time === 'string') {
        next.estimatedTime = detail.estimated_time;
      }
      if (typeof detail.eta_seconds === 'number') {
        next.etaSeconds = Math.max(0, Number(detail.eta_seconds));
      }
      if (typeof detail.throughput_fps === 'number') {
        next.throughputFps = Math.max(0, Number(detail.throughput_fps));
      }
      if (typeof detail.phase === 'string') {
        next.phase = String(detail.phase);
      }
      if (typeof detail.opaque_infer === 'boolean') {
        next.opaqueInfer = detail.opaque_infer;
      }
      if (typeof detail.status === 'string') {
        next.statusMessage = normalizeStatusText(detail.status, t);
      } else if (typeof detail.message === 'string') {
        next.statusMessage = normalizeStatusText(detail.message, t);
      }

      if (Object.keys(next).length > 0) {
        updateProcess(next);
      }
    };

    window.addEventListener('wmr-progress', onProgress as EventListener);
    return () => window.removeEventListener('wmr-progress', onProgress as EventListener);
  }, [t, updateProcess]);

  const loadAnnotations = useCallback(async (pathOverride?: string) => {
    // 从 sidecar 加载标记段；可指定路径覆盖当前视频路径。
    const targetPath = pathOverride ?? videoPath;
    if (!targetPath) return;

    try {
      const loaded = await pywebviewClient.loadAnnotations(targetPath);
      if (!loaded.success) {
        Toast.error(loaded.error ?? t('toast.loadAnnotationsFailed'));
        return;
      }
      replaceSegments(Array.isArray(loaded.segments) ? loaded.segments : []);
      if (loaded.video_meta) {
        setVideoMeta(loaded.video_meta);
      }
      if (loaded.warning) {
        Toast.warning(loaded.warning);
      }
    } catch {
      Toast.error(t('toast.loadAnnotationsFailed'));
    }
  }, [replaceSegments, setVideoMeta, t, videoPath]);

  const saveAnnotations = useCallback(async () => {
    // 把当前工作区标记段保存到磁盘 sidecar。
    if (!videoPath) return;
    try {
      const saved = await pywebviewClient.saveAnnotations({
        video_path: videoPath,
        segments,
        video_meta: videoMeta ?? EMPTY_META,
      });
      if (!saved.success) {
        Toast.error(saved.error ?? t('toast.saveAnnotationsFailed'));
        return;
      }
      replaceSegments(Array.isArray(saved.segments) ? saved.segments : segments);
      Toast.success(t('toast.saveAnnotationsSuccess'));
    } catch {
      Toast.error(t('toast.saveAnnotationsFailed'));
    }
  }, [replaceSegments, segments, t, videoMeta, videoPath]);

  const clearAnnotationsOnDisk = useCallback(async () => {
    // 先清前端，再尝试清理磁盘 sidecar。
    clearSegments();
    if (!videoPath) return;
    try {
      const resultDelete = await pywebviewClient.deleteAnnotations(videoPath);
      if (!resultDelete.success) {
        Toast.warning(resultDelete.error ?? t('toast.deleteAnnotationsFailed'));
        return;
      }
      Toast.success(t('toast.deleteAnnotationsSuccess'));
    } catch {
      Toast.warning(t('toast.deleteAnnotationsFailed'));
    }
  }, [clearSegments, t, videoPath]);

  const waitForDialogResult = useCallback(async (requestId: string, timeoutMs = 300000) => {
    // 统一处理异步文件对话框轮询与超时控制。
    const startedAt = Date.now();

    while (true) {
      const resultDialog = await pywebviewClient.pollDialogResult(requestId);
      if (resultDialog.done) {
        try {
          await pywebviewClient.clearDialogResult(requestId);
        } catch {
          // Best effort cleanup.
        }

        if (!resultDialog.success) {
          throw new Error(resultDialog.error || t('toast.dialogFailed'));
        }
        if (resultDialog.cancelled || !resultDialog.path) {
          return null;
        }
        return { path: resultDialog.path };
      }

      if (Date.now() - startedAt > timeoutMs) {
        try {
          await pywebviewClient.clearDialogResult(requestId);
        } catch {
          // Best effort cleanup.
        }
        throw new Error(t('toast.dialogTimeout'));
      }

      await sleep(120);
    }
  }, [t]);

  const ensurePreviewSession = useCallback(async (path: string, fpsHint?: number): Promise<string> => {
    // 懒创建处理页预览会话，已有会话时直接复用。
    if (previewSessionIdRef.current) return previewSessionIdRef.current;
    const session = await pywebviewClient.openVideoPreviewSession(
      path,
      Math.max(15, Math.round(fpsHint || 24)),
      1280,
    );
    if (!session.success || !session.session_id) {
      Toast.error(session.error ?? tRef.current('toast.previewSessionFailed'));
      return '';
    }
    previewSessionIdRef.current = session.session_id;
    setPreviewSessionId(session.session_id);
    setProcessPreviewFrameWidth(Math.max(0, Number(session.width || 0)));
    setProcessPreviewFrameHeight(Math.max(0, Number(session.height || 0)));
    return session.session_id;
  }, []);

  useEffect(() => {
    if (!videoPath) return;
    if (view !== 'process' && view !== 'annotate') return;
    if (previewSessionIdRef.current) return;
    void ensurePreviewSession(videoPath, videoMeta?.fps);
  }, [ensurePreviewSession, videoMeta?.fps, videoPath, view]);

  const selectVideo = useCallback(async () => {
    // 选择视频后的主流程：校验 -> 重置状态 -> 打开预览 -> 加载标注。
    if (isSelectingVideo) return;
    setIsSelectingVideo(true);
    try {
      const begin = await pywebviewClient.beginSelectFile();
      if (!begin.success || !begin.request_id) {
        throw new Error(begin.error || t('toast.dialogFailed'));
      }

      const pickedTimed = await waitForDialogResult(begin.request_id);
      if (!pickedTimed?.path) return;

      const mediaInfo = await pywebviewClient.getMediaInfo(pickedTimed.path);
      if (!mediaInfo.success || mediaInfo.type !== 'video') {
        Toast.warning(t('toast.selectVideoFirst'));
        return;
      }

      const meta: VideoMeta = {
        path: pickedTimed.path,
        basename: basename(pickedTimed.path),
        sha1: '',
        size: 0,
        mtime_ns: 0,
        width: Number(mediaInfo.width ?? 1280),
        height: Number(mediaInfo.height ?? 720),
        fps: Number(mediaInfo.fps ?? 30),
        frame_count: Number(mediaInfo.frame_count ?? 1),
      };

      await closePreviewSession();
      clearSegments();
      clearResult();
      clearUpscaleResult();
      resetUpscaleTask();
      lastUpscaleOutputPathRef.current = '';
      setVideoPath(pickedTimed.path);
      setVideoMeta(meta);
      setCurrentFrame(0);
      setFrameImageUrl('');
      setIsProcessPlaying(false);

      const sid = await ensurePreviewSession(pickedTimed.path, meta.fps);
      if (!sid) {
        return;
      }

      await loadAnnotations(pickedTimed.path);
      setView('process');
      Toast.success(t('toast.videoImported'));
    } catch (error) {
      const msg = (error as Error).message || t('toast.selectVideoFirst');
      Toast.error(msg);
    } finally {
      setIsSelectingVideo(false);
    }
  }, [
    clearResult,
    clearUpscaleResult,
    clearSegments,
    closePreviewSession,
    ensurePreviewSession,
    isSelectingVideo,
    loadAnnotations,
    resetUpscaleTask,
    setCurrentFrame,
    setVideoMeta,
    setVideoPath,
    setView,
    t,
    waitForDialogResult,
  ]);

  const selectOutputFolder = useCallback(async () => {
    // 选择输出目录并更新设置草稿。
    try {
      const begin = await pywebviewClient.beginSelectFolder();
      if (!begin.success || !begin.request_id) {
        throw new Error(begin.error || t('toast.dialogFailed'));
      }

      const resultFolder = await waitForDialogResult(begin.request_id);
      if (resultFolder?.path) {
        updateSettings({ outputPath: resultFolder.path });
      }
    } catch (error) {
      const msg = (error as Error).message || t('toast.dialogFailed');
      Toast.error(msg);
    }
  }, [t, updateSettings, waitForDialogResult]);

  const startModelDownload = useCallback(async (modelId: AppSettings['modelId'], force: boolean) => {
    // 启动模型下载，并立即把前端状态切到 running。
    try {
      const response = await pywebviewClient.startModelDownload(modelId, force);
      if (!response.success) {
        Toast.error(response.error || t('toast.modelDownloadFailed'));
        return;
      }

      previousDownloadStateRef.current = 'running';
      setDownloadTask((prev) => ({
        ...prev,
        state: 'running',
        model_id: modelId,
        message: t('toast.modelDownloadStart'),
        error: '',
      }));
      setIsPollingDownload(true);
      Toast.info(t('toast.modelDownloadStart'));
      await refreshModelDownloadStatus();
    } catch {
      Toast.error(t('toast.modelDownloadFailed'));
    }
  }, [refreshModelDownloadStatus, t]);

  const cancelModelDownload = useCallback(async () => {
    // 请求取消模型下载，后续状态由轮询结果驱动。
    try {
      const response = await pywebviewClient.cancelModelDownload();
      if (!response.success) {
        Toast.error(response.error || t('toast.modelDownloadFailed'));
        return;
      }
      setIsPollingDownload(true);
      Toast.info(t('toast.modelDownloadCancelRequested'));
      await refreshModelDownloadStatus();
    } catch {
      Toast.error(t('toast.modelDownloadFailed'));
    }
  }, [refreshModelDownloadStatus, t]);

  const startUpscaleModelDownload = useCallback(async (modelId: UpscaleModelId, force: boolean) => {
    // 启动 AI 放大模型下载，并立即切换到 polling。
    try {
      const response = await pywebviewClient.startUpscaleModelDownload(modelId, force);
      if (!response.success) {
        Toast.error(response.error || t('toast.upscaleModelDownloadFailed'));
        return;
      }

      previousUpscaleDownloadStateRef.current = 'running';
      setUpscaleDownloadTask((prev) => ({
        ...prev,
        state: 'running',
        model_id: modelId,
        message: t('toast.upscaleModelDownloadStart'),
        error: '',
      }));
      setIsPollingUpscaleDownload(true);
      Toast.info(t('toast.upscaleModelDownloadStart'));
      await refreshUpscaleModelDownloadStatus();
    } catch {
      Toast.error(t('toast.upscaleModelDownloadFailed'));
    }
  }, [refreshUpscaleModelDownloadStatus, t]);

  const cancelUpscaleModelDownload = useCallback(async () => {
    // 请求取消 AI 放大模型下载。
    try {
      const response = await pywebviewClient.cancelUpscaleModelDownload();
      if (!response.success) {
        Toast.error(response.error || t('toast.upscaleModelDownloadFailed'));
        return;
      }
      setIsPollingUpscaleDownload(true);
      Toast.info(t('toast.upscaleModelDownloadCancelRequested'));
      await refreshUpscaleModelDownloadStatus();
    } catch {
      Toast.error(t('toast.upscaleModelDownloadFailed'));
    }
  }, [refreshUpscaleModelDownloadStatus, t]);

  const startProcessing = useCallback(async () => {
    // 处理按钮主流程：先保存标注，再调用后端处理，再生成结果页数据。
    if (!videoPath || process.isProcessing) return;
    const enabledSegments = segments.filter((seg) => seg.enabled !== false);
    if (enabledSegments.length <= 0) {
      Toast.warning(t('toast.noSegments'));
      setView('annotate');
      return;
    }

    try {
      const saved = await pywebviewClient.saveAnnotations({
        video_path: videoPath,
        segments,
        video_meta: videoMeta ?? EMPTY_META,
      });
      if (!saved.success) {
        Toast.error(saved.error ?? t('toast.saveAnnotationsFailed'));
        return;
      }

      updateProcess({
        isProcessing: true,
        progress: 0,
        statusMessage: t('status.running'),
        processedFrames: 0,
        totalFrames: videoMeta?.frame_count ?? 0,
        estimatedTime: '--:--',
        etaSeconds: undefined,
        throughputFps: undefined,
        phase: 'prepare',
        opaqueInfer: false,
      });

      const processingResult = await pywebviewClient.processVideo({
        input_path: videoPath,
        output_path: settings.outputPath,
        annotation_segments: enabledSegments,
        settings: {
          model_id: settings.modelId,
        },
      });

      if (!processingResult.success || !processingResult.output_path) {
        throw new Error(processingResult.error || t('toast.processFailed'));
      }
      if (processingResult.model_warning) {
        Toast.warning(processingResult.model_warning);
      }

      const mediaInfo = await pywebviewClient.getMediaInfo(processingResult.output_path);
      let resultPlaybackPath = processingResult.output_path;

      if (mediaInfo.type === 'video') {
        // 结果视频优先准备一个更稳的预览版本（可能是转码缓存）。
        try {
          const preparedPreview = await pywebviewClient.prepareVideoPreview(processingResult.output_path);
          if (preparedPreview.success && preparedPreview.path) {
            resultPlaybackPath = preparedPreview.path;
          } else if (preparedPreview.error) {
            Toast.warning(preparedPreview.error);
          }
          if (preparedPreview.warning) {
            Toast.warning(preparedPreview.warning);
          }
        } catch {
          // Keep original output path as playback source when preview preparation fails.
        }
      }

      setResult({
        outputPath: processingResult.output_path,
        outputUrl: toFileUrl(resultPlaybackPath),
        mediaType: mediaInfo.type === 'image' ? 'image' : 'video',
        width: Number(mediaInfo.width ?? 0),
        height: Number(mediaInfo.height ?? 0),
        fps: Number(mediaInfo.fps ?? 0),
        frameCount: Number(mediaInfo.frame_count ?? 0),
        modelId: normalizeModelId(processingResult.effective_model_id || settings.modelId),
      });
      clearUpscaleResult();
      resetUpscaleTask();
      lastUpscaleOutputPathRef.current = '';

      updateProcess({
        isProcessing: false,
        progress: 1,
        statusMessage: t('status.done'),
        estimatedTime: '00:00',
        etaSeconds: 0,
        phase: 'finalize',
        opaqueInfer: false,
      });
      setView('result');
      Toast.success(t('toast.processDone'));
    } catch (error) {
      updateProcess({
        isProcessing: false,
        statusMessage: t('status.failed'),
        opaqueInfer: false,
      });
      Toast.error(`${t('toast.processFailed')}: ${(error as Error).message}`);
    }
  }, [
    clearUpscaleResult,
    process.isProcessing,
    resetUpscaleTask,
    segments,
    settings.outputPath,
    settings.modelId,
    setResult,
    setView,
    t,
    updateProcess,
    videoMeta,
    videoPath,
  ]);

  const stopProcessing = useCallback(async () => {
    // 主动停止后端处理，并把前端状态回到 idle。
    try {
      await pywebviewClient.stopProcessing();
      updateProcess({
        isProcessing: false,
        statusMessage: t('status.idle'),
        opaqueInfer: false,
      });
      Toast.info(t('toast.processStopped'));
    } catch {
      Toast.error(t('toast.processFailed'));
    }
  }, [t, updateProcess]);

  const openOutputDir = useCallback(async () => {
    // 调用系统文件管理器打开输出目录。
    try {
      await pywebviewClient.openOutputDir();
    } catch {
      Toast.warning(t('toast.resultEmpty'));
    }
  }, [t]);

  const startUpscaleTask = useCallback(async () => {
    // 手动触发 AI 放大任务（独立于去水印主流程）。
    if (!result.outputPath) {
      Toast.warning(t('toast.resultEmpty'));
      return;
    }
    if (!upscaleConfig.enabled) {
      Toast.warning(t('upscale.enableHint'));
      return;
    }

    const selectedEngine = upscaleConfig.engine;
    const selectedEngineCapability = (upscaleCapabilities?.engines || []).find((item) => item.engine === selectedEngine);
    if (selectedEngineCapability && !selectedEngineCapability.available) {
      const localizedReason = selectedEngineCapability.reason
        ? localizeSeedVRError(selectedEngineCapability.reason, t)
        : t('upscale.engineUnavailableHint');
      Toast.warning(localizedReason);
      return;
    }
    const selectedModel: UpscaleModelId = upscaleConfig.modelId;
    const selectedModelInstalled = (
      upscaleModelDownloads.find((entry) => entry.model_id === selectedModel)?.installed
      ?? upscaleCapabilities?.models?.find((entry) => entry.model_id === selectedModel)?.installed
      ?? false
    );
    if (!selectedModelInstalled) {
      Toast.warning(t('upscale.modelNotReadyHint'));
      return;
    }
    if (upscaleDownloadTask.state === 'running') {
      Toast.warning(t('upscale.modelDownloading'));
      return;
    }

    clearUpscaleResult();
    resetUpscaleTask();
    lastUpscaleOutputPathRef.current = '';
    updateUpscaleTask({
      state: 'running',
      progress: 0,
      phase: 'prepare',
      message: t('upscale.status.preparing'),
      error: '',
      warning: '',
      inputPath: result.outputPath,
      mode: upscaleConfig.mode,
      engine: selectedEngine,
      modelId: selectedModel,
    });

    const payload = {
      input_path: result.outputPath,
      output_dir: settings.outputPath,
      mode: upscaleConfig.mode,
      engine: selectedEngine,
      model_id: selectedModel,
      target_preset: upscaleConfig.mode === 'upscale_resolution' ? upscaleConfig.targetPreset : undefined,
      same_res_strength: upscaleConfig.mode === 'enhance_same_resolution' ? upscaleConfig.sameResStrength : undefined,
      denoise_strength: upscaleConfig.denoiseStrength,
      keep_audio: upscaleConfig.keepAudio,
    };

    try {
      const response = await pywebviewClient.startUpscale(payload);
      if (!response.success) {
        const errorText = localizeSeedVRError(response.error || t('upscale.status.failed'), t);
        updateUpscaleTask({
          state: 'failed',
          message: t('upscale.status.failed'),
          error: errorText,
        });
        Toast.error(errorText);
        return;
      }
      setIsPollingUpscaleTask(true);
      Toast.info(t('upscale.startRequested'));
      await refreshUpscaleTaskStatus();
    } catch {
      updateUpscaleTask({
        state: 'failed',
        message: t('upscale.status.failed'),
        error: t('upscale.status.failed'),
      });
      Toast.error(t('upscale.status.failed'));
    }
  }, [
    clearUpscaleResult,
    refreshUpscaleTaskStatus,
    resetUpscaleTask,
    result.outputPath,
    settings.outputPath,
    t,
    updateUpscaleTask,
    upscaleConfig.denoiseStrength,
    upscaleConfig.enabled,
    upscaleConfig.engine,
    upscaleConfig.keepAudio,
    upscaleConfig.mode,
    upscaleConfig.modelId,
    upscaleConfig.sameResStrength,
    upscaleConfig.targetPreset,
    upscaleCapabilities?.engines,
    upscaleDownloadTask.state,
    upscaleCapabilities?.models,
    upscaleModelDownloads,
  ]);

  const cancelUpscaleTask = useCallback(async () => {
    // 请求取消当前运行中的 AI 放大任务。
    try {
      const response = await pywebviewClient.cancelUpscaleTask();
      if (!response.success) {
        Toast.error(response.error || t('upscale.cancelFailed'));
        return;
      }
      setIsPollingUpscaleTask(true);
      Toast.info(t('upscale.cancelRequested'));
      await refreshUpscaleTaskStatus();
    } catch {
      Toast.error(t('upscale.cancelFailed'));
    }
  }, [refreshUpscaleTaskStatus, t]);

  const saveSettings = useCallback(async () => {
    // 保存设置：失败时回滚到上一次已持久化值。
    setIsSavingSettings(true);
    try {
      const response = await pywebviewClient.saveSettings({
        language: settings.language,
        theme: settings.theme,
        output: {
          path: settings.outputPath,
          model_id: settings.modelId,
        },
      });
      if (!response.success) {
        rollbackSettings();
        Toast.error(t('toast.saveSettingsFailed'));
        return;
      }
      commitSettings();
      Toast.success(t('toast.saveSettingsSuccess'));
    } catch {
      rollbackSettings();
      Toast.error(t('toast.saveSettingsFailed'));
    } finally {
      setIsSavingSettings(false);
    }
  }, [commitSettings, rollbackSettings, settings.language, settings.modelId, settings.outputPath, settings.theme, t]);

  const toggleManualPanel = useCallback(() => {
    // 说明书与设置面板互斥：打开说明书时自动关闭设置。
    setIsManualPanelOpen((open) => {
      const nextOpen = !open;
      if (nextOpen) setIsSettingsPanelOpen(false);
      return nextOpen;
    });
  }, []);

  const toggleSettingsPanel = useCallback(() => {
    // 说明书与设置面板互斥：打开设置时自动关闭说明书。
    setIsSettingsPanelOpen((open) => {
      const nextOpen = !open;
      if (nextOpen) setIsManualPanelOpen(false);
      return nextOpen;
    });
  }, []);

  const renderView = () => {
    // 按当前标签渲染对应主内容。
    if (view === 'annotate') {
      return (
        <AnnotationWorkspace
          frameImageUrl={frameImageUrl}
          previewFrameWidth={processPreviewFrameWidth}
          previewFrameHeight={processPreviewFrameHeight}
          onSaveAnnotations={saveAnnotations}
          onClearAnnotations={clearAnnotationsOnDisk}
        />
      );
    }

    if (view === 'result') {
      return (
        <ResultView
          result={result}
          sourceVideoPath={videoPath}
          upscaleConfig={upscaleConfig}
          upscaleTask={upscaleTask}
          upscaleCapabilities={upscaleCapabilities}
          upscaleModelDownloads={upscaleModelDownloads}
          upscaleDownloadTask={upscaleDownloadTask}
          onOpenOutputDir={openOutputDir}
          onUpscaleConfigChange={updateUpscaleConfig}
          onStartUpscaleModelDownload={startUpscaleModelDownload}
          onCancelUpscaleModelDownload={cancelUpscaleModelDownload}
          onStartUpscale={startUpscaleTask}
          onCancelUpscale={cancelUpscaleTask}
          onOpenUpscaleResult={() => setView('upscale')}
        />
      );
    }

    if (view === 'upscale') {
      return (
        <UpscaleView
          sourceResult={result}
          upscaleResult={upscaleResult}
          upscaleTask={upscaleTask}
          onOpenOutputDir={openOutputDir}
          onBackToResult={() => setView('result')}
        />
      );
    }

    return (
      <ProcessView
        videoPath={videoPath}
        videoMeta={videoMeta}
        frameImageUrl={frameImageUrl}
        previewFrameWidth={processPreviewFrameWidth}
        previewFrameHeight={processPreviewFrameHeight}
        currentFrame={currentFrame}
        isSelectingVideo={isSelectingVideo}
        isPlaying={isProcessPlaying}
        settings={settings}
        process={process}
        deviceInfo={deviceInfo}
        segments={segments}
        onSelectVideo={selectVideo}
        onSetCurrentFrame={setCurrentFrame}
        onTogglePlay={() => {
          if (!videoPath) return;

          if (isProcessPlaying) {
            setIsProcessPlaying(false);
            return;
          }

          if (currentFrame >= frameMax) {
            setCurrentFrame(0);
          }
          setIsProcessPlaying(true);
          void ensurePreviewSession(videoPath, videoMeta?.fps);
        }}
        onSelectOutputFolder={selectOutputFolder}
        onChangeModelId={(modelId) => updateSettings({ modelId })}
        onStartProcessing={startProcessing}
        onStopProcessing={stopProcessing}
        onGoAnnotate={() => setView('annotate')}
      />
    );
  };

  return (
    // 顶部导航 + 主内容 + 右侧说明书/设置面板。
    <ConfigProvider locale={semiLocale}>
      <Layout className="app-shell">
        <Header className="top-nav-header">
          <div className="top-nav-left">
            <Title heading={5} style={{ margin: 0 }}>{t('app.title')}</Title>
            <div className="top-nav-flow">
              {NAV_ITEMS.map((item, index) => (
                <div className="top-nav-step" key={item.key}>
                  <Button
                    className="top-nav-tab-btn"
                    theme={view === item.key ? 'solid' : 'light'}
                    onClick={() => setView(item.key)}
                  >
                    {t(item.labelKey)}
                  </Button>
                  {index < NAV_ITEMS.length - 1 ? (
                    <span
                      aria-hidden="true"
                      className={`top-nav-step-arrow${
                        index === currentNavIndex && currentNavIndex < NAV_ITEMS.length - 1
                          ? ' is-active'
                          : ''
                      }`}
                    />
                  ) : null}
                </div>
              ))}
            </div>
          </div>
          <div className="top-nav-actions">
            <Button
              icon={<IconBulb />}
              theme={isManualPanelOpen ? 'solid' : 'borderless'}
              onClick={toggleManualPanel}
              aria-label={t('nav.manual')}
            />
            <Button
              icon={<IconSetting />}
              theme={isSettingsPanelOpen ? 'solid' : 'borderless'}
              onClick={toggleSettingsPanel}
              aria-label={t('nav.settings')}
            />
          </div>
        </Header>
        <Content className="app-main-content">{renderView()}</Content>
        <SideSheet
          className="manual-sidesheet"
          title={t('manual.title')}
          visible={isManualPanelOpen}
          placement="right"
          mask
          maskClosable
          closeOnEsc
          width="clamp(320px, 40vw, 680px)"
          bodyStyle={{ padding: 0 }}
          onCancel={() => setIsManualPanelOpen(false)}
        >
          <ManualView />
        </SideSheet>
        <SideSheet
          className="settings-sidesheet"
          title={t('settings.title')}
          visible={isSettingsPanelOpen}
          placement="right"
          mask
          maskClosable
          closeOnEsc
          width="clamp(320px, 30vw, 460px)"
          bodyStyle={{ padding: 0 }}
          onCancel={() => setIsSettingsPanelOpen(false)}
        >
          <SettingsView
            settings={settings}
            saving={isSavingSettings}
            modelDownloads={modelDownloads}
            downloadTask={downloadTask}
            onChangeLanguage={(value) => updateSettings({ language: value })}
            onChangeTheme={(value) => updateSettings({ theme: value })}
            onChangeOutputPath={(value) => updateSettings({ outputPath: value })}
            onSelectOutputFolder={selectOutputFolder}
            onStartModelDownload={startModelDownload}
            onCancelModelDownload={cancelModelDownload}
            onSave={saveSettings}
            onReset={rollbackSettings}
          />
        </SideSheet>
      </Layout>
    </ConfigProvider>
  );
}
