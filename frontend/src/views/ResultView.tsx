import { useEffect, useMemo } from 'react';
import { Button, Card, Progress, Select, Slider, Space, Switch, Tag, Typography } from '@douyinfe/semi-ui';
import type { ResultState, UpscaleCapabilities, UpscaleConfig, UpscaleTaskState } from '../types/app';
import { useI18n } from '../i18n/useI18n';
import type { UpscaleModelDownloadEntry, UpscaleModelDownloadTask } from '../services/pywebview';
import { FrameComparePreview } from '../components/FrameComparePreview';

const { Text, Title } = Typography;

interface ResultViewProps {
  result: ResultState;
  onOpenOutputDir: () => void;
  sourceVideoPath?: string;
  upscaleConfig: UpscaleConfig;
  upscaleTask: UpscaleTaskState;
  upscaleCapabilities: UpscaleCapabilities | null;
  upscaleModelDownloads: UpscaleModelDownloadEntry[];
  upscaleDownloadTask: UpscaleModelDownloadTask;
  onUpscaleConfigChange: (patch: Partial<UpscaleConfig>) => void;
  onStartUpscaleModelDownload: (modelId: UpscaleConfig['modelId'], force: boolean) => void;
  onCancelUpscaleModelDownload: () => void;
  onStartUpscale: () => void;
  onCancelUpscale: () => void;
  onOpenUpscaleResult: () => void;
}

export function ResultView({
  result,
  onOpenOutputDir,
  sourceVideoPath,
  upscaleConfig,
  upscaleTask,
  upscaleCapabilities,
  upscaleModelDownloads,
  upscaleDownloadTask,
  onUpscaleConfigChange,
  onStartUpscaleModelDownload,
  onCancelUpscaleModelDownload,
  onStartUpscale,
  onCancelUpscale,
  onOpenUpscaleResult,
}: ResultViewProps) {
  const { t } = useI18n();
  const hasResult = !!result.outputUrl;
  const isVideo = result.mediaType === 'video';
  const modelLabel =
    result.modelId === 'lama_roi'
      ? t('process.model.lama')
      : result.modelId === 'propainter_roi'
        ? t('process.model.propainter')
        : t('process.model.lama');
  const engineOptionsRaw = (upscaleCapabilities?.engines && upscaleCapabilities.engines.length > 0)
    ? upscaleCapabilities.engines
    : [
        {
          engine: 'realesrgan' as const,
          display_name: 'Real-ESRGAN',
          available: true,
        },
        {
          engine: 'seedvr2' as const,
          display_name: 'SeedVR2',
          available: true,
        },
      ];
  const engineReasonMap = new Map(engineOptionsRaw.map((entry) => [entry.engine, String(entry.reason || '')]));
  const availableEngines = engineOptionsRaw.filter((entry) => entry.available);
  const selectedEngineAvailable = availableEngines.some((entry) => entry.engine === upscaleConfig.engine);
  const fallbackEngine = availableEngines.length > 0 ? availableEngines[0].engine : 'realesrgan';
  const effectiveEngine = selectedEngineAvailable ? upscaleConfig.engine : fallbackEngine;
  const effectiveEngineAvailable = !!engineOptionsRaw.find((entry) => entry.engine === effectiveEngine)?.available;
  const effectiveEngineReason = engineReasonMap.get(effectiveEngine) || '';
  const localizeSeedVRText = (reason: string): string => {
    const text = String(reason || '');
    const lower = text.toLowerCase();
    if (lower.includes('real-esrgan runtime') || lower.includes('realesrgan runtime')) {
      return t('upscale.realesrgan.runtimeMissing');
    }
    if (lower.includes('unsupported real-esrgan model_id')) {
      return t('upscale.realesrgan.invalidModel');
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
    if (lower.includes('python 3.12') || lower.includes('seedvr runtime')) return t('upscale.seedvr.runtimeMissing');
    if (lower.includes('requires at least') && lower.includes('memory')) return t('upscale.seedvr.lowMemory');
    if (lower.includes('memory guard triggered') || lower.includes('out of memory')) return t('upscale.seedvr.memoryGuard');
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
    if (lower.includes('auto-preprocessed input to 720p')) return t('upscale.seedvr.auto720');
    if (lower.includes('capped internal target short side')) return t('upscale.seedvr.auto1080cap');
    if (lower.includes('applied low-memory profile')) return t('upscale.seedvr.memoryPolicyApplied');
    if (lower.includes('applied mps-first execution policy')) return t('upscale.seedvr.policy.mpsFirst');
    if (lower.includes('emergency low-memory profile') || lower.includes('memory pressure detected')) {
      return t('upscale.seedvr.retryEmergency');
    }
    if (lower.includes('scene split fallback') || lower.includes('scene split review fallback')) {
      return t('upscale.seedvr.sceneFallback');
    }
    if (lower.includes('concat copy failed')) return t('upscale.seedvr.concatFallback');
    if (lower.includes('load governor active') || lower.includes('throttling to 80%')) {
      return t('upscale.seedvr.loadGovernor');
    }
    return text;
  };
  const localizeSeedVRMultiText = (reason: string): string => (
    String(reason || '')
      .split(';')
      .map((entry) => localizeSeedVRText(entry.trim()))
      .filter(Boolean)
      .join('；')
  );
  const effectiveEngineReasonText = localizeSeedVRText(effectiveEngineReason);
  const localizedTaskError = localizeSeedVRText(upscaleTask.error);
  const localizedTaskWarning = localizeSeedVRMultiText(upscaleTask.warning);
  const getLocalizedModelName = (modelId: string, fallback: string): string => {
    if (modelId === 'realesrgan_general_x4v3') return t('upscale.realesrgan.model.general');
    if (modelId === 'realesrgan_x2plus') return t('upscale.realesrgan.model.x2plus');
    if (modelId === 'seedvr2_3b_q8_0_gguf') return t('upscale.seedvr.model.q8');
    if (modelId === 'seedvr2_3b_q4_k_m_gguf') return t('upscale.seedvr.model.q4');
    return fallback;
  };

  const installedMap = useMemo(() => {
    const next = new Map<UpscaleConfig['modelId'], boolean>();
    for (const entry of upscaleModelDownloads) {
      next.set(entry.model_id, !!entry.installed);
    }
    return next;
  }, [upscaleModelDownloads]);

  const modelOptions = (upscaleCapabilities?.models || [])
    .filter((model) => model.engine === effectiveEngine)
    .map((model) => ({
      value: model.model_id,
      label: (
        (installedMap.get(model.model_id) ?? !!model.installed)
          ? getLocalizedModelName(model.model_id, model.display_name)
          : `${getLocalizedModelName(model.model_id, model.display_name)} (${t('upscale.modelNotInstalled')})`
      ),
      installed: installedMap.get(model.model_id) ?? !!model.installed,
    }));

  const selectedModelValid = modelOptions.some((entry) => entry.value === upscaleConfig.modelId);
  const effectiveModel = selectedModelValid
    ? upscaleConfig.modelId
    : (modelOptions[0]?.value || 'realesrgan_general_x4v3');
  const effectiveModelInstalled = modelOptions.find((entry) => entry.value === effectiveModel)?.installed ?? false;
  const effectiveModelDownloadEntry = upscaleModelDownloads.find((entry) => entry.model_id === effectiveModel);
  const runningModelDownload = upscaleDownloadTask.state === 'running';
  const runningCurrentModelDownload = runningModelDownload && upscaleDownloadTask.model_id === effectiveModel;
  const allowRedownload = effectiveModelDownloadEntry?.can_redownload !== false;
  const downloadForce = effectiveModelInstalled && allowRedownload;

  const startDisabled = !upscaleConfig.enabled
    || !hasResult
    || upscaleTask.state === 'running'
    || runningModelDownload
    || !effectiveEngineAvailable
    || !effectiveModelInstalled;

  useEffect(() => {
    if (upscaleConfig.engine !== effectiveEngine) {
      onUpscaleConfigChange({ engine: effectiveEngine });
    }
  }, [effectiveEngine, onUpscaleConfigChange, upscaleConfig.engine]);

  useEffect(() => {
    if (upscaleConfig.modelId !== effectiveModel) {
      onUpscaleConfigChange({ modelId: effectiveModel });
    }
  }, [effectiveModel, onUpscaleConfigChange, upscaleConfig.modelId]);

  return (
    <Card
      className="result-card"
      title={t('result.title')}
      headerExtraContent={(
        <Space>
          <Button onClick={onOpenOutputDir}>{t('common.openOutput')}</Button>
        </Space>
      )}
    >
      <Text type="tertiary">{t('result.subtitle')}</Text>
      <div className="result-layout">
        <div className="result-left-pane">
          {!hasResult && (
            <div className="result-empty">
              <Title heading={5}>{t('result.empty')}</Title>
            </div>
          )}

          {hasResult && (
            <>
              <Space className="result-meta-tags" wrap>
                <Tag>{`${t('common.fileName')}: ${result.outputPath.split('/').pop()}`}</Tag>
                <Tag>{`${t('common.resolution')}: ${result.width}×${result.height}`}</Tag>
                <Tag>{`${t('common.fps')}: ${result.fps ? result.fps.toFixed(2) : '-'}`}</Tag>
                <Tag>{`${t('common.frames')}: ${result.frameCount}`}</Tag>
                <Tag>{`${t('result.model')}: ${modelLabel}`}</Tag>
              </Space>

              <FrameComparePreview
                outputPath={result.outputPath}
                outputUrlFallback={result.outputUrl}
                sourcePath={sourceVideoPath}
                isVideo={isVideo}
                fpsHint={result.fps}
                frameCountHint={result.frameCount}
                widthHint={result.width}
                heightHint={result.height}
                beforeLabel={t('result.compare.before')}
                afterLabel={t('result.compare.after')}
                compareUnavailableText={t('result.compare.unavailable')}
              />
            </>
          )}
        </div>

        <div className="result-right-pane">
          <div className="result-upscale-panel">
            <Title heading={6}>{t('upscale.panelTitle')}</Title>
            <div className="result-upscale-row">
              <Text>{t('upscale.enable')}</Text>
              <Switch checked={upscaleConfig.enabled} onChange={(value) => onUpscaleConfigChange({ enabled: !!value })} />
            </div>
            <div className="result-upscale-field">
              <Text className="process-field-label">{t('upscale.mode')}</Text>
              <Select
                value={upscaleConfig.mode}
                onChange={(value) => onUpscaleConfigChange({ mode: String(value) as UpscaleConfig['mode'] })}
                optionList={[
                  { value: 'upscale_resolution', label: t('upscale.mode.upscale') },
                  { value: 'enhance_same_resolution', label: t('upscale.mode.sameRes') },
                ]}
                disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
              />
            </div>
            <div className="result-upscale-field">
              <Text className="process-field-label">{t('upscale.engine')}</Text>
              <Select
                value={effectiveEngine}
                onChange={(value) => onUpscaleConfigChange({ engine: String(value) as UpscaleConfig['engine'] })}
                optionList={engineOptionsRaw.map((entry) => ({
                  value: entry.engine,
                  label: entry.available ? entry.display_name : `${entry.display_name} (${t('upscale.unavailable')})`,
                  disabled: !entry.available,
                }))}
                disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
              />
              {!effectiveEngineAvailable && (
                <Text type="tertiary">{effectiveEngineReasonText || t('upscale.engineUnavailableHint')}</Text>
              )}
            </div>
            <div className="result-upscale-field">
              <Text className="process-field-label">{t('upscale.model')}</Text>
              <Select
                value={effectiveModel}
                onChange={(value) => onUpscaleConfigChange({ modelId: String(value) as UpscaleConfig['modelId'] })}
                optionList={modelOptions}
                disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
              />
            </div>
            <div className="result-upscale-model-state">
              <Space wrap>
                <Tag color={effectiveModelInstalled ? 'green' : 'orange'}>
                  {effectiveModelInstalled ? t('upscale.modelInstalled') : t('upscale.modelNotInstalled')}
                </Tag>
                <Button
                  theme="light"
                  disabled={runningModelDownload && !runningCurrentModelDownload}
                  loading={runningCurrentModelDownload}
                  onClick={() => onStartUpscaleModelDownload(effectiveModel, downloadForce)}
                >
                  {runningCurrentModelDownload
                    ? t('upscale.modelDownloading')
                    : (downloadForce ? t('upscale.modelRedownload') : t('upscale.modelDownload'))}
                </Button>
                <Button
                  theme="light"
                  type="danger"
                  disabled={!runningModelDownload}
                  onClick={onCancelUpscaleModelDownload}
                >
                  {t('settings.modelDownload.cancel')}
                </Button>
              </Space>
              {!effectiveModelInstalled && (
                <Text type="tertiary">{t('upscale.modelNotReadyHint')}</Text>
              )}
            </div>
            {upscaleConfig.mode === 'upscale_resolution' ? (
              <div className="result-upscale-field">
                <Text className="process-field-label">{t('upscale.target')}</Text>
                <Select
                  value={upscaleConfig.targetPreset}
                  onChange={(value) => onUpscaleConfigChange({ targetPreset: String(value) as UpscaleConfig['targetPreset'] })}
                  optionList={[
                    { value: '1080p', label: '1080p' },
                  ]}
                  disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
                />
              </div>
            ) : (
              <div className="result-upscale-field">
                <Text className="process-field-label">{t('upscale.sameResStrength')}</Text>
                <Select
                  value={upscaleConfig.sameResStrength}
                  onChange={(value) => onUpscaleConfigChange({ sameResStrength: String(value) as UpscaleConfig['sameResStrength'] })}
                  optionList={[
                    { value: 'x2_then_downscale', label: t('upscale.sameRes.x2') },
                  ]}
                  disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
                />
              </div>
            )}
            <div className="result-upscale-field">
              <Text className="process-field-label">{`${t('upscale.denoise')}: ${upscaleConfig.denoiseStrength.toFixed(2)}`}</Text>
              <Slider
                min={0}
                max={1}
                step={0.05}
                value={upscaleConfig.denoiseStrength}
                onChange={(value) => onUpscaleConfigChange({ denoiseStrength: Number(value) })}
                disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
              />
            </div>
            <div className="result-upscale-row">
              <Text>{t('upscale.keepAudio')}</Text>
              <Switch
                checked={upscaleConfig.keepAudio}
                onChange={(value) => onUpscaleConfigChange({ keepAudio: !!value })}
                disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
              />
            </div>

            <Space className="result-upscale-actions">
              <Button type="primary" disabled={startDisabled} onClick={onStartUpscale}>
                {t('upscale.start')}
              </Button>
              <Button
                theme="light"
                type="danger"
                disabled={upscaleTask.state !== 'running'}
                onClick={onCancelUpscale}
              >
                {t('upscale.cancel')}
              </Button>
              <Button
                disabled={upscaleTask.state !== 'success' || !upscaleTask.outputPath}
                onClick={onOpenUpscaleResult}
              >
                {t('upscale.openResult')}
              </Button>
            </Space>

            <div className="result-upscale-progress">
              <Progress percent={Math.round(upscaleTask.progress * 100)} showInfo />
              <Text type="tertiary">{upscaleTask.message || t('upscale.status.idle')}</Text>
              {upscaleTask.state === 'running' && (upscaleTask.segmentTotal || 0) > 0 && (
                <Text type="tertiary">{`${t('upscale.segmentProgress')}: ${upscaleTask.segmentIndex || 0}/${upscaleTask.segmentTotal || 0}`}</Text>
              )}
              {typeof upscaleTask.etaSeconds === 'number' && upscaleTask.state === 'running' && (
                <Text type="tertiary">{`${t('process.etaLabel')}: ${Math.ceil(upscaleTask.etaSeconds)}s`}</Text>
              )}
              {upscaleTask.warning && <Text type="tertiary">{localizedTaskWarning}</Text>}
              {upscaleTask.error && <Text type="danger">{localizedTaskError}</Text>}
            </div>
            {(runningModelDownload || upscaleDownloadTask.state === 'failed' || upscaleDownloadTask.state === 'cancelled') && (
              <div className="result-upscale-download-progress">
                <Progress percent={Math.round((upscaleDownloadTask.progress || 0) * 100)} showInfo />
                <Text type="tertiary">{upscaleDownloadTask.message}</Text>
                {upscaleDownloadTask.current_file && (
                  <Text type="tertiary">
                    {`${t('settings.modelDownload.currentFile')}: ${upscaleDownloadTask.current_file}`}
                  </Text>
                )}
                {upscaleDownloadTask.error && <Text type="danger">{upscaleDownloadTask.error}</Text>}
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
