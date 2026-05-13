import { useEffect, useMemo, useState } from 'react';
import type { ResultState, UpscaleCapabilities, UpscaleConfig, UpscaleTaskState } from '../types/app';
import { useI18n } from '../i18n/useI18n';
import type { UpscaleModelDownloadEntry, UpscaleModelDownloadTask } from '../services/desktop';
import { FrameComparePreview } from '../components/FrameComparePreview';
import {
  MdButton,
  MdChip,
  MdEmptyState,
  MaterialIcon,
  MdLinearProgress,
  MdSelect,
  MdSlider,
  MdSurface,
  MdSwitch,
  MdTaskPanel,
} from '../material';

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
  const [showUpscaleAdvanced, setShowUpscaleAdvanced] = useState(false);
  const isVideo = result.mediaType === 'video';
  const modelLabel = t('process.model.lama');
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
    if (
      lower.includes('real-esrgan runtime') ||
      lower.includes('realesrgan runtime') ||
      lower.includes('realesrgan native') ||
      lower.includes('real-esrgan native')
    ) {
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
    if (
      lower.includes('seedvr runtime') ||
      lower.includes('seedvr native') ||
      lower.includes('gguf runner')
    ) return t('upscale.seedvr.runtimeMissing');
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
    <MdSurface className="result-card">
      <div className="surface-header result-header">
        <div>
          <h2>{t('result.title')}</h2>
          <p>{t('result.subtitle')}</p>
        </div>
        <MdButton variant="outlined" icon="folder_open" onClick={onOpenOutputDir}>
          {t('common.openOutput')}
        </MdButton>
      </div>
      <div className="result-layout">
        <div className={`result-left-pane ${hasResult ? 'has-result' : 'is-empty'}`}>
          {!hasResult && (
            <div className="result-empty">
              <MdEmptyState
                icon="movie_filter"
                title={t('result.empty')}
                description={t('result.subtitle')}
              />
            </div>
          )}

          {hasResult && (
            <>
              <div className="result-meta-tags">
                <MdChip>{`${t('common.fileName')}: ${result.outputPath.split('/').pop()}`}</MdChip>
                <MdChip>{`${t('common.resolution')}: ${result.width}×${result.height}`}</MdChip>
                <MdChip>{`${t('common.fps')}: ${result.fps ? result.fps.toFixed(2) : '-'}`}</MdChip>
                <MdChip>{`${t('common.frames')}: ${result.frameCount}`}</MdChip>
                <MdChip>{`${t('result.model')}: ${modelLabel}`}</MdChip>
              </div>

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

        <div className="result-right-pane action-pane">
          <div className="result-upscale-panel">
            <h3>{t('upscale.panelTitle')}</h3>
            <div className="result-upscale-row">
              <span>{t('upscale.enable')}</span>
              <MdSwitch
                checked={upscaleConfig.enabled}
                label={t('upscale.enable')}
                onChange={(value) => onUpscaleConfigChange({ enabled: value })}
              />
            </div>
            <section className="result-upscale-config-card">
              <div className="result-upscale-section-head">
                <span className="result-upscale-section-icon">
                  <MaterialIcon name="tune" />
                </span>
                <h4>{t('upscale.basicConfig')}</h4>
              </div>

              <div className="result-upscale-form-grid">
                <div className="result-upscale-field">
                  <span className="process-field-label">{t('upscale.mode')}</span>
                  <MdSelect
                    value={upscaleConfig.mode}
                    onChange={(value) => onUpscaleConfigChange({ mode: value })}
                    options={[
                      { value: 'upscale_resolution', label: t('upscale.mode.upscale') },
                      { value: 'enhance_same_resolution', label: t('upscale.mode.sameRes') },
                    ]}
                    disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
                  />
                </div>

                <div className="result-upscale-field">
                  <span className="process-field-label">{t('upscale.engine')}</span>
                  <MdSelect
                    value={effectiveEngine}
                    onChange={(value) => onUpscaleConfigChange({ engine: value })}
                    options={engineOptionsRaw.map((entry) => ({
                      value: entry.engine,
                      label: entry.available ? entry.display_name : `${entry.display_name} (${t('upscale.unavailable')})`,
                      disabled: !entry.available,
                    }))}
                    disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
                  />
                </div>

                {!effectiveEngineAvailable && (
                  <div className="result-upscale-inline-status">
                    <MaterialIcon name="info" />
                    <span>{effectiveEngineReasonText || t('upscale.engineUnavailableHint')}</span>
                  </div>
                )}

              <div className="result-upscale-field">
                <span className="process-field-label">{t('upscale.model')}</span>
                <MdSelect
                  value={effectiveModel}
                  onChange={(value) => onUpscaleConfigChange({ modelId: value })}
                  options={modelOptions}
                  disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
                />
              </div>
              </div>
            </section>
            <div className="result-upscale-model-state">
              <div className="button-row wrap">
                <MdChip tone={effectiveModelInstalled ? 'success' : 'warning'}>
                  {effectiveModelInstalled ? t('upscale.modelInstalled') : t('upscale.modelNotInstalled')}
                </MdChip>
                <MdButton
                  variant="tonal"
                  icon="download"
                  disabled={runningModelDownload && !runningCurrentModelDownload}
                  loading={runningCurrentModelDownload}
                  onClick={() => onStartUpscaleModelDownload(effectiveModel, downloadForce)}
                >
                  {runningCurrentModelDownload
                    ? t('upscale.modelDownloading')
                    : (downloadForce ? t('upscale.modelRedownload') : t('upscale.modelDownload'))}
                </MdButton>
                <MdButton
                  variant="outlined"
                  tone="danger"
                  icon="cancel"
                  disabled={!runningModelDownload}
                  onClick={onCancelUpscaleModelDownload}
                >
                  {t('settings.modelDownload.cancel')}
                </MdButton>
              </div>
              {!effectiveModelInstalled && (
                <p className="metadata-text">{t('upscale.modelNotReadyHint')}</p>
              )}
            </div>
            {upscaleConfig.enabled ? (
              <button
                type="button"
                className="compact-disclosure"
                aria-expanded={showUpscaleAdvanced}
                onClick={() => setShowUpscaleAdvanced((value) => !value)}
              >
                <span>{t('upscale.target')}</span>
                <span className="metadata-text">
                  {upscaleConfig.mode === 'upscale_resolution'
                    ? upscaleConfig.targetPreset
                    : t('upscale.sameRes.x2')}
                </span>
                <MaterialIcon name={showUpscaleAdvanced ? 'expand_less' : 'expand_more'} />
              </button>
            ) : null}

            {upscaleConfig.enabled && showUpscaleAdvanced ? (
              <MdTaskPanel icon="instant_mix" title={t('upscale.target')} className="result-upscale-advanced">
                {upscaleConfig.mode === 'upscale_resolution' ? (
                  <div className="result-upscale-field">
                    <MdSelect
                      value={upscaleConfig.targetPreset}
                      onChange={(value) => onUpscaleConfigChange({ targetPreset: value })}
                      options={[
                        { value: '1080p', label: '1080p' },
                      ]}
                      disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
                    />
                  </div>
                ) : (
                  <div className="result-upscale-field">
                    <span className="process-field-label">{t('upscale.sameResStrength')}</span>
                    <MdSelect
                      value={upscaleConfig.sameResStrength}
                      onChange={(value) => onUpscaleConfigChange({ sameResStrength: value })}
                      options={[
                        { value: 'x2_then_downscale', label: t('upscale.sameRes.x2') },
                      ]}
                      disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
                    />
                  </div>
                )}
                <div className="result-upscale-field">
                  <span className="process-field-label">{`${t('upscale.denoise')}: ${upscaleConfig.denoiseStrength.toFixed(2)}`}</span>
                  <MdSlider
                    min={0}
                    max={1}
                    step={0.05}
                    value={upscaleConfig.denoiseStrength}
                    onChange={(value) => onUpscaleConfigChange({ denoiseStrength: Number(value) })}
                    disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
                  />
                </div>
                <div className="result-upscale-row">
                  <span>{t('upscale.keepAudio')}</span>
                  <MdSwitch
                    checked={upscaleConfig.keepAudio}
                    label={t('upscale.keepAudio')}
                    onChange={(value) => onUpscaleConfigChange({ keepAudio: value })}
                    disabled={!upscaleConfig.enabled || upscaleTask.state === 'running'}
                  />
                </div>
              </MdTaskPanel>
            ) : null}

            {upscaleConfig.enabled ? (
              <div className="result-upscale-actions">
                <MdButton variant="filled" icon="auto_awesome" disabled={startDisabled} onClick={onStartUpscale}>
                  {t('upscale.start')}
                </MdButton>
                <MdButton
                  variant="outlined"
                  tone="danger"
                  icon="cancel"
                  disabled={upscaleTask.state !== 'running'}
                  onClick={onCancelUpscale}
                >
                  {t('upscale.cancel')}
                </MdButton>
                <MdButton
                  variant="tonal"
                  icon="open_in_new"
                  disabled={upscaleTask.state !== 'success' || !upscaleTask.outputPath}
                  onClick={onOpenUpscaleResult}
                >
                  {t('upscale.openResult')}
                </MdButton>
              </div>
            ) : null}

            {upscaleConfig.enabled || upscaleTask.state !== 'idle' ? (
              <div className="result-upscale-progress">
                <div className="progress-with-value">
                  <MdLinearProgress value={upscaleTask.progress} />
                  <span>{Math.round(upscaleTask.progress * 100)}%</span>
                </div>
                <p className="metadata-text">{upscaleTask.message || t('upscale.status.idle')}</p>
                {upscaleTask.state === 'running' && (upscaleTask.segmentTotal || 0) > 0 && (
                  <p className="metadata-text">{`${t('upscale.segmentProgress')}: ${upscaleTask.segmentIndex || 0}/${upscaleTask.segmentTotal || 0}`}</p>
                )}
                {typeof upscaleTask.etaSeconds === 'number' && upscaleTask.state === 'running' && (
                  <p className="metadata-text">{`${t('process.etaLabel')}: ${Math.ceil(upscaleTask.etaSeconds)}s`}</p>
                )}
                {upscaleTask.warning && <p className="metadata-text">{localizedTaskWarning}</p>}
                {upscaleTask.error && <p className="form-error">{localizedTaskError}</p>}
              </div>
            ) : null}
            {(runningModelDownload || upscaleDownloadTask.state === 'failed' || upscaleDownloadTask.state === 'cancelled') && (
              <div className="result-upscale-download-progress">
                <div className="progress-with-value">
                  <MdLinearProgress value={upscaleDownloadTask.progress || 0} />
                  <span>{Math.round((upscaleDownloadTask.progress || 0) * 100)}%</span>
                </div>
                <p className="metadata-text">{upscaleDownloadTask.message}</p>
                {upscaleDownloadTask.current_file && (
                  <p className="metadata-text">
                    {`${t('settings.modelDownload.currentFile')}: ${upscaleDownloadTask.current_file}`}
                  </p>
                )}
                {upscaleDownloadTask.error && <p className="form-error">{upscaleDownloadTask.error}</p>}
              </div>
            )}
          </div>
        </div>
      </div>
    </MdSurface>
  );
}
