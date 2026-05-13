// 设置侧栏视图：
// 语言/主题/输出目录，以及模型下载管理。
import type { AppSettings } from '../types/app';
import type { ModelDownloadEntry, ModelDownloadTask } from '../services/desktop';
import { useI18n } from '../i18n/useI18n';
import { MdButton, MdChip, MdLinearProgress, MdSelect, MdTextField } from '../material';
import { useState } from 'react';

interface SettingsViewProps {
  settings: AppSettings;
  saving: boolean;
  modelDownloads: ModelDownloadEntry[];
  downloadTask: ModelDownloadTask;
  onChangeLanguage: (language: AppSettings['language']) => void;
  onChangeTheme: (theme: AppSettings['theme']) => void;
  onChangeOutputPath: (path: string) => void;
  onSelectOutputFolder: () => void;
  onStartModelDownload: (modelId: AppSettings['modelId'], force: boolean) => void;
  onCancelModelDownload: () => void;
  onSave: () => void;
  onReset: () => void;
}

function formatBytes(raw: number): string {
  // 把字节数转换成易读单位。
  const value = Math.max(0, Number(raw) || 0);
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let unitIndex = 0;
  let n = value;
  while (n >= 1024 && unitIndex < units.length - 1) {
    n /= 1024;
    unitIndex += 1;
  }
  return `${n.toFixed(unitIndex === 0 ? 0 : 1)}${units[unitIndex]}`;
}

function formatSpeed(raw: number): string {
  // 下载速度统一显示为 MB/s。
  const speed = Math.max(0, Number(raw) || 0);
  return `${(speed / (1024 * 1024)).toFixed(2)} MB/s`;
}

type SettingsSection = 'general' | 'output' | 'models';

export function SettingsView({
  settings,
  saving,
  modelDownloads,
  downloadTask,
  onChangeLanguage,
  onChangeTheme,
  onChangeOutputPath,
  onSelectOutputFolder,
  onStartModelDownload,
  onCancelModelDownload,
  onSave,
  onReset,
}: SettingsViewProps) {
  const { t } = useI18n();
  const isDownloadRunning = downloadTask.state === 'running';
  const [activeSection, setActiveSection] = useState<SettingsSection>('general');
  const sectionTabs: Array<{ key: SettingsSection; label: string }> = [
    { key: 'general', label: t('settings.section.general') },
    { key: 'output', label: t('settings.section.output') },
    { key: 'models', label: t('settings.section.models') },
  ];

  const downloadStateLabelMap: Record<ModelDownloadTask['state'], string> = {
    idle: t('settings.modelDownload.status.idle'),
    running: t('settings.modelDownload.status.running'),
    success: t('settings.modelDownload.status.success'),
    failed: t('settings.modelDownload.status.failed'),
    cancelled: t('settings.modelDownload.status.cancelled'),
  };

  const totalText = downloadTask.total_bytes > 0
    ? `${formatBytes(downloadTask.downloaded_bytes)} / ${formatBytes(downloadTask.total_bytes)}`
    : formatBytes(downloadTask.downloaded_bytes);

  return (
    // 上方介绍 + 中间字段区 + 底部保存/重置操作。
    <div className="settings-panel">
      <div className="settings-panel-intro">
        <p className="settings-subtitle">{t('settings.subtitle')}</p>
      </div>

      <div className="settings-tabs" role="tablist" aria-label={t('settings.title')}>
        {sectionTabs.map((item) => (
          <button
            key={item.key}
            type="button"
            role="tab"
            aria-selected={activeSection === item.key}
            className={activeSection === item.key ? 'is-selected' : ''}
            onClick={() => setActiveSection(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="settings-body">
        {activeSection === 'general' ? (
          <div className="settings-section-page" role="tabpanel">
            <div className="settings-field settings-field-card">
              <span className="settings-label">{t('settings.language')}</span>
              <MdSelect
                value={settings.language}
                onChange={onChangeLanguage}
                options={[
                  { value: 'zh', label: t('settings.language.zh') },
                  { value: 'en', label: t('settings.language.en') },
                ]}
              />
            </div>

            <div className="settings-field settings-field-card">
              <span className="settings-label">{t('settings.theme')}</span>
              <div className="md-segmented-control" role="radiogroup" aria-label={t('settings.theme')}>
                <button
                  type="button"
                  className={settings.theme === 'light' ? 'is-selected' : ''}
                  aria-checked={settings.theme === 'light'}
                  role="radio"
                  onClick={() => onChangeTheme('light')}
                >
                  {t('settings.theme.light')}
                </button>
                <button
                  type="button"
                  className={settings.theme === 'dark' ? 'is-selected' : ''}
                  aria-checked={settings.theme === 'dark'}
                  role="radio"
                  onClick={() => onChangeTheme('dark')}
                >
                  {t('settings.theme.dark')}
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {activeSection === 'output' ? (
          <div className="settings-section-page" role="tabpanel">
            <div className="settings-field settings-field-card">
              <span className="settings-label">{t('settings.outputPath')}</span>
              <div className="settings-output-row">
                <MdTextField
                  value={settings.outputPath}
                  onChange={onChangeOutputPath}
                  placeholder={t('settings.outputPath')}
                />
                <MdButton variant="outlined" icon="folder_open" onClick={onSelectOutputFolder}>
                  {t('common.browse')}
                </MdButton>
              </div>
            </div>
          </div>
        ) : null}

        {activeSection === 'models' ? (
          <div className="settings-section-page" role="tabpanel">
            <div className="settings-field settings-field-card">
              <span className="settings-label">{t('settings.modelDownload.title')}</span>

              <div className="settings-model-download-list">
                {modelDownloads.map((model) => {
                  const modelId = model.model_id as AppSettings['modelId'];
                  const runningThisModel = isDownloadRunning && downloadTask.model_id === modelId;
                  const disableAction = isDownloadRunning && !runningThisModel;
                  const isRedownload = !!model.installed;
                  const actionLabel = isRedownload
                    ? t('settings.modelDownload.redownload')
                    : t('settings.modelDownload.download');

                  return (
                    <div className="settings-model-download-row" key={model.model_id}>
                      <div className="settings-model-download-meta">
                        <strong>{model.display_name}</strong>
                        <span className="metadata-text">{model.install_hint}</span>
                        <MdChip tone={model.installed ? 'success' : 'neutral'}>
                          {model.installed
                            ? t('settings.modelDownload.installed')
                            : t('settings.modelDownload.notInstalled')}
                        </MdChip>
                      </div>

                      <MdButton
                        variant={isRedownload ? 'tonal' : 'filled'}
                        icon="download"
                        loading={runningThisModel}
                        disabled={disableAction}
                        onClick={() => onStartModelDownload(modelId, isRedownload)}
                      >
                        {actionLabel}
                      </MdButton>
                    </div>
                  );
                })}
              </div>

              {downloadTask.state !== 'idle' ? (
                <div className="settings-model-download-progress">
                  <div className="settings-model-download-progress-head">
                    <strong>{t('settings.modelDownload.progress')}</strong>
                    <span>{downloadStateLabelMap[downloadTask.state]}</span>
                  </div>

                  <div className="progress-with-value">
                    <MdLinearProgress value={downloadTask.progress} />
                    <span>{Math.round(Math.max(0, Math.min(1, downloadTask.progress)) * 100)}%</span>
                  </div>

                  <div className="settings-model-download-stats">
                    <span className="metadata-text">
                      {t('settings.modelDownload.speed')}: {formatSpeed(downloadTask.speed_bps)}
                    </span>
                    <span className="metadata-text">
                      {t('settings.modelDownload.downloaded')}: {totalText}
                    </span>
                    <span className="metadata-text" title={downloadTask.current_file || '--'}>
                      {t('settings.modelDownload.currentFile')}: {downloadTask.current_file || '--'}
                    </span>
                  </div>

                  {downloadTask.message ? (
                    <p className="metadata-text">{downloadTask.message}</p>
                  ) : null}
                  {downloadTask.error ? (
                    <p className="form-error">{downloadTask.error}</p>
                  ) : null}

                  {isDownloadRunning ? (
                    <MdButton variant="text" tone="danger" icon="cancel" onClick={onCancelModelDownload}>
                      {t('settings.modelDownload.cancel')}
                    </MdButton>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>

      <div className="settings-actions-wrap">
        <div className="settings-actions">
          <MdButton variant="filled" icon="save" loading={saving} onClick={onSave}>
            {t('settings.save')}
          </MdButton>
          <MdButton variant="outlined" icon="restart_alt" onClick={onReset}>
            {t('settings.reset')}
          </MdButton>
        </div>
      </div>
    </div>
  );
}
