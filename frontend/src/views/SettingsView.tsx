// 设置侧栏视图：
// 语言/主题/输出目录，以及模型下载管理。
import { Button, Input, Progress, Radio, Select, Space, Typography } from '@douyinfe/semi-ui';
import type { AppSettings } from '../types/app';
import type { ModelDownloadEntry, ModelDownloadTask } from '../services/pywebview';
import { useI18n } from '../i18n/useI18n';

const { Text } = Typography;

function normalizeLanguageValue(
  value: unknown,
  fallback: AppSettings['language'],
): AppSettings['language'] {
  // 兼容 Select 可能返回的多种值结构。
  if (value === 'zh' || value === 'en') return value;

  if (Array.isArray(value) && value.length > 0) {
    return normalizeLanguageValue(value[0], fallback);
  }

  if (value && typeof value === 'object' && 'value' in value) {
    const optionValue = (value as { value?: unknown }).value;
    return normalizeLanguageValue(optionValue, fallback);
  }

  return fallback;
}

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
        <Text type="tertiary" className="settings-subtitle">{t('settings.subtitle')}</Text>
      </div>

      <div className="settings-body">
        <div className="settings-field settings-field-card">
          <Text className="settings-label">{t('settings.language')}</Text>
          <Select
            value={settings.language}
            onChange={(value) => onChangeLanguage(normalizeLanguageValue(value, settings.language))}
            optionList={[
              { value: 'zh', label: t('settings.language.zh') },
              { value: 'en', label: t('settings.language.en') },
            ]}
          />
        </div>

        <div className="settings-field settings-field-card">
          <Text className="settings-label">{t('settings.theme')}</Text>
          <Radio.Group
            type="button"
            value={settings.theme}
            onChange={(event) => onChangeTheme(String(event.target.value) as AppSettings['theme'])}
          >
            <Radio value="light">{t('settings.theme.light')}</Radio>
            <Radio value="dark">{t('settings.theme.dark')}</Radio>
          </Radio.Group>
        </div>

        <div className="settings-field settings-field-card">
          <Text className="settings-label">{t('settings.outputPath')}</Text>
          <Space className="settings-output-row">
            <Input
              value={settings.outputPath}
              onChange={(value) => onChangeOutputPath(String(value))}
              placeholder={t('settings.outputPath')}
            />
            <Button onClick={onSelectOutputFolder}>{t('common.browse')}</Button>
          </Space>
        </div>

        <div className="settings-field settings-field-card">
          <Text className="settings-label">{t('settings.modelDownload.title')}</Text>

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
                    <Text strong>{model.display_name}</Text>
                    <Text type="tertiary">{model.install_hint}</Text>
                    <Text type={model.installed ? 'success' : 'tertiary'}>
                      {model.installed
                        ? t('settings.modelDownload.installed')
                        : t('settings.modelDownload.notInstalled')}
                    </Text>
                  </div>

                  <Button
                    type="primary"
                    theme={isRedownload ? 'light' : 'solid'}
                    loading={runningThisModel}
                    disabled={disableAction}
                    onClick={() => onStartModelDownload(modelId, isRedownload)}
                  >
                    {actionLabel}
                  </Button>
                </div>
              );
            })}
          </div>

          {downloadTask.state !== 'idle' ? (
            <div className="settings-model-download-progress">
              <div className="settings-model-download-progress-head">
                <Text strong>{t('settings.modelDownload.progress')}</Text>
                <Text>{downloadStateLabelMap[downloadTask.state]}</Text>
              </div>

              <Progress
                percent={Math.round(Math.max(0, Math.min(1, downloadTask.progress)) * 100)}
                showInfo
              />

              <div className="settings-model-download-stats">
                <Text type="tertiary">
                  {t('settings.modelDownload.speed')}: {formatSpeed(downloadTask.speed_bps)}
                </Text>
                <Text type="tertiary">
                  {t('settings.modelDownload.downloaded')}: {totalText}
                </Text>
                <Text type="tertiary" ellipsis={{ showTooltip: true }}>
                  {t('settings.modelDownload.currentFile')}: {downloadTask.current_file || '--'}
                </Text>
              </div>

              {downloadTask.message ? (
                <Text type="tertiary">{downloadTask.message}</Text>
              ) : null}
              {downloadTask.error ? (
                <Text type="danger">{downloadTask.error}</Text>
              ) : null}

              {isDownloadRunning ? (
                <Button type="danger" theme="borderless" onClick={onCancelModelDownload}>
                  {t('settings.modelDownload.cancel')}
                </Button>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>

      <div className="settings-actions-wrap">
        <Space className="settings-actions" wrap>
          <Button theme="solid" type="primary" loading={saving} onClick={onSave}>
            {t('settings.save')}
          </Button>
          <Button onClick={onReset}>{t('settings.reset')}</Button>
        </Space>
      </div>
    </div>
  );
}
